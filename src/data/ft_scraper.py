"""FT.com sitemap-based news scraper for training pipeline.

Architecture:
    1. Parse index sitemap → discover monthly archive URLs
    2. Parse monthly archives → extract article URLs + lastmod dates
    3. Filter articles by date range and gold/macro keywords
    4. Fetch individual articles → extract headline + body via DOM xpath
    5. Cache everything to disk to respect rate limits

This is for TRAINING DATA ONLY. For live inference, use the GDELT/RSS pipeline.
FT.com will rate-limit aggressive scraping — use polite delays and caching.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from loguru import logger


# Sitemap XML namespaces
NS_SITEMAP = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
NS_IMAGE = "{http://www.google.com/schemas/sitemap-image/1.1}"

# Gold/macro keywords for filtering relevant articles
GOLD_MACRO_KEYWORDS = [
    # Direct gold
    "gold", "xauusd", "bullion", "precious metal",
    # Central banks / monetary policy
    "federal reserve", "fomc", "interest rate", "rate decision", "rate cut",
    "rate hike", "monetary policy", "quantitative", "tightening", "easing",
    "central bank", "ecb", "boj", "pboc",
    # Inflation / yields
    "inflation", "cpi", "ppi", "treasury yield", "bond yield", "real yield",
    "deflation", "stagflation",
    # Dollar / FX
    "dollar index", "dxy", "dollar strength", "dollar weakness",
    "currency", "forex", "usd",
    # Geopolitical / safe haven
    "safe haven", "geopolitical", "war", "conflict", "sanctions",
    "trade war", "tariff",
    # Economic indicators
    "nonfarm", "payroll", "unemployment", "gdp", "retail sales",
    "manufacturing", "pmi", "consumer confidence",
    # Commodities context
    "commodity", "oil price", "silver", "copper",
]

# Compile keyword patterns for fast matching
_KEYWORD_PATTERNS = [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in GOLD_MACRO_KEYWORDS]


@dataclass
class FTArticle:
    """Parsed FT.com article."""

    url: str
    published_at: str  # ISO format
    headline: str = ""
    summary: str = ""
    body_text: str = ""
    image_caption: str = ""
    fetch_timestamp: str = ""
    relevance_score: float = 0.0  # keyword match density

    @property
    def text_for_embedding(self) -> str:
        """Combined text for FinBERT embedding."""
        parts = [self.headline]
        if self.summary:
            parts.append(self.summary)
        if self.body_text:
            # Use first 500 chars of body to stay within BERT token limits
            parts.append(self.body_text[:500])
        return ". ".join(p for p in parts if p)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "published_at": self.published_at,
            "headline": self.headline,
            "summary": self.summary,
            "body_text": self.body_text[:1000],  # truncate for storage
            "image_caption": self.image_caption,
            "fetch_timestamp": self.fetch_timestamp,
            "relevance_score": self.relevance_score,
        }


def _relevance_score(text: str) -> float:
    """Score text by keyword match density. Higher = more relevant to gold trading."""
    if not text:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for pat in _KEYWORD_PATTERNS if pat.search(text_lower))
    return matches / len(GOLD_MACRO_KEYWORDS)


class FTSitemapScraper:
    """Scrape FT.com articles via sitemap XML feeds.

    Hybrid approach:
        - curl_cffi with browser impersonation for fast sitemap XML crawling
          (bypasses bot detection that blocks plain requests)
        - Playwright CDP with session cookies for paywalled article pages
          (extracts headline + body via JSON-LD structured data, the most
           reliable method since FT.com DOM classes change frequently)

    Requires:
        pip install curl_cffi playwright
        playwright install chromium

    Usage:
        scraper = FTSitemapScraper(cookie_string="FTSession=...; ...")
        articles = await scraper.fetch_articles("2024-01", "2025-03")
        await scraper.close()
    """

    INDEX_URL = "https://www.ft.com/sitemaps/index.xml"

    def __init__(
        self,
        cookie_string: str = "",
        cache_dir: str = "data/ft_cache",
        request_delay: tuple[float, float] = (1.0, 3.0),
        max_retries: int = 3,
        headless: bool = True,
    ):
        self.cookie_string = cookie_string
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "sitemaps").mkdir(exist_ok=True)
        (self.cache_dir / "articles").mkdir(exist_ok=True)
        (self.cache_dir / "processed").mkdir(exist_ok=True)

        self.request_delay = request_delay
        self.max_retries = max_retries
        self.headless = headless

        # curl_cffi session for sitemap XML (fast, no JS needed)
        try:
            from curl_cffi.requests import Session as CurlSession
            self._curl = CurlSession(impersonate="chrome")
            self._use_curl = True
        except ImportError:
            logger.warning(
                "curl_cffi not installed — falling back to requests for sitemaps. "
                "Install for better reliability: pip install curl_cffi"
            )
            self._curl = requests.Session()
            self._curl.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
                ),
            })
            self._use_curl = False

        # Playwright browser (lazy-initialized via setup())
        self._playwright = None
        self._browser = None
        self._context = None
        self._browser_ready = False

    async def setup(self) -> None:
        """Initialize Playwright browser with FT.com cookies.

        Must be called before fetch_articles() / extract_article_content().
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.error(
                "Playwright not installed. Run:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            )
            return

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36 Edg/146.0.0.0"
            )
        )

        # Inject cookies from cookie string
        if self.cookie_string:
            cookie_list = []
            for c in self.cookie_string.split(";"):
                c = c.strip()
                if "=" in c:
                    name, value = c.split("=", 1)
                    cookie_list.append({
                        "name": name.strip(),
                        "value": value.strip(),
                        "domain": ".ft.com",
                        "path": "/",
                    })
            if cookie_list:
                await self._context.add_cookies(cookie_list)

        self._browser_ready = True
        logger.info("Playwright browser + cookies initialized")

    async def close(self) -> None:
        """Shutdown Playwright browser."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._browser_ready = False
        logger.info("Playwright browser closed")

    # ------------------------------------------------------------------
    # Sitemap fetching: curl_cffi (fast, handles bot detection)
    # ------------------------------------------------------------------

    def _polite_delay(self) -> None:
        """Random delay between requests."""
        delay = random.uniform(*self.request_delay)
        time.sleep(delay)

    def _fetch_sitemap_xml(self, url: str) -> Optional[str]:
        """Fetch sitemap XML using curl_cffi with browser impersonation.

        curl_cffi impersonates a real Chrome TLS fingerprint, which bypasses
        Cloudflare and bot detection that blocks plain requests/urllib.
        """
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_path = self.cache_dir / "sitemaps" / f"{cache_key}.xml"

        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        for attempt in range(self.max_retries):
            try:
                self._polite_delay()
                resp = self._curl.get(url, timeout=15)
                if resp.status_code == 200:
                    text = resp.text
                    cache_path.write_text(text, encoding="utf-8")
                    return text
                elif resp.status_code == 429:
                    wait = 30 * (attempt + 1)
                    logger.warning(f"Rate limited on {url}, waiting {wait}s")
                    time.sleep(wait)
                else:
                    logger.warning(f"HTTP {resp.status_code} on sitemap {url}")
            except Exception as e:
                logger.warning(f"Sitemap fetch failed ({attempt+1}/{self.max_retries}): {e}")
                time.sleep(5 * (attempt + 1))

        return None

    # ------------------------------------------------------------------
    # Article fetching: Playwright + JSON-LD (handles paywall + JS)
    # ------------------------------------------------------------------

    async def extract_article_content(self, url: str) -> Optional[dict]:
        """Fetch an article page via Playwright and extract content from JSON-LD.

        JSON-LD (structured data) is the most reliable extraction method:
        - FT.com embeds <script type="application/ld+json"> in every article
        - Contains headline, alternativeHeadline, articleBody, datePublished
        - Immune to DOM class/structure changes
        - Works even behind paywall when cookies are set

        Falls back to og:title / <title> if JSON-LD is missing.
        """
        # Check cache first
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_path = self.cache_dir / "articles" / f"{cache_key}.json"

        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        if not self._browser_ready:
            logger.warning("Browser not initialized — call await scraper.setup() first")
            return None

        import asyncio

        page = await self._context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            # Brief pause for JSON-LD scripts to populate
            await asyncio.sleep(1.5)

            # Primary: extract from JSON-LD structured data
            json_ld = await page.evaluate("""
                () => {
                    const scripts = document.querySelectorAll('script[type="application/ld+json"]');
                    for (let s of scripts) {
                        try {
                            const data = JSON.parse(s.textContent);
                            if (data["@type"] === "NewsArticle" || data["@type"] === "Article") {
                                return {
                                    headline: data.headline || data.alternativeHeadline || "",
                                    summary: data.description || "",
                                    body_text: (data.articleBody || "").substring(0, 1000),
                                    date_published: data.datePublished || "",
                                    date_modified: data.dateModified || "",
                                    author: (data.author && data.author.name) ? data.author.name : "",
                                    section: data.articleSection || ""
                                };
                            }
                            // Handle @graph arrays (some pages nest it)
                            if (Array.isArray(data["@graph"])) {
                                for (let item of data["@graph"]) {
                                    if (item["@type"] === "NewsArticle" || item["@type"] === "Article") {
                                        return {
                                            headline: item.headline || item.alternativeHeadline || "",
                                            summary: item.description || "",
                                            body_text: (item.articleBody || "").substring(0, 1000),
                                            date_published: item.datePublished || "",
                                            date_modified: item.dateModified || "",
                                            author: (item.author && item.author.name) ? item.author.name : "",
                                            section: item.articleSection || ""
                                        };
                                    }
                                }
                            }
                        } catch(e) {}
                    }
                    return null;
                }
            """)

            result = None

            if json_ld and json_ld.get("headline"):
                result = {
                    "headline": json_ld["headline"],
                    "summary": json_ld.get("summary", ""),
                    "body_text": json_ld.get("body_text", ""),
                    "date_published": json_ld.get("date_published", ""),
                }
            else:
                # Fallback: extract from meta tags / DOM
                fallback = await page.evaluate("""
                    () => {
                        const ogTitle = document.querySelector('meta[property="og:title"]');
                        const ogDesc = document.querySelector('meta[property="og:description"]');
                        const titleTag = document.querySelector('title');
                        const h1 = document.querySelector('main h1') || document.querySelector('h1');
                        return {
                            headline: (h1 && h1.textContent.trim()) ||
                                      (ogTitle && ogTitle.content) ||
                                      (titleTag && titleTag.textContent.replace(/\\s*\\|.*$/, '').trim()) ||
                                      "",
                            summary: (ogDesc && ogDesc.content) || "",
                        };
                    }
                """)
                if fallback and fallback.get("headline"):
                    result = {
                        "headline": fallback["headline"],
                        "summary": fallback.get("summary", ""),
                        "body_text": "",
                        "date_published": "",
                    }

            # Cache the result
            if result:
                with open(cache_path, "w") as f:
                    json.dump(result, f)

            return result

        except Exception as e:
            logger.debug(f"Failed to extract {url}: {e}")
            return None
        finally:
            await page.close()

    # ------------------------------------------------------------------
    # Sitemap parsing (same logic, but using curl_cffi for fetching)
    # ------------------------------------------------------------------

    def discover_archives(self) -> list[dict]:
        """Parse the index sitemap to find all monthly archive URLs."""
        xml_text = self._fetch_sitemap_xml(self.INDEX_URL)
        if not xml_text:
            logger.error("Failed to fetch index sitemap")
            return []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"Failed to parse index sitemap: {e}")
            return []

        archives = []
        for sitemap_elem in root.findall(f"{NS_SITEMAP}sitemap"):
            loc = sitemap_elem.findtext(f"{NS_SITEMAP}loc", "")
            lastmod = sitemap_elem.findtext(f"{NS_SITEMAP}lastmod", "")
            if loc:
                archives.append({"url": loc, "lastmod": lastmod})

        logger.info(f"Discovered {len(archives)} monthly archives from index sitemap")
        return archives

    def filter_archives_by_date(
        self,
        archives: list[dict],
        start_month: str = "2020-01",
        end_month: Optional[str] = None,
    ) -> list[dict]:
        """Filter archives to a date range (YYYY-MM format)."""
        if end_month is None:
            end_month = datetime.utcnow().strftime("%Y-%m")

        filtered = []
        for arch in archives:
            match = re.search(r"archive-(\d{4})-(\d{2})", arch["url"])
            if match:
                archive_month = f"{match.group(1)}-{match.group(2)}"
                if start_month <= archive_month <= end_month:
                    filtered.append(arch)

        logger.info(f"Filtered to {len(filtered)} archives in [{start_month} → {end_month}]")
        return filtered

    def parse_archive(self, archive_url: str) -> list[dict]:
        """Parse a monthly archive sitemap to extract article URLs."""
        xml_text = self._fetch_sitemap_xml(archive_url)
        if not xml_text:
            return []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"Failed to parse archive {archive_url}: {e}")
            return []

        articles = []
        for url_elem in root.findall(f"{NS_SITEMAP}url"):
            loc = url_elem.findtext(f"{NS_SITEMAP}loc", "")
            lastmod = url_elem.findtext(f"{NS_SITEMAP}lastmod", "")

            image_caption = ""
            image_elem = url_elem.find(f"{NS_IMAGE}image")
            if image_elem is not None:
                caption_elem = image_elem.find(f"{NS_IMAGE}caption")
                if caption_elem is not None and caption_elem.text:
                    image_caption = caption_elem.text.strip()

            if loc and "/content/" in loc:
                articles.append({
                    "url": loc,
                    "lastmod": lastmod,
                    "image_caption": image_caption,
                })

        logger.info(f"Parsed {len(articles)} article URLs from {archive_url}")
        return articles

    # ------------------------------------------------------------------
    # Full async pipeline
    # ------------------------------------------------------------------

    async def fetch_articles(
        self,
        start_month: str = "2024-01",
        end_month: Optional[str] = None,
        min_relevance: float = 0.01,
        max_articles_per_month: int = 500,
        keywords_prefilter: bool = True,
    ) -> list[FTArticle]:
        """Full pipeline: discover → filter → fetch → score.

        This is an async method because article extraction uses Playwright.
        Sitemap crawling (curl_cffi) is synchronous but fast.

        Args:
            start_month: Start date "YYYY-MM"
            end_month: End date "YYYY-MM" (default: current month)
            min_relevance: Minimum keyword relevance score to keep
            max_articles_per_month: Cap articles per archive to manage load
            keywords_prefilter: Pre-filter by image_caption keywords before
                              fetching full articles (saves bandwidth)

        Returns:
            List of FTArticle objects with headline, body, relevance score.
        """
        # Check processed cache
        cache_key = f"{start_month}_{end_month or 'now'}_{min_relevance}"
        processed_path = self.cache_dir / "processed" / f"articles_{cache_key}.json"
        if processed_path.exists():
            logger.info(f"Loading cached articles from {processed_path}")
            with open(processed_path) as f:
                data = json.load(f)
            return [FTArticle(**d) for d in data]

        if not self._browser_ready:
            await self.setup()

        # Discover and filter archives (sync, fast via curl_cffi)
        all_archives = self.discover_archives()
        archives = self.filter_archives_by_date(all_archives, start_month, end_month)

        all_articles: list[FTArticle] = []

        for archive in archives:
            logger.info(f"Processing archive: {archive['url']}")
            article_entries = self.parse_archive(archive["url"])

            # Pre-filter by image caption keywords to avoid fetching irrelevant pages
            if keywords_prefilter:
                article_entries = [
                    a for a in article_entries
                    if _relevance_score(a.get("image_caption", "")) > 0
                    or not a.get("image_caption")
                ]

            if len(article_entries) > max_articles_per_month:
                article_entries = article_entries[:max_articles_per_month]

            fetched_count = 0
            for entry in article_entries:
                content = await self.extract_article_content(entry["url"])
                if content is None:
                    continue

                # Use JSON-LD datePublished if available, fall back to sitemap lastmod
                published_at = (
                    content.get("date_published")
                    or entry.get("lastmod", "")
                )

                article = FTArticle(
                    url=entry["url"],
                    published_at=published_at,
                    headline=content.get("headline", ""),
                    summary=content.get("summary", ""),
                    body_text=content.get("body_text", ""),
                    image_caption=entry.get("image_caption", ""),
                    fetch_timestamp=datetime.utcnow().isoformat(),
                )

                # Score relevance
                combined = f"{article.headline} {article.summary} {article.image_caption}"
                article.relevance_score = _relevance_score(combined)

                if article.relevance_score >= min_relevance:
                    all_articles.append(article)
                    fetched_count += 1
                    logger.debug(f"Relevant [{article.relevance_score:.3f}]: {article.headline[:80]}")

            logger.info(
                f"Archive complete: {fetched_count} relevant articles "
                f"(of {len(article_entries)} total)"
            )

        # Sort by date
        all_articles.sort(key=lambda a: a.published_at)

        # Cache processed results
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(processed_path, "w") as f:
            json.dump([a.to_dict() for a in all_articles], f, indent=2)

        logger.info(
            f"Pipeline complete: {len(all_articles)} relevant articles "
            f"from {start_month} to {end_month or 'now'}"
        )
        return all_articles

    async def fetch_month(
        self, year: int, month: int, min_relevance: float = 0.01
    ) -> list[FTArticle]:
        """Convenience: fetch articles for a single month."""
        month_str = f"{year}-{month:02d}"
        return await self.fetch_articles(
            start_month=month_str,
            end_month=month_str,
            min_relevance=min_relevance,
        )


class FTArticleCache:
    """Manage cached FT articles for fast access during training."""

    def __init__(self, cache_dir: str = "data/ft_cache/processed"):
        self.cache_dir = Path(cache_dir)

    def load_date_range(
        self,
        start_date: str,
        end_date: str,
    ) -> list[FTArticle]:
        """Load cached articles within a date range.

        Args:
            start_date: "YYYY-MM-DD" format
            end_date: "YYYY-MM-DD" format
        """
        all_articles = []

        for json_file in sorted(self.cache_dir.glob("articles_*.json")):
            with open(json_file) as f:
                data = json.load(f)

            for d in data:
                pub = d.get("published_at", "")[:10]  # YYYY-MM-DD
                if start_date <= pub <= end_date:
                    all_articles.append(FTArticle(**d))

        logger.info(f"Loaded {len(all_articles)} cached articles for [{start_date} → {end_date}]")
        return all_articles

    def get_articles_for_timestamp(
        self,
        target_time: datetime,
        lookback_hours: int = 24,
        max_articles: int = 20,
    ) -> list[FTArticle]:
        """Get articles published within lookback window of a timestamp.

        Useful for aligning news with specific trading bars during training.
        """
        cutoff = target_time - timedelta(hours=lookback_hours)

        all_articles = []
        for json_file in sorted(self.cache_dir.glob("articles_*.json")):
            with open(json_file) as f:
                data = json.load(f)

            for d in data:
                try:
                    pub_time = datetime.fromisoformat(d["published_at"].replace("Z", "+00:00"))
                    pub_time = pub_time.replace(tzinfo=None)
                    if cutoff <= pub_time <= target_time:
                        all_articles.append(FTArticle(**d))
                except (ValueError, KeyError):
                    continue

        # Sort by relevance, take top N
        all_articles.sort(key=lambda a: a.relevance_score, reverse=True)
        return all_articles[:max_articles]
