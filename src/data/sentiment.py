"""Sentiment pipeline: FT.com (training) + GDELT (live) → FinBERT embeddings.

Architecture:
    TRAINING: FT.com sitemaps → cached articles → FinBERT → embeddings aligned to bars
    LIVE:     GDELT API (free, no key) → recent headlines → FinBERT → consensus embedding

NewsAPI has been replaced because:
    - FT.com sitemaps provide deeper historical coverage (1995–present)
    - GDELT is truly free (no API key, no rate limit tier)
    - Both provide higher quality financial content than NewsAPI free tier
"""

from __future__ import annotations

import json
import os
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from loguru import logger


# =====================================================================
# GDELT: Free real-time news API (for LIVE inference)
# =====================================================================

class GDELTFetcher:
    """Fetch gold/macro news from GDELT DOC API.

    GDELT is completely free — no API key, no rate limits for reasonable usage.
    Returns structured article metadata with titles and URLs.

    Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Queries optimized for gold/macro news
    QUERIES = [
        "gold price OR gold market OR XAUUSD OR bullion",
        "Federal Reserve OR FOMC OR interest rate decision",
        "inflation CPI OR treasury yield OR dollar index",
    ]

    def __init__(self, cache_dir: str = "data/sentiment_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        hours_back: int = 24,
        max_articles: int = 30,
        query_override: Optional[str] = None,
    ) -> list[dict]:
        """Fetch recent news articles from GDELT.

        Args:
            hours_back: How far back to search.
            max_articles: Maximum articles to return.
            query_override: Custom query string (overrides defaults).

        Returns:
            List of {"text": str, "source": str, "published_at": str, "url": str}
        """
        all_articles = []

        queries = [query_override] if query_override else self.QUERIES

        for query in queries:
            try:
                params = {
                    "query": query,
                    "mode": "ArtList",
                    "maxrecords": min(max_articles, 75),
                    "timespan": f"{hours_back}h",
                    "format": "json",
                    "sort": "DateDesc",
                }

                resp = requests.get(self.BASE_URL, params=params, timeout=15)
                if resp.status_code != 200:
                    logger.warning(f"GDELT returned {resp.status_code} for query: {query[:50]}")
                    continue

                data = resp.json()
                articles = data.get("articles", [])

                for a in articles:
                    title = a.get("title", "").strip()
                    if not title:
                        continue

                    all_articles.append({
                        "text": title,
                        "source": a.get("domain", "unknown"),
                        "published_at": a.get("seendate", ""),
                        "url": a.get("url", ""),
                        "language": a.get("language", "English"),
                    })

            except Exception as e:
                logger.warning(f"GDELT fetch failed for query '{query[:40]}': {e}")
                continue

            # Small delay between queries
            time.sleep(0.5)

        # Deduplicate by URL
        seen_urls = set()
        unique = []
        for a in all_articles:
            if a["url"] not in seen_urls:
                seen_urls.add(a["url"])
                unique.append(a)

        # Filter to English only
        unique = [a for a in unique if a.get("language", "").startswith("English")]

        # Sort by recency
        unique.sort(key=lambda a: a.get("published_at", ""), reverse=True)
        result = unique[:max_articles]

        # Cache
        self._save_cache(result)
        logger.info(f"GDELT: fetched {len(result)} articles (from {len(all_articles)} raw)")
        return result

    def _save_cache(self, articles: list[dict]) -> None:
        today = datetime.utcnow().strftime("%Y-%m-%d_%H")
        path = self.cache_dir / f"gdelt_{today}.json"
        with open(path, "w") as f:
            json.dump(articles, f, indent=2)

    def load_cache(self) -> list[dict]:
        """Load most recent GDELT cache."""
        caches = sorted(self.cache_dir.glob("gdelt_*.json"), reverse=True)
        if caches:
            with open(caches[0]) as f:
                return json.load(f)
        return []


# =====================================================================
# FinBERT embedding extractor
# =====================================================================

class FinBERTSentiment:
    """Extract rich sentiment embeddings from FinBERT.

    Instead of just positive/negative/neutral labels, extracts intermediate
    hidden state representations for richer sentiment features.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = "cpu",
        max_length: int = 512,
        cache_dir: str = "data/sentiment_cache",
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazy-load model and tokenizer."""
        if self._model is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        logger.info(f"Loading FinBERT model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, output_hidden_states=True
        )
        self._model.to(self.device)
        self._model.eval()
        logger.info("FinBERT loaded successfully")

    def get_embeddings(
        self,
        texts: list[str],
        use_hidden_states: bool = True,
        layer: int = -2,
    ) -> np.ndarray:
        """Extract embeddings from FinBERT.

        Args:
            texts: List of text strings.
            use_hidden_states: If True, use intermediate hidden states (richer).
                             If False, use classification logits (simpler).
            layer: Which hidden layer to use (-1 = last, -2 = second to last).

        Returns:
            Embeddings array. Shape: (n_texts, 768) if hidden states, (n_texts, 3) if logits.
        """
        self._load_model()
        import torch

        cache_key = self._cache_key(texts, use_hidden_states, layer)
        cached = self._load_embedding_cache(cache_key)
        if cached is not None:
            return cached

        embeddings = []
        batch_size = 8

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                outputs = self._model(**inputs)

                if use_hidden_states and outputs.hidden_states:
                    hidden = outputs.hidden_states[layer]
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
                    embeddings.append(pooled.cpu().numpy())
                else:
                    logits = torch.softmax(outputs.logits, dim=-1)
                    embeddings.append(logits.cpu().numpy())

        result = np.concatenate(embeddings, axis=0).astype(np.float32)
        self._save_embedding_cache(cache_key, result)

        logger.info(f"Generated embeddings: shape={result.shape}")
        return result

    def get_consensus_embedding(
        self,
        texts: list[str],
        use_hidden_states: bool = True,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get a single consensus embedding from multiple news articles.

        Args:
            texts: List of article texts.
            use_hidden_states: Use hidden states (768d) or logits (3d).
            weights: Optional per-article weights (e.g., relevance scores).

        Returns:
            Single consensus embedding vector.
        """
        if not texts:
            dim = 768 if use_hidden_states else 3
            return np.zeros(dim, dtype=np.float32)

        embeddings = self.get_embeddings(texts, use_hidden_states)

        if weights is not None:
            weights = np.array(weights, dtype=np.float32)
            weights = weights / (weights.sum() + 1e-8)
            consensus = (embeddings * weights[:, np.newaxis]).sum(axis=0)
        else:
            consensus = embeddings.mean(axis=0)

        return consensus

    def _cache_key(self, texts: list[str], use_hidden: bool, layer: int) -> str:
        content = json.dumps(texts, sort_keys=True) + f"_{use_hidden}_{layer}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_embedding_cache(self, key: str) -> Optional[np.ndarray]:
        path = self.cache_dir / f"emb_{key}.npy"
        if path.exists():
            return np.load(str(path))
        return None

    def _save_embedding_cache(self, key: str, arr: np.ndarray) -> None:
        path = self.cache_dir / f"emb_{key}.npy"
        np.save(str(path), arr)


# =====================================================================
# Training sentiment builder: FT.com → DAILY consensus embeddings
# =====================================================================

class TrainingSentimentBuilder:
    """Build DAILY time-aligned sentiment embeddings from FT.com article cache.

    Instead of a single global consensus or per-15-min refresh, this computes
    one 768-dim embedding per calendar day using all articles published on or
    before that day (within the lookback window). All M1 bars within the same
    calendar day share the same embedding — this is the correct granularity
    because news sentiment shifts daily, not per-minute.

    Output shape: (n_bars, 768) — each row is the daily consensus for that bar's day.
    """

    def __init__(
        self,
        ft_cache_dir: str = "data/ft_cache/processed",
        model_device: str = "cpu",
        lookback_hours: int = 24,
        max_articles_per_day: int = 30,
    ):
        self.lookback_hours = lookback_hours
        self.max_articles = max_articles_per_day
        self.sentiment_model = FinBERTSentiment(device=model_device)

        # Load all cached FT articles into memory for fast lookup
        self._articles: list[dict] = []
        self._load_ft_cache(ft_cache_dir)

    def _load_ft_cache(self, cache_dir: str) -> None:
        """Load all FT article cache files."""
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            logger.warning(f"FT cache dir not found: {cache_dir}")
            return

        for json_file in sorted(cache_path.glob("articles_*.json")):
            with open(json_file) as f:
                data = json.load(f)
                self._articles.extend(data)

        # Also try parquet files
        for pq_file in sorted(cache_path.glob("*.parquet")):
            try:
                import polars as pl
                df = pl.read_parquet(str(pq_file))
                for row in df.iter_rows(named=True):
                    self._articles.append(row)
            except Exception:
                pass

        # Parse dates
        for a in self._articles:
            try:
                dt_str = a.get("published_at", "")
                if dt_str:
                    a["_parsed_dt"] = datetime.fromisoformat(
                        dt_str.replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                else:
                    a["_parsed_dt"] = None
            except ValueError:
                a["_parsed_dt"] = None

        self._articles = [a for a in self._articles if a["_parsed_dt"] is not None]
        self._articles.sort(key=lambda a: a["_parsed_dt"])
        logger.info(f"Loaded {len(self._articles)} FT articles into training sentiment builder")

    def _get_articles_for_day(self, day: datetime) -> list[dict]:
        """Get articles published within lookback window ending at end-of-day.

        For a given calendar day, collects articles published between
        [day_start - lookback_hours, day_end] — this means the embedding
        for Monday includes articles from Sunday evening if lookback=24h.
        """
        day_end = day.replace(hour=23, minute=59, second=59)
        cutoff = day_end - timedelta(hours=self.lookback_hours)

        # Binary search for start index (articles are sorted by date)
        lo, hi = 0, len(self._articles)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._articles[mid]["_parsed_dt"] < cutoff:
                lo = mid + 1
            else:
                hi = mid

        relevant = []
        for i in range(lo, len(self._articles)):
            a = self._articles[i]
            if a["_parsed_dt"] > day_end:
                break
            relevant.append(a)

        # Sort by relevance, take top N
        relevant.sort(key=lambda a: a.get("relevance_score", 0), reverse=True)
        return relevant[: self.max_articles]

    def get_embedding_for_day(self, day: datetime) -> np.ndarray:
        """Compute consensus sentiment embedding for a single calendar day."""
        relevant = self._get_articles_for_day(day)

        if not relevant:
            return np.zeros(768, dtype=np.float32)

        texts = []
        weights = []
        for a in relevant:
            text = a.get("headline", "") or a.get("title", "")
            summary = a.get("summary", "") or a.get("body_snippet", "")
            if summary:
                text = f"{text}. {summary}"
            if text.strip():
                texts.append(text)
                weights.append(max(a.get("relevance_score", 0.01), 0.001))

        if not texts:
            return np.zeros(768, dtype=np.float32)

        return self.sentiment_model.get_consensus_embedding(
            texts, use_hidden_states=True, weights=np.array(weights)
        )

    def build_embedding_series(
        self,
        timestamps: list[datetime],
        cache_path: Optional[str] = None,
    ) -> np.ndarray:
        """Build daily consensus embeddings for a series of bar timestamps.

        Groups all bars by calendar day, computes one FinBERT embedding per day,
        then broadcasts that embedding to all bars within the same day.

        This is ~100x faster than per-bar or per-15-min computation and
        semantically correct — news sentiment is a daily-frequency signal.

        Args:
            timestamps: List of bar timestamps (M1 frequency).
            cache_path: If set, save/load intermediate daily embeddings here.

        Returns:
            Array of shape (n_timestamps, 768).
        """
        # Extract unique calendar days
        day_to_indices: dict[str, list[int]] = {}
        for i, ts in enumerate(timestamps):
            day_key = ts.strftime("%Y-%m-%d")
            if day_key not in day_to_indices:
                day_to_indices[day_key] = []
            day_to_indices[day_key].append(i)

        unique_days = sorted(day_to_indices.keys())
        logger.info(
            f"Building daily embeddings: {len(unique_days)} unique days "
            f"for {len(timestamps)} bars"
        )

        # Check cache for daily embeddings
        daily_cache: dict[str, np.ndarray] = {}
        daily_cache_path = Path(cache_path) if cache_path else None
        if daily_cache_path and daily_cache_path.exists():
            cached = np.load(str(daily_cache_path), allow_pickle=True).item()
            if isinstance(cached, dict):
                daily_cache = cached
                logger.info(f"Loaded {len(daily_cache)} cached daily embeddings")

        # Compute embeddings for each unique day
        for idx, day_key in enumerate(unique_days):
            if day_key in daily_cache:
                continue

            day_dt = datetime.strptime(day_key, "%Y-%m-%d")
            emb = self.get_embedding_for_day(day_dt)
            daily_cache[day_key] = emb

            if (idx + 1) % 50 == 0:
                logger.info(f"Daily embedding progress: {idx+1}/{len(unique_days)}")

        # Save daily cache
        if daily_cache_path:
            daily_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(daily_cache_path), daily_cache)
            logger.info(f"Saved daily embedding cache: {daily_cache_path}")

        # Broadcast daily embeddings to all bars
        n = len(timestamps)
        embeddings = np.zeros((n, 768), dtype=np.float32)
        zero_emb = np.zeros(768, dtype=np.float32)

        for day_key, indices in day_to_indices.items():
            emb = daily_cache.get(day_key, zero_emb)
            for i in indices:
                embeddings[i] = emb

        # Stats
        nonzero_days = sum(1 for d in unique_days if np.any(daily_cache.get(d, zero_emb) != 0))
        logger.info(
            f"Built embedding series: shape={embeddings.shape}, "
            f"{nonzero_days}/{len(unique_days)} days have non-zero embeddings"
        )
        return embeddings


def load_sentiment_embeddings(
    embeddings_path: str,
    n_bars: int,
    offset: int = 0,
) -> np.ndarray:
    """Load pre-built sentiment embeddings and align with model sequences.

    The embeddings file has one row per original bar. After preprocessing
    (window trimming + sequence creation), the model's sequences correspond
    to bars [offset : offset + n_bars]. This function extracts the matching
    slice.

    Handles both formats:
        - Correct: ndarray of shape (N, 768) saved by build_embedding_series
        - Legacy:  pickled dict {day_key: embedding} saved by daily cache

    Args:
        embeddings_path: Path to sentiment_embeddings.npy
        n_bars: Number of sequences (after create_sequences)
        offset: Number of bars trimmed by preprocessing (window_size + seq_length)

    Returns:
        Array of shape (n_bars, 768), aligned 1:1 with the model's X sequences.
    """
    if not Path(embeddings_path).exists():
        logger.warning(f"Sentiment embeddings not found: {embeddings_path}")
        return np.zeros((n_bars, 768), dtype=np.float32)

    try:
        raw = np.load(embeddings_path, allow_pickle=False)
    except ValueError:
        # File contains a pickled object (dict from old daily cache format)
        raw = np.load(embeddings_path, allow_pickle=True)

    # Handle dict format: this is a daily cache, not a bar-level array
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        obj = raw.item()
        if isinstance(obj, dict):
            logger.error(
                f"Sentiment file {embeddings_path} contains a daily cache dict, "
                f"not a bar-level array. Re-run: python scripts/build_embeddings.py"
            )
            return np.zeros((n_bars, 768), dtype=np.float32)

    all_embeddings = raw
    if all_embeddings.ndim != 2 or all_embeddings.shape[1] != 768:
        logger.error(
            f"Unexpected embedding shape: {all_embeddings.shape}. "
            f"Expected (N, 768). Re-run: python scripts/build_embeddings.py"
        )
        return np.zeros((n_bars, 768), dtype=np.float32)

    logger.info(f"Loaded sentiment embeddings: {all_embeddings.shape}")

    # Slice to match the sequence indices
    end = offset + n_bars
    if end > len(all_embeddings):
        logger.warning(
            f"Embedding array too short ({len(all_embeddings)}) for "
            f"requested range [{offset}:{end}]. Padding with zeros."
        )
        sliced = all_embeddings[offset:] if offset < len(all_embeddings) else np.empty((0, 768))
        pad_rows = end - max(offset, 0) - len(sliced)
        if pad_rows > 0:
            pad = np.zeros((pad_rows, 768), dtype=np.float32)
            sliced = np.concatenate([sliced, pad])
    else:
        sliced = all_embeddings[offset:end]

    return sliced.astype(np.float32)


# =====================================================================
# Live sentiment service: GDELT → FinBERT → consensus embedding
# =====================================================================

class SentimentService:
    """Live sentiment service for inference.

    Uses GDELT (free, no API key) for real-time news,
    with FinBERT for embedding generation.
    """

    def __init__(
        self,
        model_device: str = "cpu",
        update_interval: int = 900,  # 15 minutes
    ):
        self.gdelt = GDELTFetcher()
        self.sentiment_model = FinBERTSentiment(device=model_device)
        self.update_interval = update_interval
        self._last_update = 0.0
        self._cached_embedding: Optional[np.ndarray] = None

    def get_current_sentiment(self, force: bool = False) -> np.ndarray:
        """Get current consensus sentiment embedding, refreshing if stale."""
        now = time.time()
        if not force and self._cached_embedding is not None:
            if now - self._last_update < self.update_interval:
                return self._cached_embedding

        articles = self.gdelt.fetch(hours_back=24, max_articles=20)
        texts = [a["text"] for a in articles if a.get("text")]

        if not texts:
            logger.warning("No GDELT articles available, using zero embedding")
            self._cached_embedding = np.zeros(768, dtype=np.float32)
        else:
            self._cached_embedding = self.sentiment_model.get_consensus_embedding(texts)

        self._last_update = now
        return self._cached_embedding
