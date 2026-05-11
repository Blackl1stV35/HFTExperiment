"""Human-in-the-loop interface v2 — richer context for mid-hold reviews.

PATCH v2 changes (Strategy 5b):
    - SignalContext extended with DrawdownContext dataclass
    - _console_approval() renders DrawdownContext block when present
    - Mid-hold review layout: PnL hero + drawdown stats + consecutive losses
    - Auto-approve threshold tightened from 0.7→0.72 (higher bar for auto-pass)
    - check_exit() no longer auto-approves ALL profitable exits — exits in
      consecutive-loss streaks still get reviewed to catch regime breaks.

Layout priority (top to bottom):
    1. Unrealized PnL (largest, highlighted)
    2. Pips from entry
    3. Agent recommendation (one-line)
    4. Risk % of account  +  drawdown context (if mid-hold review)
    5. Position details (price, lots, hold time)
    6. Confidence bar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from loguru import logger


PIP_VALUE = 0.10


@dataclass
class DrawdownContext:
    """Rich drawdown state passed to HITL for mid-hold reviews (Strategy 5b)."""

    consecutive_losses: int = 0          # current losing streak
    daily_pnl_usd: float = 0.0           # today's running PnL
    session_volatility_pips: float = 0.0 # ATR proxy for current session
    peak_balance: float = 10_000.0
    current_balance: float = 10_000.0

    @property
    def account_drawdown_pct(self) -> float:
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance * 100.0


@dataclass
class SignalContext:
    """Full context for human approval decisions."""

    action: str
    confidence: float
    current_price: float
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    hold_time_bars: int = 0
    exit_reason: str = ""
    position_size_lots: float = 0.01
    account_balance: float = 10_000.0

    pips_from_entry: float = 0.0
    risk_pct: float = 0.0

    # Strategy 5b: optional rich drawdown context
    drawdown_ctx: Optional[DrawdownContext] = None

    # Optional technicals
    rsi: float = 0.0
    atr: float = 0.0
    sentiment_score: float = 0.0
    regime: str = ""

    def __post_init__(self):
        if self.entry_price > 0 and self.pips_from_entry == 0:
            price_diff = self.current_price - self.entry_price
            if "CLOSE_LONG" in self.action or (
                "CLOSE" in self.action and self.unrealized_pnl >= 0
            ):
                self.pips_from_entry = price_diff / PIP_VALUE
            elif "CLOSE_SHORT" in self.action or (
                "CLOSE" in self.action and self.unrealized_pnl < 0
            ):
                self.pips_from_entry = -price_diff / PIP_VALUE
            else:
                self.pips_from_entry = price_diff / PIP_VALUE
        if self.account_balance > 0 and self.risk_pct == 0:
            self.risk_pct = abs(self.unrealized_pnl) / self.account_balance * 100

    @property
    def recommendation(self) -> str:
        if "mid_hold_drawdown" in self.exit_reason:
            if self.drawdown_ctx and self.drawdown_ctx.consecutive_losses >= 3:
                return "CAUTION — possible regime break, consecutive losses detected"
            return f"Mid-hold review — loss at ${self.unrealized_pnl:.0f}, consider cutting"
        elif "max_hold" in self.exit_reason or "max_time" in self.exit_reason:
            return "Time-based close — position exceeded max hold period"
        elif "signal_reverse" in self.exit_reason:
            if self.unrealized_pnl > 0:
                return "Take profit — model sees reversal, locking in gains"
            elif self.unrealized_pnl < -5:
                return "Cut loss — model sees reversal, limiting damage"
            else:
                return "Signal reversal — model direction changed"
        elif "circuit" in self.exit_reason.lower() or "drawdown" in self.exit_reason.lower():
            return "RISK EXIT — circuit breaker triggered"
        elif self.action in ("BUY", "SELL"):
            if self.confidence >= 0.72:
                return f"Strong {self.action.lower()} signal — high confidence"
            elif self.confidence >= 0.5:
                return f"Moderate {self.action.lower()} signal"
            else:
                return f"Weak {self.action.lower()} signal — consider skipping"
        elif "end_of_data" in self.exit_reason:
            return "End of session — closing all positions"
        return "Agent suggests this action"


class HITLGate:
    """Human approval gate with auto-approve rules.

    PATCH v2 (Strategy 5):
        - Auto-approve threshold raised to 0.72 (was 0.70)
        - Profitable exits in consecutive-loss streaks (>=3) are reviewed
          even if in profit — detects regime breaks early
        - check_exit() receives DrawdownContext when available

    Auto-approve (no prompt):
        - High-confidence (>=0.72) small entries
        - Profitable exits with no active losing streak

    Requires human approval:
        - Any entry below 0.72 confidence
        - Large position entries
        - Any losing exit
        - Forced exits (max hold, circuit breaker)
        - Profitable exits during consecutive-loss streak >=3
        - Mid-hold drawdown reviews (new in v2)
    """

    def __init__(
        self,
        enabled: bool = True,
        auto_approve_confidence: float = 0.72,    # PATCH: was 0.70
        auto_approve_max_lots: float = 0.03,
        auto_approve_loss_streak_cap: int = 3,    # PATCH: review profits during streaks
        show_technicals: bool = False,
        approval_fn: Optional[Callable[[SignalContext], bool]] = None,
        telegram_hitl: "TelegramHITL | None" = None,   # Phase 8b: Telegram approval
    ):
        self.enabled = enabled
        self.auto_approve_confidence = auto_approve_confidence
        self.auto_approve_max_lots = auto_approve_max_lots
        self.auto_approve_loss_streak_cap = auto_approve_loss_streak_cap
        self.show_technicals = show_technicals
        self._telegram = telegram_hitl
        # If TelegramHITL provided, use it instead of console
        if telegram_hitl is not None:
            self._approval_fn = telegram_hitl.request_approval
        else:
            self._approval_fn = approval_fn or self._console_approval

        self.stats = {
            "approved": 0, "vetoed": 0, "auto_approved": 0, "total_prompted": 0,
        }
        self._consecutive_losses = 0  # internal streak tracker

    def record_trade_result(self, pnl_usd: float) -> None:
        """Call after each closed trade to maintain streak state."""
        if pnl_usd <= 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def check_entry(self, ctx: SignalContext) -> bool:
        if not self.enabled:
            self.stats["auto_approved"] += 1
            return True
        if (
            ctx.confidence >= self.auto_approve_confidence
            and ctx.position_size_lots <= self.auto_approve_max_lots
        ):
            self.stats["auto_approved"] += 1
            return True
        return self._request(ctx)

    def check_exit(self, ctx: SignalContext) -> bool:
        if not self.enabled:
            self.stats["auto_approved"] += 1
            return True

        # Auto-approve profitable exits UNLESS in a losing streak
        if (
            ctx.unrealized_pnl > 0
            and self._consecutive_losses < self.auto_approve_loss_streak_cap
        ):
            self.stats["auto_approved"] += 1
            return True

        return self._request(ctx)

    def _request(self, ctx: SignalContext) -> bool:
        self.stats["total_prompted"] += 1
        result = self._approval_fn(ctx)
        self.stats["approved" if result else "vetoed"] += 1
        return result

    def _console_approval(self, ctx: SignalContext) -> bool:
        """Console prompt — PnL-first layout with optional DrawdownContext block."""
        is_exit = "CLOSE" in ctx.action or ctx.entry_price > 0
        is_mid_hold = "mid_hold" in ctx.exit_reason

        print(f"\n{'━'*60}")

        if is_exit:
            pnl_sign = "+" if ctx.unrealized_pnl >= 0 else ""
            pnl_color = "32" if ctx.unrealized_pnl >= 0 else "31"
            print(
                f"  \033[1;{pnl_color}m  PnL: {pnl_sign}${ctx.unrealized_pnl:.2f}  "
                f"({pnl_sign}{ctx.pips_from_entry:.0f} pips)\033[0m"
            )
            print(f"  {'─'*56}")
            risk_label = "RISK" if ctx.unrealized_pnl < 0 else "GAIN"
            print(f"  {risk_label}: {ctx.risk_pct:.1f}% of account")

            # Strategy 5b: DrawdownContext block for mid-hold reviews
            if is_mid_hold and ctx.drawdown_ctx:
                dc = ctx.drawdown_ctx
                print(f"  {'─'*56}")
                print(f"  DRAWDOWN CONTEXT")
                print(f"  Account DD:       {dc.account_drawdown_pct:.1f}%")
                print(f"  Consecutive L:    {dc.consecutive_losses}")
                print(f"  Daily PnL:        ${dc.daily_pnl_usd:.2f}")
                if dc.session_volatility_pips > 0:
                    print(f"  Session ATR:      {dc.session_volatility_pips:.1f} pips")
        else:
            print(f"  NEW {ctx.action}  |  Lots: {ctx.position_size_lots}")
            print(f"  {'─'*56}")

        print(f"  >> {ctx.recommendation}")
        print(f"  {'─'*56}")

        if is_exit:
            print(f"  Action:     {ctx.action}")
            print(f"  Entry:      {ctx.entry_price:.2f}  →  Now: {ctx.current_price:.2f}")
            print(f"  Lots:       {ctx.position_size_lots}")
            print(f"  Hold:       {ctx.hold_time_bars} bars ({ctx.hold_time_bars} min)")
            print(f"  Reason:     {ctx.exit_reason}")
        else:
            print(f"  Price:      {ctx.current_price:.2f}")

        conf_filled = int(ctx.confidence * 20)
        conf_bar = "█" * conf_filled + "░" * (20 - conf_filled)
        print(f"  Confidence: [{conf_bar}] {ctx.confidence:.1%}")

        if self.show_technicals:
            parts = []
            if ctx.rsi > 0: parts.append(f"RSI:{ctx.rsi:.0f}")
            if ctx.atr > 0: parts.append(f"ATR:{ctx.atr:.2f}")
            if ctx.sentiment_score != 0: parts.append(f"Sent:{ctx.sentiment_score:.3f}")
            if ctx.regime: parts.append(f"Regime:{ctx.regime}")
            if parts:
                print(f"  Tech: {' | '.join(parts)}")

        print(f"{'━'*60}")

        while True:
            resp = input("  Approve? (y/n/s=skip all): ").strip().lower()
            if resp in ("y", "yes"):
                return True
            if resp in ("n", "no"):
                print("  → VETOED")
                return False
            if resp in ("s", "skip"):
                self.enabled = False
                print("  → Auto-approving all remaining")
                return True


class TelegramHITL:
    """Telegram-based approval gate.

    Sends a signal card to your Telegram chat when a trade needs approval.
    Polls for /approve or /veto reply (or inline button) for up to `timeout_s`.
    Falls back to auto-approve if no reply within timeout.

    Setup:
        1. Create a bot via @BotFather → get TELEGRAM_BOT_TOKEN
        2. Send /start to your bot → get TELEGRAM_CHAT_ID via getUpdates
        3. Set env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

    Commands (reply to bot):
        /approve  or  /y  → approve the trade
        /veto     or  /n  → reject the trade
        /skip            → auto-approve everything for this session
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        timeout_s: int = 120,            # wait up to 2 min before auto-approve
        auto_approve_on_timeout: bool = True,
    ):
        import os
        self.bot_token  = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id    = chat_id   or os.getenv("TELEGRAM_CHAT_ID", "")
        self.timeout_s  = timeout_s
        self.auto_approve_on_timeout = auto_approve_on_timeout
        self._enabled   = bool(self.bot_token and self.chat_id)
        self._skip_all  = False      # set True when user sends /skip
        self._last_update_id = 0     # Telegram getUpdates offset

        if not self._enabled:
            logger.warning("TelegramHITL: no token/chat_id — falling back to console")

    def request_approval(self, ctx: "SignalContext") -> bool:
        """Send card to Telegram, poll for reply. Returns True=approve."""
        if self._skip_all:
            return True
        if not self._enabled:
            return self._console_fallback(ctx)
        try:
            msg = self._format_card(ctx)
            self._send_message(msg, parse_mode="Markdown")
            return self._poll_reply()
        except Exception as e:
            logger.error(f"TelegramHITL error: {e} — auto-approving")
            return True

    def _format_card(self, ctx: "SignalContext") -> str:
        is_exit = "CLOSE" in ctx.action or ctx.entry_price > 0
        conf_filled = int(ctx.confidence * 10)
        conf_bar    = "█" * conf_filled + "░" * (10 - conf_filled)

        if is_exit:
            pnl_sign = "+" if ctx.unrealized_pnl >= 0 else ""
            emoji    = "💰" if ctx.unrealized_pnl >= 0 else "📉"
            card = (
                f"{emoji} *EXIT SIGNAL*\n"
                f"PnL: `{pnl_sign}${ctx.unrealized_pnl:.2f}` "
                f"({pnl_sign}{ctx.pips_from_entry:.0f} pips)\n"
                f"Entry: `{ctx.entry_price:.2f}` → Now: `{ctx.current_price:.2f}`\n"
                f"Hold: {ctx.hold_time_bars} bars | Lots: {ctx.position_size_lots}\n"
                f"Reason: `{ctx.exit_reason}`\n"
            )
        else:
            emoji = "📈" if ctx.action == "BUY" else "📉"
            card = (
                f"{emoji} *{ctx.action} SIGNAL*\n"
                f"Price: `{ctx.current_price:.2f}` | Lots: `{ctx.position_size_lots}`\n"
            )

        card += (
            f"Conf: `[{conf_bar}] {ctx.confidence:.1%}`\n"
            f">> _{ctx.recommendation}_\n\n"
            f"Reply: /approve ✅  or  /veto ❌  or  /skip ⏭\n"
            f"_(auto-approve in {self.timeout_s}s if no reply)_"
        )
        return card

    def _send_message(self, text: str, parse_mode: str = "Markdown") -> dict:
        import requests
        url  = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id":    self.chat_id,
            "text":       text,
            "parse_mode": parse_mode,
        }, timeout=5)
        return resp.json()

    def _poll_reply(self) -> bool:
        """Poll getUpdates for /approve, /veto or /skip. Returns True=approve."""
        import requests, time
        url      = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        deadline = time.time() + self.timeout_s
        approve_cmds = {"/approve", "/y", "approve", "y", "yes"}
        veto_cmds    = {"/veto",    "/n", "veto",    "n", "no"}
        skip_cmds    = {"/skip", "skip", "s"}

        while time.time() < deadline:
            try:
                resp = requests.get(url, params={
                    "offset":  self._last_update_id + 1,
                    "timeout": 5,
                }, timeout=10).json()
                for update in resp.get("result", []):
                    self._last_update_id = max(self._last_update_id,
                                               update["update_id"])
                    text = (
                        update.get("message", {}).get("text", "")
                        or update.get("callback_query", {}).get("data", "")
                    ).strip().lower()
                    # Only accept replies from our chat
                    from_chat = (
                        str(update.get("message", {}).get("chat", {}).get("id", ""))
                        or str(update.get("callback_query", {}).get("message", {})
                               .get("chat", {}).get("id", ""))
                    )
                    if from_chat != str(self.chat_id):
                        continue
                    if text in approve_cmds:
                        logger.info("TelegramHITL: APPROVED via Telegram")
                        self._send_message("✅ Trade APPROVED")
                        return True
                    if text in veto_cmds:
                        logger.info("TelegramHITL: VETOED via Telegram")
                        self._send_message("❌ Trade VETOED")
                        return False
                    if text in skip_cmds:
                        logger.info("TelegramHITL: SKIP ALL via Telegram")
                        self._send_message("⏭ Auto-approving all remaining signals")
                        self._skip_all = True
                        return True
            except Exception as e:
                logger.warning(f"TelegramHITL poll error: {e}")
                time.sleep(2)
            time.sleep(1)

        logger.info(f"TelegramHITL: timeout ({self.timeout_s}s) — "
                    f"{'auto-approving' if self.auto_approve_on_timeout else 'vetoing'}")
        msg = "⏰ Timeout — " + ("trade AUTO-APPROVED" if self.auto_approve_on_timeout
                                 else "trade VETOED (timeout)")
        self._send_message(msg)
        return self.auto_approve_on_timeout

    @staticmethod
    def _console_fallback(ctx: "SignalContext") -> bool:
        """Use if Telegram not configured."""
        print(f"\nApprove {ctx.action} @ {ctx.current_price:.2f} "
              f"conf={ctx.confidence:.2%}? (y/n): ", end="", flush=True)
        return input().strip().lower() in ("y", "yes", "")


class TelegramCommander:
    """Full remote control of the trading bot via Telegram commands.

    Send commands to your bot from your phone to control the running process.

    Commands:
        /start          Resume trading (un-pause)
        /stop           Pause new entries (holds existing positions)
        /kill           Emergency stop — close all positions and exit process
        /close          Close current position immediately
        /status         Show current position, PnL, regime, last signal
        /config         Show current config (lot size, gate, thresholds)
        /setlot <n>     Change lot size (e.g. /setlot 0.02)
        /setconf <n>    Change confidence gate (e.g. /setconf 0.75)
        /gate on|off    Enable/disable Bear+HIGH deploy gate
        /autoapprove    Auto-approve all trades for this session
        /help           Show this list

    Usage in paper_trade.py:
        commander = TelegramCommander(state=self)   # pass TradingLoop as state
        threading.Thread(target=commander.listen, daemon=True).start()
    """

    def __init__(self, state, bot_token: str | None = None, chat_id: str | None = None):
        import os
        self.state      = state          # reference to TradingLoop instance
        self.bot_token  = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id    = chat_id   or os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled   = bool(self.bot_token and self.chat_id)
        self._offset    = 0
        self._running   = False

        if not self._enabled:
            logger.warning("TelegramCommander: env vars not set — remote control disabled")

    # ── public ────────────────────────────────────────────────────────────────

    def listen(self) -> None:
        """Blocking loop — run in a daemon thread."""
        if not self._enabled:
            return
        self._running = True
        self.send("🤖 *Commander online*\n" + self._help_text())
        logger.info("TelegramCommander: listening for commands")
        import time
        while self._running:
            try:
                self._poll_commands()
            except Exception as e:
                logger.warning(f"TelegramCommander poll error: {e}")
            time.sleep(1)

    def send(self, text: str) -> None:
        if not self._enabled:
            return
        try:
            import requests
            requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text,
                      "parse_mode": "Markdown"},
                timeout=5,
            )
        except Exception as e:
            logger.warning(f"TelegramCommander send error: {e}")

    # ── private ───────────────────────────────────────────────────────────────

    def _poll_commands(self) -> None:
        import requests
        resp = requests.get(
            f"https://api.telegram.org/bot{self.bot_token}/getUpdates",
            params={"offset": self._offset + 1, "timeout": 3},
            timeout=8,
        ).json()
        for update in resp.get("result", []):
            self._offset = max(self._offset, update["update_id"])
            msg  = update.get("message", {})
            text = msg.get("text", "").strip()
            from_chat = str(msg.get("chat", {}).get("id", ""))
            if from_chat != str(self.chat_id) or not text.startswith("/"):
                continue
            self._handle(text)

    def _handle(self, text: str) -> None:
        s    = self.state
        cmd  = text.split()[0].lower()
        args = text.split()[1:]

        if cmd == "/help":
            self.send(self._help_text())

        elif cmd == "/status":
            self.send(self._status_text())

        elif cmd == "/start":
            s._paused = False
            self.send("▶️ *Trading RESUMED* — new entries enabled")
            logger.info("TelegramCommander: RESUMED")

        elif cmd == "/stop":
            s._paused = True
            self.send("⏸ *Trading PAUSED* — holding existing positions, no new entries")
            logger.info("TelegramCommander: PAUSED")

        elif cmd == "/kill":
            self.send("🔴 *KILL received* — closing all positions and shutting down...")
            logger.warning("TelegramCommander: KILL command received")
            self._emergency_close_all()
            s.running = False

        elif cmd == "/close":
            if s.position is not None:
                self.send("🔒 Closing current position...")
                s._force_close = True
            else:
                self.send("ℹ️ No open position to close")

        elif cmd == "/config":
            self.send(self._config_text())

        elif cmd == "/setlot":
            if args:
                try:
                    v = float(args[0])
                    if 0.001 <= v <= 10.0:
                        s.cfg["broker"]["lot_size"] = v
                        self.send(f"✅ Lot size set to `{v}`")
                    else:
                        self.send("❌ Lot size must be between 0.001 and 10.0")
                except ValueError:
                    self.send("❌ Usage: /setlot 0.02")

        elif cmd == "/setconf":
            if args:
                try:
                    v = float(args[0])
                    if 0.5 <= v <= 0.99:
                        s.hitl.auto_approve_confidence = v
                        self.send(f"✅ Confidence gate set to `{v:.2f}` ({v:.0%})")
                    else:
                        self.send("❌ Confidence must be between 0.50 and 0.99")
                except ValueError:
                    self.send("❌ Usage: /setconf 0.75")

        elif cmd == "/gate":
            if args:
                on = args[0].lower() in ("on", "true", "1", "yes")
                s.cfg.setdefault("deploy_gate", {})["enabled"] = on
                self.send(f"✅ Deploy gate: {'ON (Bear+HIGH blocked)' if on else 'OFF (all regimes trade)'}")
            else:
                self.send("❌ Usage: /gate on  or  /gate off")

        elif cmd == "/autoapprove":
            if s.hitl._telegram:
                s.hitl._telegram._skip_all = True
            s.hitl.enabled = False
            self.send("⏭ *Auto-approving all* — HITL disabled for this session")

        else:
            self.send(f"❓ Unknown command: `{cmd}`\nSend /help for command list")

    def _emergency_close_all(self) -> None:
        """Close all open positions via broker."""
        try:
            positions = self.state.broker.get_open_positions()
            if not positions:
                self.send("ℹ️ No open positions to close")
                return
            for p in positions:
                result = self.state.broker.close_position(p["ticket"], comment="kill")
                if result.success:
                    self.send(f"✅ Closed ticket {p['ticket']} @ {result.price:.2f}")
                else:
                    self.send(f"❌ Failed to close {p['ticket']}: {result.comment}")
            self.state.position  = None
            self.state.hold_bars = 0
        except Exception as e:
            self.send(f"❌ Emergency close error: {e}")

    def _status_text(self) -> str:
        s = self.state
        pos_str = "FLAT"
        if s.position is not None:
            dir_str = "LONG" if s.position["dir"] == 1 else "SHORT"
            pos_str = (
                f"{dir_str} {s.position['lots']} lots @ {s.position['entry_price']:.2f}\n"
                f"  Hold: {s.hold_bars} bars | Ticket: {s.position['ticket']}"
            )
        paused  = getattr(s, "_paused", False)
        gate_on = s.cfg.get("deploy_gate", {}).get("enabled", True)
        acct    = s.broker.get_account_info() if s.broker else {}

        return (
            f"📊 *Status*\n"
            f"Position: `{pos_str}`\n"
            f"Trading: {'⏸ PAUSED' if paused else '▶️ ACTIVE'}\n"
            f"Deploy gate: {'ON' if gate_on else 'OFF'}\n"
            f"Lot size: `{s.cfg.get('broker', {}).get('lot_size', 0.01)}`\n"
            f"Conf gate: `{s.hitl.auto_approve_confidence:.0%}`\n"
            f"Balance: `${acct.get('balance', 0):.2f}` | "
            f"Equity: `${acct.get('equity', 0):.2f}`"
        )

    def _config_text(self) -> str:
        s = self.state
        return (
            f"⚙️ *Config*\n"
            f"Symbol: `{s.cfg.get('broker', {}).get('symbol', 'XAUUSD')}`\n"
            f"Lot size: `{s.cfg.get('broker', {}).get('lot_size', 0.01)}`\n"
            f"Max hold: `{s.cfg.get('risk', {}).get('max_hold_bars', 80)} bars`\n"
            f"Conf gate: `{s.hitl.auto_approve_confidence:.0%}`\n"
            f"Deploy gate: `{s.cfg.get('deploy_gate', {}).get('enabled', True)}`\n"
            f"Latency kill: `{s.cfg.get('risk', {}).get('latency_kill_ms', 500)}ms`\n"
            f"Mode: `{'SYNTHETIC' if s.synthetic else 'LIVE MT5'}`"
        )

    @staticmethod
    def _help_text() -> str:
        return (
            "🤖 *HFT Bot Commands*\n\n"
            "`/status`       — position, PnL, regime\n"
            "`/start`        — resume trading\n"
            "`/stop`         — pause new entries\n"
            "`/kill`         — close all + shutdown\n"
            "`/close`        — close current position\n"
            "`/config`       — show current settings\n"
            "`/setlot 0.02`  — change lot size\n"
            "`/setconf 0.75` — change confidence gate\n"
            "`/gate on|off`  — enable/disable Bear+HIGH gate\n"
            "`/autoapprove`  — disable HITL for session\n"
            "`/help`         — show this list"
        )


class RiskDisplay:
    """Compact signal display for periodic logging."""

    @staticmethod
    def format_signal(action: int, confidence: float, price: float) -> str:
        action_map = {0: "SELL ↓", 1: "HOLD ─", 2: "BUY ↑"}
        conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
        return f"{action_map.get(action, '???')} [{conf_bar}] {confidence:.0%} @ {price:.2f}"