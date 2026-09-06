from __future__ import annotations

from typing import Literal, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Kite Connect ──────────────────────────────────────────────────────────
    kite_api_key: str = ""
    kite_api_secret: str = ""
    kite_access_token: str = ""
    kite_redirect_url: str = "http://localhost:3000/kite/callback"

    # Arms the static-IP / daily-token automation: a scheduled worker task
    # validates the Kite session shortly before the market opens and FAILS
    # CLOSED (a "failed" result, never a silent pass) when the configured
    # access token no longer authenticates.
    #
    # Kite access tokens are single-day; they cannot be refreshed
    # server-side, only re-issued through the interactive login flow. The
    # flag is the explicit human decision that automation may check sessions
    # against this deployment's static IP. Off by default, and the worker
    # guard refuses to do anything when it is off.
    kite_static_ip_enabled: bool = False

    # Clearly-labelled MOCK of the Zerodha Kite credential surface for local
    # development without real credentials.
    #
    # When true:
    #   * health reports the broker as "mock" (never "connected")
    #   * market-data routes return deterministic mock quotes tagged mock
    #   * the mock client has NO order surface — anything that would place an
    #     order raises
    #   * live trading is in no way enabled: trading_mode still defaults to
    #     "paper", is_live_trading_enabled stays False, and the eligibility
    #     gate would still block live even if someone flipped the mode.
    # Mock mode is a development convenience, never a route to a live account.
    zerodha_mock_mode: bool = False

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://algodollar:algodollar@localhost:5432/algodollar"
    database_echo: bool = False

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Application ───────────────────────────────────────────────────────────
    # "test" is included because CI sets APP_ENV=test. Without it, Settings
    # raised a pydantic literal_error at import time and the ENTIRE backend
    # test job aborted with exit 2 before collecting a single test.
    #
    # This is APP_ENV — environment labelling only. It is NOT TRADING_MODE,
    # which independently gates paper vs live and still accepts only "paper"
    # or "live". Adding "test" here cannot enable live trading.
    app_env: Literal[
        "development", "test", "staging", "paper", "live"
    ] = "development"

    # ── Auth ──────────────────────────────────────────────────────────────────
    secret_key: str = "change-me-in-production-at-least-32-chars-long"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 hours

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # ── Trading ───────────────────────────────────────────────────────────────
    trading_mode: Literal["paper", "live"] = "paper"

    # Where the paper broker persists its book. Without it the simulated
    # positions and cash are lost on restart, and the next start would find the
    # broker disagreeing with the database — reconciliation would refuse to open
    # the trading gate, which is correct but makes paper trading unrestartable.
    paper_state_path: Optional[str] = "data/paper_broker_state.json"

    # Append-only JSONL record of every execution attempt, including the ones
    # that were refused. Refusals are the more interesting half: they are the
    # evidence that the gates are doing something.
    execution_audit_path: Optional[str] = "data/execution_audit.jsonl"

    # ── AI ───────────────────────────────────────────────────────────────────
    anthropic_api_key: Optional[str] = None

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── Risk parameters ───────────────────────────────────────────────────────
    max_daily_loss_pct: float = 0.02       # 2%
    max_weekly_loss_pct: float = 0.05      # 5%
    max_monthly_loss_pct: float = 0.10     # 10%
    max_portfolio_drawdown_pct: float = 0.15  # 15%
    max_single_stock_pct: float = 0.10     # 10%
    max_sector_pct: float = 0.25           # 25%
    max_intraday_capital_pct: float = 0.10  # 10%
    max_positions: int = 20

    # ── Automated intraday trading (paper only) ───────────────────────────────
    # Master switch for the automated intraday loop. Off by default: the process
    # must never start proposing its own orders without an explicit human
    # decision, and this flag IS that decision. The loop additionally refuses to
    # arm unless trading_mode == "paper".
    auto_trade_enabled: bool = False

    # Tick feed for the intraday sleeve. "mock" synthesises a deterministic,
    # seeded NSE-style tick stream around the previous daily close — paper only.
    # "replay" (recorded session) and "live" (KiteTicker) are reserved for later
    # phases and are not implemented yet.
    tick_mode: Literal["mock", "replay", "live"] = "mock"

    # Mock feed cadence: one tick per universe symbol per interval.
    tick_interval_ms: int = 1000

    # Seed for the deterministic mock session. Fixed default so an unexplained
    # paper session can be reproduced exactly.
    tick_seed: int = 42

    # How often the auto-trader re-evaluates the book (entries, exits,
    # square-off). Seconds.
    trader_cycle_seconds: int = 60

    # Size of the intraday watchlist the mock feed prices and the loop watches.
    trader_universe_size: int = 50

    # Enables the intraday sleeve of the runtime loop. Requires a working tick
    # feed; the auto-trader refuses to arm without one.
    intraday_enabled: bool = False

    # Attach a broker-side SL-M stop to every intraday entry so that a dead
    # process still stops the position. Both the paper and Zerodha adapters
    # support SL-M; a safety test proves the broker refused the order if it
    # doesn't.
    intraday_use_broker_stop: bool = True

    # ── Computed properties ───────────────────────────────────────────────────
    @property
    def is_live_trading_enabled(self) -> bool:
        return self.trading_mode == "live"

    @property
    def kite_credentials_configured(self) -> bool:
        """Real credentials, not the mock surface."""
        return bool(self.kite_api_key and self.kite_api_secret)


settings = Settings()
