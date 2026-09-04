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

    # ── Computed properties ───────────────────────────────────────────────────
    @property
    def is_live_trading_enabled(self) -> bool:
        return self.trading_mode == "live"


settings = Settings()
