from __future__ import annotations

from typing import Literal, Optional

from pydantic import AnyHttpUrl, field_validator
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
    app_env: Literal["development", "staging", "paper", "live"] = "development"

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
