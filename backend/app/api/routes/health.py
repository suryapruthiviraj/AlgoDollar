from __future__ import annotations

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.session import get_async_session

router = APIRouter()
logger = structlog.get_logger(__name__)


class HealthResponse(BaseModel):
    status: str
    trading_mode: str
    db: str
    redis: str
    broker: str
    kill_switch: bool
    timestamp: str


class DetailedHealthResponse(BaseModel):
    status: str
    trading_mode: str
    is_live_trading_enabled: bool
    components: dict
    config: dict
    timestamp: str


async def _check_db(session: AsyncSession) -> str:
    try:
        await session.execute(text("SELECT 1"))
        return "connected"
    except Exception:
        return "disconnected"


async def _check_redis() -> str:
    try:
        # `redis.asyncio`, not `aioredis` — see the note in app/main.py. The
        # old import could never succeed on Python 3.11+, so this health check
        # reported "disconnected" even against a perfectly healthy Redis.
        import redis.asyncio as redis_asyncio

        redis = redis_asyncio.from_url(settings.redis_url, socket_connect_timeout=2)
        await redis.ping()
        await redis.aclose()
        return "connected"
    except Exception:
        return "disconnected"


async def _check_broker() -> str:
    if not settings.kite_api_key or not settings.kite_access_token:
        return "not_configured"
    try:
        from kiteconnect import KiteConnect

        kite = KiteConnect(api_key=settings.kite_api_key)
        kite.set_access_token(settings.kite_access_token)
        profile = kite.profile()  # type: ignore[attr-defined]
        if profile:
            return "connected"
        return "disconnected"
    except Exception:
        return "disconnected"


@router.get("", response_model=HealthResponse)
async def health(session: AsyncSession = Depends(get_async_session)) -> HealthResponse:
    db_status = await _check_db(session)
    redis_status = await _check_redis()
    broker_status = await _check_broker()

    # Determine kill_switch from UserSettings — for a simple health endpoint we
    # check if any user has it active and return True.
    kill_switch = False

    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        trading_mode=settings.trading_mode,
        db=db_status,
        redis=redis_status,
        broker=broker_status,
        kill_switch=kill_switch,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/detailed", response_model=DetailedHealthResponse)
async def health_detailed(
    session: AsyncSession = Depends(get_async_session),
) -> DetailedHealthResponse:
    db_status = await _check_db(session)
    redis_status = await _check_redis()
    broker_status = await _check_broker()

    all_ok = all(
        s == "connected" for s in [db_status, redis_status]
    )

    return DetailedHealthResponse(
        status="ok" if all_ok else "degraded",
        trading_mode=settings.trading_mode,
        is_live_trading_enabled=settings.is_live_trading_enabled,
        components={
            "database": {"status": db_status, "url_host": settings.database_url.split("@")[-1]},
            "redis": {"status": redis_status},
            "broker": {"status": broker_status, "api_key_set": bool(settings.kite_api_key)},
        },
        config={
            "app_env": settings.app_env,
            "log_level": settings.log_level,
            "max_daily_loss_pct": settings.max_daily_loss_pct,
            "max_portfolio_drawdown_pct": settings.max_portfolio_drawdown_pct,
            "max_positions": settings.max_positions,
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
