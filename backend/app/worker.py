"""
AlgoDollar Celery application and its scheduled tasks.

WHY THIS MODULE EXISTS
----------------------
The compose file shipped a ``worker`` service that could never start: it
pointed at ``python -m app.worker`` but no ``app/worker.py`` existed and no
Celery application was defined anywhere. Task queues were infrastructure
without a worker. This module is the missing entrypoint.

WHAT RUNS HERE VS WHAT RUNS IN THE API
--------------------------------------
The API process owns the real-time execution path — order submission goes
through the in-process ExecutionService, deliberately. The worker runs the
reporting and housekeeping that should outlive a restart and do not belong on
the request path:

* ``daily_pnl_summary``  — end-of-day realised P&L recap written to the
  database (Celery Beat, 15:35 IST weekdays).
* ``refresh_kite_token`` — the static-IP / daily-token automation surface.
  Kite access tokens are single-day; this task validates the configured
  session while the flag is armed and FAILS CLOSED when it cannot, rather
  than pretending a dead token still works.
* ``ping``              — a queue round-trip probe for health checks.

Anything here that can place an order is a bug. The only broker interaction
is read-only session validation, gated by ``zerodha_mock_mode`` and the
explicit ``kite_static_ip_enabled`` switch.

SAFETY NOTES
------------
* The worker never runs migrations and never creates schema — that is an
  operator action (``alembic upgrade head``), applied before the worker is
  started (deployment the same as the API).
* Scheduled tasks are idempotent: rerunning the daily summary for a day it
  already recorded does not duplicate the notification.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from celery import Celery
from celery.schedules import crontab
from sqlalchemy import func, select

from app.core.config import settings
from app.database.session import async_session_maker

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# The session factory tasks write through. Tests substitute this boundary; a
# production worker uses the same Postgres as the API.
_session_factory = async_session_maker


celery_app = Celery(
    "algodollar",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.worker"],
)

celery_app.conf.update(
    timezone="Asia/Kolkata",
    enable_utc=False,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    result_expires=60 * 60 * 24,
    broker_connection_retry_on_startup=True,
    task_default_queue="algodollar",
    task_default_exchange="algodollar",
    task_default_routing_key="algodollar",
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    beat_schedule={
        "daily-pnl-summary": {
            "task": "app.worker.daily_pnl_summary",
            # NSE close plus margin: the recap is written ~35 minutes after the
            # 15:30 IST session ends.
            "schedule": crontab(hour=15, minute=35, day_of_week="mon-fri"),
        },
        "kite-session-health": {
            "task": "app.worker.refresh_kite_token",
            # Just before the session, while there is still time for a human
            # to re-login if the previous day's token is dead.
            "schedule": crontab(hour=8, minute=45, day_of_week="mon-fri"),
        },
    },
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _previous_trading_day(today: Optional[date] = None) -> date:
    """The most recent Monday–Friday before ``today`` (IST by default)."""
    day = today or datetime.now(IST).date()
    day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _ist_day_bounds(day: date) -> tuple[datetime, datetime]:
    """UTC instants covering the full IST calendar ``day``."""
    start = datetime.combine(day, time.min, tzinfo=IST).astimezone(timezone.utc)
    end = datetime.combine(day, time.max, tzinfo=IST).astimezone(timezone.utc)
    return start, end


async def _resolve_system_user(session_factory: Any, mode: str) -> int:
    from app.execution.persistence import get_or_create_system_user

    return await get_or_create_system_user(session_factory)


# ---------------------------------------------------------------------------
# Daily P&L summary
# ---------------------------------------------------------------------------


async def _compute_daily_summary(
    session_factory: Any, *, day: date, mode: str, user_id: int
) -> dict:
    """Read-only day recap: realised P&L + costs + open book from the DB.

    ``realized_pnl`` is booked only on the closing side of a trade (see the
    Trade model), so this sums what the session actually realised, not a
    mark-to-market figure (equity/mark-to-market is the EquityTracker's job
    on the campaign path).
    """
    from app.database.models import AccountCash, Notification, Position, Trade

    start_utc, end_utc = _ist_day_bounds(day)

    realized = 0.0
    costs = 0.0
    num_trades = 0
    num_buys = 0
    num_sells = 0
    by_strategy: dict[str, dict[str, float | int]] = {}

    async with session_factory() as session:
        trades = (
            (
                await session.execute(
                    select(Trade)
                    .where(Trade.created_at >= start_utc, Trade.created_at < end_utc)
                    .order_by(Trade.created_at)
                )
            )
            .scalars()
            .all()
        )

        for t in trades:
            pnl = float(t.realized_pnl or 0.0)
            cost = float(t.total_costs or 0.0)
            realized += pnl
            costs += cost
            num_trades += 1
            if t.transaction_type == "BUY":
                num_buys += 1
            elif t.transaction_type == "SELL":
                num_sells += 1
            strat = t.strategy or "unknown"
            agg = by_strategy.setdefault(
                strat, {"realized_pnl_rupees": 0.0, "total_costs_rupees": 0.0, "num_trades": 0}
            )
            agg["realized_pnl_rupees"] += pnl  # type: ignore[operator]
            agg["total_costs_rupees"] += cost  # type: ignore[operator]
            agg["num_trades"] += 1  # type: ignore[operator]

        cash_row = (
            await session.execute(
                select(AccountCash).where(
                    AccountCash.user_id == user_id, AccountCash.trading_mode == mode
                )
            )
        ).scalar_one_or_none()
        cash = float(cash_row.cash) if cash_row is not None else None

        open_positions = (
            await session.execute(
                select(func.count())
                .select_from(Position)
                .where(Position.user_id == user_id, Position.is_open)
            )
        ).scalar() or 0

        summary = {
            "day": day.isoformat(),
            "mode": mode,
            "realized_pnl_rupees": round(realized, 2),
            "total_costs_rupees": round(costs, 2),
            "realized_net_pnl_rupees": round(realized - costs, 2),
            "num_trades": num_trades,
            "num_buys": num_buys,
            "num_sells": num_sells,
            "by_strategy": by_strategy,
            "open_positions": open_positions,
            "cash_rupees": round(cash, 2) if cash is not None else None,
        }

        # Idempotent recap: one notification per day. A beat retry or a manual
        # re-run of the same day must not stack notifications.
        existing = (
            await session.execute(
                select(Notification).where(
                    Notification.user_id == user_id,
                    Notification.type == "daily_pnl_summary",
                    Notification.title.like(f"[{day.isoformat()}] %"),
                )
            )
        ).scalar_one_or_none()

        if existing is None:
            session.add(
                Notification(
                    user_id=user_id,
                    type="daily_pnl_summary",
                    title=f"[{day.isoformat()}] Daily P&L summary",
                    message=(
                        f"{num_trades} trades, realised {realized:+,.2f} Rs "
                        f"- total costs {costs:,.2f} Rs "
                        f"= net {realized - costs:+,.2f} Rs "
                        f"({num_buys} buys / {num_sells} sells). "
                        f"Open positions: {open_positions}."
                    ),
                )
            )
            await session.commit()

    return summary


@celery_app.task(
    name="app.worker.daily_pnl_summary",
    autoretry_for=(Exception,),
    retry_backoff=60,
    max_retries=5,
)
def daily_pnl_summary(day: Optional[str] = None, mode: Optional[str] = None) -> dict:
    """
    Compute and record the previous trading day's realised P&L recap.

    ``day`` is an ISO date string (defaults to the previous trading day) and
    ``mode`` to ``paper``/``live`` (defaults to the configured trading mode).
    Fail-closed: if the database cannot be read the task retries and, failing
    that, surfaced as a failed task — an EOD recap that silently vanished is
    worse than one that visibly failed.
    """
    resolved_day = date.fromisoformat(day) if day else _previous_trading_day()
    resolved_mode = mode or str(settings.trading_mode)

    async def _run() -> dict:
        user_id = await _resolve_system_user(_session_factory, resolved_mode)
        return await _compute_daily_summary(
            _session_factory, day=resolved_day, mode=resolved_mode, user_id=user_id
        )

    try:
        summary = asyncio.run(_run())
    except Exception:
        logger.exception("daily_pnl_summary failed for %s", resolved_day.isoformat())
        raise
    logger.info(
        "daily_pnl_summary",
        extra={
            "day": summary["day"],
            "mode": summary["mode"],
            "realized": summary["realized_pnl_rupees"],
            "net": summary["realized_net_pnl_rupees"],
            "trades": summary["num_trades"],
        },
    )
    return summary


# ---------------------------------------------------------------------------
# Kite session health / daily-token automation
# ---------------------------------------------------------------------------


def _build_kite_broker():
    """The live-session handle tasks interact with. Swappable in tests."""
    from app.broker.zerodha import ZerodhaBroker

    return ZerodhaBroker(
        settings.kite_api_key,
        settings.kite_api_secret,
        settings.kite_access_token,
    )


_broker_factory = _build_kite_broker


@celery_app.task(name="app.worker.refresh_kite_token")
def refresh_kite_token() -> dict:
    """
    Validate the Kite session ahead of the trading day; fail closed.

    Kite access tokens are single-day: they cannot be refreshed server-side,
    only re-issued through the interactive login flow. What automation CAN do
    daily is confirm the configured token still authenticates BEFORE the
    market opens, so a dead session is surfaced before it matters instead of
    finding out at the first order.

    Armed only when all three switches agree: not mock mode, real credentials
    configured, and `kite_static_ip_enabled` explicitly set. Anything less
    returns {"status": "skipped"}; a validation failure returns
    {"status": "failed"} — never a silent pass.
    """
    if settings.zerodha_mock_mode:
        return {"status": "skipped", "reason": "zerodha_mock_mode is set; no live session surface"}
    if not settings.kite_credentials_configured:
        return {"status": "skipped", "reason": "no Kite credentials configured"}
    if not settings.kite_static_ip_enabled:
        return {
            "status": "skipped",
            "reason": "kite_static_ip_enabled is False; daily-token automation not armed",
        }

    broker = _broker_factory()
    try:
        asyncio.run(broker.connect())
    except Exception as exc:  # noqa: BLE001
        logger.exception("kite session validation failed")
        return {
            "status": "failed",
            "error": repr(exc),
            "detail": (
                "The configured Kite access token could not be validated. "
                "A human must re-run the login flow before live trading "
                "resumes. Failing closed — no refresh was attempted."
            ),
        }
    finally:
        try:
            asyncio.run(broker.disconnect(invalidate_token=False))
        except Exception:  # noqa: BLE001
            logger.exception("kite disconnect during health check failed")

    logger.info("kite session validated; access token is live for today")
    return {
        "status": "ok",
        "detail": (
            "Kite session validated — the access token authenticates. Tokens "
            "are single-day; the next scheduled check runs before the next "
            "session."
        ),
    }


# ---------------------------------------------------------------------------
# Health probe
# ---------------------------------------------------------------------------


@celery_app.task(name="app.worker.ping")
def ping() -> dict:
    """A broker round-trip probe for ``celery -A app.worker inspect ping``."""
    return {"status": "ok", "service": "algodollar-worker"}
