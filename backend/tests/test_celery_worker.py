"""
Celery worker — app/worker.py.

Until this file existed there was no ``app/worker.py`` and the compose
``worker`` service (which ran ``celery -A app.worker worker``) could not start;
that is the CI_SECURITY_AUDIT.md limitation the module was written to close.

These tests pin down the parts that matter operationally:

* the Celery application topology (name, broker/backend, queue, IST schedule),
* the beat schedule — exactly the two intended jobs, and their task names
  resolve on the app,
* ``daily_pnl_summary`` — correct day scoping across IST boundaries, the
  realised-P&L/+costs arithmetic, strategy aggregation, the report notification
  written once (idempotent rerun), and that out-of-day trades are excluded,
* the Kite session-health task — skipped unless every guard is armed, and
  fail-closed (never a silent pass) when validation fails.

Tasks are invoked through ``.run()`` (the sync body) rather than the broker,
so no Redis is required and every case is deterministic.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import app.worker as worker
from app.database.models import AccountCash, Notification, Position, Trade

DAY = "2026-09-04"


def _make_session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return engine, async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
def migrated_sf():
    import asyncio

    from app.database.session import apply_migrations

    engine, sf = _make_session_factory()
    asyncio.run(apply_migrations(engine))
    yield sf
    asyncio.run(engine.dispose())


async def _seed_book(sf, user_id: int) -> None:
    async with sf() as s:
        s.add(
            AccountCash(
                user_id=user_id,
                trading_mode="paper",
                cash=1_000_000,
                reserved=0.0,
                realized_pnl=0.0,
                total_costs=0.0,
            )
        )
        s.add(
            Position(
                user_id=user_id,
                symbol="RELIANCE",
                exchange="NSE",
                quantity=10,
                average_price=2400.0,
                strategy="intraday",
                entry_date=datetime(2026, 9, 4, 4, 0, tzinfo=timezone.utc),
                is_open=True,
            )
        )
        trades = [
            {
                "symbol": "RELIANCE",
                "exchange": "NSE",
                "transaction_type": "SELL",
                "quantity": 10,
                "price": 2401.25,
                "value": 24012.5,
                "total_costs": 3.15,
                "net_value": 24009.35,
                "strategy": "intraday",
                "realized_pnl": 12.5,
                "created_at": datetime(2026, 9, 4, 4, 45, tzinfo=timezone.utc),
            },
            {
                "symbol": "HDFCBANK",
                "exchange": "NSE",
                "transaction_type": "SELL",
                "quantity": 5,
                "price": 1620.0,
                "value": 8100.0,
                "total_costs": 2.4,
                "net_value": 8097.6,
                "strategy": "swing",
                "realized_pnl": -8.0,
                "created_at": datetime(2026, 9, 4, 5, 30, tzinfo=timezone.utc),
            },
            {
                "symbol": "RELIANCE",
                "exchange": "NSE",
                "transaction_type": "BUY",
                "quantity": 5,
                "price": 2390.0,
                "value": 11950.0,
                "total_costs": 2.0,
                "net_value": 11952.0,
                "strategy": "longterm",
                "realized_pnl": None,
                "created_at": datetime(2026, 9, 4, 4, 15, tzinfo=timezone.utc),
            },
            {
                # Outside the target IST day (2026-09-03 09:30 IST) — must not
                # be counted in the 2026-09-04 summary.
                "symbol": "TATASTEEL",
                "exchange": "NSE",
                "transaction_type": "SELL",
                "quantity": 2,
                "price": 140.0,
                "value": 280.0,
                "total_costs": 0.2,
                "net_value": 279.8,
                "strategy": "intraday",
                "realized_pnl": 999.0,
                "created_at": datetime(2026, 9, 3, 4, 0, tzinfo=timezone.utc),
            },
        ]
        for t in trades:
            s.add(Trade(user_id=user_id, **t))
        await s.commit()


async def _count_notifications(sf, user_id: int) -> int:
    from sqlalchemy import func, select

    async with sf() as s:
        return int(
            (
                await s.execute(
                    select(func.count())
                    .select_from(Notification)
                    .where(
                        Notification.user_id == user_id,
                        Notification.type == "daily_pnl_summary",
                    )
                )
            ).scalar()
            or 0
        )


# ---------------------------------------------------------------------------
# Celery application topology
# ---------------------------------------------------------------------------


class TestCeleryAppTopology:
    def test_app_name_and_broker(self):
        assert worker.celery_app.main == "algodollar"
        assert worker.celery_app.conf.broker_url == worker.settings.redis_url
        assert worker.celery_app.conf.result_backend == worker.settings.redis_url

    def test_ist_timezone_and_json_serialization(self):
        assert worker.celery_app.conf.timezone == "Asia/Kolkata"
        assert worker.celery_app.conf.enable_utc is False
        assert worker.celery_app.conf.task_serializer == "json"
        assert worker.celery_app.conf.accept_content == ["json"]
        assert worker.celery_app.conf.task_default_queue == "algodollar"

    def test_beat_schedule_registers_both_jobs_and_names_resolve(self):
        sched = worker.celery_app.conf.beat_schedule
        assert set(sched) == {"daily-pnl-summary", "kite-session-health"}
        assert sched["daily-pnl-summary"]["task"] == "app.worker.daily_pnl_summary"
        assert sched["kite-session-health"]["task"] == "app.worker.refresh_kite_token"
        for job in sched.values():
            assert job["task"] in worker.celery_app.tasks

    def test_tasks_are_registered_on_the_app(self):
        for name in (
            "app.worker.daily_pnl_summary",
            "app.worker.refresh_kite_token",
            "app.worker.ping",
        ):
            assert name in worker.celery_app.tasks


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------


class TestTradingCalendarHelper:
    def test_previous_trading_day_skips_weekends(self):
        assert worker._previous_trading_day(date(2026, 9, 7)) == date(2026, 9, 4)  # Mon -> Fri
        assert worker._previous_trading_day(date(2026, 9, 6)) == date(2026, 9, 4)  # Sun -> Fri
        assert worker._previous_trading_day(date(2026, 9, 5)) == date(2026, 9, 4)  # Sat -> Fri
        assert worker._previous_trading_day(date(2026, 9, 4)) == date(2026, 9, 3)  # Fri -> Thu

    def test_ist_day_bounds_are_utc_anchored(self):
        from zoneinfo import ZoneInfo

        ist = ZoneInfo("Asia/Kolkata")
        start, end = worker._ist_day_bounds(date(2026, 9, 4))
        assert start.tzinfo is not None and end.tzinfo is not None
        assert start == datetime(2026, 9, 3, 18, 30, tzinfo=timezone.utc)
        # The *local* wall clock is 2026-09-04 from 00:00 to 23:59:59.999 IST.
        assert end.astimezone(ist).year == 2026
        assert end.astimezone(ist).month == 9
        assert end.astimezone(ist).day == 4
        assert end.astimezone(ist).hour == 23


# ---------------------------------------------------------------------------
# daily_pnl_summary
# ---------------------------------------------------------------------------


class TestDailyPnlSummary:
    async def test_summary_arithmetic_and_day_scoping(self, migrated_sf):
        from app.execution.persistence import get_or_create_system_user

        user_id = await get_or_create_system_user(migrated_sf)
        await _seed_book(migrated_sf, user_id)

        summary = await worker._compute_daily_summary(
            migrated_sf, day=date.fromisoformat(DAY), mode="paper", user_id=user_id
        )

        assert summary["day"] == DAY
        assert summary["mode"] == "paper"
        assert summary["realized_pnl_rupees"] == pytest.approx(4.5)  # 12.5 - 8.0 (+0 for open leg)
        assert summary["total_costs_rupees"] == pytest.approx(7.55)
        assert summary["realized_net_pnl_rupees"] == pytest.approx(-3.05)
        assert summary["num_trades"] == 3  # the 2026-09-03 trade is excluded
        assert summary["num_buys"] == 1
        assert summary["num_sells"] == 2
        assert summary["open_positions"] == 1
        assert summary["cash_rupees"] == pytest.approx(1_000_000.0)
        assert summary["by_strategy"]["intraday"]["realized_pnl_rupees"] == pytest.approx(12.5)
        assert summary["by_strategy"]["swing"]["realized_pnl_rupees"] == pytest.approx(-8.0)
        assert summary["by_strategy"]["longterm"]["num_trades"] == 1

    async def test_empty_day_is_not_an_error(self, migrated_sf):
        from app.execution.persistence import get_or_create_system_user

        user_id = await get_or_create_system_user(migrated_sf)
        summary = await worker._compute_daily_summary(
            migrated_sf, day=date.fromisoformat(DAY), mode="paper", user_id=user_id
        )
        assert summary["num_trades"] == 0
        assert summary["realized_pnl_rupees"] == 0.0
        assert summary["realized_net_pnl_rupees"] == 0.0
        assert summary["cash_rupees"] is None  # no AccountCash row seeded

    def test_worker_task_records_notification_once(self, migrated_sf, monkeypatch):
        # The task body runs its own event loop (asyncio.run), so it must be
        # invoked from a sync test — never while pytest-asyncio's loop is live.
        import asyncio

        from app.execution.persistence import get_or_create_system_user

        monkeypatch.setattr(worker, "_session_factory", migrated_sf)
        user_id = asyncio.run(get_or_create_system_user(migrated_sf))
        asyncio.run(_seed_book(migrated_sf, user_id))

        result_a = worker.daily_pnl_summary.run(day=DAY, mode="paper")
        result_b = worker.daily_pnl_summary.run(day=DAY, mode="paper")

        assert result_a == result_b
        assert result_a["realized_pnl_rupees"] == pytest.approx(4.5)
        # Rerunning the same day must not stack a second recap.
        assert asyncio.run(_count_notifications(migrated_sf, user_id)) == 1

    def test_task_defaults_to_previous_trading_day_and_configured_mode(
        self, migrated_sf, monkeypatch
    ):
        import asyncio

        from app.execution.persistence import get_or_create_system_user

        monkeypatch.setattr(worker, "_session_factory", migrated_sf)
        user_id = asyncio.run(get_or_create_system_user(migrated_sf))
        asyncio.run(_seed_book(migrated_sf, user_id))

        result = worker.daily_pnl_summary.run()
        assert result["day"] == "2026-09-04"  # previous trading day from today
        assert result["mode"] == worker.settings.trading_mode


# ---------------------------------------------------------------------------
# refresh_kite_token (Kite session health / daily-token automation)
# ---------------------------------------------------------------------------


class _FakeBroker:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.connected = False
        self.disconnected_with_invalidate: bool | None = None

    async def connect(self) -> None:
        if self.fail:
            raise ConnectionError("boom")
        self.connected = True

    async def disconnect(self, invalidate_token: bool = False) -> None:
        self.disconnected_with_invalidate = invalidate_token
        self.connected = False


def _arm(monkeypatch, *, mock: bool = False, creds: bool = True, static_ip: bool = True):
    monkeypatch.setattr(worker.settings, "zerodha_mock_mode", mock)
    monkeypatch.setattr(worker.settings, "kite_api_key", "k" if creds else "")
    monkeypatch.setattr(worker.settings, "kite_api_secret", "s" if creds else "")
    monkeypatch.setattr(worker.settings, "kite_static_ip_enabled", static_ip)


class TestRefreshKiteTokenGuards:
    def test_skips_in_mock_mode(self, monkeypatch):
        _arm(monkeypatch, mock=True, static_ip=True)
        assert worker.refresh_kite_token()["status"] == "skipped"

    def test_skips_when_no_credentials(self, monkeypatch):
        _arm(monkeypatch, mock=False, creds=False, static_ip=True)
        assert worker.refresh_kite_token()["status"] == "skipped"

    def test_skips_when_static_ip_feature_not_armed(self, monkeypatch):
        _arm(monkeypatch, mock=False, creds=True, static_ip=False)
        assert worker.refresh_kite_token()["status"] == "skipped"

    def test_task_function_is_registered(self):
        assert "app.worker.refresh_kite_token" in worker.celery_app.tasks


class TestRefreshKiteTokenArmed:
    def test_validates_session_and_returns_ok(self, monkeypatch):
        _arm(monkeypatch)
        broker = _FakeBroker(fail=False)
        monkeypatch.setattr(worker, "_broker_factory", lambda: broker)

        result = worker.refresh_kite_token()
        assert result["status"] == "ok"
        assert broker.connected is False  # closed after the check
        assert broker.disconnected_with_invalidate is False  # token retained

    def test_fails_closed_when_session_invalid(self, monkeypatch):
        _arm(monkeypatch)
        broker = _FakeBroker(fail=True)
        monkeypatch.setattr(worker, "_broker_factory", lambda: broker)

        result = worker.refresh_kite_token()
        assert result["status"] == "failed"
        assert "login flow" in result["detail"]
        assert broker.disconnected_with_invalidate is False  # token preserved


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------


class TestPing:
    def test_ping_returns_ok(self):
        assert worker.ping.run() == {"status": "ok", "service": "algodollar-worker"}
