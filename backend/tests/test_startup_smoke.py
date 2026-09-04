"""
Smoke test of the REAL application startup, end to end.

``tests/test_e2e_paper_trade.py`` proves the execution graph works when
assembled by ``build_production_stack``. This proves that ``app.main`` actually
assembles it — that a process started the way a deployment starts it arrives at
a state where it can trade.

The distinction is the whole point. Every defect in this area so far has been a
wiring defect rather than a component defect: the objects were correct and were
simply never connected, or were connected with every collaborator defaulted to
None. A test that constructs the stack itself cannot catch that. This one runs
``app.main``'s lifespan.

WHAT IS SUBSTITUTED
-------------------
Only the database (SQLite instead of PostgreSQL) and the market data feed. The
lifespan, the router set, the settings object, the execution stack builder, the
startup reconciliation and the trading gate are all the production ones.

The lifespan is entered directly via ``app.router.lifespan_context`` rather
than through ``starlette.testclient.TestClient``, which under Starlette 1.x
requires an ``httpx2`` package this project does not depend on. Entering the
context is also the more honest test: it runs the same startup coroutine
uvicorn runs, with no test client in between.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database.models import Base, Order, Trade
from app.execution.audit import ExecutionOutcome
from tests.test_e2e_paper_trade import (
    MARKET_OPEN_IST,
    OPENING_CASH,
    PRICE,
    WIDE_RISK,
    DeterministicFeed,
    buy_signal,
    db_cash,
    db_orders,
    db_positions,
    db_trades,
)

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def sqlite_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


class TestRealStartupWiring:

    async def test_the_application_lifespan_builds_a_tradeable_stack(
        self, sqlite_factory, monkeypatch
    ):
        """
        Start the app the way uvicorn does, then trade through what it built.

        Nothing is reached into: the ExecutionService used here is the one
        ``app.state`` published during startup.
        """
        import httpx

        import app.execution.runtime as runtime_mod
        from app.main import create_app

        real_build = runtime_mod.build_production_stack

        async def build_with_test_boundaries(**kwargs):
            # Substitute ONLY the two external boundaries. Every internal
            # collaborator is still resolved by the production builder.
            kwargs["session_factory"] = sqlite_factory
            kwargs["data_broker"] = DeterministicFeed()
            kwargs["paper_clock"] = lambda: MARKET_OPEN_IST
            kwargs["paper_state_path"] = None
            return await real_build(**kwargs)

        monkeypatch.setattr(
            runtime_mod, "build_production_stack", build_with_test_boundaries
        )

        app = create_app()
        async with app.router.lifespan_context(app):
            # ---- what startup actually produced ------------------------ #
            stack = getattr(app.state, "execution_stack", None)
            assert stack is not None, (
                "startup did not build an execution stack; the application "
                "would serve read-only and no order could ever be placed"
            )
            assert app.state.execution_service is stack.service, (
                "the service on app.state is not the one that was built"
            )
            assert stack.trading_permitted, (
                f"the app started but cannot trade: {stack.startup_reason}"
            )
            assert stack.persistence is not None, (
                "the stack has no persistence, so nothing an order does would "
                "be written down"
            )
            assert type(stack.broker).__name__ == "PaperBroker", (
                f"paper must be the default; got {type(stack.broker).__name__}"
            )

            # ---- the API is up ----------------------------------------- #
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                health = await client.get("/api/v1/health")
            assert health.status_code < 500, f"health returned {health.status_code}"

            # ---- a real paper trade through the started stack ---------- #
            result = await app.state.execution_service.submit_signal(
                buy_signal(), 10, reference_price=PRICE, **WIDE_RISK
            )
            assert result.outcome is ExecutionOutcome.SUBMITTED, (
                f"the started application could not place a paper order: "
                f"{result.outcome} — {result.reason}"
            )

        # ---- and it was persisted ---------------------------------------- #
        orders = await db_orders(sqlite_factory)
        trades = await db_trades(sqlite_factory)
        positions = await db_positions(sqlite_factory)
        cash = await db_cash(sqlite_factory)

        assert len(orders) == 1 and orders[0].status == "COMPLETE"
        assert len(trades) == 1 and trades[0].quantity == 10
        assert positions and positions[0].quantity == 10
        assert float(cash.cash) < OPENING_CASH, "the fill did not reduce cash"

    async def test_startup_publishes_a_trading_pipeline(
        self, sqlite_factory, monkeypatch
    ):
        """The signal pipeline must be reachable, not just the execution stack."""
        import app.execution.runtime as runtime_mod
        from app.main import create_app

        real_build = runtime_mod.build_production_stack

        async def build_with_test_boundaries(**kwargs):
            kwargs["session_factory"] = sqlite_factory
            kwargs["data_broker"] = DeterministicFeed()
            kwargs["paper_clock"] = lambda: MARKET_OPEN_IST
            kwargs["paper_state_path"] = None
            return await real_build(**kwargs)

        monkeypatch.setattr(
            runtime_mod, "build_production_stack", build_with_test_boundaries
        )

        app = create_app()
        async with app.router.lifespan_context(app):
            pipeline = getattr(app.state, "trading_pipeline", None)
            assert pipeline is not None, (
                "startup built an execution stack but no pipeline, so nothing "
                "in the running application can produce a signal"
            )
            assert pipeline.service is app.state.execution_service, (
                "the pipeline routes to a different service than the one "
                "startup published — there would be two order paths"
            )
            assert pipeline.universe, "the pipeline has an empty universe"
            assert pipeline.data is not None, "the pipeline has no price source"

    async def test_a_cycle_is_not_started_on_a_timer(self):
        """
        Nothing schedules itself into placing orders.

        A cycle happens because something asked for one. This is checked
        structurally because the failure mode — an app that starts trading by
        itself on deploy — is not one to discover in production.
        """
        import pathlib
        import re

        src = pathlib.Path("app/main.py").read_text()
        assert not re.search(r"create_task|ensure_future|BackgroundTasks|add_job", src), (
            "app/main.py schedules background work; a trading cycle must be "
            "triggered explicitly, never by a timer installed at startup"
        )
        assert "run_once(" not in src, "startup runs a trading cycle by itself"

    async def test_paper_is_the_default_trading_mode(self):
        """Never live unless someone explicitly asked for it."""
        from app.core.config import Settings

        assert Settings().trading_mode == "paper"
        assert Settings().is_live_trading_enabled is False

    async def test_startup_survives_a_missing_execution_stack(self, monkeypatch):
        """
        A failure to build the stack must not leave a half-configured object.

        The API still starts and serves read-only rather than crashing — but
        `execution_service` must be None, so any attempt to trade fails loudly
        instead of finding something that looks usable.
        """
        import httpx

        import app.execution.runtime as runtime_mod
        from app.main import create_app

        async def explode(**_):
            raise RuntimeError("dependency unavailable")

        monkeypatch.setattr(runtime_mod, "build_production_stack", explode)

        app = create_app()
        async with app.router.lifespan_context(app):
            assert app.state.execution_stack is None
            assert app.state.execution_service is None
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                assert (await client.get("/api/v1/health")).status_code < 500


class TestNoSecondPathToABroker:
    """
    ExecutionService must remain the only way an order reaches a venue.

    A structural check, not a behavioural one: it fails when someone adds a new
    caller, which is exactly when it should fail.
    """

    def test_only_the_execution_layer_calls_place_order(self):
        import pathlib
        import re

        allowed = {
            # The boundary itself, and the brokers that implement the call.
            "app/execution/order_manager.py",
            "app/broker/paper.py",
            "app/broker/zerodha.py",
            "app/broker/base.py",
            "app/broker/marketdata.py",
        }
        offenders: list[str] = []
        for path in pathlib.Path("app").rglob("*.py"):
            rel = str(path).replace("\\", "/")
            if rel in allowed:
                continue
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if re.search(r"\.place_order\s*\(", line) and not line.strip().startswith("#"):
                    offenders.append(f"{rel}:{i}: {line.strip()}")

        assert not offenders, (
            "a second path to a broker was introduced — orders must go through "
            "ExecutionService.submit_signal so the kill switch, risk, "
            "eligibility, idempotency and audit gates cannot be skipped:\n"
            + "\n".join(offenders)
        )

    async def test_the_market_data_feed_cannot_execute(self):
        """A price source that could also trade would be a second venue path."""
        from app.broker.marketdata import MarketDataBroker

        class _Provider:
            def fetch_symbol(self, *a, **k):
                return None

        feed = MarketDataBroker(_Provider())
        with pytest.raises(NotImplementedError):
            await feed.place_order(symbol="X")
        with pytest.raises(NotImplementedError):
            await feed.cancel_order("1")
        with pytest.raises(NotImplementedError):
            await feed.modify_order("1")


class TestSchemaSupportsTheGuarantees:
    """The durability guarantees are enforced by the schema, not by convention."""

    def test_client_order_id_is_unique_per_user(self):
        constraints = {
            tuple(sorted(c.columns.keys()))
            for c in Order.__table__.constraints
            if c.__class__.__name__ == "UniqueConstraint"
        }
        assert ("client_order_id", "user_id") in constraints, (
            "without a UNIQUE constraint the idempotency check is a "
            "check-then-insert, which two concurrent workers both pass"
        )

    def test_broker_trade_id_is_unique_per_user(self):
        constraints = {
            tuple(sorted(c.columns.keys()))
            for c in Trade.__table__.constraints
            if c.__class__.__name__ == "UniqueConstraint"
        }
        assert ("trade_id_broker", "user_id") in constraints, (
            "without this, replaying a broker trade book double-counts fills"
        )


class TestSchemaAndShutdown:
    """Deployment-surface behaviour: schema drift and clean release."""

    async def test_a_clean_database_reports_no_drift(self, sqlite_factory):
        """The models and a freshly created schema must agree."""
        import app.database.session as sess

        engine = sqlite_factory.kw["bind"]
        from sqlalchemy import inspect

        def _check(conn):
            from app.database.models import Base

            inspector = inspect(conn)
            existing = set(inspector.get_table_names())
            missing = [t for t in Base.metadata.tables if t not in existing]
            assert not missing, f"tables never created: {missing}"

        async with engine.begin() as conn:
            await conn.run_sync(_check)
        assert callable(sess.verify_schema)

    async def test_drift_is_reported_rather_than_raised(self):
        """
        A stale schema must be NAMED, not discovered at the first query.

        Reported rather than raised on purpose: refusing to start would take
        down a read-only API that is otherwise fine, and the execution path
        already fails closed on a persistence error.
        """
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        import app.database.session as sess
        from app.database.models import Base

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        original = sess.engine
        sess.engine = engine
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            assert await sess.verify_schema() == []

            async with engine.begin() as conn:
                await conn.execute(text("DROP TABLE account_cash"))
            drift = await sess.verify_schema()
            assert any("account_cash" in d for d in drift), drift
        finally:
            sess.engine = original
            await engine.dispose()

    async def test_shutdown_closes_what_startup_opened(
        self, sqlite_factory, monkeypatch
    ):
        """
        Redis clients opened by the stack must be released.

        Without a handle on them they were unreachable, so every restart leaked
        its connections and a container cycling repeatedly would exhaust the
        server's client limit.
        """
        import app.execution.runtime as runtime_mod
        from app.main import create_app

        closed: list[str] = []

        class FakeRedis:
            def __init__(self, name: str) -> None:
                self.name = name

            async def aclose(self) -> None:
                closed.append(self.name)

        real = runtime_mod.build_production_stack

        async def build(**kw):
            kw["session_factory"] = sqlite_factory
            kw["data_broker"] = DeterministicFeed()
            kw["paper_clock"] = lambda: MARKET_OPEN_IST
            kw["paper_state_path"] = None
            stack = await real(**kw)
            stack.redis_clients = (FakeRedis("async"), FakeRedis("sync"))
            return stack

        monkeypatch.setattr(runtime_mod, "build_production_stack", build)

        app = create_app()
        async with app.router.lifespan_context(app):
            assert app.state.execution_stack is not None

        assert sorted(closed) == ["async", "sync"], (
            f"shutdown released {closed}; the rest leaked"
        )

    async def test_closing_twice_is_safe(self, sqlite_factory):
        """Shutdown may run more than once; it must not raise."""
        from tests.test_e2e_paper_trade import make_stack

        stack = await make_stack(sqlite_factory)
        await stack.aclose()
        await stack.aclose()

    async def test_an_uncooperative_client_does_not_abort_shutdown(
        self, sqlite_factory
    ):
        """One bad client must not prevent the others being released."""
        from tests.test_e2e_paper_trade import make_stack

        released: list[str] = []

        class Bad:
            async def aclose(self):
                raise ConnectionError("already gone")

        class Good:
            async def aclose(self):
                released.append("good")

        stack = await make_stack(sqlite_factory)
        stack.redis_clients = (Bad(), Good())
        await stack.aclose()
        assert released == ["good"]
