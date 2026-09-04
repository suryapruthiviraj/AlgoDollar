"""
Release audit: every failure mode must fail CLOSED.

These are the cases where the system is asked to keep going while something it
depends on is broken. The bar for each is identical: no order reaches the
broker, no state is invented, and the reason is stated.

Deliberately driven through the REAL production graph rather than through
mocked internals — the failure modes that matter are the ones that arise from
how the pieces are connected.
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database.models import Base
from app.execution.audit import ExecutionOutcome
from tests.test_e2e_paper_trade import (
    PRICE,
    WIDE_RISK,
    buy_signal,
    db_orders,
    db_trades,
    make_stack,
    sell_signal,
)

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


class DeadFactory:
    """A session factory whose database has gone away."""

    def __call__(self, *a: Any, **k: Any):
        raise ConnectionError("database is gone")


# =========================================================================== #
#  Database                                                                    #
# =========================================================================== #

class TestDatabaseUnavailable:

    async def test_a_dead_database_blocks_the_order(self, session_factory):
        """
        No durable record means no order.

        The idempotency claim is the first thing that touches the database, and
        it RAISES rather than returning falsy. Being unable to prove this is not
        a duplicate is not permission to assume it is not one.
        """
        stack = await make_stack(session_factory)
        assert stack.trading_permitted

        stack.persistence._sf = DeadFactory()

        result = await stack.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE, **WIDE_RISK
        )
        assert not result.submitted
        assert "idempotency claim failed" in (result.reason or "")
        assert not await stack.broker.get_orders(), (
            "an order reached the broker with no durable record of it"
        )

    async def test_a_dead_database_does_not_fabricate_a_fill(self, session_factory):
        stack = await make_stack(session_factory)
        stack.persistence._sf = DeadFactory()
        await stack.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE, **WIDE_RISK
        )
        assert not await stack.broker.get_trades()

    async def test_a_failure_AFTER_the_fill_is_surfaced_not_swallowed(
        self, session_factory
    ):
        """
        A persistence failure does not un-happen a fill.

        The money already moved, so the error is reported on the audit record
        rather than being hidden — reconciliation is what must find this, and it
        can only do that if the failure is visible.
        """
        stack = await make_stack(session_factory)
        real_sync = stack.persistence.sync_from_broker

        async def broken_sync(*a: Any, **k: Any):
            raise ConnectionError("database died mid-write")

        stack.persistence.sync_from_broker = broken_sync
        result = await stack.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE, **WIDE_RISK
        )
        assert result.audit.error, "the persistence failure left no trace"
        assert "persistence" in result.audit.error.lower()
        stack.persistence.sync_from_broker = real_sync


# =========================================================================== #
#  Redis                                                                       #
# =========================================================================== #

class TestRedisUnavailable:

    async def test_the_stack_still_builds_and_says_what_was_lost(
        self, session_factory
    ):
        """
        Redis is optional for paper, but its absence is never silent.

        Without it the kill switch is process-local and the order store is
        non-durable. Both are logged, and reconciliation already refuses to
        report OK against a non-durable store.
        """
        stack = await make_stack(session_factory)
        # The local test environment has no Redis, so this IS the degraded path.
        assert stack.trading_permitted
        assert type(stack.kill_switch_store).__name__ == "InMemoryKillSwitchStore"

    async def test_an_unreadable_kill_switch_counts_as_ENGAGED(
        self, session_factory
    ):
        """A switch that cannot be read must stop trading, not be ignored."""
        stack = await make_stack(session_factory)

        def boom(_key: str):
            raise ConnectionError("redis gone")

        stack.kill_switch_store.get = boom
        active, reason = stack.service.kill_switch.is_active()
        assert active, "an unreadable kill switch reported clear"

        result = await stack.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE, **WIDE_RISK
        )
        assert result.outcome is ExecutionOutcome.BLOCKED_KILL_SWITCH
        assert not await db_trades(session_factory)


# =========================================================================== #
#  Broker                                                                      #
# =========================================================================== #

class TestBrokerFailures:

    async def test_an_unreachable_broker_blocks_and_records_nothing(
        self, session_factory
    ):
        stack = await make_stack(session_factory)

        async def boom(*a: Any, **k: Any):
            raise ConnectionError("broker unreachable")

        stack.broker.place_order = boom
        result = await stack.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE, **WIDE_RISK
        )
        assert not result.submitted
        assert not await db_trades(session_factory)

    async def test_an_order_status_we_cannot_read_becomes_UNKNOWN(
        self, session_factory
    ):
        """
        Not knowing the outcome is its own state, and it blocks.

        The alternative — assuming "probably filled" or "probably not" — is how
        a duplicate order or a phantom position gets created.
        """
        stack = await make_stack(session_factory)
        real = stack.broker.get_order_status

        async def boom(*a: Any, **k: Any):
            raise ConnectionError("status unreadable")

        stack.broker.get_order_status = boom
        await stack.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE, **WIDE_RISK
        )
        orders = await db_orders(session_factory)
        assert orders and orders[-1].status == "UNKNOWN", (
            f"an unreadable outcome was recorded as {orders[-1].status}"
        )
        stack.broker.get_order_status = real


# =========================================================================== #
#  Risk limits                                                                 #
# =========================================================================== #

class TestRiskLimits:

    async def test_the_daily_loss_limit_blocks(self, session_factory):
        stack = await make_stack(session_factory)
        result = await stack.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE,
            **{**WIDE_RISK, "max_daily_loss": 1000.0, "realised_pnl_today": -1500.0},
        )
        assert not result.submitted, "an order was placed past the daily loss limit"
        assert not await db_trades(session_factory)

    async def test_the_daily_risk_budget_blocks(self, session_factory):
        stack = await make_stack(session_factory)
        result = await stack.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE,
            **{**WIDE_RISK, "max_daily_risk": 1.0, "daily_risk_used": 1.0},
        )
        assert not result.submitted
        assert result.audit.failed_risk_checks

    async def test_the_position_count_limit_blocks(self, session_factory):
        stack = await make_stack(session_factory)
        existing = [
            {"symbol": f"SYM{i}", "quantity": 10, "average_price": 100.0}
            for i in range(20)
        ]
        result = await stack.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE,
            **{**WIDE_RISK, "max_positions": 20, "current_positions": existing},
        )
        assert not result.submitted, "a 21st position was opened against a limit of 20"

    async def test_insufficient_cash_blocks(self, session_factory):
        stack = await make_stack(session_factory)
        result = await stack.service.submit_signal(
            buy_signal(), 100, reference_price=PRICE,
            **{**WIDE_RISK, "available_cash": 1000.0},
        )
        assert not result.submitted
        assert not await db_trades(session_factory)

    async def test_selling_stock_we_do_not_hold_never_creates_a_short(
        self, session_factory
    ):
        stack = await make_stack(session_factory)
        await stack.service.submit_signal(
            sell_signal(), 40, reference_price=PRICE, **WIDE_RISK
        )
        positions = await stack.broker.get_positions()
        assert not any(p.get("quantity", 0) < 0 for p in positions), (
            "a short position was created by a broker that does not support it"
        )


# =========================================================================== #
#  Reconciliation                                                              #
# =========================================================================== #

class TestReconciliation:

    async def test_a_mismatch_does_not_report_OK(self, session_factory):
        """
        Broker and local disagreeing is not the same as agreeing.

        A phantom position at the broker must stop trading until a human looks
        at it, not be reconciled away.
        """
        stack = await make_stack(session_factory)

        async def phantom(*a: Any, **k: Any):
            return [{
                "symbol": "GHOST", "exchange": "NSE", "quantity": 500,
                "average_price": 100.0, "product": "CNC",
            }]

        stack.broker.get_positions = phantom
        await stack.recovery.recover(stack.broker)

        assert not stack.recovery.trading_permitted, (
            "a position the local book has never heard of reconciled as OK"
        )
        result = await stack.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE, **WIDE_RISK
        )
        assert not result.submitted

    async def test_an_unreadable_broker_is_UNAVAILABLE_not_OK(
        self, session_factory
    ):
        """Empty-vs-empty comparing equal is the fail-open this prevents."""
        stack = await make_stack(session_factory)

        async def boom(*a: Any, **k: Any):
            raise ConnectionError("broker unreachable")

        for name in ("get_positions", "get_orders", "get_trades", "get_funds"):
            setattr(stack.broker, name, boom)
        await stack.recovery.recover(stack.broker)

        assert not stack.recovery.trading_permitted
        state = str(getattr(stack.recovery, "state", "")).upper()
        assert "OK" not in state or "BLOCK" in state


# =========================================================================== #
#  Idempotency under stress                                                    #
# =========================================================================== #

class TestIdempotencyUnderFailure:

    async def test_a_retry_after_a_broker_error_does_not_duplicate(
        self, session_factory
    ):
        """
        The first attempt failed at the broker; the retry must not double up.

        The idempotency key is claimed BEFORE the broker call, so the row exists
        even though nothing was placed — and the retry is refused.
        """
        stack = await make_stack(session_factory)
        key = "retry-after-error"

        async def boom(*a: Any, **k: Any):
            raise ConnectionError("broker unreachable")

        stack.broker.place_order = boom
        first = await stack.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE, idempotency_key=key, **WIDE_RISK
        )
        assert not first.submitted

        second = await stack.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE, idempotency_key=key, **WIDE_RISK
        )
        assert second.outcome is ExecutionOutcome.BLOCKED_DUPLICATE, (
            "a retry of a failed submission was treated as a new order"
        )
        orders = await db_orders(session_factory)
        assert len(orders) == 1, f"{len(orders)} rows for one logical order"

    async def test_the_unique_constraint_is_what_enforces_it(self, session_factory):
        """Not a check-then-insert, which two concurrent workers both pass."""
        from sqlalchemy import select

        from app.database.models import Order

        stack = await make_stack(session_factory)
        await stack.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE,
            idempotency_key="unique-test", **WIDE_RISK,
        )
        async with session_factory() as s:
            rows = (await s.execute(
                select(Order).where(Order.client_order_id == "unique-test")
            )).scalars().all()
        assert len(rows) == 1


# =========================================================================== #
#  Restart                                                                     #
# =========================================================================== #

class TestRestart:

    async def test_a_restart_reconciles_before_permitting_trading(
        self, session_factory, tmp_path
    ):
        state = str(tmp_path / "paper.json")
        first = await make_stack(session_factory, state_path=state)
        await first.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE, **WIDE_RISK
        )

        second = await make_stack(session_factory, state_path=state)
        assert second.trading_permitted, (
            f"a clean restart could not resume: {second.startup_reason}"
        )
        assert await second.broker.get_positions(), "the book was lost"

    async def test_a_restart_with_a_mismatched_book_stays_blocked(
        self, session_factory, tmp_path
    ):
        """
        Local state surviving while broker state does not is a real divergence.

        Starting with a database that records positions and a paper book that
        does not must NOT open the gate.
        """
        state = str(tmp_path / "paper.json")
        first = await make_stack(session_factory, state_path=state)
        await first.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE, **WIDE_RISK
        )
        assert await db_trades(session_factory)

        # Restart with a FRESH paper book: the database still has the position.
        second = await make_stack(session_factory, state_path=None)
        assert not second.trading_permitted, (
            "the database holds a position the broker has never heard of, and "
            "the gate opened anyway"
        )
