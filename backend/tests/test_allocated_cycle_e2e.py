"""
The allocator wired to the real paper execution path.

`test_portfolio_allocation.py` tests the engine in isolation. This drives the
whole chain — market data, strategy, allocation, execution, fill, persistence —
through the production graph, so a target that the engine produces actually
becomes a persisted paper trade.

The safety assertions matter more than the happy path: an allocation must never
reach the broker when the kill switch is on, when the drawdown limit is
breached, or when reconciliation has not succeeded.
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database.models import Base
from app.engine.allocated_cycle import (
    estimate_risk_inputs,
    run_allocated_cycle,
)
from tests.test_e2e_paper_trade import db_orders, db_positions, db_trades, make_stack
from tests.test_engine_pipeline import (
    TRENDS,
    UNIVERSE,
    HistoryFeed,
    StubAlphaModel,
    _feature_builder,
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


async def make_allocated(session_factory: Any, *, with_model: bool = True):
    """The production stack plus a pipeline whose signals reach the allocator."""
    from app.engine.pipeline import build_default_pipeline
    from app.strategies.swing import SwingStrategy

    feed = HistoryFeed(TRENDS)
    stack = await make_stack(session_factory, feed=feed)
    kwargs: dict[str, Any] = {}
    if with_model:
        kwargs["strategies"] = [
            SwingStrategy(paper_mode=True, alpha_model=StubAlphaModel())
        ]
        kwargs["feature_builder"] = _feature_builder
    pipeline = build_default_pipeline(
        execution_service=stack.service, data_broker=feed,
        universe=UNIVERSE, max_orders_per_cycle=20, **kwargs,
    )
    return stack, pipeline, feed


# =========================================================================== #
#  Risk estimation from real history                                          #
# =========================================================================== #

class TestRiskEstimation:

    async def test_volatility_and_traded_value_come_from_the_data(self, session_factory):
        _, pipeline, _ = await make_allocated(session_factory)
        md = await pipeline.fetch_market_data(UNIVERSE)
        vols, traded, corr_syms, corr = estimate_risk_inputs(md, list(UNIVERSE))

        assert len(vols) == len(UNIVERSE)
        assert all(v > 0 for v in vols.values())
        assert len(traded) == len(UNIVERSE)
        assert corr is not None and len(corr) == len(corr_syms)

    async def test_a_symbol_with_no_history_gets_no_estimate(self, session_factory):
        """A defaulted volatility would size a position nobody measured."""
        _, pipeline, _ = await make_allocated(session_factory)
        md = await pipeline.fetch_market_data(UNIVERSE)
        vols, traded, _, _ = estimate_risk_inputs(md, [*UNIVERSE, "NOSUCH"])
        assert "NOSUCH" not in vols
        assert "NOSUCH" not in traded

    async def test_the_correlation_matrix_has_a_unit_diagonal(self, session_factory):
        _, pipeline, _ = await make_allocated(session_factory)
        md = await pipeline.fetch_market_data(UNIVERSE)
        _, _, syms, corr = estimate_risk_inputs(md, list(UNIVERSE))
        assert corr is not None
        for i in range(len(syms)):
            assert corr[i][i] == pytest.approx(1.0, abs=1e-9)


# =========================================================================== #
#  The full chain                                                             #
# =========================================================================== #

class TestAllocatedCycleEndToEnd:

    async def test_an_allocation_becomes_persisted_paper_trades(self, session_factory):
        """market data -> signals -> target -> orders -> fills -> database."""
        stack, pipeline, _ = await make_allocated(session_factory)
        assert stack.trading_permitted

        result = await run_allocated_cycle(
            pipeline=pipeline, execution_service=stack.service,
            total_capital=5_000_000.0, cash=5_000_000.0,
        )

        assert result.target is not None, f"no target: {result.errors}"
        assert not result.target.is_no_trade, result.target.summary()
        assert result.submitted > 0, (
            f"the allocator produced a target that placed no orders: "
            f"{result.summary()} / {result.outcomes}"
        )

        orders = await db_orders(session_factory)
        trades = await db_trades(session_factory)
        positions = await db_positions(session_factory)
        assert len(orders) == result.submitted
        assert trades, "submitted orders produced no persisted fill"
        assert positions, "fills produced no position"

    async def test_the_allocation_fingerprint_reaches_the_audit_record(
        self, session_factory
    ):
        """An order must be traceable back to the allocation that caused it."""
        stack, pipeline, _ = await make_allocated(session_factory)
        result = await run_allocated_cycle(
            pipeline=pipeline, execution_service=stack.service,
            total_capital=5_000_000.0, cash=5_000_000.0,
        )
        assert result.submitted > 0
        records = stack.audit.sinks[0].records
        with_alloc = [
            r for r in records
            if (r.portfolio_allocation or {}).get("allocation_fingerprint")
        ]
        assert with_alloc, "no audit record carries the allocation fingerprint"
        assert all(
            r.portfolio_allocation["allocation_fingerprint"]
            == result.target.input_fingerprint
            for r in with_alloc
        )

    async def test_a_dry_run_produces_a_target_but_no_orders(self, session_factory):
        stack, pipeline, _ = await make_allocated(session_factory)
        result = await run_allocated_cycle(
            pipeline=pipeline, execution_service=stack.service,
            total_capital=5_000_000.0, cash=5_000_000.0, dry_run=True,
        )
        assert result.target is not None
        assert result.submitted == 0
        assert not await db_orders(session_factory)

    async def test_every_constraint_that_bound_is_reported(self, session_factory):
        stack, pipeline, _ = await make_allocated(session_factory)
        result = await run_allocated_cycle(
            pipeline=pipeline, execution_service=stack.service,
            total_capital=200_000.0, cash=200_000.0, dry_run=True,
        )
        assert result.target is not None
        assert result.target.binding_constraints, (
            "a small book against a 10% position cap bound nothing"
        )
        for c in result.target.binding_constraints:
            assert c.requested >= c.applied - 1e-9


# =========================================================================== #
#  Safety — the allocator must never reach the broker when blocked            #
# =========================================================================== #

class TestAllocatedCycleSafety:

    async def test_the_kill_switch_stops_the_cycle_before_any_order(
        self, session_factory
    ):
        stack, pipeline, _ = await make_allocated(session_factory)
        stack.kill_switch_store.engage("operator halt")

        result = await run_allocated_cycle(
            pipeline=pipeline, execution_service=stack.service,
            total_capital=5_000_000.0, cash=5_000_000.0,
            kill_switch_active=True,
        )
        assert result.submitted == 0
        assert result.target is not None and result.target.is_no_trade
        assert not await db_trades(session_factory)

    async def test_the_execution_gate_still_stops_an_allocation_that_slips_past(
        self, session_factory
    ):
        """
        The allocator is not the safety layer; the boundary is.

        Here the allocator is told nothing is wrong while the kill switch is
        genuinely engaged, so it produces a full target. Every order must still
        be refused downstream.
        """
        stack, pipeline, _ = await make_allocated(session_factory)
        stack.kill_switch_store.engage("engaged behind the allocator's back")

        result = await run_allocated_cycle(
            pipeline=pipeline, execution_service=stack.service,
            total_capital=5_000_000.0, cash=5_000_000.0,
            kill_switch_active=False,
        )
        assert result.target is not None and not result.target.is_no_trade, (
            "the allocator did not produce a target, so the boundary was untested"
        )
        assert result.submitted == 0, "an order got through with the kill switch on"
        assert not await db_trades(session_factory)

    async def test_a_drawdown_breach_allocates_to_cash(self, session_factory):
        stack, pipeline, _ = await make_allocated(session_factory)
        result = await run_allocated_cycle(
            pipeline=pipeline, execution_service=stack.service,
            total_capital=5_000_000.0, cash=5_000_000.0,
            current_drawdown_pct=-0.25,
        )
        assert result.submitted == 0
        assert any("DRAWDOWN" in r for r in result.target.reasons)
        assert not await db_trades(session_factory)

    async def test_blocked_reconciliation_produces_no_orders(self, session_factory):
        stack, pipeline, _ = await make_allocated(session_factory)

        async def unreachable(*a: Any, **k: Any):
            raise ConnectionError("broker unreachable")

        for name in ("get_positions", "get_orders", "get_trades", "get_funds"):
            setattr(stack.broker, name, unreachable)
        await stack.recovery.recover(stack.broker)

        result = await run_allocated_cycle(
            pipeline=pipeline, execution_service=stack.service,
            total_capital=5_000_000.0, cash=5_000_000.0,
        )
        assert result.submitted == 0
        assert not await db_trades(session_factory)

    async def test_zero_capital_produces_no_orders(self, session_factory):
        stack, pipeline, _ = await make_allocated(session_factory)
        result = await run_allocated_cycle(
            pipeline=pipeline, execution_service=stack.service,
            total_capital=0.0, cash=0.0,
        )
        assert result.submitted == 0
        assert not await db_orders(session_factory)

    async def test_no_signals_produces_cash_not_orders(self, session_factory):
        """No alpha model means the momentum prior clears nothing — correctly."""
        stack, pipeline, _ = await make_allocated(session_factory, with_model=False)
        result = await run_allocated_cycle(
            pipeline=pipeline, execution_service=stack.service,
            total_capital=5_000_000.0, cash=5_000_000.0,
        )
        assert result.submitted == 0
        assert result.target is not None and result.target.is_no_trade
        assert not await db_trades(session_factory)


# =========================================================================== #
#  Existing portfolio                                                         #
# =========================================================================== #

class TestExistingPortfolioIntegration:

    async def test_a_second_cycle_trades_less_than_the_first(self, session_factory):
        """
        The second cycle must top up, not rebuild.

        The book from cycle one is fed back in as the existing portfolio, and a
        blind overwrite would show up as the same or greater turnover.
        """
        stack, pipeline, _ = await make_allocated(session_factory)

        first = await run_allocated_cycle(
            pipeline=pipeline, execution_service=stack.service,
            total_capital=5_000_000.0, cash=5_000_000.0, dry_run=True,
        )
        assert first.target is not None
        held = [
            {
                "symbol": p.symbol, "quantity": p.target_quantity,
                "average_price": p.price, "last_price": p.price,
                "strategy": p.strategy,
            }
            for p in first.target.positions if p.target_quantity > 0
        ]

        second = await run_allocated_cycle(
            pipeline=pipeline, execution_service=stack.service,
            total_capital=5_000_000.0, cash=1_000_000.0,
            positions=held, dry_run=True,
        )
        assert second.target is not None
        assert second.target.expected_turnover_pct < \
            first.target.expected_turnover_pct, (
                "the second cycle rebuilt the book instead of topping it up"
            )

    async def test_the_snapshot_reproduces_the_same_target(self, session_factory):
        """An allocation nobody can re-run is an allocation nobody can review."""
        stack, pipeline, _ = await make_allocated(session_factory)
        result = await run_allocated_cycle(
            pipeline=pipeline, execution_service=stack.service,
            total_capital=5_000_000.0, cash=5_000_000.0, dry_run=True,
        )
        assert result.snapshot is not None
        assert result.snapshot.fingerprint == result.target.input_fingerprint
        assert "positions" in result.snapshot.target


# =========================================================================== #
#  Capital sizes                                                              #
# =========================================================================== #

class TestCapitalSizes:

    @pytest.mark.parametrize("capital", [1_000.0, 50_000.0, 5_000_000.0, 500_000_000.0])
    async def test_every_capital_size_ends_in_a_valid_target_or_cash(
        self, session_factory, capital
    ):
        """
        The core invariant: a valid target portfolio, or cash. Never an unsafe
        target merely because capital exists.
        """
        stack, pipeline, _ = await make_allocated(session_factory)
        result = await run_allocated_cycle(
            pipeline=pipeline, execution_service=stack.service,
            total_capital=capital, cash=capital, dry_run=True,
        )
        t = result.target
        assert t is not None

        if t.is_no_trade:
            assert t.reasons, "a no-trade decision with no stated reason"
            return

        assert t.deployed_value <= capital + 1.0
        assert t.cash_reserve_pct >= 0.05 - 1e-9
        for p in t.positions:
            assert p.target_weight <= 0.10 + 1e-9, f"{p.symbol} breached the cap"

    async def test_tiny_capital_cannot_buy_a_share_and_says_so(self, session_factory):
        stack, pipeline, _ = await make_allocated(session_factory)
        result = await run_allocated_cycle(
            pipeline=pipeline, execution_service=stack.service,
            total_capital=100.0, cash=100.0, dry_run=True,
        )
        assert result.target is not None
        assert result.target.is_no_trade
