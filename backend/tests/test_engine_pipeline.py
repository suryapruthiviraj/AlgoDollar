"""
The signal pipeline: market data -> strategy -> sizing -> submitted order.

These tests care about two things above all:

1. **NO TRADE is reachable and is a success.** A pipeline that always finds
   something to buy is not selecting, it is deploying. Several tests here exist
   only to prove the pipeline can decline.
2. **Nothing downstream is bypassed.** The pipeline's last act is to call
   ``ExecutionService.submit_signal``; it never reaches a broker itself, so the
   risk, eligibility, kill-switch and idempotency gates cannot be skipped by
   going through it.

The market data feed is the external boundary and is substituted with a
generator that produces deterministic OHLCV. The strategy that scores it is the
real ``SwingStrategy``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database.models import Base
from app.engine.pipeline import TradingPipeline, build_default_pipeline
from app.execution.audit import ExecutionOutcome
from tests.test_e2e_paper_trade import (
    MARKET_OPEN_IST,
    DeterministicFeed,
    db_orders,
    db_trades,
    make_stack,
)

pytestmark = pytest.mark.asyncio


# =========================================================================== #
#  A market data feed with real-shaped history                                #
# =========================================================================== #

class HistoryFeed(DeterministicFeed):
    """
    ``DeterministicFeed`` plus daily OHLCV deep enough for a momentum score.

    Swing's momentum prior needs 252 trading days. ``trend`` controls each
    symbol's drift so the cross-section has genuine dispersion — without it
    every name ranks identically and the ranking step is untested.
    """

    def __init__(self, trends: dict[str, float], *, bars: int = 400, **kw: Any) -> None:
        super().__init__(**kw)
        self.trends = trends
        self.bars = bars
        self._cache: dict[str, pd.DataFrame] = {}

    def _history(self, symbol: str) -> pd.DataFrame:
        if symbol in self._cache:
            return self._cache[symbol]
        drift = self.trends.get(symbol, 0.0)
        # Seeded per symbol: the same symbol always yields the same history, so
        # a failure is reproducible rather than a coin flip.
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        steps = drift + rng.normal(0.0, 0.01, self.bars)
        close = 1000.0 * np.exp(np.cumsum(steps))
        idx = pd.date_range(end=MARKET_OPEN_IST.date(), periods=self.bars, freq="B")
        df = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": np.full(self.bars, 2_000_000),
            },
            index=idx,
        )
        self._cache[symbol] = df
        return df

    async def get_historical_data(
        self, symbol: str, exchange: str = "NSE", interval: str = "day",
        from_date: str = "", to_date: str = "",
    ) -> pd.DataFrame:
        sym = str(symbol).split(":")[-1]
        if sym not in self.trends:
            raise ValueError(f"no history for {sym}")
        return self._history(sym)

    async def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for raw in symbols or []:
            sym = str(raw).split(":")[-1]
            if sym not in self.trends:
                continue
            px = float(self._history(sym)["close"].iloc[-1])
            q = {
                "last_price": px,
                "timestamp": self.timestamp,
                "volume": self.volume,
                "ohlc": {"open": px, "high": px, "low": px, "close": px},
                "depth": {
                    "buy": [{"price": px - 0.05, "quantity": 100_000}],
                    "sell": [{"price": px + 0.05, "quantity": 100_000}],
                },
            }
            out[raw] = q
            out[sym] = q
        return out


UNIVERSE = [f"SYM{i:02d}" for i in range(20)]
#: A spread of drifts, so momentum ranking has something to rank.
TRENDS = {s: (i - 10) * 0.0006 for i, s in enumerate(UNIVERSE)}


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


class StubAlphaModel:
    """
    A model that predicts a fixed, cost-clearing horizon return.

    NOT a validated model and not a claim about edge — its only job is to put
    the strategy on its ALPHA-MODEL scoring path so the rest of the pipeline
    (ranking, sizing, submission, persistence) can be exercised. Without it the
    momentum prior correctly declines every trade, which is itself asserted in
    TestNoAlphaModelMeansNoTrade below.

    `alpha_model` is a constructor parameter of SwingStrategy, so this is a
    double at a designed seam, not a patched internal.
    """

    def __init__(self, value: float = 0.012) -> None:
        self.value = value

    def predict(self, X):
        n = int(np.asarray(X).shape[0])
        # A spread around `value` so the cross-sectional ranking has an order
        # to find, all of it above the 40 bps cost hurdle.
        return np.linspace(self.value, self.value * 2.0, n)


def _feature_builder(symbols, market_data):
    """Build the `{feat}__{symbol}` frame SwingStrategy's model path expects."""
    cols = {}
    for sym in symbols:
        df = market_data.get(sym)
        if df is None or len(df) < 30:
            continue
        px = df["close"]
        cols[f"ret21__{sym}"] = px.pct_change(21)
        cols[f"vol21__{sym}"] = px.pct_change().rolling(21).std()
    return pd.DataFrame(cols) if cols else pd.DataFrame()


async def make_pipeline(
    session_factory: Any,
    *,
    feed: Optional[HistoryFeed] = None,
    universe: Optional[list[str]] = None,
    max_orders: int = 5,
    with_alpha_model: bool = False,
):
    from app.strategies.swing import SwingStrategy

    feed = feed or HistoryFeed(TRENDS)
    stack = await make_stack(session_factory, feed=feed)
    kwargs: dict[str, Any] = {}
    if with_alpha_model:
        kwargs["strategies"] = [
            SwingStrategy(paper_mode=True, alpha_model=StubAlphaModel())
        ]
        kwargs["feature_builder"] = _feature_builder
    pipeline = build_default_pipeline(
        execution_service=stack.service,
        data_broker=feed,
        universe=universe if universe is not None else UNIVERSE,
        max_orders_per_cycle=max_orders,
        **kwargs,
    )
    return stack, pipeline, feed


# =========================================================================== #
#  Market data                                                                #
# =========================================================================== #

class TestMarketData:

    async def test_symbols_without_history_are_omitted_not_invented(
        self, session_factory
    ):
        """A symbol the feed cannot price must not get placeholder history."""
        _, pipeline, _ = await make_pipeline(
            session_factory, universe=UNIVERSE + ["DELISTED", "NOSUCH"]
        )
        data = await pipeline.fetch_market_data(pipeline.universe)
        assert set(data) == set(UNIVERSE)
        assert "DELISTED" not in data and "NOSUCH" not in data

    async def test_history_is_deep_enough_to_score(self, session_factory):
        _, pipeline, _ = await make_pipeline(session_factory)
        data = await pipeline.fetch_market_data(pipeline.universe)
        assert all(len(df) >= 252 for df in data.values()), (
            "less than 252 bars means swing silently drops the symbol"
        )
        assert all("close" in df.columns for df in data.values())


# =========================================================================== #
#  NO TRADE — the outcome that must remain reachable                          #
# =========================================================================== #

class TestNoTradeIsReachable:

    async def test_empty_universe_produces_no_trade(self, session_factory):
        _, pipeline, _ = await make_pipeline(session_factory, universe=[])
        run = await pipeline.run_once(available_capital=1_000_000.0)
        assert run.no_trade and run.submitted == 0
        assert "universe" in run.summary().lower()

    async def test_no_market_data_produces_no_trade_not_a_guess(self, session_factory):
        """No data must never be read as 'no opportunity'."""
        _, pipeline, _ = await make_pipeline(
            session_factory, universe=["NOSUCH1", "NOSUCH2"]
        )
        run = await pipeline.run_once(available_capital=1_000_000.0)
        assert run.no_trade
        assert run.symbols_with_data == 0
        assert any("NOT a signal that no opportunity exists" in w for w in run.warnings)

    async def test_capital_too_small_sizes_to_zero_and_submits_nothing(
        self, session_factory
    ):
        """Sizing to zero is the strategy declining, not an error."""
        _, pipeline, _ = await make_pipeline(session_factory, with_alpha_model=True)
        run = await pipeline.run_once(available_capital=100.0)
        assert run.submitted == 0
        assert not await db_trades(session_factory)


class TestNoAlphaModelMeansNoTrade:
    """
    THE CURRENT PRODUCTION STATE, asserted rather than assumed.

    With no alpha model, SwingStrategy scores on its 12-1 momentum prior. That
    prior shrinks momentum by IC = 0.03, so its expected 5-day return tops out
    near 0.03 * 3 * sigma_5d — roughly 20 bps at typical volatility.
    `min_signal_strength` is 40 bps, deliberately set to clear the ~34 bps
    round-trip cost.

    So the baseline essentially never clears the hurdle, and the sleeve
    proposes NOTHING. That is the strategy refusing to trade at a negative
    expected NET return, which is correct — and it means this pipeline cannot
    place a swing order until a validated model exists.

    This test exists so that if someone later lowers the threshold or inflates
    the prior, the change is visible rather than silent.
    """

    async def test_the_momentum_prior_alone_proposes_nothing(self, session_factory):
        stack, pipeline, _ = await make_pipeline(session_factory)
        assert stack.trading_permitted, "the gate must be open to isolate the strategy"

        run = await pipeline.run_once(available_capital=1_000_000.0)

        assert run.symbols_with_data == len(UNIVERSE), "the data step must have run"
        assert run.signals_generated == 0, (
            f"the no-model baseline produced {run.signals_generated} signal(s). "
            "Either the cost hurdle was lowered or the prior was inflated — "
            "both change what this system will trade on."
        )
        assert run.submitted == 0
        assert not await db_orders(session_factory)
        assert "NO TRADE" in run.summary()

    async def test_the_prior_scores_below_the_cost_hurdle(self, session_factory):
        """The mechanism, not just the outcome."""
        from app.strategies.swing import SwingStrategy

        _, pipeline, _ = await make_pipeline(session_factory)
        market_data = await pipeline.fetch_market_data(UNIVERSE)

        strat = SwingStrategy(paper_mode=True)
        scores = strat._score_universe(
            UNIVERSE, pd.DataFrame(index=UNIVERSE), market_data, None
        )
        assert (scores["score_source"] == "momentum_prior").all()
        assert scores["raw_score"].abs().max() < strat.min_signal_strength, (
            "the momentum prior now clears the cost hurdle; that is a change in "
            "what the system will trade on and must be deliberate"
        )

    async def test_a_no_trade_run_still_explains_itself(self, session_factory):
        _, pipeline, _ = await make_pipeline(session_factory, universe=[])
        run = await pipeline.run_once(available_capital=1_000_000.0)
        assert run.summary().startswith("NO TRADE"), run.summary()


# =========================================================================== #
#  The full front-to-back path                                                #
# =========================================================================== #

class TestPipelineToExecution:

    async def test_a_cycle_produces_signals_and_persisted_orders(self, session_factory):
        """market data -> signal -> sizing -> order -> fill -> database."""
        stack, pipeline, _ = await make_pipeline(session_factory, with_alpha_model=True)
        assert stack.trading_permitted

        # available_capital is the SWING SLEEVE's capital; portfolio_value is
        # the whole book. The distinction is load-bearing: swing normalises its
        # signal set to 100% gross OF THE SLEEVE, while max_single_stock_pct is
        # measured against the PORTFOLIO. Passing one number for both makes
        # 25%-of-sleeve equal 25%-of-portfolio, which correctly trips the 10%
        # cap — asserted in test_the_single_stock_cap_blocks_an_oversized_sleeve.
        run = await pipeline.run_once(
            available_capital=1_000_000.0, portfolio_value=5_000_000.0
        )

        assert run.symbols_with_data == len(UNIVERSE)
        assert run.signals_generated > 0, (
            "the real SwingStrategy produced no signal over a dispersed "
            "cross-section; the pipeline is not reaching the scoring path"
        )
        assert run.submitted > 0, f"nothing was submitted: {run.summary()}"

        orders = await db_orders(session_factory)
        trades = await db_trades(session_factory)
        assert len(orders) == run.submitted
        assert trades, "a submitted order produced no persisted fill"
        assert all(o.client_order_id for o in orders), "an order has no idempotency key"
        assert all(o.strategy for o in orders), "strategy attribution was lost"

    async def test_dry_run_submits_nothing(self, session_factory):
        stack, pipeline, _ = await make_pipeline(session_factory, with_alpha_model=True)
        run = await pipeline.run_once(
            available_capital=1_000_000.0, portfolio_value=5_000_000.0, dry_run=True
        )
        assert run.submitted == 0
        assert run.proposals, "a dry run should still say what it would propose"
        assert not await db_orders(session_factory)
        assert not await db_trades(session_factory)

    async def test_the_per_cycle_ceiling_is_enforced(self, session_factory):
        stack, pipeline, _ = await make_pipeline(
            session_factory, max_orders=2, with_alpha_model=True
        )
        run = await pipeline.run_once(
            available_capital=1_000_000.0, portfolio_value=50_000_000.0
        )
        assert run.submitted <= 2, f"the ceiling was exceeded: {run.submitted}"
        assert run.submitted > 0, "the ceiling test never reached submission"

    async def test_capital_is_not_committed_twice(self, session_factory):
        """
        Each submitted order reduces what the next may size against.

        Without that, N signals in one cycle each size against the FULL balance
        and together commit N times the cash that exists.
        """
        stack, pipeline, _ = await make_pipeline(
            session_factory, max_orders=5, with_alpha_model=True
        )
        capital = 200_000.0
        run = await pipeline.run_once(
            available_capital=capital, portfolio_value=50_000_000.0
        )

        committed = sum(
            p.quantity * p.reference_price for p in run.proposals if p.submitted
        )
        assert committed <= capital * 1.05, (
            f"the cycle committed Rs {committed:,.0f} against Rs {capital:,.0f} "
            "of capital"
        )


# =========================================================================== #
#  The pipeline must not be able to skip the safety layer                     #
# =========================================================================== #

class TestPipelineCannotBypassSafety:

    async def test_kill_switch_stops_the_whole_cycle(self, session_factory):
        stack, pipeline, _ = await make_pipeline(session_factory, with_alpha_model=True)
        stack.kill_switch_store.engage("operator halt")

        run = await pipeline.run_once(
            available_capital=1_000_000.0, portfolio_value=5_000_000.0
        )
        assert run.submitted == 0, "the pipeline traded through an engaged kill switch"
        assert run.signals_generated > 0, "the test did not reach the submission step"
        assert all(
            p.outcome == ExecutionOutcome.BLOCKED_KILL_SWITCH.value
            for p in run.proposals if p.outcome
        )
        assert not await db_trades(session_factory)

    async def test_blocked_reconciliation_stops_the_whole_cycle(self, session_factory):
        stack, pipeline, _ = await make_pipeline(session_factory, with_alpha_model=True)

        async def unreachable(*a: Any, **k: Any):
            raise ConnectionError("broker unreachable")

        for name in ("get_positions", "get_orders", "get_trades", "get_funds"):
            setattr(stack.broker, name, unreachable)
        await stack.recovery.recover(stack.broker)

        run = await pipeline.run_once(
            available_capital=1_000_000.0, portfolio_value=5_000_000.0
        )
        assert run.submitted == 0
        assert not await db_trades(session_factory)

    async def test_the_single_stock_cap_blocks_an_oversized_sleeve(
        self, session_factory
    ):
        """
        A REAL conflict between two correct policies, pinned rather than papered over.

        SwingStrategy normalises its emitted signal set to 100% gross OF THE
        SLEEVE. With only N signals that is 1/N each — 25% at N=4. The global
        `max_single_stock_pct` is 10% OF THE PORTFOLIO. So when the swing
        sleeve IS the whole portfolio, the strategy's own sizing exceeds the
        global cap and the risk engine refuses every trade.

        Blocking is the correct resolution. This test exists so that neither
        the strategy's gross budget nor the risk cap can be quietly relaxed to
        make the orders go through.
        """
        stack, pipeline, _ = await make_pipeline(session_factory, with_alpha_model=True)

        run = await pipeline.run_once(
            available_capital=1_000_000.0, portfolio_value=1_000_000.0
        )
        assert run.signals_generated > 0, "the test never reached sizing"
        assert run.submitted == 0, (
            "a position exceeding the 10% single-stock cap was submitted"
        )
        assert run.skipped_zero_size == run.signals_generated, (
            "the risk engine did not zero every oversized proposal"
        )
        assert not await db_trades(session_factory)

    async def test_the_pipeline_never_touches_a_broker_directly(self):
        """
        Structural: the pipeline module must not call place_order.

        Its only route to a venue is ExecutionService.submit_signal, which is
        what makes every downstream gate unskippable.
        """
        import pathlib
        import re

        src = pathlib.Path("app/engine/pipeline.py").read_text()
        assert not re.search(r"\.place_order\s*\(", src), (
            "the pipeline reaches a broker directly, skipping every gate"
        )
        assert "submit_signal(" in src


# =========================================================================== #
#  Honesty about what is wired                                                #
# =========================================================================== #

class TestWhichStrategiesAreEnabled:

    def test_only_swing_is_enabled_by_default(self):
        """
        longterm scores on MOCK fundamentals and intraday needs intraday bars.

        Enabling either would mean proposing real orders from data this system
        does not have.
        """
        pipeline = build_default_pipeline(
            execution_service=object(), data_broker=object(), universe=["X"]
        )
        names = {type(s).__name__ for s in pipeline.strategies}
        assert names == {"SwingStrategy"}, names

    def test_a_pipeline_without_market_data_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="market data"):
            TradingPipeline(
                execution_service=object(), data_broker=None, strategies=[]
            )

    def test_a_pipeline_without_an_execution_service_is_refused(self):
        with pytest.raises(ValueError, match="ExecutionService"):
            TradingPipeline(
                execution_service=None, data_broker=object(), strategies=[]
            )
