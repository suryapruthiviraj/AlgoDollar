"""
The front half of the runtime flow: market data -> signals -> sized orders.

    market data -> features -> strategy -> allocation -> position size
                -> ExecutionService.submit_signal

Everything downstream of that last call (risk, eligibility, idempotency, the
kill switch, the broker, persistence, audit) already exists and is not
re-implemented here. This module's entire job is to DECIDE WHAT TO PROPOSE and
hand it to the one authoritative order path.

WHY THIS DID NOT EXIST
----------------------
Strategies were reachable only from the backtester. Nothing in the running
application ever called ``generate_signals``, and ``POST /allocation/execute``
iterated an ``alloc.pending_signals`` list that was always empty. So the
execution layer could be exercised only by hand.

NO TRADE IS A VALID OUTCOME
---------------------------
A cycle that proposes nothing is a success, not a failure. Capital is never
deployed to fill a quota: if no symbol clears its strategy's own threshold, the
run reports zero proposals and that is the end of it.

WHICH STRATEGIES RUN, AND WHY THE OTHER TWO DO NOT
--------------------------------------------------
``build_default_pipeline`` enables SWING only. That is a deliberate, narrow
choice rather than an oversight:

* **longterm** scores on fundamentals, and the only fundamentals provider in
  this repository is ``_MockFundamentalProvider`` — it returns SYNTHETIC values
  and says so on every call. Letting synthetic fundamentals propose real orders
  is fabricating the input to a trading decision, so it is excluded until a
  real fundamentals source exists.
* **intraday** needs intraday bars. The available feed serves DAILY bars. Wiring
  it to daily data would silently change the strategy's meaning rather than run
  it, so it is excluded until an intraday feed exists.
* **swing** ranks on price momentum computed from real daily OHLCV, which is
  data this system actually has.

None of this says swing is profitable. It is not validated, and this pipeline
makes no claim about its edge — see docs/PRODUCTION_READINESS.md. It is wired
because it is the one sleeve whose INPUTS are real.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

#: Swing's momentum prior needs 252 trading days (12-1 momentum over 231 plus a
#: 21-day skip). Three calendar years is a comfortable margin over weekends and
#: holidays; anything less silently drops symbols from the cross-section.
HISTORY_DAYS = 3 * 365

#: Symbols fetched concurrently. Bounded so a large universe cannot open an
#: unbounded number of connections to the data vendor.
FETCH_CONCURRENCY = 8


@dataclass
class SignalOutcome:
    """One proposal and what became of it."""

    symbol: str
    strategy: str
    direction: str
    #: Rupee value the strategy's own sizer asked for.
    target_value: float = 0.0
    quantity: int = 0
    reference_price: float = 0.0
    submitted: bool = False
    outcome: Optional[str] = None
    reason: Optional[str] = None
    broker_order_id: Optional[str] = None


@dataclass
class PipelineRun:
    """
    What one cycle did, stated so a zero-order run is still fully explained.

    The distinction between "no signal was generated", "a signal was generated
    but sized to zero" and "an order was proposed and refused" matters, and
    collapsing them into a single count would hide which one happened.
    """

    started_at: datetime
    universe_size: int = 0
    symbols_with_data: int = 0
    signals_generated: int = 0
    proposals: list[SignalOutcome] = field(default_factory=list)
    submitted: int = 0
    blocked: int = 0
    skipped_zero_size: int = 0
    allocation: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def no_trade(self) -> bool:
        return self.submitted == 0

    def summary(self) -> str:
        if self.universe_size == 0:
            return "NO TRADE — the universe was empty."
        if self.symbols_with_data == 0:
            return (
                f"NO TRADE — no market data for any of {self.universe_size} "
                f"symbols."
            )
        if self.signals_generated == 0:
            return (
                f"NO TRADE — no symbol cleared its strategy threshold "
                f"({self.symbols_with_data} priced)."
            )
        if self.submitted == 0:
            return (
                f"NO TRADE — {self.signals_generated} signal(s) generated, "
                f"{self.skipped_zero_size} sized to zero, {self.blocked} refused "
                f"by the safety layer."
            )
        return (
            f"{self.submitted} order(s) submitted from "
            f"{self.signals_generated} signal(s); {self.blocked} refused."
        )


class TradingPipeline:
    """
    One cycle: fetch, score, size, submit.

    Collaborators are injected rather than constructed so a test can replace an
    external boundary. Every internal decision — risk, eligibility, gates — is
    still made by the production ``ExecutionService`` this hands off to.
    """

    def __init__(
        self,
        *,
        execution_service: Any,
        data_broker: Any,
        strategies: Sequence[Any],
        risk_engine: Any = None,
        universe: Optional[Sequence[str]] = None,
        allocator: Any = None,
        max_orders_per_cycle: int = 5,
        feature_builder: Any = None,
    ) -> None:
        if execution_service is None:
            raise ValueError("TradingPipeline requires an ExecutionService")
        if data_broker is None:
            raise ValueError(
                "TradingPipeline requires a market data source. Without prices "
                "nothing can be scored, sized or filled."
            )
        self.service = execution_service
        self.data = data_broker
        self.strategies = list(strategies)
        self.risk_engine = risk_engine
        self.universe = list(universe or [])
        self.allocator = allocator
        # Builds the {feat}__{symbol} frame an alpha model scores from.
        # None means no features are available, so a strategy configured with
        # an alpha model has nothing to score and falls back to its own prior.
        # Nothing here fabricates a feature frame: a half-built one would
        # silently change WHICH scoring path runs.
        self.feature_builder = feature_builder
        # A ceiling on how many orders one cycle may propose. Not a risk
        # control — the risk engine is — but a blast-radius limit: a bug that
        # generated a signal for every name in the universe should not be able
        # to submit hundreds of orders before anyone notices.
        self.max_orders_per_cycle = int(max_orders_per_cycle)

    # -- market data ------------------------------------------------------ #

    async def fetch_market_data(
        self, symbols: Sequence[str], *, as_of: Optional[date] = None
    ) -> dict[str, pd.DataFrame]:
        """
        Daily OHLCV per symbol. Symbols that cannot be priced are OMITTED.

        Omitted, not filled with a placeholder: a strategy scoring a symbol
        against invented history would rank it against names that have real
        history, which is worse than not considering it at all.
        """
        end = as_of or datetime.now(timezone.utc).date()
        start = end - timedelta(days=HISTORY_DAYS)
        sem = asyncio.Semaphore(FETCH_CONCURRENCY)
        out: dict[str, pd.DataFrame] = {}

        async def one(sym: str) -> None:
            async with sem:
                try:
                    df = await self.data.get_historical_data(
                        sym, "NSE", "day", start.isoformat(), end.isoformat()
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("no history for %s: %s", sym, exc)
                    return
                if df is None or len(df) == 0:
                    return
                df = _normalise_ohlcv(df)
                if df is not None:
                    out[sym] = df

        await asyncio.gather(*(one(s) for s in symbols))
        return out

    # -- one cycle -------------------------------------------------------- #

    async def run_once(
        self,
        *,
        available_capital: float,
        portfolio_value: Optional[float] = None,
        current_positions: Optional[list[dict]] = None,
        as_of: Optional[date] = None,
        dry_run: bool = False,
    ) -> PipelineRun:
        """
        Run one full cycle.

        ``dry_run`` stops after sizing and submits nothing. It is for
        inspecting what the pipeline WOULD propose; it does not weaken any gate,
        because the gates all live downstream of the submission it skips.
        """
        run = PipelineRun(started_at=datetime.now(timezone.utc))
        run.universe_size = len(self.universe)
        positions = list(current_positions or [])
        portfolio_value = (
            float(portfolio_value) if portfolio_value is not None
            else float(available_capital)
        )

        if not self.universe:
            run.warnings.append("The universe is empty; nothing can be scored.")
            return run
        if not self.strategies:
            run.warnings.append("No strategies are enabled; nothing can propose a trade.")
            return run

        # ---- 1. market data --------------------------------------------- #
        try:
            market_data = await self.fetch_market_data(self.universe, as_of=as_of)
        except Exception as exc:  # noqa: BLE001
            run.errors.append(f"market data fetch failed: {exc!r}")
            logger.exception("pipeline market data fetch failed")
            return run

        run.symbols_with_data = len(market_data)
        if not market_data:
            run.warnings.append(
                "No market data was returned for any symbol, so no signal can "
                "be produced. This is NOT a signal that no opportunity exists."
            )
            return run

        priced = [s for s in self.universe if s in market_data]

        # ---- 2. signals --------------------------------------------------- #
        # With no feature_builder the frame is EMPTY, and the enabled strategy
        # falls back to its own momentum prior. That is the current production
        # state and it has a consequence worth stating plainly:
        #
        #   SwingStrategy's momentum prior shrinks 12-1 momentum by IC=0.03,
        #   so its expected 5-day return tops out around 0.03 * 3 * sigma_5d
        #   — roughly 20 bps at typical volatility. `min_signal_strength` is
        #   40 bps, chosen to clear the ~34 bps round-trip cost. So the prior
        #   almost never clears the hurdle, and the sleeve correctly proposes
        #   NOTHING.
        #
        # That is the strategy declining to trade at a negative expected NET
        # return, which is right. It also means this pipeline cannot place a
        # swing order until a validated alpha model exists. See
        # docs/PRODUCTION_READINESS.md; PRODUCTION MODEL = NONE.
        features_df = pd.DataFrame(index=priced)
        if self.feature_builder is not None:
            try:
                built = self.feature_builder(priced, market_data)
                if built is not None and not built.empty:
                    features_df = built
            except Exception as exc:  # noqa: BLE001
                run.errors.append(f"feature build failed: {exc!r}")
                logger.exception("pipeline feature build failed")

        signals: list[tuple[Any, Any]] = []
        for strat in self.strategies:
            name = getattr(strat, "name", None) or type(strat).__name__
            try:
                produced = strat.generate_signals(priced, features_df, market_data)
            except Exception as exc:  # noqa: BLE001
                run.errors.append(f"{name}: generate_signals raised: {exc!r}")
                logger.exception("strategy %s failed to generate signals", name)
                continue
            for sig in produced or []:
                signals.append((strat, sig))

        run.signals_generated = len(signals)
        if not signals:
            run.warnings.append(
                "No symbol cleared its strategy's threshold. NO TRADE is the "
                "correct outcome; capital is not deployed to fill a quota."
            )
            return run

        # Strongest first, so the per-cycle ceiling keeps the best proposals
        # rather than whichever happened to be scored first.
        signals.sort(key=lambda p: float(getattr(p[1], "edge_score", 0.0) or 0.0), reverse=True)

        # The risk engine judges concentration against the CURRENT book, and it
        # raises rather than approving if it has not been told what that book
        # is. Set once per cycle, immediately before sizing — stale context is
        # worse than none, because a concentration limit measured against last
        # hour's portfolio value quietly stops binding.
        if self.risk_engine is not None and hasattr(
            self.risk_engine, "set_portfolio_context"
        ):
            try:
                self.risk_engine.set_portfolio_context(
                    portfolio_value=portfolio_value,
                    available_cash=available_capital,
                    positions=positions,
                )
            except Exception as exc:  # noqa: BLE001
                # Without context every approve_trade() call raises and every
                # trade blocks. That is the safe direction, but say why.
                run.errors.append(f"risk context could not be set: {exc!r}")
                logger.exception("pipeline could not set risk context")

        # ---- 3. sizing and submission ------------------------------------ #
        submitted_count = 0
        for strat, sig in signals:
            if submitted_count >= self.max_orders_per_cycle:
                run.warnings.append(
                    f"Per-cycle ceiling of {self.max_orders_per_cycle} orders "
                    f"reached; {len(signals) - len(run.proposals)} proposal(s) "
                    f"not considered this cycle."
                )
                break

            name = getattr(strat, "name", None) or type(strat).__name__
            outcome = SignalOutcome(
                symbol=str(getattr(sig, "symbol", "")),
                strategy=str(getattr(sig, "strategy_name", name) or name),
                direction=str(getattr(getattr(sig, "direction", None), "value", "")),
            )

            price = _last_close(market_data.get(outcome.symbol))
            if not price or price <= 0:
                outcome.reason = "no reference price"
                run.proposals.append(outcome)
                continue
            outcome.reference_price = price

            try:
                target_value = float(strat.calculate_position_size(
                    sig, available_capital, self.risk_engine
                ))
            except Exception as exc:  # noqa: BLE001
                outcome.reason = f"position sizing raised: {exc!r}"
                run.proposals.append(outcome)
                run.errors.append(f"{name}/{outcome.symbol}: {outcome.reason}")
                continue

            outcome.target_value = target_value
            qty = int(target_value // price)
            outcome.quantity = qty
            if qty <= 0:
                # Sizing to zero is a decision, not a failure — it is the
                # strategy declining the trade at this capital level.
                outcome.reason = (
                    f"sized to zero (target Rs {target_value:.2f} < one share "
                    f"at Rs {price:.2f})"
                )
                run.skipped_zero_size += 1
                run.proposals.append(outcome)
                continue

            if dry_run:
                outcome.reason = "dry run: not submitted"
                run.proposals.append(outcome)
                continue

            result = await self.service.submit_signal(
                sig,
                qty,
                reference_price=price,
                available_cash=available_capital,
                total_portfolio=portfolio_value,
                current_positions=positions,
            )
            outcome.submitted = bool(result.submitted)
            outcome.outcome = result.outcome.value
            outcome.reason = result.reason
            outcome.broker_order_id = result.broker_order_id
            run.proposals.append(outcome)

            if outcome.submitted:
                submitted_count += 1
                run.submitted += 1
                # Committed capital is withdrawn from what the next proposal
                # may size against. Without this, N signals in one cycle each
                # size against the FULL balance and together commit N times the
                # cash that exists.
                available_capital = max(0.0, available_capital - qty * price)
            else:
                run.blocked += 1

        return run


# --------------------------------------------------------------------------- #
#  Default assembly                                                             #
# --------------------------------------------------------------------------- #

def build_default_pipeline(
    *,
    execution_service: Any,
    data_broker: Any,
    universe: Optional[Sequence[str]] = None,
    max_orders_per_cycle: int = 5,
    universe_size: int = 50,
    strategies: Optional[Sequence[Any]] = None,
    feature_builder: Any = None,
) -> TradingPipeline:
    """
    The pipeline a running deployment uses.

    Only SWING is enabled — see this module's docstring for why longterm and
    intraday are not. That exclusion is a data-availability statement, not a
    judgement about which strategy is better.
    """
    from app.risk.engine import RiskEngine
    from app.strategies.swing import SwingStrategy

    if universe is None:
        universe = _default_universe(universe_size)

    return TradingPipeline(
        execution_service=execution_service,
        data_broker=data_broker,
        strategies=list(strategies) if strategies is not None
        else [SwingStrategy(paper_mode=True)],
        risk_engine=RiskEngine(),
        universe=universe,
        max_orders_per_cycle=max_orders_per_cycle,
        feature_builder=feature_builder,
    )


def _default_universe(n: int) -> list[str]:
    """
    The first ``n`` NSE names available.

    NOT a point-in-time index membership list: it is a present-day snapshot, so
    anything computed over it carries survivorship bias. That is acceptable for
    running a paper session and is NOT acceptable for measuring performance —
    see docs/DATA_INTEGRITY_REPORT.md.
    """
    try:
        from app.data.universe import StockUniverse

        symbols = StockUniverse.get_nifty500_symbols()
    except Exception as exc:  # noqa: BLE001
        logger.error("could not load a universe: %s", exc)
        return []
    return list(symbols)[:n]


# --------------------------------------------------------------------------- #
#  helpers                                                                      #
# --------------------------------------------------------------------------- #

def _normalise_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Lower-case the OHLCV columns strategies index by name.

    Returns None when there is no usable close, rather than a frame that looks
    valid and is not.
    """
    try:
        out = df.copy()
        out.columns = [str(c).lower() for c in out.columns]
        if "close" not in out.columns:
            return None
        return out
    except Exception:  # noqa: BLE001
        return None


def _last_close(df: Optional[pd.DataFrame]) -> float:
    if df is None or len(df) == 0 or "close" not in df.columns:
        return 0.0
    try:
        return float(df["close"].iloc[-1])
    except (TypeError, ValueError):
        return 0.0
