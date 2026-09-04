"""
The allocated cycle: signals -> target portfolio -> orders through execution.

WHY THIS IS SEPARATE FROM TradingPipeline.run_once
---------------------------------------------------
``run_once`` sizes each signal independently through the strategy's own sizer.
That is a per-signal decision and it cannot see the portfolio: it cannot cap a
sector, cannot measure correlation, cannot enforce a turnover budget across the
whole book, and cannot decide that the right answer is cash.

This path replaces that step with a PORTFOLIO decision. Every signal in the
cycle is considered together, the allocation engine produces one coherent
target, and only the DELTA from the current book is submitted.

WHAT IT DOES NOT REPLACE
------------------------
The execution boundary. Every order still goes through
``ExecutionService.submit_signal``, so the kill switch, risk validation,
eligibility, idempotency and audit all apply exactly as they do to any other
order. The allocator decides WHAT to hold; it has no route to a broker.

The allocator is also not a second safety layer. Its pre-flight gates duplicate
some of the execution gates deliberately — a target that would be refused
downstream should never be produced upstream — but the downstream gates remain
authoritative, and an allocation that slips past a pre-flight is still stopped
at the boundary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.portfolio.allocation import (
    AllocationInputs,
    AllocationSnapshot,
    PositionInput,
    RiskLimits,
    SignalInput,
    TargetPortfolio,
)
from app.portfolio.engine import PortfolioAllocationEngine

logger = logging.getLogger(__name__)

#: Sessions used to estimate volatility and traded value. One quarter — long
#: enough to be stable, short enough to reflect the current regime.
ESTIMATION_WINDOW = 63
TRADING_DAYS = 252


@dataclass
class AllocatedCycleResult:
    """What one allocated cycle decided and what became of it."""

    target: Optional[TargetPortfolio] = None
    snapshot: Optional[AllocationSnapshot] = None
    submitted: int = 0
    blocked: int = 0
    skipped: int = 0
    outcomes: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def no_trade(self) -> bool:
        return self.submitted == 0

    def summary(self) -> str:
        if self.target is None:
            return f"NO ALLOCATION — {'; '.join(self.errors) or 'no target produced'}"
        head = self.target.summary()
        return f"{head} | submitted {self.submitted}, refused {self.blocked}"


def estimate_risk_inputs(
    market_data: dict[str, pd.DataFrame], symbols: list[str]
) -> tuple[dict[str, float], dict[str, float], list[str], Optional[list[list[float]]]]:
    """
    Volatility, traded value and a correlation matrix from real price history.

    A symbol with too little history gets NO estimate rather than a default. A
    defaulted volatility would size a position against a number nobody measured,
    and the allocation engine drops such names for exactly that reason.

    The correlation matrix is computed on the overlapping window only, so names
    that were not simultaneously listed do not contribute a spurious pairing.
    """
    vols: dict[str, float] = {}
    traded: dict[str, float] = {}
    ret_cols: dict[str, pd.Series] = {}

    for sym in symbols:
        df = market_data.get(sym)
        if df is None or len(df) < ESTIMATION_WINDOW:
            continue
        cols = {str(c).lower(): c for c in df.columns}
        if "close" not in cols:
            continue
        px = df[cols["close"]].astype(float)
        r = px.pct_change().dropna()
        if len(r) < ESTIMATION_WINDOW // 2:
            continue
        sd = float(r.tail(ESTIMATION_WINDOW).std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            continue
        vols[sym] = sd * float(np.sqrt(TRADING_DAYS))
        ret_cols[sym] = r

        if "volume" in cols:
            vol_series = df[cols["volume"]].astype(float)
            tv = (vol_series * px).tail(ESTIMATION_WINDOW).median()
            if np.isfinite(tv) and tv > 0:
                traded[sym] = float(tv)

    corr_syms: list[str] = []
    corr: Optional[list[list[float]]] = None
    if len(ret_cols) >= 2:
        frame = pd.DataFrame(ret_cols).tail(ESTIMATION_WINDOW).dropna(how="any")
        if len(frame) >= 20:
            c = frame.corr()
            if c.notna().all().all():
                corr_syms = list(c.columns)
                corr = c.to_numpy().tolist()
    return vols, traded, corr_syms, corr


def build_allocation_inputs(
    *,
    signals: list[tuple[Any, Any]],
    market_data: dict[str, pd.DataFrame],
    total_capital: float,
    cash: float,
    positions: list[dict],
    strategy_health: Optional[dict[str, str]] = None,
    regime: str = "UNKNOWN",
    current_drawdown_pct: float = 0.0,
    daily_pnl_pct: float = 0.0,
    realised_vol: Optional[float] = None,
    kill_switch_active: bool = False,
    market_data_stale: bool = False,
    trading_permitted: bool = True,
    cost_bps_per_side: float = 25.0,
    limits: Optional[RiskLimits] = None,
    sector_map: Optional[dict[str, str]] = None,
    as_of: Optional[datetime] = None,
) -> AllocationInputs:
    """
    Assemble the engine's inputs from strategy output and market history.

    Everything the engine reads is materialised here, which is what makes the
    resulting allocation reproducible from the snapshot alone.
    """
    sector_map = sector_map or {}
    symbols = sorted({str(getattr(sig, "symbol", "")) for _, sig in signals})
    vols, traded, corr_syms, corr = estimate_risk_inputs(market_data, symbols)

    sig_inputs: list[SignalInput] = []
    for strat, sig in signals:
        sym = str(getattr(sig, "symbol", ""))
        df = market_data.get(sym)
        price = 0.0
        if df is not None and len(df):
            cols = {str(c).lower(): c for c in df.columns}
            if "close" in cols:
                price = float(df[cols["close"]].iloc[-1])
        sig_inputs.append(SignalInput(
            symbol=sym,
            strategy=str(getattr(sig, "strategy_name", None)
                         or getattr(strat, "name", "") or "swing"),
            direction=str(getattr(getattr(sig, "direction", None), "value", "LONG")),
            edge_score=float(getattr(sig, "edge_score", 0.0) or 0.0),
            expected_return=float(getattr(sig, "expected_return", 0.0) or 0.0),
            expected_return_std=float(getattr(sig, "expected_return_std", 0.0) or 0.0),
            price=price,
            sector=sector_map.get(sym, "UNKNOWN"),
            volatility=vols.get(sym),
            median_traded_value=traded.get(sym),
            quote_age_sec=None,
        ))

    pos_inputs: list[PositionInput] = []
    for p in positions or []:
        sym = str(p.get("symbol", ""))
        last = p.get("last_price") or p.get("average_price") or 0.0
        pos_inputs.append(PositionInput(
            symbol=sym,
            quantity=int(p.get("quantity", 0) or 0),
            average_price=float(p.get("average_price", 0.0) or 0.0),
            last_price=float(last or 0.0),
            strategy=str(p.get("strategy", "unknown")),
            sector=sector_map.get(sym, "UNKNOWN"),
        ))

    return AllocationInputs(
        as_of=as_of or datetime.now(timezone.utc),
        total_capital=float(total_capital),
        cash=float(cash),
        positions=pos_inputs,
        signals=sig_inputs,
        strategy_health=dict(strategy_health or {}),
        regime=regime,
        volatilities=vols,
        correlation_symbols=corr_syms,
        correlation_matrix=corr,
        current_drawdown_pct=float(current_drawdown_pct),
        daily_pnl_pct=float(daily_pnl_pct),
        realised_vol=realised_vol,
        cost_bps_per_side=float(cost_bps_per_side),
        limits=limits or RiskLimits(),
        kill_switch_active=bool(kill_switch_active),
        market_data_stale=bool(market_data_stale),
        trading_permitted=bool(trading_permitted),
    )


async def submit_target(
    target: TargetPortfolio,
    *,
    execution_service: Any,
    signal_by_symbol: dict[str, Any],
    total_capital: float,
    available_cash: float,
    current_positions: list[dict],
) -> AllocatedCycleResult:
    """
    Submit the DELTA between the target and the current book.

    Only names whose delta is non-zero produce an order, and each order goes
    through ``ExecutionService.submit_signal`` — the single authoritative path.
    A position with no originating signal (an exit) is skipped here rather than
    forced through: exits need a Signal object the strategy layer did not
    produce, and inventing one would put a fabricated signal into the audit
    trail. That gap is reported, not hidden.
    """
    result = AllocatedCycleResult(target=target)

    for pt in target.positions:
        if pt.delta_quantity == 0:
            continue
        sig = signal_by_symbol.get(pt.symbol)
        if sig is None:
            result.skipped += 1
            result.outcomes.append({
                "symbol": pt.symbol,
                "delta_quantity": pt.delta_quantity,
                "submitted": False,
                "reason": (
                    "no originating signal for this symbol, so no order was "
                    "created. Exits of held names require a signal the strategy "
                    "layer did not produce; fabricating one would corrupt the "
                    "audit trail."
                ),
            })
            continue

        qty = abs(int(pt.delta_quantity))
        if qty <= 0:
            continue

        outcome = await execution_service.submit_signal(
            sig,
            qty,
            reference_price=pt.price,
            portfolio_allocation={
                "target_weight": pt.target_weight,
                "risk_contribution_pct": pt.risk_contribution_pct,
                "strategy_bucket": pt.strategy,
                "binding_constraints": pt.constraints,
                "allocation_fingerprint": target.input_fingerprint,
            },
            available_cash=available_cash,
            total_portfolio=total_capital,
            current_positions=current_positions,
        )
        result.outcomes.append({
            "symbol": pt.symbol,
            "delta_quantity": pt.delta_quantity,
            "submitted": bool(outcome.submitted),
            "outcome": outcome.outcome.value,
            "reason": outcome.reason,
            "broker_order_id": outcome.broker_order_id,
        })
        if outcome.submitted:
            result.submitted += 1
            available_cash = max(0.0, available_cash - qty * pt.price)
        else:
            result.blocked += 1

    return result


async def run_allocated_cycle(
    *,
    pipeline: Any,
    execution_service: Any,
    total_capital: float,
    cash: float,
    positions: Optional[list[dict]] = None,
    strategy_health: Optional[dict[str, str]] = None,
    current_drawdown_pct: float = 0.0,
    daily_pnl_pct: float = 0.0,
    kill_switch_active: bool = False,
    trading_permitted: bool = True,
    limits: Optional[RiskLimits] = None,
    sector_map: Optional[dict[str, str]] = None,
    as_of: Optional[date] = None,
    dry_run: bool = False,
) -> AllocatedCycleResult:
    """
    One full cycle: market data -> signals -> allocation -> orders.

    ``dry_run`` stops after the target is produced. It weakens nothing: the
    gates it skips all live downstream of the submission it does not make.
    """
    result = AllocatedCycleResult()

    if not pipeline.universe:
        result.errors.append("the pipeline universe is empty; nothing to allocate")
        return result

    try:
        market_data = await pipeline.fetch_market_data(pipeline.universe, as_of=as_of)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"market data fetch failed: {exc!r}")
        return result

    if not market_data:
        result.errors.append(
            "no market data was returned. This is NOT a statement that no "
            "opportunity exists."
        )
        return result

    priced = [s for s in pipeline.universe if s in market_data]
    features = pd.DataFrame(index=priced)
    if getattr(pipeline, "feature_builder", None) is not None:
        try:
            built = pipeline.feature_builder(priced, market_data)
            if built is not None and not built.empty:
                features = built
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"feature build failed: {exc!r}")

    pairs: list[tuple[Any, Any]] = []
    for strat in pipeline.strategies:
        try:
            for sig in strat.generate_signals(priced, features, market_data) or []:
                pairs.append((strat, sig))
        except Exception as exc:  # noqa: BLE001
            result.errors.append(
                f"{type(strat).__name__}: generate_signals raised: {exc!r}"
            )

    inputs = build_allocation_inputs(
        signals=pairs,
        market_data=market_data,
        total_capital=total_capital,
        cash=cash,
        positions=positions or [],
        strategy_health=strategy_health,
        current_drawdown_pct=current_drawdown_pct,
        daily_pnl_pct=daily_pnl_pct,
        kill_switch_active=kill_switch_active,
        trading_permitted=trading_permitted,
        limits=limits,
        sector_map=sector_map,
    )

    target = PortfolioAllocationEngine(limits).allocate(inputs)
    result.target = target
    result.snapshot = AllocationSnapshot.build(inputs, target)

    if dry_run or target.is_no_trade:
        return result

    submitted = await submit_target(
        target,
        execution_service=execution_service,
        signal_by_symbol={str(getattr(s, "symbol", "")): s for _, s in pairs},
        total_capital=total_capital,
        available_cash=cash,
        current_positions=positions or [],
    )
    submitted.target = target
    submitted.snapshot = result.snapshot
    submitted.errors = result.errors
    return submitted
