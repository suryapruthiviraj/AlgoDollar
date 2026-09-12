"""
app/engine/replay.py — offline playback of the paper intraday auto-trader.

Feeds historical 1-minute bars through the EXACT production pipeline — a
fresh ``TickBarAggregator`` -> ``TickQuoteSource`` -> ``PaperBroker`` and a
fresh ``IntradayAutoTrader`` per session — so a recorded bar file is answered
the same way a live mock session would be:

  bars -> aggregator -> strategy signals -> ExecutionService -> PaperBroker
       -> protective SL-M stops -> monitor exits (stop/target/vwap/15:15)

The output is a cost-realised trade log (paper fills already carry the full
Zerodha MIS breakdown) plus per-day realised equity.  That is the honest
input to the heavy research/statistics pipeline (DSR/PBO, which live in the
scipy-backed environment): this module deliberately performs only arithmetic
statistics so it can run anywhere.

Data source: a directory of per-symbol CSV files (columns
``time, open, high, low, close, volume``; naive ``time`` is assumed IST) or a
deterministic synthetic generator for pipeline tests.  Synthetic outputs are
flagged on every public surface so they can never be mistaken for results
against real market data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.broker.paper import PaperBroker
from app.broker.tickdata import TickBarAggregator, TickQuoteSource, to_ist
from app.engine.equity import EquityTracker, risk_breaches_for
from app.engine.trader import IntradayAutoTrader
from app.execution.audit import AuditJournal, InMemoryAuditSink
from app.execution.bootstrap import InMemoryKillSwitchStore, store_kill_switch_probe
from app.execution.lifecycle import InMemoryOrderStore
from app.execution.order_manager import OrderManager
from app.execution.reconciliation import ReconciliationEngine
from app.execution.recovery import RecoveryManager
from app.execution.safety import ExecutionSafety
from app.execution.service import ExecutionService, KillSwitch, TradingGate, TradingMode
from app.risk.limits import LimitBreached, RiskLimits, Severity
from app.strategies.intraday import IntradayStrategy

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

REPLAY_WARN = (
    "[REPLAY] Offline playback — NOT live trading. Results depend entirely on "
    "the integrity of the bar history supplied; synthetic data is marked "
    "explicitly and must never be treated as market data."
)

_SESSION_OPEN = time(9, 15)
_SESSION_CLOSE = time(15, 30)

_STRATEGY_OVERRIDABLE = {
    "min_net_edge", "max_positions", "min_intraday_volume", "cost_estimate",
}


def build_strategy_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Validate strategy overrides for the replay grid.

    Only constructor-exposed, production-honest parameters of
    ``IntradayStrategy`` are accepted, so a research sweep cannot silently
    change an internal constant that production never exposes.
    """
    cfg = dict(config or {})
    unknown = set(cfg) - _STRATEGY_OVERRIDABLE
    if unknown:
        raise ValueError(f"unknown strategy override(s): {sorted(unknown)}; "
                         f"allowed: {sorted(_STRATEGY_OVERRIDABLE)}")
    return cfg


def _next_minute_t(minute: time) -> time:
    if minute.minute == 59:
        return time(minute.hour + 1, 0)
    return time(minute.hour, minute.minute + 1)


@dataclass
class TradeRecord:
    session: str
    symbol: str
    direction: str
    entry_ts: str
    entry_price: float
    qty: int
    exit_ts: str
    exit_price: float
    exit_reason: str
    pnl_rupees: float
    costs_rupees: float
    net_pnl_rupees: float
    protection_trigger: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionResult:
    session: str
    cycles: int = 0
    entries: int = 0
    exits: int = 0
    stops_placed: int = 0
    stops_cancelled: int = 0
    rejected: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    open_at_end: List[str] = field(default_factory=list)
    cash_start: float = 0.0
    cash_end: float = 0.0
    equity_return_pct: float = 0.0
    recovery_ok: bool = True
    recovery_detail: str = "OK"
    # O2: persisted intraday P&L / high-water mark (see app/engine/equity.py).
    # Populated only when an EquityTracker is supplied to drive_session; replay
    # leaves them empty so existing single-session output is unchanged.
    equity_snapshots: List[Any] = field(default_factory=list)
    start_equity: Optional[float] = None
    end_equity: Optional[float] = None
    peak_equity: Optional[float] = None
    min_equity: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    daily_loss_rupees: Optional[float] = None


@dataclass
class ReplayResult:
    synthetic: bool
    symbols: List[str]
    sessions: List[SessionResult] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    equity: List[dict] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def write(self, prefix: str) -> str:
        trades_path = f"{prefix}_trades.json"
        equity_path = f"{prefix}_equity.json"
        with open(trades_path, "w") as fh:
            json.dump([t.as_dict() for t in self.trades], fh, indent=2, default=str)
        with open(equity_path, "w") as fh:
            json.dump(self.equity, fh, indent=2, default=str)
        return f"{trades_path},{equity_path}"


class ReplayClock:
    """A settable IST clock the trader and the paper broker both read."""

    def __init__(self, day: date, tick: Optional[datetime] = None) -> None:
        self._now = tick or datetime.combine(day, _SESSION_OPEN, tzinfo=IST)

    def set(self, now: datetime) -> None:
        self._now = to_ist(now)

    def __call__(self) -> datetime:
        return self._now


# Backwards-compatible alias for code that imported the private name.
_ReplayClock = ReplayClock


class _LocalState:
    """Minimal LocalStateStore double; cash matches the paper account."""

    def __init__(self, cash: float) -> None:
        self._cash = cash

    async def get_positions(self) -> list[dict]:
        return []

    async def get_orders(self) -> list[dict]:
        return []

    async def get_trades(self) -> list[dict]:
        return []

    async def get_cash(self) -> dict:
        return {"cash": self._cash}


def _build_stack(
    broker: PaperBroker,
    cash: float,
    local_state: Optional[Any] = None,
    *,
    kill_switch_store: Optional[Any] = None,
    order_store: Optional[Any] = None,
) -> Mapping[str, Any]:
    """Assemble the same collaborators main.py / tests assemble.

    The kill switch and order stores default to the in-memory implementations
    (identical to ``build_production_stack``'s Redis-less fallback).  The paper
    campaign overrides them with the durable file stores so engaged halts and
    reserved order claims survive between days and across restarts.
    """
    kill_store = kill_switch_store if kill_switch_store is not None else InMemoryKillSwitchStore()
    safety = ExecutionSafety(kill_store)
    order_manager = OrderManager(safety, store=order_store or InMemoryOrderStore())
    local = local_state if local_state is not None else _LocalState(cash=cash)
    engine = ReconciliationEngine(kill_store, local_state=local)
    recovery = RecoveryManager(engine, local_state=local)
    sink = InMemoryAuditSink()
    service = ExecutionService(
        broker=broker,
        order_manager=order_manager,
        trading_mode=TradingMode.PAPER,
        kill_switch=KillSwitch(store_kill_switch_probe(kill_store)),
        trading_gate=TradingGate(recovery),
        audit=AuditJournal(sink),
        live_authorized=False,
    )
    return {
        "kill_store": kill_store,
        "service": service,
        "recovery": recovery,
        "broker": broker,
    }


def sessions_in(bar_source: Mapping[str, pd.DataFrame]) -> List[date]:
    all_dates = set()
    for df in bar_source.values():
        if df is None or df.empty or "time" not in df.columns:
            continue
        for ts in pd.to_datetime(df["time"]):
            all_dates.add(to_ist(ts.to_pydatetime()).date())
    return sorted(all_dates)


def _minute_rows(df: Optional[pd.DataFrame], day: date, minute) -> Optional[pd.DataFrame]:
    if df is None or df.empty or "time" not in df.columns:
        return None
    positions: List[int] = []
    for idx, ts in enumerate(pd.to_datetime(df["time"])):
        d = to_ist(ts.to_pydatetime())
        floored = time(d.hour, d.minute)
        if d.date() == day and floored == minute:
            positions.append(idx)
    if not positions:
        return None
    return df.iloc[positions]


def _push_minute(agg: TickBarAggregator, day: date, minute, row: pd.Series, sym: str) -> None:
    ts = datetime.combine(day, minute, tzinfo=IST)
    volume = float(row.get("volume", 0) or 0) / 4.0
    for px in (float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])):
        if math.isfinite(px) and px > 0:
            agg.on_tick(sym, ts, px, volume)


async def _record_equity(
    tracker: EquityTracker,
    aggregator: TickBarAggregator,
    broker: PaperBroker,
    day: date,
    minute,
) -> None:
    """Mark the paper book to market at ``minute`` and record the snapshot."""
    funds = await broker.get_funds()
    cash = float(funds.get("total_cash") or funds.get("cash") or 0.0)
    positions = await broker.get_positions()
    prices = {}
    for sym in aggregator.symbols():
        px = aggregator.last_price(sym)
        if px is not None and px > 0:
            prices[sym] = px
    tracker.record(datetime.combine(day, minute, tzinfo=IST), cash, positions, prices)


def _prepare_aggregator(
    bar_source: Mapping[str, pd.DataFrame],
    day: date,
    symbols: Sequence[str],
) -> TickBarAggregator:
    """Push the pre-open window (09:15..09:30) into a fresh aggregator.

    Bars must already exist before the paper broker primes its PB-STALE quote
    cache (the same ordering as the tests — push bars first, then construct
    the broker, then snapshot).
    """
    agg = TickBarAggregator()
    pre = time(9, 15)
    while pre < time(9, 31):
        for sym in symbols:
            rows = _minute_rows(bar_source.get(sym), day, pre)
            if rows is not None and not rows.empty:
                for _, row in rows.iterrows():
                    _push_minute(agg, day, pre, row, sym)
        pre = _next_minute_t(pre)
    return agg


async def drive_session(
    bar_source: Mapping[str, pd.DataFrame],
    *,
    day: date,
    symbols: Sequence[str],
    aggregator: TickBarAggregator,
    broker: PaperBroker,
    stack: Mapping[str, Any],
    strategy: IntradayStrategy,
    clock: ReplayClock,
    collect_reports: bool,
    equity_tracker: Optional[EquityTracker] = None,
    risk_limits: Optional[RiskLimits] = None,
    on_breach: Optional[Callable[[Sequence[LimitBreached]], None]] = None,
) -> SessionResult:
    """Drive one full session through a caller-supplied production stack.

    Connects the broker (loading any persisted paper book), runs startup
    recovery reconciliation, then replays 09:31..15:30 minute-by-minute through
    the trader.  Reused by both ``run_replay`` (fresh broker per session) and
    the multi-day paper campaign (state-backed, continuous book).

    ``risk_limits`` + ``on_breach`` are the opt-in *enforcement* link: after
    each minute's equity snapshot, a HARD daily-loss (:data:`Severity.BREACH`)
    or drawdown (:data:`Severity.CRITICAL`) breach fires ``on_breach`` once.
    The evaluation is live in the loop precisely so the breach is handled
    *before* the next order a minute later — a hard breach halts, it is not
    merely written down.  Leave ``risk_limits`` at ``None`` for recording only.
    """
    trader = IntradayAutoTrader(
        execution_service=stack["service"],
        strategy=strategy,
        bar_source=aggregator,
        broker=broker,
        universe=list(symbols),
        clock=clock,
    )

    await broker.connect()
    rec_report = await stack["recovery"].recover(broker)
    for sym in symbols:
        if aggregator.has_ticked(sym):
            await broker._snapshot(sym, "NSE")

    funds = await broker.get_funds()
    cash_start = float(funds.get("cash") or 0.0)
    result = SessionResult(
        session=day.isoformat(),
        cycles=0,
        cash_start=cash_start,
        cash_end=cash_start,
        recovery_ok=rec_report.trading_permitted,
        recovery_detail=(
            "OK"
            if rec_report.trading_permitted
            else f"{rec_report.state.value}: {rec_report.blocked_reason}"
        ),
    )
    reports = []
    decisions: List[Any] = []
    has_bars = any(
        df is not None and not df.empty for df in bar_source.values()
    )

    minute = time(9, 31)
    while minute <= _SESSION_CLOSE:
        clock.set(datetime.combine(day, minute, tzinfo=IST))
        for sym in symbols:
            rows = _minute_rows(bar_source.get(sym), day, minute)
            if rows is not None and not rows.empty:
                for _, row in rows.iterrows():
                    _push_minute(aggregator, day, minute, row, sym)

        # Riding SL-M stops fill only when the broker polls them — the trader
        # never does.  Poll after the price moves, before the trader evaluates.
        await broker.poll_open_orders()

        if minute >= time(9, 31):
            report = await trader.run_once()
            if collect_reports:
                reports.append(report)
            decisions.extend(report.exits)
            result.cycles += 1
            result.entries += len(report.entries)
            result.exits += len(report.exits)
            result.stops_placed += len(report.stops_placed)
            result.stops_cancelled += len(report.stopped_cancelled)
            result.rejected.extend(report.rejected)
            result.errors.extend(report.errors)

        if equity_tracker is not None:
            await _record_equity(equity_tracker, aggregator, broker, day, minute)
            if risk_limits is not None and on_breach is not None:
                hard = [
                    b for b in risk_breaches_for(
                        equity_tracker.snapshots[-1], risk_limits
                    )
                    if b.severity in (Severity.BREACH, Severity.CRITICAL)
                ]
                if hard:
                    on_breach(hard)

        minute = _next_minute_t(minute)

    if not has_bars:
        return result

    result.trades = await _finish_session_trades(broker, day, decisions)

    positions = await broker.get_positions()
    result.open_at_end = [
        f"{p['symbol']}x{p.get('quantity')}"
        for p in positions
        if int(p.get("quantity", 0) or 0) != 0
    ]
    funds = await broker.get_funds()
    cash_end = float(funds.get("cash") or 0.0)
    result.cash_end = cash_end
    result.equity_return_pct = (
        100.0 * (cash_end - cash_start) / cash_start if cash_start else 0.0
    )
    if equity_tracker is not None:
        # Final snapshot AFTER the EOD fills book realised P&L, so the close
        # of the equity curve reflects the closed book, not the last tick.
        await _record_equity(
            equity_tracker, aggregator, broker, day, _SESSION_CLOSE
        )
        result.equity_snapshots = list(equity_tracker.snapshots)
        result.start_equity = equity_tracker.start_equity
        result.end_equity = equity_tracker.end_equity
        result.peak_equity = equity_tracker.peak_equity
        result.min_equity = equity_tracker.min_equity
        result.max_drawdown_pct = equity_tracker.max_drawdown_pct
        result.daily_loss_rupees = equity_tracker.max_daily_loss_rupees
    return result


async def _run_session(
    bar_source: Mapping[str, pd.DataFrame],
    day: date,
    symbols: Sequence[str],
    *,
    cash: float,
    collect_reports: bool,
    strategy_config: Optional[Mapping[str, Any]] = None,
) -> SessionResult:
    clock = ReplayClock(day)
    aggregator = _prepare_aggregator(bar_source, day, symbols)
    broker = PaperBroker(data_broker=TickQuoteSource(aggregator), initial_cash=cash,
                         clock=clock)
    stack = _build_stack(broker, cash)
    strategy = IntradayStrategy(paper_mode=True,
                                **(build_strategy_config(strategy_config)))
    return await drive_session(
        bar_source,
        day=day,
        symbols=symbols,
        aggregator=aggregator,
        broker=broker,
        stack=stack,
        strategy=strategy,
        clock=clock,
        collect_reports=collect_reports,
    )


async def _finish_session_trades(
    broker: PaperBroker, day: date, decisions: Sequence[Any],
) -> List[TradeRecord]:
    from app.broker.base import TransactionType

    raw = await broker.get_trades()
    fills = []
    for t in raw:
        if t is None:
            continue
        try:
            ts = pd.to_datetime(t.get("timestamp"))
        except Exception:
            continue
        if to_ist(ts.to_pydatetime()).date() != day:
            continue
        fills.append({
            "symbol": str(t.get("symbol")),
            "txn": str(t.get("txn_type")),
            "qty": int(t.get("qty", 0)),
            "price": float(t.get("price", 0.0) or 0.0),
            "costs_paise": float(t.get("costs_paise", 0) or 0),
            "ts": to_ist(ts.to_pydatetime()),
        })
    if not fills:
        return []

    reasons: Dict[str, List[Any]] = {}
    for d in decisions:
        reasons.setdefault(str(d.symbol), []).append(
            (to_ist(d.exit_time), str(getattr(d.reason, "value", d.reason)))
        )
    for sym in reasons:
        reasons[sym].sort(key=lambda pair: pair[0])

    def reason_for(sym: str, ts: datetime) -> str:
        queue = reasons.get(str(sym))
        if not queue:
            return "broker_exit"
        best = queue[0]
        while len(queue) > 1 and queue[1][0] <= ts:
            best = queue.pop(0)
        if best[0] <= ts:
            queue.pop(0)
            return best[1]
        return "round_trip"

    buys: Dict[str, List[dict]] = {}
    sells: Dict[str, List[dict]] = {}
    for f in fills:
        (buys if f["txn"] == TransactionType.BUY.value else sells) \
            .setdefault(f["symbol"], []).append(f)
    for b in buys.values():
        b.sort(key=lambda f: f["ts"])
    for s in sells.values():
        s.sort(key=lambda f: f["ts"])

    trades: List[TradeRecord] = []
    for sym in sorted(set(buys) | set(sells)):
        buy_queue = list(buys.get(sym, []))
        for sell in sells.get(sym, []):
            remaining = int(sell["qty"])
            while remaining > 0 and buy_queue:
                b = buy_queue[0]
                taken = min(b["qty"], remaining)
                entry = float(b["price"])
                exit_px = float(sell["price"])
                intrinsic = (exit_px - entry) * taken
                cost_share = (b["costs_paise"] + sell["costs_paise"]) / 100.0 \
                    * (taken / max(b["qty"], 1))
                trades.append(TradeRecord(
                    session=day.isoformat(),
                    symbol=sym,
                    direction="LONG",
                    entry_ts=b["ts"].isoformat(),
                    entry_price=entry,
                    qty=taken,
                    exit_ts=sell["ts"].isoformat(),
                    exit_price=exit_px,
                    exit_reason=reason_for(sym, sell["ts"]),
                    pnl_rupees=round(intrinsic, 2),
                    costs_rupees=round(cost_share, 2),
                    net_pnl_rupees=round(intrinsic - cost_share, 2),
                ))
                b["qty"] -= taken
                remaining -= taken
            buy_queue = [x for x in buy_queue if x["qty"] > 0]
    return trades


def _sharpe(returns_pct: Sequence[float], periods_per_year: float = 252.0) -> Optional[float]:
    arr = np.asarray(returns_pct, dtype=float)
    if arr.size < 2:
        return None
    sd = float(arr.std(ddof=1))
    if not math.isfinite(sd) or sd <= 1e-12:
        return None
    return float(arr.mean() / sd * math.sqrt(periods_per_year))


def _max_drawdown(equity_pct: Sequence[float]) -> float:
    eq = np.asarray(equity_pct, dtype=float)
    if eq.size == 0:
        return 0.0
    cum = np.cumprod(1.0 + eq / 100.0)
    peak = np.maximum.accumulate(cum)
    return float(np.max(peak - cum) / peak[-1]) if peak[-1] > 0 else 0.0


def summarize(result: ReplayResult) -> Dict[str, Any]:
    trades = result.trades
    sessions = result.sessions
    daily = [s.equity_return_pct for s in sessions] or [0.0]
    net = sum(t.net_pnl_rupees for t in trades)
    gross = sum(t.pnl_rupees for t in trades)
    costs = sum(t.costs_rupees for t in trades)
    wins = [t for t in trades if t.net_pnl_rupees > 0]
    losses = [t for t in trades if t.net_pnl_rupees < 0]
    summary = {
        "synthetic": result.synthetic,
        "symbols": result.symbols,
        "days": len(sessions),
        "trades": len(trades),
        "trades_per_day": round(len(trades) / len(sessions), 2) if sessions else 0.0,
        "percent_trading_days": round(
            100.0 * sum(1 for s in sessions if s.trades) / len(sessions), 1
        ) if sessions else 0.0,
        "win_rate_pct": round(100.0 * len(wins) / max(len(trades), 1), 1),
        "avg_winner": round(sum(t.net_pnl_rupees for t in wins)
                            / max(len(wins), 1), 2),
        "avg_loser": round(sum(t.net_pnl_rupees for t in losses)
                           / max(len(losses), 1), 2),
        "profit_factor": round(sum(t.net_pnl_rupees for t in wins)
                               / max(abs(sum(t.net_pnl_rupees for t in losses)), 1e-9), 2),
        "gross_pnl_rupees": round(gross, 2),
        "total_costs_rupees": round(costs, 2),
        "net_pnl_rupees": round(net, 2),
        "cost_drag_pct": round(100.0 * abs(costs) / max(abs(gross), 1e-9), 1),
        "expectancy_rupees_per_trade": round(net / max(len(trades), 1), 2),
        "mean_daily_return_pct": round(float(np.mean(daily)), 4),
        "daily_vol_pct": round(
            float(np.std(daily, ddof=1)) if len(daily) > 1 else 0.0, 4,
        ),
        "sharpe_annualized": _sharpe(daily),
        "max_drawdown_pct": round(100.0 * _max_drawdown(daily), 2),
        "sessions_with_errors": sum(1 for s in sessions if s.errors),
        "open_positions_at_sweep": sum(len(s.open_at_end) for s in sessions),
        "rejected_orders": sum(len(s.rejected) for s in sessions),
    }
    return summary


async def run_replay(
    bar_source: Mapping[str, pd.DataFrame],
    *,
    symbols: Optional[Sequence[str]] = None,
    cash: float = 1_000_000.0,
    days_back: Optional[int] = None,
    collect_reports: bool = False,
    synthetic: bool = False,
    strategy_config: Optional[Mapping[str, Any]] = None,
) -> ReplayResult:
    logger.warning(REPLAY_WARN)
    build_strategy_config(strategy_config)
    if symbols:
        bar_source = {k: v for k, v in bar_source.items() if k in set(symbols)}
    sessions = sessions_in(bar_source)
    if not sessions:
        raise ValueError("no sessions found in the bar source (empty or unreadable data)")
    if days_back is not None:
        sessions = sessions[-days_back:]
    result = ReplayResult(synthetic=synthetic, symbols=sorted(bar_source))

    for day in sessions:
        session_result = await _run_session(
            bar_source, day, sorted(bar_source), cash=cash,
            collect_reports=collect_reports, strategy_config=strategy_config,
        )
        result.sessions.append(session_result)
        result.trades.extend(session_result.trades)
        result.equity.append({
            "session": day.isoformat(),
            "cash_end": round(session_result.cash_end, 2),
            "return_pct": round(session_result.equity_return_pct, 4),
            "open_at_end": session_result.open_at_end,
        })

    result.summary = summarize(result)
    return result


def synthetic_bar_source(
    symbols: Sequence[str],
    sessions: Sequence[date],
    *,
    seed: int = 0,
    start_px: float = 100.0,
) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out: Dict[str, pd.DataFrame] = {}
    for si, sym in enumerate(symbols):
        base = start_px * (1.0 + 0.4 * si)
        rows: List[dict] = []
        for day in sessions:
            drift = rng.normal(0.0004, 0.0035)
            px = base * (1.0 + rng.normal(0.0, 0.01) * si * 0.1)
            minute = _SESSION_OPEN
            prev = px
            while minute <= _SESSION_CLOSE:
                shock = rng.normal(drift, 0.0009)
                o = prev
                c = o * math.exp(shock)
                h = max(o, c) * (1.0 + abs(rng.normal(0.0, 0.0025)))
                lo = min(o, c) * (1.0 - abs(rng.normal(0.0, 0.0025)))
                volume = int(rng.poisson(4000.0))
                rows.append({
                    "time": datetime.combine(day, minute, tzinfo=IST),
                    "open": round(o, 2),
                    "high": round(h, 2),
                    "low": round(lo, 2),
                    "close": round(c, 2),
                    "volume": volume,
                })
                prev = c
                minute = (
                    time(minute.hour + 1, 0) if minute.minute == 59
                    else time(minute.hour, minute.minute + 1)
                )
        out[sym] = pd.DataFrame(rows)
    return out


def load_bars_from_csv(data_dir: str, symbols: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for path in sorted(os.listdir(data_dir)):
        if not path.endswith(".csv"):
            continue
        sym = os.path.splitext(path)[0].upper()
        if symbols and sym not in set(symbols):
            continue
        df = pd.read_csv(os.path.join(data_dir, path))
        parsed = pd.to_datetime(df["time"])
        if parsed.dt.tz is None:
            parsed = parsed.dt.tz_localize(IST, ambiguous="NaT")
        else:
            parsed = parsed.dt.tz_convert(IST)
        df["time"] = parsed
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                raise ValueError(f"{path} missing column {col}")
        out[sym] = df
    return out


def _recent_sessions(n: int) -> List[date]:
    today = date.today()
    out: List[date] = []
    d = today
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d = date.fromordinal(d.toordinal() - 1)
    return list(reversed(out))


async def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Offline replay of the paper intraday auto-trader")
    parser.add_argument("--dir", help="directory of per-symbol 1-min CSV files (time,open,high,low,close,volume)")
    parser.add_argument("--symbols", help="comma-separated symbol filter")
    parser.add_argument("--synthetic", type=int, metavar="DAYS", help="run on N business days of synthetic bars")
    parser.add_argument("--cash", type=float, default=1_000_000.0, help="starting cash")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--days", type=int, help="only replay the last N sessions")
    parser.add_argument("--out", default="replay", help="output prefix for trades/equity JSON")
    args = parser.parse_args(list(argv))

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    synthetic = args.synthetic is not None

    if args.dir:
        bars = load_bars_from_csv(args.dir, symbols)
        if not bars:
            print("no bar files found under", args.dir, file=sys.stderr)
            return 2
    elif synthetic:
        bars = synthetic_bar_source(
            symbols or ["RELIANCE", "TCS", "INFY"],
            _recent_sessions(args.synthetic), seed=args.seed,
        )
    else:
        parser.print_usage(sys.stderr)
        print("provide --dir (real data) or --synthetic DAYS", file=sys.stderr)
        return 2

    result = await run_replay(
        bars, symbols=symbols, cash=args.cash,
        days_back=args.days, synthetic=synthetic,
    )
    written = result.write(args.out)
    print(json.dumps(result.summary, indent=2))
    print("wrote", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
