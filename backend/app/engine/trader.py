"""
trader.py — the paper intraday auto-trader.

A loop that turns the live (mock) tick stream into a managed intraday book:

        MockTickFeed ──▶ TickBarAggregator ──▶ IntradayAutoTrader ──▶ ExecutionService
                          (1-minute bars)       (this module)          (the boundary:
                                                                        kill switch,
                                                                        gates, audit)

WHY IT EXISTS
-------------
The intraday strategy was always a decision engine: it produces signals, but
nothing ever ran it continuously, so its signals were never turned into a
book.  This loop does that exactly once per cycle:

  1. Session gate — before the opening grace window ends, or after market
     close, there is nothing to do.
  2. Read the bars the aggregator has built from the tick stream.
  3. Reconcile the in-memory ledger against the broker's actual MIS book.
  4. EXITS first — stop / target / vwap reversal / square-off, via
     `PositionExitMonitor`.  A manual exit CANCELS the protective SL-M stop
     first: a lingering stop would otherwise fill later and open a spurious
     short in a book that cannot short.
  5. ENTRIES — `IntradayStrategy.generate_signals`, sized by the strategy,
     checked by the risk engine, and submitted for MIS.
  6. PROTECTIVE STOPS — an SL-M sell stop on every live entry, so a dead
     process still stops the position.  If the stop cannot be placed, the
     entry is immediately exited again: an unprotected position is not an
     acceptable state.

WHY EXITS RUN BEFORE ENTRIES
----------------------------
Freeing risk and cash first means an exiting position's capital is available
to new entries in the same cycle, and it keeps the position-count cap honest.

SAFETY CONTRACT
---------------
`main.py` arms this loop ONLY after verifying:
  * settings.auto_trade_enabled is True   (a human turned it on)
  * settings.trading_mode == "paper"       (never a live account)
  * settings.tick_mode == "mock"           (never a live feed)
Every order it emits STILL travels through `ExecutionService`, which
re-checks the kill switch, trading gate, mode and eligibility on each request.
The ledger is in-memory by design: a restart loses the monitor's view of
stop/target levels, but the broker-side SL-M stops and the broker's own
intraday square-off survive as the backstop.  A lost ledger is safe; a
lost stop would not be.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from app.broker.base import BrokerInterface
from app.engine.monitor import ExitDecision, ExitReason, PositionExitMonitor
from app.execution.audit import ExecutionOutcome
from app.execution.service import ExecutionService
from app.risk.engine import RiskEngine
from app.strategies.base import BaseStrategy, Signal, SignalDirection

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# IST wall-clock gates (must agree with app.strategies.intraday).
_OPENING_GRACE_END = time(9, 30)   # no new positions in the first 15 minutes
_SIGNAL_CUTOFF = time(14, 45)      # no new positions after this
_SQUARE_OFF_TIME = time(15, 15)    # force-exit all intraday positions
_SESSION_CLOSE = time(15, 30)      # nothing to do after market close
_PRODUCT_INTRADAY = "MIS"

# An adopted position whose stop/target the ledger cannot know uses a no-op
# stop (never triggers) and an unreachable target, leaving VWAP-reversal and
# the square-off as its only exits.  Explicit, so the playback stays honest.
_NO_STOP = 0.0
_NO_TARGET = 1.0e18
# Risk anchor for a manual exit order.  The protective SL-M is cancelled
# *before* the market exit so its later fill cannot mint a fresh short;
# between cancel and fill the position is naked.  Rather than charge the full
# notional as risk (which blocks every profitable exit against the daily-risk
# budget), the exit signal declares a small adverse-gap budget — the maximum
# move we are willing to absorb between decision and fill.
_EXIT_GAP_FRACTION = 0.01


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class LedgerPosition:
    """The trade's own view of one intraday position and its protective stop."""

    symbol: str
    quantity: int
    entry_price: float
    stop_price: float       # absolute SL-M trigger (below entry for a long)
    target_price: float
    direction: str = "LONG"
    strategy: str = ""
    entry_time: Optional[datetime] = None
    entry_order_id: Optional[str] = None
    stop_order_id: Optional[str] = None


class IntradayPositionLedger:
    """
    In-memory record of the positions this trader opened and their stop levels.

    Deliberately NOT durable: see the module docstring.  `reconcile` keeps it
    consistent with the broker's actual MIS book, so a position closed by the
    broker (SL-M filled, manual trade elsewhere) drops out on the next cycle.
    """

    def __init__(self) -> None:
        self._rows: Dict[str, LedgerPosition] = {}

    # ------------------------------------------------------------------ #
    #  Writes                                                            #
    # ------------------------------------------------------------------ #

    def record_entry(
        self,
        signal: Signal,
        quantity: int,
        entry_price: float,
        entry_order_id: Optional[str] = None,
    ) -> LedgerPosition:
        stop_pct = float(signal.stop_loss_pct or 0.0)
        target_pct = float(signal.target_pct or 0.0)
        row = LedgerPosition(
            symbol=signal.symbol,
            quantity=int(quantity),
            entry_price=float(entry_price),
            stop_price=float(entry_price * (1 - stop_pct)) if stop_pct > 0 else _NO_STOP,
            target_price=float(entry_price * (1 + target_pct)) if target_pct > 0 else _NO_TARGET,
            direction=str(getattr(signal.direction, "value", "LONG")),
            strategy=str(signal.strategy_name),
            entry_time=signal.timestamp,
            entry_order_id=entry_order_id,
        )
        self._rows[row.symbol] = row
        return row

    def record_stop_order(self, symbol: str, stop_order_id: Optional[str]) -> None:
        row = self._rows.get(symbol)
        if row is not None:
            row.stop_order_id = stop_order_id

    def drop(self, symbol: str) -> None:
        self._rows.pop(symbol, None)

    def clear(self) -> None:
        self._rows.clear()

    # ------------------------------------------------------------------ #
    #  Reads                                                             #
    # ------------------------------------------------------------------ #

    def open(self) -> List[LedgerPosition]:
        return list(self._rows.values())

    def get(self, symbol: str) -> Optional[LedgerPosition]:
        return self._rows.get(symbol)

    def symbols(self) -> List[str]:
        return list(self._rows.keys())

    # ------------------------------------------------------------------ #
    #  Broker reconciliation                                             #
    # ------------------------------------------------------------------ #

    def reconcile(self, broker_positions: Sequence[dict], now: datetime) -> List[str]:
        """
        Sync the ledger with the broker's live MIS book.

        * A symbol present at the broker but unknown in the ledger is ADOPTED
          (stop/target unknown: stop=0, target=unreachable).  This happens on
          a mid-session restart.
        * A ledger row whose broker position has gone is dropped — the exit
          already happened (usually the SL-M stop filling).
        * Matching rows get the broker's quantity.

        Returns the symbols adopted, so the caller can report them.
        """
        adopted: List[str] = []
        by_symbol: Dict[str, int] = {}
        for pos in broker_positions:
            if str(pos.get("product", "")).upper() != _PRODUCT_INTRADAY:
                continue
            sym = str(pos.get("symbol", ""))
            qty = int(pos.get("quantity", 0) or 0)
            if qty > 0:
                by_symbol[sym] = qty

        for sym, qty in by_symbol.items():
            row = self._rows.get(sym)
            if row is None:
                avg = float(pos_avg(broker_positions, sym) or 0.0)
                self._rows[sym] = LedgerPosition(
                    symbol=sym,
                    quantity=qty,
                    entry_price=avg or 0.0,
                    stop_price=_NO_STOP,
                    target_price=_NO_TARGET,
                    direction="LONG",
                    strategy="adopted",
                    entry_time=now,
                )
                adopted.append(sym)
                logger.info(
                    "Ledger adopted %s qty=%d (restart mid-session); "
                    "monitor keeps only VWAP+square-off exits for it.",
                    sym, qty,
                )
            else:
                row.quantity = qty

        self._rows = {s: r for s, r in self._rows.items() if s in by_symbol}
        return adopted


def pos_avg(positions: Sequence[dict], symbol: str) -> Optional[float]:
    for p in positions:
        if str(p.get("symbol", "")) == symbol:
            return float(p.get("average_price", 0.0) or 0.0) or None
    return None


@dataclass
class CycleReport:
    """What one trader cycle did — for tests, logs and the dashboard."""

    started_at: datetime
    session_active: bool = True
    adopted: List[str] = field(default_factory=list)
    exits: List[ExitDecision] = field(default_factory=list)
    entries: List[dict] = field(default_factory=list)
    stops_placed: List[str] = field(default_factory=list)
    stopped_cancelled: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class IntradayAutoTrader:
    """
    The continuous intraday loop.  One `run_once()` = one full evaluate-and-act
    pass; `run(stop_event)` wraps it with a `cycle_seconds` cadence.
    """

    def __init__(
        self,
        *,
        execution_service: ExecutionService,
        strategy: BaseStrategy,
        bar_source: Any,
        broker: BrokerInterface,
        universe: Sequence[str],
        risk_engine: Optional[RiskEngine] = None,
        clock: Callable[[], datetime] = _utc_now,
        cycle_seconds: float = 60.0,
        use_broker_stop: bool = True,
        max_entries_per_cycle: int = 3,
        ledger: Optional[IntradayPositionLedger] = None,
        monitor: Optional[PositionExitMonitor] = None,
    ) -> None:
        self.execution = execution_service
        self.strategy = strategy
        self.bar_source = bar_source
        self.broker = broker
        self.universe = list(universe)
        self.risk_engine = risk_engine or RiskEngine()
        self.clock = clock
        self.cycle_seconds = float(cycle_seconds)
        self.use_broker_stop = bool(use_broker_stop)
        self.max_entries_per_cycle = int(max_entries_per_cycle)
        self.ledger = ledger or IntradayPositionLedger()
        self.monitor = monitor or PositionExitMonitor(strategy)

    # ------------------------------------------------------------------ #
    #  Loop                                                              #
    # ------------------------------------------------------------------ #

    async def run(self, stop_event: Any) -> int:
        cycles = 0
        while not stop_event.is_set():
            try:
                await self.run_once()
            except Exception as exc:  # noqa: BLE001
                # A broken cycle must never kill the trader silently.  The
                # broker-side stops are the backstop while we keep going;
                # stopping the loop entirely would take that away.
                logger.exception("auto-trader cycle failed (%s); continuing", exc)
            cycles += 1
            await asyncio.sleep(self.cycle_seconds)
        return cycles

    # ------------------------------------------------------------------ #
    #  One pass                                                          #
    # ------------------------------------------------------------------ #

    async def run_once(self) -> CycleReport:
        now = to_ist(self.clock())
        report = CycleReport(started_at=now)

        if now.time() < _OPENING_GRACE_END or now.time() >= _SESSION_CLOSE:
            report.session_active = False
            logger.debug("Auto-trader idle at %s IST (outside trading gate).", now.time())
            return report

        bars = {sym: self.bar_source.bars(sym) for sym in self.universe}

        positions = await self.broker.get_positions()
        report.adopted = self.ledger.reconcile(positions, now)

        # ---- exits first ------------------------------------------------
        monitor_positions = [self._monitor_position(row) for row in self.ledger.open()]
        decisions = self.monitor.evaluate(monitor_positions, bars, now)
        positions = await self.broker.get_positions()
        for decision in decisions:
            await self._execute_exit(decision, positions, bars, report)

        # ---- entries (only while the signal window is open) -------------
        if now.time() < _SIGNAL_CUTOFF:
            await self._execute_entries(bars, now, report)

        logger.info(
            "Cycle @ %s IST: %d exit(s), %d entry(ies), %d stop(s), "
            "%d rejected, %d error(s), %d open.",
            now.time(), len(report.exits), len(report.entries),
            len(report.stops_placed), len(report.rejected), len(report.errors),
            len(self.ledger.open()),
        )
        return report

    # ------------------------------------------------------------------ #
    #  Exits                                                             #
    # ------------------------------------------------------------------ #

    async def _execute_exit(
        self,
        decision: ExitDecision,
        positions: Sequence[dict],
        bars: Mapping[str, pd.DataFrame],
        report: CycleReport,
    ) -> None:
        symbol = decision.symbol
        row = self.ledger.get(symbol)
        live_qty = pos_quantity(positions, symbol)
        if live_qty <= 0:
            # The protective stop already did the job; nothing left to exit.
            self.ledger.drop(symbol)
            report.exits.append(decision)
            logger.info("Exit %s: position already closed by its stop.", symbol)
            return

        # 1. Remove the protective stop BEFORE exiting, or its later fill
        #    would mint a fresh short.
        if row is not None and row.stop_order_id and self.use_broker_stop:
            res = await self.execution.cancel_order(
                row.stop_order_id, symbol=symbol, strategy=row.strategy,
            )
            if res.outcome == ExecutionOutcome.BLOCKED_KILL_SWITCH:
                report.errors.append(
                    f"cancel-stop:{symbol}:kill switch active — keeping stop"
                )
                return  # keep the protective stop; the book is healthier with it
            report.stopped_cancelled.append(row.stop_order_id)

        # 2. Exit at market through the SAME boundary as everything else.
        exit_signal = self._exit_signal(symbol, decision.exit_time)
        bar = bars.get(symbol)
        ref_price = None
        if bar is not None and not bar.empty:
            ref_price = float(bar["close"].iloc[-1])
        positions_now = await self.broker.get_positions()
        qty = pos_quantity(positions_now, symbol)
        if qty <= 0:
            self.ledger.drop(symbol)
            report.exits.append(decision)
            return
        funds_now = await self.broker.get_funds()
        pv = self._portfolio_value(funds_now, positions_now)
        # Exits carry the SAME portfolio context as entries: without it the
        # risk gates fail closed (total_portfolio 0 -> single-stock exposure
        # cannot be sized) and the no-stop exit is risked at its full notional,
        # which trips the daily-risk budget. A risk-reducing exit must not be
        # blocked by a bookkeeping default.
        res = await self.execution.submit_signal(
            exit_signal, qty, product=_PRODUCT_INTRADAY,
            reference_price=ref_price or 0.0,
            portfolio_allocation={
                "intent": "auto_intraday_exit",
                "reason": decision.reason.value,
            },
            available_cash=max(float(funds_now.get("cash") or 0.0) + ref_price * qty, 0.0),
            total_portfolio=pv,
            current_positions=positions_now,
        )
        if res.submitted:
            self.ledger.drop(symbol)
            report.exits.append(decision)
            logger.info(
                "Exit %s [%s] qty=%d @ %s submitted (%s).",
                symbol, decision.reason.value, qty, ref_price, res.broker_order_id,
            )
        else:
            report.errors.append(f"exit:{symbol}:{res.reason}")

    # ------------------------------------------------------------------ #
    #  Entries + protective stops                                        #
    # ------------------------------------------------------------------ #

    async def _execute_entries(
        self,
        bars: Mapping[str, pd.DataFrame],
        now: datetime,
        report: CycleReport,
    ) -> None:
        positions = await self.broker.get_positions()
        existing = {p["symbol"]: p for p in positions}
        funds = await self.broker.get_funds()
        cash = float(funds.get("cash") or 0.0)
        pv = self._portfolio_value(funds, positions)

        self.risk_engine.set_portfolio_context(
            portfolio_value=pv, available_cash=cash, positions=positions,
        )

        signals = self.strategy.generate_signals(
            self.universe,
            pd.DataFrame(),
            {},
            current_time=now,
            intraday_data={s: bars[s] for s in bars if not bars[s].empty},
            existing_positions=existing,
        )

        attempted: set[str] = set()
        for sig in signals[: self.max_entries_per_cycle]:
            if sig.symbol in attempted or self.ledger.get(sig.symbol) is not None:
                continue
            attempted.add(sig.symbol)
            bar = bars.get(sig.symbol)
            if bar is None or bar.empty:
                report.rejected.append(f"entry:{sig.symbol}:no bar")
                continue
            ref_price = float(bar["close"].iloc[-1])
            if ref_price <= 0:
                report.rejected.append(f"entry:{sig.symbol}:bad price")
                continue

            target_value = self.strategy.calculate_position_size(sig, cash, self.risk_engine)
            if target_value <= 0 or target_value > cash:
                report.rejected.append(f"entry:{sig.symbol}:sizing {target_value:.0f}")
                continue
            qty = int(target_value // ref_price)
            if qty <= 0:
                report.rejected.append(f"entry:{sig.symbol}:qty {qty}")
                continue

            res = await self.execution.submit_signal(
                sig, qty, product=_PRODUCT_INTRADAY,
                reference_price=ref_price,
                idempotency_key=self._idem("ENT", sig, now),
                portfolio_allocation={"intent": "auto_intraday_entry"},
                available_cash=cash,
                total_portfolio=pv,
                current_positions=positions,
            )
            if not res.submitted:
                report.rejected.append(f"entry:{sig.symbol}:{res.reason}")
                continue

            report.entries.append({
                "symbol": sig.symbol, "qty": qty, "price": ref_price,
                "order_id": res.broker_order_id,
            })
            row = self.ledger.record_entry(sig, qty, ref_price, res.broker_order_id)

            if self.use_broker_stop:
                await self._protect(row, sig, qty, ref_price, pv, positions, now, report)

    async def _protect(
        self,
        row: LedgerPosition,
        entry_signal: Signal,
        qty: int,
        fill_price: float,
        pv: float,
        positions: Sequence[dict],
        now: datetime,
        report: CycleReport,
    ) -> None:
        """Place the SL-M protective stop; if it cannot be placed, exit again."""
        positions_now = await self.broker.get_positions()
        res = await self.execution.submit_stop(
            entry_signal, qty, product=_PRODUCT_INTRADAY,
            fill_price=fill_price,
            idempotency_key=self._idem("STP", entry_signal, now),
            portfolio_allocation={
                "intent": "auto_intraday_stop",
                "parent": row.entry_order_id,
            },
            available_cash=pv,
            total_portfolio=pv,
            current_positions=positions_now,
        )
        if res.submitted:
            self.ledger.record_stop_order(row.symbol, res.broker_order_id)
            report.stops_placed.append(f"{row.symbol}:{res.broker_order_id}")
            logger.info(
                "Protective SL-M stop for %s qty=%d trigger=%.2f (%s).",
                row.symbol, qty, row.stop_price, res.broker_order_id,
            )
            return

        # Fail-closed: an un-protected position is one decision away from an
        # uncontrolled loss.  Undo the entry that just passed the boundary.
        report.errors.append(f"stop:{row.symbol}:{res.reason}")
        logger.error(
            "Protective stop for %s could not be placed (%s); exiting position.",
            row.symbol, res.reason,
        )
        await self._execute_exit(
            ExitDecision(
                symbol=row.symbol,
                reason=ExitReason.STOP_PLACEMENT_FAILED,
                price=fill_price, vwap=None, exit_time=now,
            ),
            positions, {}, report,
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #

    def _portfolio_value(self, funds: dict, positions: Sequence[dict]) -> float:
        base = float(funds.get("total_cash") or funds.get("cash") or 0.0)
        pv = base + sum(
            float(p.get("quantity", 0) or 0) * float(p.get("last_price", 0) or 0)
            for p in positions
        )
        return max(pv, base)

    @staticmethod
    def _idem(prefix: str, signal: Signal, now: datetime) -> str:
        from app.execution.lifecycle import deterministic_client_order_id

        return deterministic_client_order_id(
            prefix, signal.symbol, "NSE",
            str(getattr(signal.direction, "value", "")),
            str(int(signal.stop_loss_pct or 0.0)),
            now.date().isoformat(),
            prefix=prefix,
        )

    def _exit_signal(self, symbol: str, now: datetime) -> Signal:
        return Signal(
            symbol=symbol,
            direction=SignalDirection.EXIT,
            strategy_name=self.strategy.name,
            timestamp=now,
            signal_date=now.replace(hour=0, minute=0, second=0, microsecond=0),
            edge_score=0.0,
            expected_return=0.0,
            expected_return_std=0.0,
            stop_loss_pct=_EXIT_GAP_FRACTION,
            target_pct=0.0,
            holding_period_days=0,
            metadata={"intent": "auto_intraday_exit"},
        )

    @staticmethod
    def _monitor_position(row: LedgerPosition) -> dict:
        return {
            "symbol": row.symbol,
            "direction": row.direction,
            "entry_price": row.entry_price,
            "stop_loss": row.stop_price,
            "target": row.target_price,
            "quantity": row.quantity,
            "last_price": row.entry_price,
        }


# --------------------------------------------------------------------------- #
# Module helpers
# --------------------------------------------------------------------------- #


def to_ist(ts: datetime) -> datetime:
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError("auto-trader timestamps must be timezone-aware")
    return ts.astimezone(IST)


def pos_quantity(positions: Sequence[dict], symbol: str) -> int:
    for p in positions:
        if str(p.get("symbol", "")) == symbol:
            return int(p.get("quantity", 0) or 0)
    return 0
