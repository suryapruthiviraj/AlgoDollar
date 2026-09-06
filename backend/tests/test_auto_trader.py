"""
Tests for the paper intraday auto-trader: IntradayAutoTrader, the position
ledger and the exit monitor (app/engine/trader.py, app/engine/monitor.py), plus
the two ExecutionService additions it depends on — submit_stop (SL-M protective
stops) and cancel_order (removing them) in app/execution/service.py.

Everything below uses the REAL production stack: ExecutionService, OrderManager,
ExecutionSafety, PaperBroker, the kill-switch store and the audit journal.  The
only doubles are the two genuine external boundaries — the market-data feed (the
tick source) and a clock — exactly as in test_execution_integration.py.

TIMEZONE: every timestamp is aware IST, pinned to 2025-06-10 (a Tuesday, not an
NSE holiday), so the suite behaves identically under TZ=UTC and TZ=Asia/Kolkata.
Bars are pushed into a real TickBarAggregator, so the prices the strategy
signals on, the prices the broker fills on and the prices the exit monitor
decides on are all the SAME stream — that is the property under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import pytest

from app.broker.base import OrderType, Product, TransactionType
from app.broker.paper import IST, OrderStatus, PaperBroker
from app.broker.tickdata import TickBarAggregator, TickQuoteSource
from app.engine.monitor import ExitReason, PositionExitMonitor
from app.engine.trader import IntradayAutoTrader, IntradayPositionLedger
from app.execution.audit import AuditJournal, ExecutionOutcome, InMemoryAuditSink
from app.execution.bootstrap import InMemoryKillSwitchStore, store_kill_switch_probe
from app.execution.lifecycle import InMemoryOrderStore
from app.execution.order_manager import OrderManager
from app.execution.reconciliation import (
    ReconciliationEngine,
)
from app.execution.recovery import RecoveryManager
from app.execution.safety import ExecutionSafety
from app.execution.service import ExecutionService, KillSwitch, TradingGate, TradingMode
from app.strategies.base import Signal, SignalDirection
from app.strategies.intraday import IntradayStrategy

SYMBOL = "RELIANCE"
TRADING_DAY = 10                                  # 2025-06-10 (Tuesday)
OPEN_IST = datetime(2025, 6, 10, 11, 0, tzinfo=IST)
INITIAL_CASH = 1_000_000.0


def _at(day: int, h: int, m: int = 0) -> datetime:
    return datetime(2025, 6, day, h, m, tzinfo=IST)


class _FakeClock:
    """A controllable UTC-correct clock; the trader feeds it IST values."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@dataclass
class _Stack:
    broker: PaperBroker
    kill_store: InMemoryKillSwitchStore
    order_manager: OrderManager
    order_store: InMemoryOrderStore
    audit: AuditJournal
    sink: InMemoryAuditSink
    service: ExecutionService


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


async def _build_stack(
    *,
    data_broker: Any,
    clock: _FakeClock,
    initial_cash: float = INITIAL_CASH,
) -> _Stack:
    """Assemble the same objects main.py's execution stack assembles."""
    broker = PaperBroker(data_broker=data_broker, initial_cash=initial_cash, clock=clock)
    kill_store = InMemoryKillSwitchStore()
    safety = ExecutionSafety(kill_store)
    order_store = InMemoryOrderStore()
    order_manager = OrderManager(safety, store=order_store)
    local = _LocalState(cash=initial_cash)
    engine = ReconciliationEngine(kill_store, local_state=local)
    recovery = RecoveryManager(engine, local_state=local)
    sink = InMemoryAuditSink()
    audit = AuditJournal(sink)
    service = ExecutionService(
        broker=broker,
        order_manager=order_manager,
        trading_mode=TradingMode.PAPER,
        kill_switch=KillSwitch(store_kill_switch_probe(kill_store)),
        trading_gate=TradingGate(recovery),
        audit=audit,
        live_authorized=False,
    )
    await broker.connect()
    await recovery.recover(broker)
    # Break PB-STALE (PaperBroker refuses the first order in a symbol that has
    # never been snapshotted): prime the quote-staleness cache exactly the way
    # the paper broker itself does, with the symbol's real price.  This is the
    # same test seam test_execution_integration.py uses.
    await broker._snapshot(SYMBOL, "NSE")
    return _Stack(
        broker=broker, kill_store=kill_store, order_manager=order_manager,
        order_store=order_store, audit=audit, sink=sink, service=service,
    )


def _push_bars(
    agg: TickBarAggregator,
    count: int,
    *,
    step: float = 0.002,
    volume: float = 50_000,
    span: float = 0.0,
    start_px: float = 100.0,
    at: Optional[datetime] = None,
) -> float:
    """Push `count` one-minute bars of a +`step` per-minute climb."""
    t = at or _at(TRADING_DAY, 9, 31)
    px = float(start_px)
    for _ in range(count):
        agg.on_tick(SYMBOL, t, px, volume)                       # open
        if span > 0:
            agg.on_tick(SYMBOL, t, px * (1 + span), volume)     # high
            agg.on_tick(SYMBOL, t, px * (1 - span), volume)     # low
        agg.on_tick(SYMBOL, t, px, volume)                       # close
        px *= (1 + step)
        t = t + timedelta(minutes=1)
    return px / (1 + step)                                       # last close


def _push_at(agg: TickBarAggregator, minute: int, close: float, span: float = 0.0) -> None:
    t = _at(TRADING_DAY, 10, minute)
    agg.on_tick(SYMBOL, t, close, 50_000)
    if span > 0:
        agg.on_tick(SYMBOL, t, close * (1 + span), 50_000)
        agg.on_tick(SYMBOL, t, close * (1 - span), 50_000)
    agg.on_tick(SYMBOL, t, close, 50_000)


def _signal(
    *,
    direction: SignalDirection = SignalDirection.LONG,
    strategy: str = "intraday",
    stop_loss_pct: float = 0.005,
    target_pct: float = 0.01,
    symbol: str = SYMBOL,
    now: Optional[datetime] = None,
) -> Signal:
    ts = now or _at(TRADING_DAY, 10, 31)
    return Signal(
        symbol=symbol,
        direction=direction,
        strategy_name=strategy,
        timestamp=ts,
        signal_date=ts.replace(hour=0, minute=0, second=0, microsecond=0),
        edge_score=0.01,
        expected_return=0.01,
        expected_return_std=0.005,
        stop_loss_pct=stop_loss_pct,
        target_pct=target_pct,
        holding_period_days=0,
        metadata={"paper": True},
    )


async def _entered_stack(
    clock: _FakeClock,
) -> tuple[_Stack, TickBarAggregator, IntradayAutoTrader, object]:
    """A stack with one protected position open: one entry + its stop."""
    agg = TickBarAggregator()
    _push_bars(agg, 60)                                    # 09:31..10:30 climb
    src = TickQuoteSource(agg)
    stack = await _build_stack(data_broker=src, clock=clock)
    clock.now = _at(TRADING_DAY, 10, 31)
    trader = IntradayAutoTrader(
        execution_service=stack.service,
        strategy=IntradayStrategy(paper_mode=True),
        bar_source=agg,
        broker=stack.broker,
        universe=[SYMBOL],
        clock=clock,
    )
    report = await trader.run_once()
    return stack, agg, trader, report


# =========================================================================== #
#  Ledger                                                                      #
# =========================================================================== #

class TestLedger:
    def test_reconcile_adopts_unknown_mis_positions(self) -> None:
        ledger = IntradayPositionLedger()
        broker_positions = [
            {"symbol": "TCS", "product": "MIS", "quantity": 50, "average_price": 200.0},
        ]
        adopted = ledger.reconcile(broker_positions, _at(TRADING_DAY, 10, 31))
        assert adopted == ["TCS"]
        row = ledger.get("TCS")
        assert row is not None
        assert row.quantity == 50
        assert row.stop_price == 0.0          # cannot know the stop
        assert row.target_price == 1.0e18     # unreachable target
        assert row.strategy == "adopted"

    def test_reconcile_ignores_non_intraday_products(self) -> None:
        ledger = IntradayPositionLedger()
        broker_positions = [
            {"symbol": "TCS", "product": "CNC", "quantity": 50, "average_price": 200.0},
        ]
        assert ledger.reconcile(broker_positions, _at(TRADING_DAY, 10, 31)) == []
        assert ledger.open() == []

    def test_reconcile_drops_rows_closed_at_the_broker(self) -> None:
        ledger = IntradayPositionLedger()
        ledger.record_entry(_signal(), 100, 100.0)
        assert ledger.reconcile([], _at(TRADING_DAY, 10, 31)) == []
        assert ledger.open() == []

    def test_reconcile_syncs_quantity_for_known_rows(self) -> None:
        ledger = IntradayPositionLedger()
        ledger.record_entry(_signal(), 100, 100.0)
        ledger.reconcile(
            [{"symbol": SYMBOL, "product": "MIS", "quantity": 35, "average_price": 100.0}],
            _at(TRADING_DAY, 10, 31),
        )
        assert ledger.get(SYMBOL).quantity == 35


# =========================================================================== #
#  Exit monitor (pure decisions — no broker)                                   #
# =========================================================================== #

class TestExitMonitor:
    def _monitor(self) -> PositionExitMonitor:
        return PositionExitMonitor(IntradayStrategy(paper_mode=True))

    def _pos(self, **kw) -> dict:
        base = {
            "symbol": SYMBOL, "direction": "LONG", "entry_price": 100.0,
            "stop_loss": 99.0, "target": 110.0, "quantity": 10, "last_price": 100.0,
        }
        base.update(kw)
        return base

    def _bars(self, last_close: float) -> dict[str, pd.DataFrame]:
        df = pd.DataFrame({
            "time": [_at(TRADING_DAY, 10, 0), _at(TRADING_DAY, 10, 1)],
            "open": [99.0, last_close], "high": [101.0, last_close],
            "low": [98.0, last_close], "close": [100.5, last_close],
            "volume": [1000, 1000],
        })
        return {SYMBOL: df}

    def test_stop_hit(self) -> None:
        mon = self._monitor()
        decisions = mon.evaluate([self._pos()], self._bars(98.5), _at(TRADING_DAY, 10, 31))
        assert [d.reason for d in decisions] == [ExitReason.STOP]

    def test_target_hit(self) -> None:
        mon = self._monitor()
        decisions = mon.evaluate([self._pos()], self._bars(111.0), _at(TRADING_DAY, 10, 31))
        assert [d.reason for d in decisions] == [ExitReason.TARGET]

    def test_square_off_wins_over_everything(self) -> None:
        mon = self._monitor()
        decisions = mon.evaluate([self._pos()], self._bars(111.0), _at(TRADING_DAY, 15, 16))
        assert [d.reason for d in decisions] == [ExitReason.SQUARE_OFF]

    def test_vwap_reversal_before_the_stop_or_target(self) -> None:
        mon = self._monitor()
        # A long that crossed back under VWAP and under entry, but is still
        # above its stop.
        pos = self._pos(stop_loss=90.0, last_price=99.0)
        decisions = mon.evaluate([pos], self._bars(98.0), _at(TRADING_DAY, 10, 31))
        assert [d.reason for d in decisions] == [ExitReason.VWAP_REVERSAL]

    def test_unpushable_positions_are_left_to_the_broker_stop(self) -> None:
        mon = self._monitor()
        pos = self._pos(last_price=0.0)                    # never priced
        decisions = mon.evaluate([pos], {SYMBOL: pd.DataFrame()}, _at(TRADING_DAY, 10, 31))
        assert decisions == []

    def test_no_exit_when_price_is_between_vwap_and_entry(self) -> None:
        mon = self._monitor()
        decisions = mon.evaluate(
            [self._pos()], self._bars(103.0), _at(TRADING_DAY, 10, 31)
        )
        assert decisions == []


# =========================================================================== #
#  ExecutionService additions: submit_stop / cancel_order                      #
# =========================================================================== #

class TestProtectiveStops:
    async def _stack(self, clock: Optional[_FakeClock] = None) -> _Stack:
        agg = TickBarAggregator()
        _push_bars(agg, 30)
        stack = await _build_stack(
            data_broker=TickQuoteSource(agg), clock=clock or _FakeClock(OPEN_IST),
        )
        # A protective stop is a SELL: it needs the long on the book first
        # (production order of operations: enter MIS, then protect it).
        await stack.broker.place_order(
            symbol=SYMBOL, exchange="NSE", txn_type=TransactionType.BUY,
            qty=100, price=0.0, order_type=OrderType.MARKET, product=Product.MIS,
        )
        return stack

    async def test_submit_stop_places_an_sl_m_sell_below_the_fill(self) -> None:
        stack = await self._stack()
        sig = _signal(stop_loss_pct=0.005)
        res = await stack.service.submit_stop(
            sig, 100,
            product="MIS", fill_price=100.0, idempotency_key="STOP-1",
            available_cash=INITIAL_CASH, total_portfolio=INITIAL_CASH,
            current_positions=[], open_orders=[],
        )
        assert res.submitted
        assert res.outcome == ExecutionOutcome.SUBMITTED
        orders = await stack.broker.get_orders()
        stops = [o for o in orders if o["order_type"] == "SL-M"]
        assert len(stops) == 1
        assert stops[0]["txn_type"] == "SELL"
        assert stops[0]["product"] == "MIS"
        assert stops[0]["qty"] == 100
        # TTL the trigger: SELL protecting a LONG sits BELOW the fill, not
        # above it (a stop the market is already through is an instant fill).
        assert stops[0]["price"] == pytest.approx(99.5, abs=0.05)
        assert stops[0]["status"] == OrderStatus.OPEN

    async def test_submit_stop_is_idempotent(self) -> None:
        stack = await self._stack()
        sig = _signal()
        kw = dict(
            product="MIS", fill_price=100.0, idempotency_key="STOP-1",
            available_cash=INITIAL_CASH, total_portfolio=INITIAL_CASH,
            current_positions=[], open_orders=[],
        )
        first = await stack.service.submit_stop(sig, 100, **kw)
        second = await stack.service.submit_stop(sig, 100, **kw)
        assert first.submitted
        # A replay of the SAME intent is suppressed at the OrderManager: the
        # service reports merge-success (SUBMITTED) rather than a fresh
        # submission.  What matters is that no second order ever reaches the
        # broker.
        assert second.outcome in (ExecutionOutcome.SUBMITTED, ExecutionOutcome.BLOCKED_DUPLICATE)
        orders = await stack.broker.get_orders()
        assert len([o for o in orders if o["order_type"] == "SL-M"]) == 1

    async def test_cancel_order_surfaces_a_cancelled_audit_record(self) -> None:
        stack = await self._stack()
        sig = _signal()
        placed = await stack.service.submit_stop(
            sig, 100, product="MIS", fill_price=100.0,
            available_cash=INITIAL_CASH, total_portfolio=INITIAL_CASH,
            current_positions=[], open_orders=[],
        )
        cancelled = await stack.service.cancel_order(
            placed.broker_order_id, symbol=sig.symbol, strategy=sig.strategy_name,
        )
        assert cancelled.outcome == ExecutionOutcome.CANCELLED
        status = await stack.broker.get_order_status(placed.broker_order_id)
        assert status["status"] == OrderStatus.CANCELLED
        assert any(
            r.outcome == ExecutionOutcome.CANCELLED for r in stack.sink.records
        )


# =========================================================================== #
#  The trader loop                                                             #
# =========================================================================== #

class TestTrader:
    async def test_idles_outside_the_signal_window(self) -> None:
        agg = TickBarAggregator()
        _push_bars(agg, 60)
        clock = _FakeClock(_at(TRADING_DAY, 9, 0))         # before 09:30
        stack = await _build_stack(data_broker=TickQuoteSource(agg), clock=clock)
        trader = IntradayAutoTrader(
            execution_service=stack.service,
            strategy=IntradayStrategy(paper_mode=True),
            bar_source=agg, broker=stack.broker, universe=[SYMBOL], clock=clock,
        )
        report = await trader.run_once()
        assert report.session_active is False
        assert report.entries == []
        assert report.exits == []
        assert await stack.broker.get_orders() == []

        clock.now = _at(TRADING_DAY, 15, 45)               # after market close
        report = await trader.run_once()
        assert report.session_active is False

    async def test_goes_from_ticks_to_a_protected_position_in_one_cycle(self) -> None:
        clock = _FakeClock(_at(TRADING_DAY, 10, 31))
        stack, agg, trader, report = await _entered_stack(clock)
        assert report.session_active is True
        assert len(report.entries) == 1
        assert len(report.stops_placed) == 1
        assert report.rejected == []
        assert report.exits == []

        orders = await stack.broker.get_orders()
        trades = [o for o in orders if o["order_type"] == "MARKET"]
        stops = [o for o in orders if o["order_type"] == "SL-M"]
        assert len(trades) == 1 and trades[0]["txn_type"] == "BUY"
        assert len(stops) == 1 and stops[0]["txn_type"] == "SELL"

        entry = report.entries[0]
        stop = stops[0]
        assert stop["price"] < entry["price"]               # trigger BELOW fill
        assert stop["qty"] == entry["qty"]

        positions = await stack.broker.get_positions()
        assert len(positions) == 1
        assert [p for p in positions if p["product"] == "MIS"] != []
        assert positions[0]["symbol"] == SYMBOL

        # The stop order id is remembered so a later manual exit can remove it.
        row = trader.ledger.get(SYMBOL)
        assert row is not None
        assert row.stop_order_id == stop["order_id"]

        # The audit journal proves the sequence: entry SUBMITTED then stop
        # SUBMITTED; nothing was ever blocked or duplicated.
        reached = stack.sink.records
        assert len(reached) == 2
        assert [r.side for r in reached] == ["BUY", "SELL"]

    async def test_stop_hit_cancels_the_protective_stop_then_exits(self) -> None:
        clock = _FakeClock(_at(TRADING_DAY, 10, 31))
        stack, agg, trader, report = await _entered_stack(clock)
        entry = report.entries[0]
        stop_id = trader.ledger.get(SYMBOL).stop_order_id
        assert stop_id is not None

        # Crash the price well through the stop trigger.
        _push_at(agg, 32, entry["price"] * 0.80)
        _push_at(agg, 33, entry["price"] * 0.78)
        clock.now = _at(TRADING_DAY, 10, 34)

        report = await trader.run_once()
        assert [d.reason for d in report.exits] == [ExitReason.STOP]
        assert report.stopped_cancelled == [stop_id]

        orders = await stack.broker.get_orders()
        statuses = {o["order_id"]: o["status"] for o in orders}
        assert statuses[stop_id] == OrderStatus.CANCELLED
        sells = [o for o in orders if o["order_type"] == "MARKET" and o["txn_type"] == "SELL"]
        assert len(sells) == 1
        assert sells[0]["qty"] == entry["qty"]

        assert await stack.broker.get_positions() == []
        assert trader.ledger.open() == []

        # Ordering proven from the audit journal: the CANCELLED stop precedes
        # the exiting SELL, so no window ever existed where the stop could have
        # filled and opened a spurious short.
        recs = stack.sink.records
        idx_cancel = next(i for i, r in enumerate(recs) if r.outcome == ExecutionOutcome.CANCELLED.value)
        idx_sell = next(i for i, r in enumerate(recs) if r.outcome == ExecutionOutcome.SUBMITTED.value and r.side == "SELL" and r.order_type == "MARKET")
        assert idx_cancel < idx_sell

    async def test_target_hit_exits_at_market(self) -> None:
        clock = _FakeClock(_at(TRADING_DAY, 10, 31))
        stack, agg, trader, report = await _entered_stack(clock)
        entry = report.entries[0]
        row = trader.ledger.get(SYMBOL)

        _push_at(agg, 32, entry["price"] * 1.02)       # clear the target
        clock.now = _at(TRADING_DAY, 10, 33)
        report = await trader.run_once()
        assert [d.reason for d in report.exits] == [ExitReason.TARGET]
        assert report.stopped_cancelled == [row.stop_order_id]
        assert await stack.broker.get_positions() == []

    async def test_vwap_reversal_exits_before_the_stop_or_target(self) -> None:
        clock = _FakeClock(_at(TRADING_DAY, 10, 31))
        # Wide intra-bar ranges push the ATR-based stop OUT far enough that the
        # the same climb with MACD/volatility noise (span) so the strategy
        # still fires a signal but carries a realistic wide stop: with
        # span=0.02 the stop trigger sits ~5.9% below the fill, and VWAP is
        # ~5.3% below it — enough room for price to fall under VWAP without
        # ever hitting the stop.
        agg = TickBarAggregator()
        _push_bars(agg, 60, span=0.02)
        stack = await _build_stack(data_broker=TickQuoteSource(agg), clock=clock)
        trader = IntradayAutoTrader(
            execution_service=stack.service,
            strategy=IntradayStrategy(paper_mode=True),
            bar_source=agg, broker=stack.broker, universe=[SYMBOL], clock=clock,
        )
        report = await trader.run_once()
        assert len(report.entries) == 1
        row = trader.ledger.get(SYMBOL)
        stop_price = row.stop_price
        vwap = agg.vwap(SYMBOL)
        entry_price = row.entry_price
        # The reversal bar must sit strictly between VWAP and the stop
        # trigger (and therefore below the entry) — guaranteed by
        # construction, independent of the exact strategy numbers.
        assert vwap < entry_price
        reversal = (vwap + stop_price) / 2.0
        assert stop_price < reversal < min(vwap, entry_price)

        _push_at(agg, 32, reversal, span=0.02)
        _push_at(agg, 33, reversal, span=0.02)
        clock.now = _at(TRADING_DAY, 10, 34)
        report = await trader.run_once()
        assert [d.reason for d in report.exits] == [ExitReason.VWAP_REVERSAL]
        assert stop_price < reversal               # sanity: not a STOP exit

    async def test_squares_off_after_1515(self) -> None:
        clock = _FakeClock(_at(TRADING_DAY, 10, 31))
        stack, agg, trader, report = await _entered_stack(clock)
        stop_id = trader.ledger.get(SYMBOL).stop_order_id
        clock.now = _at(TRADING_DAY, 15, 20)
        report = await trader.run_once()
        assert [d.reason for d in report.exits] == [ExitReason.SQUARE_OFF]
        assert report.stopped_cancelled == [stop_id]
        assert await stack.broker.get_positions() == []
        # No NEW entries after the signal cutoff, and no errors.
        assert report.entries == []
        assert report.errors == []

    async def test_position_already_closed_by_its_broker_stop_is_left_alone(self) -> None:
        """The dead-process backstop: SL-M fills autonomously; the trader must
        not try to exit a position that is already gone."""
        clock = _FakeClock(_at(TRADING_DAY, 10, 31))
        stack, agg, trader, report = await _entered_stack(clock)

        _push_at(agg, 32, 90.0)                       # crash through the trigger
        filled = await stack.broker.poll_open_orders()   # broker closes it alone
        assert filled == 1
        assert await stack.broker.get_positions() == []

        report = await trader.run_once()
        # The ledger row is gone and nothing was submitted — no stale exit,
        # no new entry, nothing errored.
        assert trader.ledger.open() == []
        assert report.exits == []
        assert report.entries == []
        assert report.errors == []
        sells = [o for o in await stack.broker.get_orders()
                 if o["order_type"] == "MARKET" and o["txn_type"] == "SELL"]
        assert sells == []

    async def test_an_active_kill_switch_keeps_the_protective_stop_in_place(self) -> None:
        """A kill switch engaged between cycles must NOT remove the protective
        stop: leaving an exposed position is worse than leaving a stale one."""
        clock = _FakeClock(_at(TRADING_DAY, 10, 31))
        stack, agg, trader, report = await _entered_stack(clock)
        stop_id = trader.ledger.get(SYMBOL).stop_order_id

        _push_at(agg, 32, 90.0)
        stack.kill_store.engage("halt everything")
        clock.now = _at(TRADING_DAY, 10, 33)
        report = await trader.run_once()

        # The stop and the position both SURVIVE; the exit was blocked by the
        # same kill switch that refused to remove the stop.
        status = await stack.broker.get_order_status(stop_id)
        assert status["status"] == OrderStatus.OPEN
        positions = await stack.broker.get_positions()
        assert len(positions) == 1
        assert any("kill switch" in e.lower() or "kill_switch" in e.lower() for e in report.errors)
        assert not report.exits
        assert trader.ledger.open() != []

    async def test_flat_market_produces_no_orders(self) -> None:
        agg = TickBarAggregator()
        _push_bars(agg, 60, step=0.0)                     # a completely flat session
        clock = _FakeClock(_at(TRADING_DAY, 10, 31))
        stack = await _build_stack(data_broker=TickQuoteSource(agg), clock=clock)
        trader = IntradayAutoTrader(
            execution_service=stack.service,
            strategy=IntradayStrategy(paper_mode=True),
            bar_source=agg, broker=stack.broker, universe=[SYMBOL], clock=clock,
        )
        report = await trader.run_once()
        assert report.entries == []
        assert report.exits == []
        assert await stack.broker.get_orders() == []
