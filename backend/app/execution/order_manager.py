"""
Order lifecycle management: reserve, submit, monitor, fill, emergency flatten.

The invariants this module enforces:

*   Nothing is sent to a broker before a durable record of the intent exists
    (``OrderStore.reserve`` — an atomic set-if-not-exists on a caller-generated
    client order id).  A crash between reserve and submit is recoverable.
*   ``place_order`` is called **at most once per intent**.  An ambiguous outcome
    is never retried blind; it is reconciled by querying the broker's order book
    for the client tag.
*   A MARKET order's size is measured against the *last traded price*, never
    against a placeholder ``Signal.price`` of 0.0.
*   Risk is ``qty * |entry - stop|`` (or the full notional when no stop exists),
    never the commission.
*   Every order state is explicit; see :mod:`app.execution.lifecycle`.

FOLLOW-UP REQUIRED IN ``app/broker/zerodha.py`` (not owned by this change):

1.  ``ZerodhaBroker.place_order`` must accept ``trigger_price: float = 0.0`` and
    forward it to Kite (``kwargs["trigger_price"]``) for ``OrderType.SL`` **and**
    ``OrderType.SL_M`` — Kite rejects SL-M without one.  Until it does,
    :meth:`OrderManager.submit_order` REFUSES to place SL / SL-M orders rather
    than sending an order with no stop attached (see ``_broker_supports_trigger``).
2.  ``ZerodhaBroker._call_kite`` must NOT be used with retries for
    ``place_order``: retrying a timed-out order submission places duplicate
    live orders.  Order submission must be ``retries=1``; ambiguity is resolved
    by ``get_orders()`` + tag matching, which this module already drives.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from ..broker.base import (
    BrokerInterface,
    OrderType,
    Product,
    TransactionType,
)
from ..core.exceptions import AmbiguousOrderStateError
from .lifecycle import (
    LEGAL_TRANSITIONS,
    OrderBlockedError,
    OrderRecord,
    OrderState,
    OrderStore,
    RedisOrderStore,
    deterministic_client_order_id,
    make_client_order_id,
    map_broker_status,
)
from .safety import (
    ExecutionSafety,
    KillSwitchActiveError,
    OrderValidationResult,
    SafetyCheckError,
    StaleDataError,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Transaction cost constants (Zerodha 2024)                                   #
# --------------------------------------------------------------------------- #

BROKERAGE_INTRADAY_RATE = 0.0003
BROKERAGE_INTRADAY_MAX = 20.0
BROKERAGE_DELIVERY = 0.0
STT_INTRADAY_SELL = 0.00025
STT_DELIVERY = 0.001
NSE_EXCHANGE_CHARGE = 0.0000322
BSE_EXCHANGE_CHARGE = 0.0000375
GST_RATE = 0.18
STAMP_DUTY_BUY = 0.00003
SEBI_CHARGE = 0.000001

#: Order types whose execution price is not known in advance and must therefore
#: be sized against the last traded price.
_MARKET_PRICED = (OrderType.MARKET, OrderType.SL_M)
#: Order types that require a broker-side trigger price.
_TRIGGERED = (OrderType.SL, OrderType.SL_M)


def calculate_costs(
    symbol: str,
    qty: int,
    price: float,
    txn_type: TransactionType,
    product: Product,
    exchange: str = "NSE",
) -> dict[str, float]:
    """Compute full Zerodha transaction-cost breakdown."""
    turnover = qty * price
    is_intraday = product == Product.MIS

    if is_intraday:
        brokerage = min(BROKERAGE_INTRADAY_RATE * turnover, BROKERAGE_INTRADAY_MAX)
    else:
        brokerage = BROKERAGE_DELIVERY

    if is_intraday:
        stt = STT_INTRADAY_SELL * turnover if txn_type == TransactionType.SELL else 0.0
    else:
        stt = STT_DELIVERY * turnover

    exc_rate = NSE_EXCHANGE_CHARGE if exchange.upper() == "NSE" else BSE_EXCHANGE_CHARGE
    exchange_charges = exc_rate * turnover
    sebi = SEBI_CHARGE * turnover
    gst = GST_RATE * (brokerage + exchange_charges + sebi)
    stamp = STAMP_DUTY_BUY * turnover if txn_type == TransactionType.BUY else 0.0
    total = brokerage + stt + exchange_charges + sebi + gst + stamp

    return {
        "brokerage": round(brokerage, 4),
        "stt": round(stt, 4),
        "exchange_charges": round(exchange_charges, 4),
        "sebi": round(sebi, 4),
        "gst": round(gst, 4),
        "stamp_duty": round(stamp, 4),
        "total": round(total, 4),
    }


# --------------------------------------------------------------------------- #
#  Signal                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class Signal:
    """
    One trading intent.

    ``client_order_id`` is generated once, at intent creation, and is the
    idempotency key AND the broker tag for the whole life of the order.
    Re-submitting the *same* Signal object (a retry) can therefore never place a
    second live order; a genuinely new intent must be a new Signal.
    """

    symbol: str
    exchange: str
    txn_type: TransactionType
    order_type: OrderType
    product: Product
    price: float = 0.0          # limit price; ignored (and unused) for MARKET
    stop_price: float = 0.0     # broker-side trigger for SL / SL-M; risk anchor
    strategy: str = ""
    tag: str = ""
    client_order_id: str = ""
    intent_kind: str = "entry"  # entry | exit | flatten

    def __post_init__(self) -> None:
        if not self.client_order_id:
            self.client_order_id = make_client_order_id(self.strategy or self.tag)


class OrderSubmissionError(RuntimeError):
    """Raised when an order cannot be submitted safely."""


class AmbiguousOrderError(OrderSubmissionError, AmbiguousOrderStateError):
    """
    Raised when a submission's outcome could not be established.

    The order is left in ``UNKNOWN``; nothing further may happen to it until it
    has been reconciled against the broker.

    It inherits from BOTH `OrderSubmissionError` (this module's hierarchy) and
    `core.exceptions.AmbiguousOrderStateError` (the shared one) deliberately.
    Two parallel hierarchies existed for the same concept, and
    `ExecutionService` caught only the shared one — so its ambiguous-order
    handler was dead code, and a lost broker response would have fallen
    through to the generic handler and been misclassified as a risk rejection.

    That is precisely the defect class that produced the original fail-open
    gates: an exception that does not subclass what the handler catches. One
    concept gets one catchable type.
    """


#: Exceptions that obviously mean "we do not know whether the exchange got the
#: order".  Used only for log wording — see the handler in ``submit_order``,
#: which treats EVERY exception as ambiguous.  kiteconnect's NetworkException
#: derives from plain ``Exception``, not from OSError, so any allowlist of
#: "definitely not sent" errors would misclassify a real Kite timeout as a
#: terminal rejection.  The only safe default is: we do not know, so ask.
_AMBIGUOUS_EXC = (
    TimeoutError,
    asyncio.TimeoutError,
    ConnectionError,
    OSError,
)


# --------------------------------------------------------------------------- #
#  Live-eligibility enforcement                                                 #
# --------------------------------------------------------------------------- #

def _is_paper_broker(broker) -> bool:
    """
    True only for a broker that cannot move real money.

    Two independent ways to qualify, because relying on either alone is worse:

    1. `isinstance(broker, PaperBroker)` — definitive for the real simulator.
    2. `broker.trading_mode == "paper"` — the declaration every
       `BrokerInterface` implementation is required to make. This admits
       legitimate test doubles and any future simulator without weakening the
       rule for real adapters: `ZerodhaBroker.trading_mode` returns "live".

    Anything else — an unrecognised class, a missing property, a property that
    raises — is treated as LIVE. An object we cannot identify is exactly the
    case where guessing wrong is expensive, so the unknown case is the
    expensive-to-be-wrong one.
    """
    try:
        from app.broker.paper import PaperBroker
        if isinstance(broker, PaperBroker):
            return True
    except Exception:  # pragma: no cover - import failure
        pass

    try:
        return str(getattr(broker, "trading_mode", "")).lower() == "paper"
    except Exception:
        # A broker whose own mode cannot be read is not a broker we will
        # assume is safe.
        return False


def _require_eligible_for_live(broker) -> None:
    """
    Assert live-trading eligibility when the order is bound for a real broker.

    Paper brokers skip this. Requiring LIVE_ELIGIBLE for paper trading would
    block the very activity that produces the evidence the gates demand, and
    a paper order cannot lose money.

    Everything else must pass `require_live_eligible()`, which re-derives the
    verdict from individual gate results rather than trusting a report's own
    `state` field — an audit demonstrated that a hand-built or deserialized
    report could otherwise assert its own eligibility.

    Any failure to evaluate is a BLOCK. A safety assertion that cannot be
    completed has not passed.
    """
    if _is_paper_broker(broker):
        return

    from app.governance.eligibility import require_live_eligible

    try:
        require_live_eligible(action="place a live order")
    except Exception:
        # Includes LiveTradingBlocked and any evaluation failure. Both stop
        # the order; neither is downgraded to a warning.
        logger.error(
            "LIVE ORDER BLOCKED by eligibility gate (broker=%s)",
            type(broker).__name__,
        )
        raise


# --------------------------------------------------------------------------- #
#  OrderManager                                                                #
# --------------------------------------------------------------------------- #

class OrderManager:
    """
    Orchestrates the full order lifecycle against a durable store.

    Parameters
    ----------
    safety : ExecutionSafety
    store : OrderStore
        REQUIRED durable persistence.  Constructing an OrderManager without one
        is refused: idempotency and crash recovery are not optional.
        ``redis_client=`` is accepted as a convenience and wrapped in a
        :class:`~app.execution.lifecycle.RedisOrderStore`.
    """

    def __init__(
        self,
        safety: ExecutionSafety,
        store: Optional[OrderStore] = None,
        *,
        redis_client=None,
        cost_settings: Optional[dict] = None,
        max_qty: Optional[int] = None,
        max_price: Optional[float] = None,
        max_notional: Optional[float] = None,
        require_stop: bool = False,
        max_tick_age_sec: float = 30.0,
    ) -> None:
        if store is None and redis_client is not None:
            store = RedisOrderStore(redis_client)
        if store is None:
            raise ValueError(
                "OrderManager requires a durable OrderStore (or redis_client). "
                "Without persistence there is no idempotency and no crash "
                "recovery: a retry places a second live order and a restart "
                "loses every in-flight order. Refusing to construct."
            )
        for method in ("reserve", "get", "save", "list_open"):
            if not callable(getattr(store, method, None)):
                raise ValueError(
                    f"OrderStore implementation lacks .{method}(); got "
                    f"{type(store).__name__}"
                )
        self._safety = safety
        self._store = store
        self._cost_settings = cost_settings or {}
        self._max_qty = max_qty
        self._max_price = max_price
        self._max_notional = max_notional
        self._require_stop = require_stop
        self._max_tick_age_sec = max_tick_age_sec

    # ------------------------------------------------------------------ #
    #  Introspection helpers                                               #
    # ------------------------------------------------------------------ #

    @property
    def store(self) -> OrderStore:
        return self._store

    async def get_record(self, order_ref: str) -> Optional[OrderRecord]:
        """Look up a record by client order id, falling back to broker id."""
        rec = await self._store.get(order_ref)
        if rec is not None:
            return rec
        for candidate in await self._store.list_open():
            if candidate.broker_order_id == order_ref:
                return candidate
        return None

    async def blocked_orders(self) -> list[OrderRecord]:
        """Every order whose broker outcome is still ambiguous."""
        return [r for r in await self._store.list_open() if r.is_blocked]

    # ------------------------------------------------------------------ #
    #  Pricing / risk                                                      #
    # ------------------------------------------------------------------ #

    async def resolve_reference_price(
        self,
        signal: Signal,
        broker: BrokerInterface,
    ) -> float:
        """
        The price the capital / exposure / risk arithmetic is done against.

        For LIMIT and SL the limit price is the worst-case execution price.
        For MARKET and SL-M there is no such price, so the last traded price is
        used — and a stale or missing tick REJECTS the order instead of
        silently sizing it against 0.0.
        """
        if signal.order_type not in _MARKET_PRICED:
            price = float(signal.price)
            if not math.isfinite(price) or price <= 0:
                raise SafetyCheckError(
                    f"{signal.order_type.value} order needs a positive limit "
                    f"price, got {signal.price!r}."
                )
            return price

        # MARKET / SL-M: the tick must be fresh.
        #
        # ORDER MATTERS: fetch the quote FIRST, then judge its freshness.
        #
        # The reverse order deadlocked the whole system. `is_stale_tick` reads
        # a cache that is populated by fetching a quote, so asking "is your
        # data fresh?" before ever fetching any meant every symbol reported
        # stale, the order was refused, the fetch never happened, the cache
        # stayed empty, and the next order was refused identically. No symbol
        # was ever tradeable. It failed CLOSED, so nothing was ever at risk —
        # but nothing could trade either.
        #
        # Freshness is a property of data you have actually fetched. This is
        # not a relaxation: the staleness check below still runs, still uses
        # the configured max age, and a feed returning old or unstamped
        # timestamps is still refused.
        key = f"{signal.exchange}:{signal.symbol}"
        try:
            quotes = await broker.get_quote([key])
        except Exception as exc:
            raise StaleDataError(
                f"Refusing MARKET order in {signal.symbol}: quote lookup "
                f"failed ({type(exc).__name__}: {exc})."
            ) from exc

        # RESIDUAL RISK, DELIBERATELY LEFT TO THE BROKER'S OWN POLICY:
        #
        # A quote with no timestamp has unverifiable age. `PaperBroker` treats
        # that as fresh-but-unverified by default (`strict_quote_staleness=
        # False`) — a documented choice, made because the current data adapter
        # does not stamp its quotes.
        #
        # Enforcing a timestamp here instead would override that policy from
        # the order path. It is therefore configured, not hardcoded: any real
        # deployment must construct the broker with
        # `strict_quote_staleness=True`, after which an unstamped quote is
        # stale and the check below refuses the order.
        #
        # This is recorded as an open item rather than silently accepted: see
        # docs/EXECUTION_ARCHITECTURE.md, "What is still not wired".
        if hasattr(broker, "is_stale_tick"):
            if broker.is_stale_tick(signal.symbol, max_age_seconds=self._max_tick_age_sec):
                raise StaleDataError(
                    f"Refusing MARKET order in {signal.symbol}: the quote just "
                    f"fetched is stale (older than {self._max_tick_age_sec}s, "
                    f"or its age could not be verified)."
                )
        quote = (quotes or {}).get(key) or (quotes or {}).get(signal.symbol)
        if not quote:
            raise StaleDataError(
                f"Refusing MARKET order in {signal.symbol}: no quote returned."
            )
        last = quote.get("last_price")
        if last is None or isinstance(last, bool) or not isinstance(last, (int, float)):
            raise StaleDataError(
                f"Refusing MARKET order in {signal.symbol}: last_price={last!r}."
            )
        last = float(last)
        if not math.isfinite(last) or last <= 0:
            raise StaleDataError(
                f"Refusing MARKET order in {signal.symbol}: last_price={last!r} "
                f"is not a usable price."
            )
        return last

    def _compute_trade_risk(
        self,
        qty: int,
        entry_price: float,
        stop_price: float,
    ) -> float:
        """
        Capital at risk for this trade.

        ``qty * |entry - stop|`` when a stop exists; the **full notional**
        otherwise (an unstopped position can lose all of it).  It is never the
        commission — that is a cost, not a risk.
        """
        notional = qty * entry_price
        if stop_price and math.isfinite(stop_price) and stop_price > 0:
            risk = qty * abs(entry_price - stop_price)
            if risk <= 0:
                raise SafetyCheckError(
                    f"Stop price {stop_price} equals entry {entry_price}: "
                    f"degenerate risk."
                )
            return min(risk, notional)
        if self._require_stop:
            raise SafetyCheckError(
                "No stop price defined and require_stop=True; refusing to size "
                "an unstopped position."
            )
        logger.warning(
            "No stop defined for this order; risk budgeted at the full "
            "notional Rs %.0f.", notional,
        )
        return notional

    # ------------------------------------------------------------------ #
    #  Submit                                                             #
    # ------------------------------------------------------------------ #

    async def submit_order(
        self,
        signal: Signal,
        position_size: int,
        broker: BrokerInterface,
        *,
        daily_risk_used: float = 0.0,
        max_daily_risk: float = 50_000.0,
        available_cash: float = 0.0,
        total_portfolio: float = 0.0,
        current_positions: Optional[list[dict]] = None,
        open_orders: Optional[list[dict]] = None,
        max_positions: int = 20,
        sector: Optional[str] = None,
        sector_value: float = 0.0,
        realised_pnl_today: float = 0.0,
        max_daily_loss: float = 50_000.0,
    ) -> Optional[str]:
        """
        Reserve, validate, and submit an order.  Returns the broker order id.

        Order of operations matters:

        1. RESERVE the client order id durably (atomic set-if-not-exists).
        2. Resolve a reference price (rejects stale/missing ticks).
        3. Run the safety gates.
        4. Persist ``SUBMITTED`` **before** calling the broker.
        5. Call ``place_order`` exactly once.
        6. Ambiguous outcome -> ``UNKNOWN`` -> reconcile by querying the order
           book for the client tag.  Never a blind retry.
        """
        current_positions = current_positions or []
        open_orders = open_orders or []
        cid = signal.client_order_id

        # -- 0. live-eligibility enforcement ----------------------------- #
        #
        # ExecutionService enforces this at the application boundary too. It
        # is repeated here deliberately: this is the last function before
        # `place_order`, and therefore the last place a bypass can be caught.
        # An eligibility verdict that is merely *reported* stops nothing — it
        # has to be asserted on the path that actually reaches a broker.
        #
        # The trigger is the BROKER'S ACTUAL TYPE, not a configuration flag.
        # A mode flag can be wrong, unset, or forgotten; the object about to
        # receive the order cannot be. If it is not a paper broker, this is
        # real money and eligibility is required.
        _require_eligible_for_live(broker)

        # -- 1. reserve ------------------------------------------------- #
        record = OrderRecord(
            client_order_id=cid,
            symbol=signal.symbol,
            exchange=signal.exchange,
            side=signal.txn_type.value,
            qty=int(position_size),
            order_type=signal.order_type.value,
            product=signal.product.value,
            strategy=signal.strategy,
            limit_price=float(signal.price or 0.0),
            trigger_price=float(signal.stop_price or 0.0),
            intent_kind=signal.intent_kind,
        )
        reserved = await self._store.reserve(record)
        if not reserved:
            return await self._handle_existing_intent(cid, broker)

        # An UNKNOWN order in the same instrument/strategy blocks new work.
        blocked = await self._blocking_record_for(signal)
        if blocked is not None:
            record.transition(
                OrderState.RISK_REJECTED,
                reason=f"blocked by unresolved order {blocked.client_order_id}",
            )
            await self._log_rejected_order(record, [record.reason])
            raise OrderBlockedError(
                f"Order {blocked.client_order_id} in {blocked.symbol} is in "
                f"UNKNOWN; reconcile it before submitting new orders."
            )

        record.transition(OrderState.RISK_CHECK_PENDING, reason="running gates")
        await self._store.save(record)

        # -- 2. reference price ----------------------------------------- #
        try:
            ref_price = await self.resolve_reference_price(signal, broker)
        except SafetyCheckError as exc:
            record.transition(OrderState.RISK_REJECTED, reason=str(exc))
            await self._log_rejected_order(record, [f"reference_price: {exc}"])
            logger.error("Order rejected before validation: %s", exc)
            return None
        record.reference_price = ref_price

        trade_value = position_size * ref_price
        costs = calculate_costs(
            signal.symbol, position_size, ref_price,
            signal.txn_type, signal.product, signal.exchange,
        )
        try:
            trade_risk = self._compute_trade_risk(
                position_size, ref_price, signal.stop_price)
        except SafetyCheckError as exc:
            record.transition(OrderState.RISK_REJECTED, reason=str(exc))
            await self._log_rejected_order(record, [f"trade_risk: {exc}"])
            return None

        # -- 3. safety gates -------------------------------------------- #
        validation: OrderValidationResult = await self._safety.validate_order(
            broker=broker,
            symbol=signal.symbol,
            exchange=signal.exchange,
            qty=position_size,
            price=ref_price,
            trade_value=trade_value,
            trade_risk=trade_risk,
            strategy=signal.strategy,
            current_positions=current_positions,
            open_orders=open_orders,
            daily_risk_used=daily_risk_used,
            max_daily_risk=max_daily_risk,
            available_cash=available_cash,
            total_portfolio=total_portfolio,
            max_positions=max_positions,
            sector=sector,
            sector_value=sector_value,
            realised_pnl_today=realised_pnl_today,
            max_daily_loss=max_daily_loss,
            client_order_id=cid,
            max_qty=self._max_qty,
            max_price=self._max_price,
            max_notional=self._max_notional,
        )
        if not validation.passed:
            logger.error("Order rejected by safety checks: %s", validation.failed_checks)
            record.transition(
                OrderState.RISK_REJECTED, reason="; ".join(validation.failed_checks))
            await self._log_rejected_order(record, validation.failed_checks)
            return None

        record.transition(OrderState.RISK_APPROVED, reason="all gates passed")
        await self._store.save(record)

        # -- 4. trigger price plumbing ---------------------------------- #
        place_kwargs: dict[str, Any] = {}
        if signal.order_type in _TRIGGERED:
            trigger = float(signal.stop_price or 0.0)
            if not math.isfinite(trigger) or trigger <= 0:
                record.transition(
                    OrderState.RISK_REJECTED,
                    reason=f"{signal.order_type.value} needs a positive stop_price",
                )
                await self._log_rejected_order(
                    record, [f"trigger_price: missing for {signal.order_type.value}"])
                return None
            if not _broker_supports_trigger(broker):
                reason = (
                    f"{type(broker).__name__}.place_order does not accept "
                    f"trigger_price; refusing to place a "
                    f"{signal.order_type.value} order that would carry no stop."
                )
                record.transition(OrderState.RISK_REJECTED, reason=reason)
                await self._log_rejected_order(record, [f"trigger_price: {reason}"])
                logger.critical(reason)
                return None
            place_kwargs["trigger_price"] = trigger

        # -- 5. durable SUBMITTED, then exactly one broker call ---------- #
        record.transition(OrderState.SUBMITTED, reason="sending to broker")
        await self._persist_order(record, costs)

        # MARKET / SL-M carry no price; LIMIT / SL carry the caller's limit.
        submit_price = (
            0.0 if signal.order_type in _MARKET_PRICED else float(signal.price or 0.0)
        )
        try:
            order_id = await broker.place_order(
                symbol=signal.symbol,
                exchange=signal.exchange,
                txn_type=signal.txn_type,
                qty=position_size,
                price=submit_price,
                order_type=signal.order_type,
                product=signal.product,
                tag=cid,
                **place_kwargs,
            )
        except Exception as exc:
            # We do NOT know whether the exchange received this, and we must not
            # guess: a Kite NetworkException is a plain Exception, so "looks
            # like a client error" is not evidence that nothing was sent.
            # Enter UNKNOWN and ASK the broker. Never retry.
            obviously_ambiguous = isinstance(exc, _AMBIGUOUS_EXC)
            logger.critical(
                "%s submission outcome for %s (%s: %s) — entering UNKNOWN and "
                "reconciling against the broker order book.",
                "AMBIGUOUS" if obviously_ambiguous else "UNVERIFIED",
                cid, type(exc).__name__, exc,
            )
            record.transition(OrderState.UNKNOWN, reason=f"{type(exc).__name__}: {exc}")
            await self._store.save(record)
            return await self._resolve_unknown(record, broker)

        if not order_id:
            logger.critical(
                "Broker returned an empty order id for %s — treating as "
                "AMBIGUOUS, not as success.", cid,
            )
            record.transition(OrderState.UNKNOWN, reason="empty broker order id")
            await self._store.save(record)
            return await self._resolve_unknown(record, broker)

        record.broker_order_id = str(order_id)
        record.transition(OrderState.ACKNOWLEDGED, reason="broker accepted")
        await self._update_order_status(record)

        # Some venues fill synchronously: `place_order` returns and the order
        # is already COMPLETE or PARTIAL. The paper broker does exactly this.
        #
        # Without this step the durable record stayed at filled_qty=0 while the
        # broker reported a completed fill, so local state diverged from broker
        # state the instant an order filled — and the very next reconciliation
        # would raise a quantity mismatch for an order that behaved perfectly.
        # The fill has to be pulled in before returning.
        await self._apply_any_immediate_fill(record, broker)

        logger.info(
            "Order submitted: cid=%s broker_id=%s %s %s/%s qty=%d ref_price=%.2f",
            cid, order_id, signal.txn_type.value, signal.symbol, signal.exchange,
            position_size, ref_price,
        )
        return record.broker_order_id

    # -- submit helpers ------------------------------------------------- #

    async def _handle_existing_intent(
        self, cid: str, broker: BrokerInterface
    ) -> Optional[str]:
        """
        The client order id was already reserved: this is a retry / duplicate.

        Never place another order.  If the earlier attempt is ambiguous, resolve
        it by querying the broker.
        """
        existing = await self._store.get(cid)
        if existing is None:                       # reservation without a record
            raise AmbiguousOrderError(
                f"Client order id {cid} is reserved but unreadable; refusing to "
                f"submit."
            )
        if existing.is_blocked:
            logger.critical(
                "Retry of %s while its outcome is UNKNOWN — reconciling instead "
                "of resubmitting.", cid,
            )
            return await self._resolve_unknown(existing, broker)
        if existing.broker_order_id:
            logger.warning(
                "Duplicate submission of %s suppressed; order %s is already at "
                "the broker.", cid, existing.broker_order_id,
            )
            return existing.broker_order_id
        logger.warning(
            "Duplicate submission of %s suppressed (state=%s).",
            cid, existing.state.value,
        )
        return None

    async def _blocking_record_for(self, signal: Signal) -> Optional[OrderRecord]:
        """An UNKNOWN order in the same symbol+strategy blocks new submissions."""
        for rec in await self._store.list_open():
            if not rec.is_blocked:
                continue
            if rec.symbol == signal.symbol and rec.strategy == signal.strategy:
                return rec
        return None

    async def _resolve_unknown(
        self, record: OrderRecord, broker: BrokerInterface
    ) -> Optional[str]:
        """
        Resolve an ambiguous submission by QUERYING the broker's order book for
        the client tag.  This is the only legal way out of ``UNKNOWN``.
        """
        found = await self._query_broker_for_tag(record, broker)
        if found is None:
            logger.critical(
                "Order %s remains UNKNOWN: the broker order book could not be "
                "read. NO retry will be attempted; this order blocks further "
                "action until reconciled.", record.client_order_id,
            )
            await self._store.save(record)
            raise AmbiguousOrderError(
                f"Order {record.client_order_id} is in UNKNOWN and could not be "
                f"reconciled. Do not retry; reconcile first."
            )
        if found == "absent":
            logger.warning(
                "Order %s was NOT found in the broker order book: it never "
                "reached the exchange.", record.client_order_id,
            )
            record.transition(
                OrderState.REJECTED,
                reason="not present in broker order book after ambiguous submit",
                reconciled=True,
            )
            await self._update_order_status(record)
            return None

        broker_order = found
        record.broker_order_id = str(
            broker_order.get("order_id") or record.broker_order_id or "")
        filled = int(broker_order.get("filled_quantity") or 0)
        new_state = map_broker_status(broker_order.get("status"), filled, record.qty)
        if new_state is OrderState.UNKNOWN:
            await self._store.save(record)
            raise AmbiguousOrderError(
                f"Order {record.client_order_id} found at the broker with an "
                f"uninterpretable status {broker_order.get('status')!r}."
            )
        record.filled_qty = min(filled, record.qty)
        record.transition(
            new_state, reason="resolved from broker order book", reconciled=True)
        await self._update_order_status(record)
        logger.critical(
            "Order %s reconciled from UNKNOWN -> %s (broker_id=%s). The order "
            "WAS live; no second order was placed.",
            record.client_order_id, new_state.value, record.broker_order_id,
        )
        return record.broker_order_id or None

    async def _query_broker_for_tag(
        self, record: OrderRecord, broker: BrokerInterface
    ):
        """
        Return the broker order dict for this client tag, the string
        ``"absent"`` when the book was read and does not contain it, or ``None``
        when the book could not be read at all.
        """
        try:
            orders = await broker.get_orders()
        except Exception as exc:
            logger.error(
                "Reconciliation query failed for %s: %s", record.client_order_id, exc)
            return None
        if orders is None:
            return None
        for order in orders:
            if not isinstance(order, dict):
                continue
            tag = order.get("tag") or ""
            if tag == record.client_order_id:
                return order
            if record.broker_order_id and str(
                    order.get("order_id") or "") == record.broker_order_id:
                return order
        return "absent"

    # ------------------------------------------------------------------ #
    #  Crash recovery                                                      #
    # ------------------------------------------------------------------ #

    async def recover(self, broker: BrokerInterface) -> list[OrderRecord]:
        """
        Startup recovery: reconcile every non-terminal order against the broker.

        This is what makes a crash between "reserve" and "submit" survivable —
        the durable record names a client tag, and the broker's order book is
        the authority on whether that tag exists.
        """
        recovered: list[OrderRecord] = []
        for record in await self._store.list_open():
            if record.state in (
                OrderState.INTENT_CREATED,
                OrderState.RISK_CHECK_PENDING,
                OrderState.RISK_APPROVED,
            ):
                # Never reached the broker call.  Confirm against the book
                # anyway, then close the intent out.
                found = await self._query_broker_for_tag(record, broker)
                if found in (None, "absent"):
                    record.transition(
                        OrderState.RISK_REJECTED,
                        reason="abandoned intent recovered at startup",
                    )
                    await self._update_order_status(record)
                    recovered.append(record)
                    continue
                # Present at the broker despite a pre-submit local state: walk
                # the legal path rather than jumping states.
                _walk_to_submitted(record)

            if record.state in (
                OrderState.SUBMITTED,
                OrderState.ACKNOWLEDGED,
                OrderState.PARTIALLY_FILLED,
                OrderState.CANCEL_PENDING,
                OrderState.UNKNOWN,
            ):
                found = await self._query_broker_for_tag(record, broker)
                if found is None:
                    logger.critical(
                        "Recovery could not read the broker order book for %s; "
                        "it stays blocked.", record.client_order_id,
                    )
                    if record.state is not OrderState.UNKNOWN:
                        record.transition(
                            OrderState.UNKNOWN, reason="broker unreachable at recovery")
                        await self._store.save(record)
                    recovered.append(record)
                    continue
                if found == "absent":
                    reason = "absent from broker order book at recovery"
                    if record.state is OrderState.UNKNOWN:
                        record.transition(
                            OrderState.REJECTED, reason=reason, reconciled=True)
                    else:
                        record.transition(OrderState.REJECTED, reason=reason)
                    await self._update_order_status(record)
                    recovered.append(record)
                    continue
                filled = int(found.get("filled_quantity") or 0)
                new_state = map_broker_status(
                    found.get("status"), filled, record.qty)
                record.broker_order_id = str(
                    found.get("order_id") or record.broker_order_id or "") or None
                record.filled_qty = min(filled, record.qty)
                if new_state is OrderState.UNKNOWN:
                    if record.state is not OrderState.UNKNOWN:
                        record.transition(
                            OrderState.UNKNOWN, reason="uninterpretable broker status")
                    await self._store.save(record)
                elif new_state is record.state:
                    await self._store.save(record)
                else:
                    record.transition(
                        new_state, reason="recovered from broker", reconciled=True)
                    await self._update_order_status(record)
                recovered.append(record)
        logger.info("Recovery reconciled %d open order(s).", len(recovered))
        return recovered

    # ------------------------------------------------------------------ #
    #  Cancel                                                              #
    # ------------------------------------------------------------------ #

    async def cancel_order(
        self,
        order_ref: str,
        broker: BrokerInterface,
    ) -> bool:
        """Cancel a live order.  Blocked while its state is ambiguous."""
        record = await self.get_record(order_ref)
        if record is not None:
            if record.is_blocked:
                raise OrderBlockedError(
                    f"Cannot cancel {record.client_order_id}: its state is "
                    f"UNKNOWN. Reconcile first."
                )
            if record.is_terminal:
                logger.info(
                    "Cancel ignored: %s is already %s.",
                    record.client_order_id, record.state.value,
                )
                return False
            if record.state is not OrderState.CANCEL_PENDING:
                record.transition(OrderState.CANCEL_PENDING, reason="cancel requested")
                await self._update_order_status(record)
        broker_id = (record.broker_order_id if record else None) or order_ref
        try:
            success = bool(await broker.cancel_order(broker_id))
        except Exception as exc:
            logger.error("Cancel failed for %s: %s", broker_id, exc)
            if record is not None:
                record.transition(OrderState.UNKNOWN, reason=f"cancel error: {exc}")
                await self._store.save(record)
            return False
        if success and record is not None:
            record.transition(OrderState.CANCELLED, reason="broker confirmed cancel")
            await self._update_order_status(record)
        return success

    # ------------------------------------------------------------------ #
    #  Monitor                                                             #
    # ------------------------------------------------------------------ #

    async def monitor_order(
        self,
        order_ref: str,
        broker: BrokerInterface,
        timeout: float = 30.0,
        poll_interval: float = 1.0,
    ) -> dict:
        """
        Poll until the order reaches a terminal state or the timeout expires.

        The returned dict always carries ``timed_out``, ``state`` and
        ``poll_errors`` so a timeout is never mistaken for a terminal state, and
        a stale poll result is never re-read as a fresh one.
        """
        record = await self.get_record(order_ref)
        broker_id = (record.broker_order_id if record else None) or order_ref
        deadline = time.monotonic() + timeout
        last_good: Optional[dict] = None
        poll_errors: list[str] = []
        state = record.state if record else OrderState.UNKNOWN

        while True:
            fresh: Optional[dict] = None
            try:
                fresh = await broker.get_order_status(broker_id)
            except Exception as exc:
                poll_errors.append(f"{type(exc).__name__}: {exc}")
                logger.warning("monitor_order poll error: %s", exc)

            if fresh is not None:
                last_good = fresh
                filled = int(fresh.get("filled_quantity")
                             or fresh.get("filled_qty") or 0)
                qty = int(fresh.get("quantity") or fresh.get("qty")
                          or (record.qty if record else 0))
                state = map_broker_status(fresh.get("status"), filled, qty)
                if record is not None:
                    record.filled_qty = min(filled, record.qty)
                    if state is not record.state and not record.is_terminal:
                        try:
                            record.transition(
                                state, reason="observed by monitor",
                                reconciled=record.is_blocked,
                            )
                            await self._update_order_status(record)
                        except Exception as exc:      # illegal transition
                            logger.error(
                                "monitor_order: refusing state change %s -> %s "
                                "for %s (%s)",
                                record.state.value, state.value,
                                record.client_order_id, exc,
                            )
                if state in (OrderState.FILLED, OrderState.CANCELLED,
                             OrderState.REJECTED, OrderState.EXPIRED):
                    logger.info(
                        "Order %s reached terminal state: %s", order_ref, state.value)
                    return _monitor_result(
                        last_good, state, timed_out=False, poll_errors=poll_errors)

            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(poll_interval)

        if last_good is None:
            # Never got a single successful poll: genuinely ambiguous.
            state = OrderState.UNKNOWN
            if record is not None and not record.is_terminal and not record.is_blocked:
                record.transition(
                    OrderState.UNKNOWN, reason="no successful status poll")
                await self._store.save(record)
        logger.warning(
            "Order %s monitor timed out after %.0fs in state %s",
            order_ref, timeout, state.value,
        )
        return _monitor_result(last_good, state, timed_out=True, poll_errors=poll_errors)

    # ------------------------------------------------------------------ #
    #  Fill handling                                                       #
    # ------------------------------------------------------------------ #

    async def handle_fill(self, order_ref: str, fill_data: dict) -> dict:
        """
        Record one fill exactly once.

        A partial fill is recorded as ``PARTIALLY_FILLED`` — never COMPLETE —
        so the residual quantity stays visible.  A duplicate fill message (same
        ``fill_id``) changes neither the order nor the position.
        """
        record = await self.get_record(order_ref)
        if record is None:
            raise OrderSubmissionError(
                f"Fill for unknown order {order_ref}: refusing to book a trade "
                f"with no local record. This is a reconciliation event."
            )
        if record.is_blocked:
            raise OrderBlockedError(
                f"Fill for {record.client_order_id} while its state is UNKNOWN; "
                f"reconcile before booking."
            )

        qty = int(fill_data.get("qty", 0))
        price = float(fill_data.get("price", 0.0))
        txn_type = TransactionType(fill_data.get("txn_type", record.side))
        product = Product(fill_data.get("product", record.product))
        exchange = fill_data.get("exchange", record.exchange)
        symbol = fill_data.get("symbol", record.symbol)
        fill_id = str(fill_data.get("fill_id") or _derive_fill_id(
            record.client_order_id, fill_data))
        fill_data["fill_id"] = fill_id

        costs = calculate_costs(symbol, qty, price, txn_type, product, exchange)
        fill_data["costs"] = costs

        # The durable record is the authority on whether this fill was already
        # applied. apply_fill() raises on an over-fill BEFORE anything is
        # written, so a bad message cannot corrupt the position.
        applied = record.apply_fill(qty, price, fill_id)
        if not applied:
            logger.warning(
                "Duplicate fill %s for order %s ignored.", fill_id,
                record.client_order_id,
            )
            fill_data.update({
                "duplicate": True,
                "state": record.state.value,
                "filled_qty": record.filled_qty,
                "remaining_qty": record.remaining_qty,
            })
            return fill_data

        if not await self._record_trade(record, fill_data):
            logger.error(
                "Trade log already had fill %s for order %s while the order "
                "record did not — investigate.", fill_id, record.client_order_id,
            )
        new_state = (
            OrderState.FILLED if record.remaining_qty == 0
            else OrderState.PARTIALLY_FILLED
        )
        record.transition(
            new_state,
            reason=f"fill {record.filled_qty}/{record.qty} @ {price}",
        )
        await self._update_order_status(record)
        await self._update_position(record, fill_data)

        logger.info(
            "Fill recorded: order=%s %s %s qty=%d price=%.2f costs=Rs %.2f "
            "state=%s (%d/%d)",
            record.client_order_id, txn_type.value, symbol, qty, price,
            costs["total"], new_state.value, record.filled_qty, record.qty,
        )
        fill_data.update({
            "duplicate": False,
            "state": new_state.value,
            "filled_qty": record.filled_qty,
            "remaining_qty": record.remaining_qty,
        })
        return fill_data

    # ------------------------------------------------------------------ #
    #  Emergency flatten                                                   #
    # ------------------------------------------------------------------ #

    async def emergency_flatten_all(
        self,
        broker: BrokerInterface,
        *,
        override_kill_switch: bool = False,
        flatten_session: Optional[str] = None,
    ) -> dict:
        """
        Engage the kill switch, cancel pending orders, flatten open positions.

        Kill-switch semantics (deliberate):

        * If the kill switch is ALREADY active, this refuses to trade — an
          engaged kill switch means trading is halted, and an automated loop
          must not fight it.  A human may pass ``override_kill_switch=True``.
        * Otherwise the kill switch is engaged **first** (and the write is
          verified), so the strategy loop cannot re-enter what we just exited.

        Idempotent: every flatten order carries a deterministic client order id
        derived from (session, position), reserved atomically, so calling this
        twice cannot double-sell.
        """
        logger.critical("EMERGENCY FLATTEN INITIATED.")
        session = flatten_session or datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        report: dict[str, Any] = {
            "cancelled": [], "flattened": [], "skipped": [], "errors": [],
        }

        if await self._safety.is_kill_switch_active():
            if not override_kill_switch:
                logger.critical(
                    "Kill switch already ACTIVE — emergency flatten refuses to "
                    "place orders. Pass override_kill_switch=True to force.")
                raise KillSwitchActiveError(
                    "Kill switch is active; emergency flatten blocked."
                )
            logger.critical("Kill switch active but override requested; proceeding.")
        else:
            await self._safety.engage_kill_switch(reason="emergency_flatten_all")

        # 1. Cancel pending orders.
        try:
            orders = await broker.get_orders()
        except Exception as exc:
            logger.error("Could not fetch orders for flatten: %s", exc)
            report["errors"].append(f"get_orders: {exc}")
            orders = []

        pending_statuses = {"OPEN", "TRIGGER PENDING", "AMO REQ RECEIVED"}
        pending = [o for o in orders
                   if (o.get("status") or "") in pending_statuses]
        results = await asyncio.gather(
            *[self.cancel_order(str(o.get("order_id")), broker) for o in pending],
            return_exceptions=True,
        )
        for order, result in zip(pending, results):   # aligned by construction
            oid = order.get("order_id")
            if isinstance(result, BaseException) or result is False:
                logger.error("Cancel failed for %s: %s", oid, result)
                report["errors"].append(f"cancel {oid}: {result}")
            else:
                report["cancelled"].append(oid)

        # 2. Flatten open positions.
        try:
            positions = await broker.get_positions()
        except Exception as exc:
            logger.error("Could not fetch positions for flatten: %s", exc)
            report["errors"].append(f"get_positions: {exc}")
            positions = []

        live = []
        for pos in positions:
            qty = int(pos.get("quantity", 0) or 0)
            if qty == 0:
                continue
            live.append((pos, qty))

        for pos, qty in live:
            symbol = pos.get("tradingsymbol") or pos.get("symbol", "")
            exchange = pos.get("exchange", "NSE")
            try:
                product = Product(pos.get("product", "MIS"))
            except ValueError:
                product = Product.MIS
            txn_type = TransactionType.SELL if qty > 0 else TransactionType.BUY
            cid = deterministic_client_order_id(
                session, exchange, symbol, product.value, qty, prefix="FLAT")
            record = OrderRecord(
                client_order_id=cid, symbol=symbol, exchange=exchange,
                side=txn_type.value, qty=abs(qty), order_type=OrderType.MARKET.value,
                product=product.value, strategy="EMERGENCY_FLATTEN",
                intent_kind="flatten",
            )
            if not await self._store.reserve(record):
                logger.warning(
                    "Flatten for %s already reserved this session (%s) — "
                    "skipping to avoid a double-sell.", symbol, cid,
                )
                report["skipped"].append(symbol)
                continue
            record.transition(OrderState.RISK_CHECK_PENDING, reason="flatten")
            record.transition(OrderState.RISK_APPROVED, reason="flatten override")
            record.transition(OrderState.SUBMITTED, reason="flatten")
            await self._store.save(record)
            try:
                order_id = await broker.place_order(
                    symbol=symbol, exchange=exchange, txn_type=txn_type,
                    qty=abs(qty), price=0.0, order_type=OrderType.MARKET,
                    product=product, tag=cid,
                )
            except Exception as exc:
                # Same rule as submit_order: an un-acknowledged submission is
                # UNKNOWN, never assumed not-placed.
                logger.critical(
                    "Flatten submission UNVERIFIED for %s (%s: %s)",
                    symbol, type(exc).__name__, exc,
                )
                record.transition(OrderState.UNKNOWN, reason=str(exc))
                await self._store.save(record)
                found = await self._query_broker_for_tag(record, broker)
                if found == "absent":
                    record.transition(
                        OrderState.REJECTED, reason=str(exc), reconciled=True)
                    await self._store.save(record)
                    report["errors"].append(f"flatten {symbol}: rejected ({exc})")
                else:
                    report["errors"].append(f"flatten {symbol}: UNKNOWN ({exc})")
                continue
            record.broker_order_id = str(order_id)
            record.transition(OrderState.ACKNOWLEDGED, reason="flatten accepted")
            await self._store.save(record)
            report["flattened"].append({"symbol": symbol, "order_id": order_id})
            logger.info("Flatten order placed: %s -> %s", symbol, order_id)

        logger.critical("EMERGENCY FLATTEN COMPLETE: %s", report)
        return report

    # ------------------------------------------------------------------ #
    #  Persistence helpers — every one performs real, awaited I/O          #
    # ------------------------------------------------------------------ #

    async def _persist_order(self, record: OrderRecord, costs: dict) -> None:
        """Durably write the order record before it is sent to the broker."""
        record.history.append({"costs": costs, "at": record.updated_at})
        await self._store.save(record)
        logger.debug("Order persisted: %s (%s)", record.client_order_id,
                     record.state.value)

    async def _log_rejected_order(
        self, record: OrderRecord, reasons: list[str]
    ) -> None:
        """Durably record why an order was refused."""
        record.history.append({"rejected": list(reasons), "at": record.updated_at})
        await self._store.save(record)
        logger.info(
            "Rejected order persisted for %s/%s qty=%d reasons=%s",
            record.symbol, record.exchange, record.qty, reasons,
        )

    async def _apply_any_immediate_fill(
        self, record: OrderRecord, broker: BrokerInterface
    ) -> None:
        """
        Pull in a fill that already happened by the time `place_order` returned.

        Synchronous venues (the paper broker, and some live order types) fill
        during submission. Nothing else in the flow polls at that moment, so
        without this the durable record reports filled_qty=0 for an order the
        broker has already completed — local and broker state diverge on every
        successful fill, and the next reconciliation flags a mismatch that is
        entirely our own bookkeeping.

        Deliberately best-effort: a failure here must not undo a submission
        that genuinely succeeded. It is logged and left for reconciliation,
        which is the component whose job this is. Not raising is safe because
        the fill is not invented — an unapplied fill shows up as a mismatch and
        blocks trading, which is the conservative direction.
        """
        if not record.broker_order_id:
            return
        try:
            status = await broker.get_order_status(record.broker_order_id)
        except Exception as exc:
            logger.warning(
                "Could not read post-submit status for %s: %r. Leaving the fill "
                "to reconciliation.", record.client_order_id, exc,
            )
            return

        if not isinstance(status, dict):
            return

        filled = int(status.get("filled_qty") or status.get("filled_quantity") or 0)
        if filled <= 0 or filled <= record.filled_qty:
            return

        price = float(
            status.get("average_price")
            or status.get("avg_fill_price")
            or record.reference_price
            or 0.0
        )
        if price <= 0:
            logger.warning(
                "Immediate fill for %s reported no usable price; leaving it to "
                "reconciliation.", record.client_order_id,
            )
            return

        try:
            await self.handle_fill(record.client_order_id, {
                "symbol": record.symbol,
                "exchange": record.exchange,
                "qty": filled - record.filled_qty,
                "price": price,
                "txn_type": record.side,
                "product": record.product,
                "fill_id": f"{record.broker_order_id}:immediate:{filled}",
            })
        except Exception as exc:
            logger.warning(
                "Could not apply immediate fill for %s: %r. Reconciliation will "
                "surface the difference.", record.client_order_id, exc,
            )

    async def _update_order_status(self, record: OrderRecord) -> None:
        """Durably write the order's new state."""
        await self._store.save(record)
        logger.debug(
            "Order %s status -> %s", record.client_order_id, record.state.value)

    async def _record_trade(self, record: OrderRecord, fill_data: dict) -> bool:
        """Durably append a fill; False when it was already recorded."""
        first_time = await self._store.record_trade(
            record.client_order_id, fill_data)
        logger.debug(
            "Trade %s for order %s recorded=%s",
            fill_data.get("fill_id"), record.client_order_id, first_time,
        )
        return bool(first_time)

    async def _update_position(self, record: OrderRecord, fill_data: dict) -> dict:
        """Durably apply the position delta implied by a fill."""
        qty = int(fill_data.get("qty", 0))
        price = float(fill_data.get("price", 0.0))
        side = str(fill_data.get("txn_type", record.side)).upper()
        delta = qty if side == TransactionType.BUY.value else -qty
        pos = await self._store.apply_position_delta(
            record.symbol, record.exchange, record.product, delta, price)
        logger.debug("Position updated for %s: %s", record.symbol, pos)
        return pos


# --------------------------------------------------------------------------- #
#  Module helpers                                                              #
# --------------------------------------------------------------------------- #

def _monitor_result(
    last: Optional[dict],
    state: OrderState,
    *,
    timed_out: bool,
    poll_errors: list[str],
) -> dict:
    out = dict(last or {})
    out["state"] = state.value
    out["timed_out"] = timed_out
    out["poll_errors"] = list(poll_errors)
    out["is_terminal"] = state in (
        OrderState.FILLED, OrderState.CANCELLED,
        OrderState.REJECTED, OrderState.EXPIRED,
    )
    return out


def _walk_to_submitted(record: OrderRecord) -> None:
    """Advance a pre-submit record to SUBMITTED via declared transitions only."""
    chain = [OrderState.RISK_CHECK_PENDING, OrderState.RISK_APPROVED,
             OrderState.SUBMITTED]
    for state in chain:
        if record.state is state:
            continue
        if state in LEGAL_TRANSITIONS.get(record.state, frozenset()):
            record.transition(state, reason="found at broker during recovery")


def _derive_fill_id(client_order_id: str, fill_data: dict) -> str:
    """Deterministic fill id when the broker payload carries none."""
    payload = "|".join(str(fill_data.get(k, "")) for k in
                       ("trade_id", "qty", "price", "timestamp", "fill_time"))
    return hashlib.sha256(f"{client_order_id}|{payload}".encode()).hexdigest()[:24]


def _broker_supports_trigger(broker: BrokerInterface) -> bool:
    """
    True only when ``broker.place_order`` explicitly accepts ``trigger_price``.

    Deliberately strict: a ``**kwargs`` signature that quietly discards the
    trigger would leave a stop-loss order with no stop, which is the exact
    failure this check exists to prevent.
    """
    try:
        params = inspect.signature(broker.place_order).parameters
    except (TypeError, ValueError):      # pragma: no cover - exotic callables
        return False
    return "trigger_price" in params
