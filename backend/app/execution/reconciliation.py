"""
Reconciliation engine — compare broker state vs local persistent state.

DESIGN RULE: THIS MODULE FAILS CLOSED.
=====================================
Reconciliation is the gate that decides whether a live-trading process is
allowed to start sending orders after a restart.  The only acceptable
failure mode is "refuse to trade".  Concretely:

* Every data source (broker positions/orders/trades/funds AND the local
  persistent state) is fetched through a wrapper that records
  *unavailability* as an explicit fact.  A fetch failure NEVER degrades to
  an empty list, because ``[] vs []`` compares equal and would report OK
  while the process knows nothing about the real broker state.
* The result is an explicit four-valued enum.  ``RECONCILIATION_UNAVAILABLE``
  is NOT a success value.  Only ``RECONCILIATION_OK`` permits trading.
* ``ReconciliationResult.__bool__`` and ``ReconciliationStatus.__bool__``
  are overridden so that a careless ``if result:`` / ``if status:`` cannot
  read an unknown or degraded state as success.
* If the kill switch cannot be activated (no store, store write raised, or
  the write could not be verified by read-back) the failure PROPAGATES.
  The engine never claims a safety action it did not perform.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

from ..broker.base import BrokerInterface

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  States                                                                      #
# --------------------------------------------------------------------------- #

class ReconciliationStatus(str, Enum):
    """
    Explicit reconciliation outcome.

    There is deliberately no "warning"/"degraded-but-fine" value: anything
    that is not ``RECONCILIATION_OK`` blocks trading.
    """

    RECONCILIATION_OK = "RECONCILIATION_OK"
    RECONCILIATION_MISMATCH = "RECONCILIATION_MISMATCH"
    RECONCILIATION_UNAVAILABLE = "RECONCILIATION_UNAVAILABLE"
    RECONCILIATION_ERROR = "RECONCILIATION_ERROR"

    # Short aliases.  Same value => real Enum aliases, not extra members.
    OK = "RECONCILIATION_OK"
    MISMATCH = "RECONCILIATION_MISMATCH"
    UNAVAILABLE = "RECONCILIATION_UNAVAILABLE"
    ERROR = "RECONCILIATION_ERROR"

    def __bool__(self) -> bool:
        """Unknown/degraded state must never read as success."""
        return self is ReconciliationStatus.RECONCILIATION_OK

    @property
    def permits_trading(self) -> bool:
        return self is ReconciliationStatus.RECONCILIATION_OK


class DiscrepancyKind(str, Enum):
    """What kind of disagreement was found."""

    MISSING_LOCAL = "missing_local"            # at broker, absent locally (most dangerous)
    MISSING_BROKER = "missing_broker"          # locally, absent at broker
    MISMATCHED_QTY = "mismatched_qty"
    MISMATCHED_PRICE = "mismatched_price"      # average / fill price beyond tolerance
    MISMATCHED_CASH = "mismatched_cash"        # cash / margin beyond tolerance
    UNKNOWN_ORDER_STATE = "unknown_order_state"
    PARTIAL_FILL_MISMATCH = "partial_fill_mismatch"
    DATA_UNAVAILABLE = "data_unavailable"


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"


# --------------------------------------------------------------------------- #
#  Order-status vocabulary                                                     #
# --------------------------------------------------------------------------- #

#: Broker statuses that mean "this order can still trade".
OPEN_STATUSES: frozenset[str] = frozenset({
    "OPEN",
    "TRIGGER PENDING",
    "AMO REQ RECEIVED",
    "OPEN PENDING",
    "MODIFY PENDING",
    "MODIFY VALIDATION PENDING",
    "CANCEL PENDING",
    "VALIDATION PENDING",
    "PUT ORDER REQ RECEIVED",
    # A partially filled order is still live: the residual can trade.
    #
    # This vocabulary previously covered Kite's statuses only, so the paper
    # broker's "PARTIAL" fell through to UNKNOWN. Because an unrecognised
    # status is treated as ambiguous (correctly — fail closed), a perfectly
    # ordinary partial fill raised a CRITICAL discrepancy and blocked all
    # trading. The fail-closed default was right; the gap was that the paper
    # venue's vocabulary was never included.
    "PARTIAL",
    "PARTIALLY FILLED",
    "PARTIALLY_FILLED",
})

#: Statuses that mean "this order is finished and cannot trade again".
TERMINAL_STATUSES: frozenset[str] = frozenset({
    "COMPLETE",
    "CANCELLED",
    "REJECTED",
})

#: Local-only statuses that mean "we do not know what the broker did".
#: A crash between "about to submit" and "acknowledgement stored" leaves an
#: order in one of these; it MUST block trading until resolved.
AMBIGUOUS_STATUSES: frozenset[str] = frozenset({
    "",
    "PENDING",
    "SUBMITTING",
    "SUBMITTED",
    "UNKNOWN",
    "IN_FLIGHT",
    "CANCELLING",
})


def classify_order_status(status: Optional[str]) -> str:
    """Return one of 'open' | 'terminal' | 'unknown'."""
    s = (status or "").strip().upper()
    if s in OPEN_STATUSES:
        return "open"
    if s in TERMINAL_STATUSES:
        return "terminal"
    return "unknown"


# --------------------------------------------------------------------------- #
#  Exceptions                                                                  #
# --------------------------------------------------------------------------- #

class ReconciliationError(RuntimeError):
    """Raised when reconciliation did not end in RECONCILIATION_OK."""

    def __init__(self, message: str, result: Optional["ReconciliationResult"] = None) -> None:
        super().__init__(message)
        self.result = result


class KillSwitchActivationError(ReconciliationError):
    """
    Raised when the kill switch could NOT be activated.

    This is itself a critical condition: the process believes it must stop
    trading but cannot persist that fact, so nothing downstream (and no
    subsequent process start) will honour it.
    """


class LocalStateUnavailable(RuntimeError):
    """Raised by a LocalStateStore when local persistent state cannot be read."""


class BrokerStateUnavailable(RuntimeError):
    """Raised internally when a broker fetch fails."""


# --------------------------------------------------------------------------- #
#  Local persistence surface                                                   #
# --------------------------------------------------------------------------- #

@runtime_checkable
class LocalStateStore(Protocol):
    """
    The narrow persistence surface reconciliation REQUIRES.

    Every method must either return real data or raise
    :class:`LocalStateUnavailable`.  Returning ``[]`` to mean "I could not
    read the database" is the defect this Protocol exists to prevent: an
    empty local list compares equal to an empty broker list and reports OK.

    Position dicts:  symbol, exchange, quantity, average_price[, product]
    Order dicts:     order_id, symbol, exchange, status, quantity
                     [, filled_quantity, price, product]
    Trade dicts:     order_id, symbol, quantity, price[, trade_id, exchange]
                     -- ONE DICT PER FILL.  Do not pre-aggregate.
    Cash dict:       cash[, margin_available, margin_used]
    """

    async def get_positions(self) -> list[dict]: ...

    async def get_orders(self) -> list[dict]: ...

    async def get_trades(self) -> list[dict]: ...

    async def get_cash(self) -> dict: ...


class SqlAlchemyLocalStateStore:
    """
    Real ORM-backed implementation of :class:`LocalStateStore`.

    This replaces the ``return []`` stubs that used to live on
    ``ReconciliationEngine``.  Any ORM failure raises
    :class:`LocalStateUnavailable` so that reconciliation reports
    RECONCILIATION_UNAVAILABLE instead of silently comparing against
    nothing.

    Notes
    -----
    * ``user_id=None`` means "this deployment maps one broker account to the
      whole database"; a warning is logged because it is rarely what a
      multi-tenant deployment wants.
    * Local cash is the ``AccountCash`` balance for this trading mode. If
      there is no such row, cash is UNAVAILABLE -- deliberately, because
      "no cash record" must not be reconciled as "cash is 0.0".
    * ``Position`` has no ``product`` column, so the broker's product
      (MIS/CNC/NRML) is inferred from ``strategy``.  A wrong inference splits
      the position key and shows up as a MISSING_LOCAL + MISSING_BROKER pair
      -- a false alarm that blocks trading rather than permitting it.  Add a
      real ``product`` column to remove the guess.
    """

    def __init__(
        self,
        session: Any,
        user_id: Optional[int] = None,
        *,
        trading_mode: Optional[str] = None,
    ) -> None:
        if session is None:
            raise ValueError("SqlAlchemyLocalStateStore requires a session")
        self._session = session
        self._user_id = user_id
        # Defaults to the configured mode rather than being hard-coded, so a
        # paper process reads the paper balance and a live one the live balance.
        if trading_mode is None:
            from app.core.config import settings as _settings
            trading_mode = str(_settings.trading_mode)
        self._trading_mode = trading_mode
        if user_id is None:
            logger.warning(
                "SqlAlchemyLocalStateStore built without a user_id: reconciling "
                "ALL rows against a single broker account."
            )

    def _scoped(self, stmt, model):
        if self._user_id is not None:
            return stmt.where(model.user_id == self._user_id)
        return stmt

    async def _execute(self, stmt, what: str):
        try:
            return await self._session.execute(stmt)
        except Exception as exc:  # noqa: BLE001 -- re-raised as unavailability
            logger.error("Local state fetch failed (%s): %s", what, exc)
            raise LocalStateUnavailable(f"local {what} unreadable: {exc}") from exc

    async def get_positions(self) -> list[dict]:
        from sqlalchemy import select

        from ..database.models import Position

        stmt = self._scoped(
            select(Position).where(Position.is_open.is_(True)), Position
        )
        rows = (await self._execute(stmt, "positions")).scalars().all()
        return [
            {
                "symbol": r.symbol,
                "exchange": r.exchange,
                "quantity": int(r.quantity or 0),
                "average_price": float(r.average_price or 0.0),
                "product": getattr(r, "product", None) or _product_for_strategy(r.strategy),
                "strategy": r.strategy,
            }
            for r in rows
            if int(r.quantity or 0) != 0
        ]

    async def get_orders(self) -> list[dict]:
        from sqlalchemy import select

        from ..database.models import Order

        stmt = self._scoped(select(Order), Order)
        rows = (await self._execute(stmt, "orders")).scalars().all()
        return [
            {
                "order_id": r.order_id_broker,
                "local_id": r.id,
                "symbol": r.symbol,
                "exchange": r.exchange,
                "status": r.status,
                "quantity": int(r.quantity or 0),
                "price": float(r.price or 0.0),
                "transaction_type": r.transaction_type,
                "strategy": r.strategy,
            }
            for r in rows
        ]

    async def get_trades(self) -> list[dict]:
        """One dict PER FILL — never pre-aggregated by order id."""
        from sqlalchemy import select

        from ..database.models import Order, Trade

        stmt = self._scoped(
            select(Trade, Order.order_id_broker).join(
                Order, Trade.order_id == Order.id, isouter=True
            ),
            Trade,
        )
        rows = (await self._execute(stmt, "trades")).all()
        out: list[dict] = []
        for trade, broker_order_id in rows:
            out.append({
                "order_id": broker_order_id,
                "local_order_id": trade.order_id,
                "trade_id": trade.id,
                "symbol": trade.symbol,
                "exchange": trade.exchange,
                "quantity": int(trade.quantity or 0),
                "price": float(trade.price or 0.0),
            })
        return out

    async def get_cash(self) -> dict:
        """
        The account's cash BALANCE, from ``AccountCash``.

        This used to read the newest ``CapitalAllocation`` row, which is a
        different quantity entirely: a monthly capital budget, not a balance.
        Reconciling the broker's cash against a budget compares two unrelated
        numbers, so it produced a permanent false MISMATCH — and a permanent
        false mismatch blocks trading forever while telling you nothing.

        Scoped to the trading mode, so a paper balance can never be compared
        against a live account or vice versa.
        """
        from sqlalchemy import select

        from ..database.models import AccountCash

        stmt = self._scoped(
            select(AccountCash).where(AccountCash.trading_mode == self._trading_mode),
            AccountCash,
        )
        row = (await self._execute(stmt, "cash")).scalars().first()
        if row is None:
            raise LocalStateUnavailable(
                f"no AccountCash row for trading_mode={self._trading_mode!r}: "
                "local cash is unknown (which is NOT the same as zero)"
            )
        return {
            "cash": float(row.cash or 0.0),
            "margin_available": float(row.cash or 0.0) - float(row.reserved or 0.0),
            "margin_used": float(row.reserved or 0.0),
        }


def _product_for_strategy(strategy: Optional[str]) -> str:
    return "MIS" if (strategy or "").lower() == "intraday" else "CNC"


# --------------------------------------------------------------------------- #
#  Result types                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class Discrepancy:
    kind: DiscrepancyKind
    symbol: str
    order_id: Optional[str]
    broker_value: Any
    local_value: Any
    details: str = ""
    severity: Severity = Severity.CRITICAL

    def __str__(self) -> str:
        return (
            f"[{self.kind.value}/{self.severity.value}] {self.symbol} "
            f"oid={self.order_id} broker={self.broker_value} local={self.local_value} "
            f"— {self.details}"
        )


@dataclass
class ReconciliationSnapshot:
    """Everything that was (or could not be) fetched for one pass."""

    broker_positions: Optional[list[dict]] = None
    broker_orders: Optional[list[dict]] = None
    broker_trades: Optional[list[dict]] = None
    broker_cash: Optional[dict] = None
    local_positions: Optional[list[dict]] = None
    local_orders: Optional[list[dict]] = None
    local_trades: Optional[list[dict]] = None
    local_cash: Optional[dict] = None
    unavailable: list[str] = field(default_factory=list)

    def is_complete(self) -> bool:
        return not self.unavailable

    def _missing(self, *names: str) -> list[str]:
        return [n for n in names if n in self.unavailable]

    def positions_comparable(self) -> bool:
        return not self._missing("broker_positions", "local_positions")

    def orders_comparable(self) -> bool:
        return not self._missing("broker_orders", "local_orders")

    def trades_comparable(self) -> bool:
        return not self._missing("broker_trades", "local_trades")

    def cash_comparable(self) -> bool:
        return not self._missing("broker_cash", "local_cash")


@dataclass
class ReconciliationResult:
    status: ReconciliationStatus
    discrepancies: list[Discrepancy] = field(default_factory=list)
    unavailable_sources: list[str] = field(default_factory=list)
    recommendation: str = ""
    snapshot: Optional[ReconciliationSnapshot] = None
    kill_switch_activated: bool = False
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status is ReconciliationStatus.RECONCILIATION_OK

    @property
    def permits_trading(self) -> bool:
        return self.ok

    def __bool__(self) -> bool:
        """A non-OK result must never read as success in an ``if``."""
        return self.ok

    def of_kind(self, kind: DiscrepancyKind) -> list[Discrepancy]:
        return [d for d in self.discrepancies if d.kind is kind]

    def summary(self) -> str:
        parts = [f"status={self.status.value}"]
        if self.unavailable_sources:
            parts.append(f"unavailable={sorted(self.unavailable_sources)}")
        if self.discrepancies:
            parts.append(f"discrepancies={len(self.discrepancies)}")
        if self.error:
            parts.append(f"error={self.error}")
        return " ".join(parts)


# --------------------------------------------------------------------------- #
#  Engine                                                                      #
# --------------------------------------------------------------------------- #

class ReconciliationEngine:
    """
    Compares broker-reported positions / orders / trades / cash against the
    local persistent state.

    Runs at application startup before ANY trading begins (see
    :mod:`app.execution.recovery`) and periodically during the session.
    """

    #: statuses that trip the kill switch.  Default: everything that is not OK.
    DEFAULT_KILL_SWITCH_ON = frozenset({
        ReconciliationStatus.RECONCILIATION_MISMATCH,
        ReconciliationStatus.RECONCILIATION_UNAVAILABLE,
        ReconciliationStatus.RECONCILIATION_ERROR,
    })

    def __init__(
        self,
        kill_switch_store=None,
        *,
        local_state: Optional[LocalStateStore] = None,
        user_id: Optional[int] = None,
        qty_tolerance: int = 0,
        price_tolerance_pct: float = 0.001,      # 0.1 %
        cash_tolerance_abs: float = 1.0,         # ₹1
        cash_tolerance_pct: float = 0.001,       # 0.1 %
        activate_kill_switch_on: Optional[frozenset] = None,
        kill_switch_key: str = "kill_switch",
    ) -> None:
        """
        Parameters
        ----------
        kill_switch_store
            Redis-like client with ``.set(key, value)`` (and ideally
            ``.get(key)`` so activation can be VERIFIED).  If this is None,
            a required activation raises :class:`KillSwitchActivationError`
            rather than silently doing nothing.
        local_state
            A :class:`LocalStateStore`.  If omitted, ``reconcile()`` must be
            given one (or a usable ``db_session``); otherwise local state is
            reported UNAVAILABLE and trading is blocked.
        """
        self._store = kill_switch_store
        self._local_state = local_state
        self._user_id = user_id
        self._qty_tol = qty_tolerance
        self._price_tol = price_tolerance_pct
        self._cash_tol_abs = cash_tolerance_abs
        self._cash_tol_pct = cash_tolerance_pct
        self._kill_on = activate_kill_switch_on or self.DEFAULT_KILL_SWITCH_ON
        self._kill_key = kill_switch_key

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    async def reconcile(
        self,
        broker: BrokerInterface,
        db_session=None,
        *,
        local_state: Optional[LocalStateStore] = None,
        raise_on_failure: bool = True,
    ) -> ReconciliationResult:
        """
        Full reconciliation pass.

        Returns a :class:`ReconciliationResult`.  When ``raise_on_failure``
        (the default) and the status is not OK, the kill switch is activated
        FIRST and then :class:`ReconciliationError` is raised.  If the kill
        switch cannot be activated, :class:`KillSwitchActivationError` is
        raised INSTEAD — the caller is never told a switch was thrown when it
        was not.
        """
        result = await self.evaluate(broker, db_session, local_state=local_state)
        return self.enforce(result, raise_on_failure=raise_on_failure)

    async def evaluate(
        self,
        broker: BrokerInterface,
        db_session=None,
        *,
        local_state: Optional[LocalStateStore] = None,
    ) -> ReconciliationResult:
        """
        Run a full pass and classify it, WITHOUT touching the kill switch.

        Use this when the caller owns the fail-closed policy (as
        :class:`app.execution.recovery.RecoveryManager` does).
        """
        logger.info("Reconciliation started.")
        try:
            snapshot = await self.gather(broker, db_session, local_state=local_state)
            discrepancies: list[Discrepancy] = []
            discrepancies += self.reconcile_orders(snapshot)
            discrepancies += self.reconcile_positions(snapshot)
            discrepancies += self.reconcile_cash(snapshot)
            discrepancies += self.identify_unknown_orders(snapshot)
            result = self.classify(snapshot, discrepancies)
        except Exception as exc:  # noqa: BLE001 -- unexpected failure => ERROR, fail closed
            logger.exception("Reconciliation raised an unexpected error.")
            result = ReconciliationResult(
                status=ReconciliationStatus.RECONCILIATION_ERROR,
                recommendation=(
                    "ERROR: reconciliation itself failed. Trading must stay blocked "
                    "until the cause is understood."
                ),
                error=f"{type(exc).__name__}: {exc}",
            )

        self._log_result(result)
        return result

    def enforce(
        self,
        result: ReconciliationResult,
        *,
        raise_on_failure: bool = True,
    ) -> ReconciliationResult:
        """
        Apply the fail-closed policy to an already-computed result.

        Activates the kill switch for any status configured in
        ``activate_kill_switch_on``.  An activation failure propagates.
        """
        if result.status.permits_trading:
            logger.info("Reconciliation complete: %s", result.summary())
            return result

        if result.status in self._kill_on:
            # Raises KillSwitchActivationError if it did not actually happen.
            self._activate_kill_switch(reason=result.summary(), result=result)
            result.kill_switch_activated = True

        if raise_on_failure:
            raise ReconciliationError(
                f"Reconciliation failed: {result.summary()}. "
                + (
                    "Kill switch activated."
                    if result.kill_switch_activated
                    else "Kill switch NOT activated (not configured for this status)."
                ),
                result=result,
            )
        return result

    # ------------------------------------------------------------------ #
    #  Step 1 — gather (broker + local), recording unavailability          #
    # ------------------------------------------------------------------ #

    async def gather(
        self,
        broker: BrokerInterface,
        db_session=None,
        *,
        local_state: Optional[LocalStateStore] = None,
    ) -> ReconciliationSnapshot:
        """
        Fetch both sides.

        A failing fetch is recorded in ``snapshot.unavailable`` and the
        corresponding field stays ``None``.  It is NEVER replaced by ``[]``.
        """
        snap = ReconciliationSnapshot()
        store = self.resolve_local_state(db_session, local_state, snap)
        await self.fetch_local(store, snap)
        await self.fetch_broker(broker, snap)
        return snap

    async def fetch_local(
        self,
        store: Optional[LocalStateStore],
        snap: ReconciliationSnapshot,
    ) -> ReconciliationSnapshot:
        """Load persistent local state into ``snap`` (recovery step 1)."""
        if store is None:
            return snap  # already marked unavailable by resolve_local_state
        snap.local_positions = await self._safe_fetch(
            store.get_positions, "local_positions", snap, list
        )
        snap.local_orders = await self._safe_fetch(
            store.get_orders, "local_orders", snap, list
        )
        snap.local_trades = await self._safe_fetch(
            store.get_trades, "local_trades", snap, list
        )
        snap.local_cash = await self._safe_fetch(
            store.get_cash, "local_cash", snap, dict
        )
        return snap

    async def fetch_broker(
        self,
        broker: BrokerInterface,
        snap: ReconciliationSnapshot,
    ) -> ReconciliationSnapshot:
        """Query the broker into ``snap`` (recovery step 2)."""
        snap.broker_positions = await self._safe_fetch(
            broker.get_positions, "broker_positions", snap, list
        )
        snap.broker_orders = await self._safe_fetch(
            broker.get_orders, "broker_orders", snap, list
        )
        snap.broker_trades = await self._safe_fetch(
            broker.get_trades, "broker_trades", snap, list
        )
        snap.broker_cash = await self._safe_fetch(
            broker.get_funds, "broker_cash", snap, dict
        )
        return snap

    def resolve_local_state(
        self,
        db_session,
        local_state: Optional[LocalStateStore],
        snap: ReconciliationSnapshot,
    ) -> Optional[LocalStateStore]:
        store = local_state or self._local_state
        if store is None and db_session is not None:
            try:
                store = SqlAlchemyLocalStateStore(db_session, self._user_id)
            except Exception as exc:  # noqa: BLE001
                logger.error("Cannot build local state store from db_session: %s", exc)
                store = None
        if store is None:
            logger.error(
                "NO LOCAL STATE SOURCE. Reconciliation cannot compare anything; "
                "reporting UNAVAILABLE (fail closed)."
            )
            for name in (
                "local_positions", "local_orders", "local_trades", "local_cash",
            ):
                snap.unavailable.append(name)
        return store

    @staticmethod
    async def _safe_fetch(
        fetch: Callable[[], Any],
        name: str,
        snap: ReconciliationSnapshot,
        expected_type: type,
    ):
        """
        Run one fetch.  On ANY failure record unavailability and return None.

        This is the fix for the original fail-open bug: the old code caught
        the exception and substituted ``[]``, which compared equal to an
        empty local list and produced status OK while the broker was down.
        """
        try:
            value = await fetch()
        except Exception as exc:  # noqa: BLE001
            logger.error("UNAVAILABLE: %s could not be fetched: %s", name, exc)
            snap.unavailable.append(name)
            return None
        if value is None or not isinstance(value, expected_type):
            logger.error(
                "UNAVAILABLE: %s returned %r (expected %s)",
                name, type(value).__name__, expected_type.__name__,
            )
            snap.unavailable.append(name)
            return None
        # Narrowed against the concrete types rather than the `expected_type`
        # variable: isinstance(x, some_variable) tells mypy nothing, so `value`
        # stayed `object` here. The guard above already established which one
        # it is, so these branches are checks, not casts.
        if isinstance(value, list):
            return list(value)
        if isinstance(value, dict):
            return dict(value)
        # expected_type is only ever list or dict today. If a caller ever passes
        # something else, treat it as UNAVAILABLE rather than handing back an
        # object the comparison steps cannot interpret — the whole point of this
        # helper is that an unusable broker response never reads as agreement.
        logger.error(
            "UNAVAILABLE: %s expected list or dict, got %s",
            name, type(value).__name__,
        )
        snap.unavailable.append(name)
        return None

    # ------------------------------------------------------------------ #
    #  Step 2 — orders                                                     #
    # ------------------------------------------------------------------ #

    def reconcile_orders(self, snap: ReconciliationSnapshot) -> list[Discrepancy]:
        """Compare OPEN orders on both sides, preserving partial-fill detail."""
        if not snap.orders_comparable():
            return [
                self._unavailable_discrepancy(
                    "orders", snap, ("broker_orders", "local_orders")
                ),
                *self._reconcile_fills(snap),
            ]

        broker_open = {
            str(o.get("order_id")): o
            for o in (snap.broker_orders or [])
            if classify_order_status(o.get("status")) == "open" and o.get("order_id")
        }
        local_open = {
            str(o.get("order_id")): o
            for o in (snap.local_orders or [])
            if classify_order_status(o.get("status")) == "open" and o.get("order_id")
        }

        out: list[Discrepancy] = []

        for oid, bo in broker_open.items():
            if oid not in local_open:
                out.append(Discrepancy(
                    kind=DiscrepancyKind.MISSING_LOCAL,
                    symbol=_symbol_of(bo),
                    order_id=oid,
                    broker_value=bo.get("status"),
                    local_value=None,
                    details=(
                        "LIVE order at broker is not tracked locally — it can "
                        "still fill and create an untracked position."
                    ),
                ))

        for oid, lo in local_open.items():
            if oid not in broker_open:
                out.append(Discrepancy(
                    kind=DiscrepancyKind.MISSING_BROKER,
                    symbol=_symbol_of(lo),
                    order_id=oid,
                    broker_value=None,
                    local_value=lo.get("status"),
                    details="Order is open locally but the broker does not report it.",
                ))

        for oid in broker_open.keys() & local_open.keys():
            bo, lo = broker_open[oid], local_open[oid]
            bq, lq = _int_of(bo, "quantity"), _int_of(lo, "quantity")
            if abs(bq - lq) > self._qty_tol:
                out.append(Discrepancy(
                    kind=DiscrepancyKind.MISMATCHED_QTY,
                    symbol=_symbol_of(bo),
                    order_id=oid,
                    broker_value=bq,
                    local_value=lq,
                    details=f"Open-order quantity mismatch: broker={bq}, local={lq}.",
                ))
            bf, lf = _int_of(bo, "filled_quantity"), _int_of(lo, "filled_quantity")
            if abs(bf - lf) > self._qty_tol:
                out.append(Discrepancy(
                    kind=DiscrepancyKind.PARTIAL_FILL_MISMATCH,
                    symbol=_symbol_of(bo),
                    order_id=oid,
                    broker_value=bf,
                    local_value=lf,
                    details=f"Filled quantity mismatch: broker={bf}, local={lf}.",
                ))

        out.extend(self._reconcile_fills(snap))
        return out

    def _reconcile_fills(self, snap: ReconciliationSnapshot) -> list[Discrepancy]:
        """
        Compare executions WITHOUT collapsing partial fills.

        The original code did ``{t["order_id"]: t for t in trades}``, so three
        fills of one order became one entry: the first two vanished and only
        the last price was compared.  Here every fill is kept; per order we
        compare fill count, total filled quantity and quantity-weighted
        average price.
        """
        if not snap.trades_comparable():
            return [
                self._unavailable_discrepancy(
                    "fills", snap, ("broker_trades", "local_trades")
                )
            ]

        broker_fills = _group_fills(snap.broker_trades or [])
        local_fills = _group_fills(snap.local_trades or [])
        out: list[Discrepancy] = []

        for oid, bg in broker_fills.items():
            lg = local_fills.get(oid)
            if lg is None:
                out.append(Discrepancy(
                    kind=DiscrepancyKind.MISSING_LOCAL,
                    symbol=bg.symbol,
                    order_id=oid,
                    broker_value=bg.total_qty,
                    local_value=None,
                    details=(
                        f"{bg.fill_count} fill(s) totalling {bg.total_qty} executed at "
                        "the broker are absent locally."
                    ),
                ))
                continue

            if abs(bg.total_qty - lg.total_qty) > self._qty_tol:
                out.append(Discrepancy(
                    kind=DiscrepancyKind.MISMATCHED_QTY,
                    symbol=bg.symbol,
                    order_id=oid,
                    broker_value=bg.total_qty,
                    local_value=lg.total_qty,
                    details=(
                        f"Executed quantity mismatch: broker={bg.total_qty} across "
                        f"{bg.fill_count} fill(s), local={lg.total_qty} across "
                        f"{lg.fill_count} fill(s)."
                    ),
                ))
            elif bg.fill_count != lg.fill_count:
                out.append(Discrepancy(
                    kind=DiscrepancyKind.PARTIAL_FILL_MISMATCH,
                    symbol=bg.symbol,
                    order_id=oid,
                    broker_value=bg.fill_count,
                    local_value=lg.fill_count,
                    details=(
                        f"Same total quantity but a different number of fills: "
                        f"broker={bg.fill_count}, local={lg.fill_count}."
                    ),
                    severity=Severity.WARNING,
                ))

            if _price_differs(bg.vwap, lg.vwap, self._price_tol):
                out.append(Discrepancy(
                    kind=DiscrepancyKind.MISMATCHED_PRICE,
                    symbol=bg.symbol,
                    order_id=oid,
                    broker_value=round(bg.vwap, 4),
                    local_value=round(lg.vwap, 4),
                    details=(
                        f"Quantity-weighted fill price mismatch: broker={bg.vwap:.4f}, "
                        f"local={lg.vwap:.4f} (tolerance {self._price_tol:.4%})."
                    ),
                ))

        for oid, lg in local_fills.items():
            if oid not in broker_fills:
                out.append(Discrepancy(
                    kind=DiscrepancyKind.MISSING_BROKER,
                    symbol=lg.symbol,
                    order_id=oid,
                    broker_value=None,
                    local_value=lg.total_qty,
                    details=(
                        f"{lg.fill_count} local fill(s) totalling {lg.total_qty} are "
                        "not in the broker's trade book."
                    ),
                ))
        return out

    # ------------------------------------------------------------------ #
    #  Step 3 — positions                                                  #
    # ------------------------------------------------------------------ #

    def reconcile_positions(self, snap: ReconciliationSnapshot) -> list[Discrepancy]:
        if not snap.positions_comparable():
            return [
                self._unavailable_discrepancy(
                    "positions", snap, ("broker_positions", "local_positions")
                )
            ]

        broker_map = _position_map(snap.broker_positions or [])
        local_map = _position_map(snap.local_positions or [])
        out: list[Discrepancy] = []

        for key, bp in broker_map.items():
            if _int_of(bp, "quantity") == 0:
                continue
            if key not in local_map:
                out.append(Discrepancy(
                    kind=DiscrepancyKind.MISSING_LOCAL,
                    symbol=key,
                    order_id=None,
                    broker_value=_int_of(bp, "quantity"),
                    local_value=None,
                    details=(
                        "UNTRACKED LIVE POSITION: the broker holds this position and "
                        "the local store knows nothing about it."
                    ),
                ))

        for key, lp in local_map.items():
            if _int_of(lp, "quantity") == 0:
                continue
            if key not in broker_map or _int_of(broker_map[key], "quantity") == 0:
                out.append(Discrepancy(
                    kind=DiscrepancyKind.MISSING_BROKER,
                    symbol=key,
                    order_id=None,
                    broker_value=None,
                    local_value=_int_of(lp, "quantity"),
                    details="Position held locally does not exist at the broker.",
                ))

        for key in broker_map.keys() & local_map.keys():
            bq = _int_of(broker_map[key], "quantity")
            lq = _int_of(local_map[key], "quantity")
            if abs(bq - lq) > self._qty_tol:
                out.append(Discrepancy(
                    kind=DiscrepancyKind.MISMATCHED_QTY,
                    symbol=key,
                    order_id=None,
                    broker_value=bq,
                    local_value=lq,
                    details=f"Position quantity mismatch: broker={bq}, local={lq}.",
                ))
            bp_price = _float_of(broker_map[key], "average_price")
            lp_price = _float_of(local_map[key], "average_price")
            if _price_differs(bp_price, lp_price, self._price_tol):
                out.append(Discrepancy(
                    kind=DiscrepancyKind.MISMATCHED_PRICE,
                    symbol=key,
                    order_id=None,
                    broker_value=round(bp_price, 4),
                    local_value=round(lp_price, 4),
                    details=(
                        f"Average price mismatch: broker={bp_price:.4f}, "
                        f"local={lp_price:.4f} (tolerance {self._price_tol:.4%})."
                    ),
                ))
        return out

    # ------------------------------------------------------------------ #
    #  Step 4 — cash / margin                                              #
    # ------------------------------------------------------------------ #

    def reconcile_cash(self, snap: ReconciliationSnapshot) -> list[Discrepancy]:
        if not snap.cash_comparable():
            return [
                self._unavailable_discrepancy(
                    "cash", snap, ("broker_cash", "local_cash")
                )
            ]

        broker_cash = snap.broker_cash or {}
        local_cash = snap.local_cash or {}
        out: list[Discrepancy] = []

        for field_name in ("cash", "margin_available", "margin_used"):
            if field_name not in broker_cash or field_name not in local_cash:
                continue
            b = _to_float(broker_cash.get(field_name))
            lo = _to_float(local_cash.get(field_name))
            if b is None or lo is None:
                out.append(Discrepancy(
                    kind=DiscrepancyKind.DATA_UNAVAILABLE,
                    symbol=field_name,
                    order_id=None,
                    broker_value=broker_cash.get(field_name),
                    local_value=local_cash.get(field_name),
                    details=f"Non-numeric {field_name}; cannot reconcile.",
                ))
                continue
            tolerance = max(self._cash_tol_abs, abs(b) * self._cash_tol_pct)
            if abs(b - lo) > tolerance:
                out.append(Discrepancy(
                    kind=DiscrepancyKind.MISMATCHED_CASH,
                    symbol=field_name,
                    order_id=None,
                    broker_value=b,
                    local_value=lo,
                    details=(
                        f"{field_name} mismatch: broker={b:.2f}, local={lo:.2f}, "
                        f"delta={b - lo:.2f} > tolerance {tolerance:.2f}."
                    ),
                ))
        return out

    # ------------------------------------------------------------------ #
    #  Step 5 — ambiguous / unknown orders                                 #
    # ------------------------------------------------------------------ #

    def identify_unknown_orders(self, snap: ReconciliationSnapshot) -> list[Discrepancy]:
        """
        Find orders whose true state nobody knows.

        These are the crash-window orders: submitted but never acknowledged,
        acknowledged but never stored, cancelled locally while the broker may
        still hold them.  They must block trading until resolved.
        """
        out: list[Discrepancy] = []
        if snap.local_orders is None and "local_orders" in snap.unavailable:
            return out  # already reported by reconcile_orders

        for lo in snap.local_orders or []:
            status = (lo.get("status") or "").strip().upper()
            oid = lo.get("order_id")
            if classify_order_status(status) != "unknown":
                if oid in (None, "", "None") and status not in TERMINAL_STATUSES:
                    out.append(Discrepancy(
                        kind=DiscrepancyKind.UNKNOWN_ORDER_STATE,
                        symbol=_symbol_of(lo),
                        order_id=None,
                        broker_value=None,
                        local_value=status,
                        details=(
                            "Local order has no broker order id: it may or may not "
                            "have reached the exchange."
                        ),
                    ))
                continue
            out.append(Discrepancy(
                kind=DiscrepancyKind.UNKNOWN_ORDER_STATE,
                symbol=_symbol_of(lo),
                order_id=str(oid) if oid else None,
                broker_value=_broker_status_for(snap, oid),
                local_value=status or "<empty>",
                details=(
                    f"Local order is in ambiguous state {status or '<empty>'!r}; "
                    "its real state at the broker is unknown."
                ),
            ))

        for bo in snap.broker_orders or []:
            status = (bo.get("status") or "").strip().upper()
            if classify_order_status(status) == "unknown":
                out.append(Discrepancy(
                    kind=DiscrepancyKind.UNKNOWN_ORDER_STATE,
                    symbol=_symbol_of(bo),
                    order_id=str(bo.get("order_id")) if bo.get("order_id") else None,
                    broker_value=status or "<empty>",
                    local_value=None,
                    details=f"Broker reports unrecognised order status {status or '<empty>'!r}.",
                ))
        return out

    # ------------------------------------------------------------------ #
    #  Step 6 — classification                                             #
    # ------------------------------------------------------------------ #

    def classify(
        self,
        snap: ReconciliationSnapshot,
        discrepancies: Sequence[Discrepancy],
    ) -> ReconciliationResult:
        """
        Precedence: ERROR > UNAVAILABLE > MISMATCH > OK.

        Unavailability outranks a confirmed mismatch because when an input is
        missing no conclusion of "reconciled" is possible at all — the mismatch
        list is necessarily incomplete.
        """
        discrepancies = list(discrepancies)

        if snap.unavailable:
            return ReconciliationResult(
                status=ReconciliationStatus.RECONCILIATION_UNAVAILABLE,
                discrepancies=discrepancies,
                unavailable_sources=sorted(set(snap.unavailable)),
                snapshot=snap,
                recommendation=(
                    "UNAVAILABLE: could not read "
                    f"{sorted(set(snap.unavailable))}. Broker/local state is UNKNOWN — "
                    "this is NOT 'no discrepancies'. Trading must stay blocked until "
                    "every source can be read and compared."
                ),
            )

        if discrepancies:
            return ReconciliationResult(
                status=ReconciliationStatus.RECONCILIATION_MISMATCH,
                discrepancies=discrepancies,
                snapshot=snap,
                recommendation=(
                    f"MISMATCH: {len(discrepancies)} discrepancy(ies) between broker and "
                    "local state. Halt trading, verify at the broker, repair local state, "
                    "then reconcile again."
                ),
            )

        return ReconciliationResult(
            status=ReconciliationStatus.RECONCILIATION_OK,
            snapshot=snap,
            recommendation="Orders, positions, fills and cash all match.",
        )

    # ------------------------------------------------------------------ #
    #  Kill switch — honest activation                                     #
    # ------------------------------------------------------------------ #

    def _activate_kill_switch(
        self,
        reason: str = "",
        result: Optional[ReconciliationResult] = None,
    ) -> None:
        """
        Activate the kill switch, or raise.

        NEVER returns normally unless the switch is actually set.  A missing
        store, a failing write, or a write that cannot be verified all raise
        :class:`KillSwitchActivationError`.
        """
        logger.critical("RECONCILIATION: activating kill switch — %s", reason)

        if self._store is None:
            raise KillSwitchActivationError(
                "KILL SWITCH NOT ACTIVATED: no kill_switch_store is configured, so the "
                f"halt cannot be persisted. Triggering condition: {reason}",
                result=result,
            )

        try:
            self._store.set(self._kill_key, "1")
        except Exception as exc:  # noqa: BLE001
            raise KillSwitchActivationError(
                f"KILL SWITCH NOT ACTIVATED: store write failed ({exc}). The halt was "
                f"NOT persisted. Triggering condition: {reason}",
                result=result,
            ) from exc

        # Verify by read-back where possible: "wrote without error" is not the
        # same as "the switch is on".
        getter = getattr(self._store, "get", None)
        if callable(getter):
            try:
                value = getter(self._kill_key)
            except Exception as exc:  # noqa: BLE001
                raise KillSwitchActivationError(
                    f"KILL SWITCH UNVERIFIED: the write appeared to succeed but the "
                    f"read-back failed ({exc}); activation cannot be confirmed. "
                    f"Triggering condition: {reason}",
                    result=result,
                ) from exc
            if value in (None, "", b"", "0", b"0", 0, False):
                raise KillSwitchActivationError(
                    f"KILL SWITCH NOT ACTIVATED: read-back returned {value!r} after the "
                    f"write. Triggering condition: {reason}",
                    result=result,
                )

        logger.critical("KILL SWITCH ACTIVATED and verified (key=%s).", self._kill_key)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _unavailable_discrepancy(
        what: str,
        snap: ReconciliationSnapshot,
        sources: tuple[str, ...],
    ) -> Discrepancy:
        missing = sorted({n for n in sources if n in snap.unavailable})
        return Discrepancy(
            kind=DiscrepancyKind.DATA_UNAVAILABLE,
            symbol=what,
            order_id=None,
            broker_value=None,
            local_value=None,
            details=(
                f"Cannot reconcile {what}: {missing} could not be read. "
                "This is UNKNOWN, not equal."
            ),
        )

    def _log_result(self, result: ReconciliationResult) -> None:
        for d in result.discrepancies:
            level = logging.WARNING if d.severity is Severity.WARNING else logging.ERROR
            logger.log(level, "Discrepancy %s", d)
        if result.status.permits_trading:
            logger.info("Reconciliation: %s", result.summary())
        else:
            logger.critical("Reconciliation: %s", result.summary())

    @staticmethod
    def _pos_key(position: dict) -> str:
        return _position_key(position)


# --------------------------------------------------------------------------- #
#  Module-level helpers                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class FillGroup:
    """All fills belonging to one order — partial fills are NOT collapsed."""

    order_id: str
    symbol: str
    fills: list[dict] = field(default_factory=list)

    @property
    def fill_count(self) -> int:
        return len(self.fills)

    @property
    def total_qty(self) -> int:
        return sum(_fill_qty(f) for f in self.fills)

    @property
    def vwap(self) -> float:
        total = self.total_qty
        if total == 0:
            return 0.0
        notional = sum(_fill_qty(f) * _fill_price(f) for f in self.fills)
        return notional / total


def _group_fills(trades: Sequence[dict]) -> dict[str, FillGroup]:
    groups: dict[str, FillGroup] = {}
    for t in trades:
        oid = t.get("order_id")
        if oid in (None, ""):
            continue
        oid = str(oid)
        g = groups.get(oid)
        if g is None:
            g = FillGroup(order_id=oid, symbol=_symbol_of(t))
            groups[oid] = g
        g.fills.append(dict(t))
    return groups


def _fill_qty(fill: dict) -> int:
    """
    Quantity from a fill, whichever key the venue used.

    BUG THIS FIXES: broker fills carry ``qty`` (PaperBroker and Kite both emit
    that), while this module read ``quantity`` — the key the LOCAL store uses.
    Every broker fill therefore counted as ZERO, so after any fill a restart
    compared broker=0 against local=10 and reported a MISMATCH that was not
    real. Reconciliation then latched the kill switch and the process could
    never trade again.

    It failed CLOSED, so nothing was ever at risk — but it made restart-
    after-trading impossible, which is one of the properties reconciliation
    exists to provide.

    Aliases are matched in the same style as :func:`_fill_price`, so a venue
    using either spelling reconciles correctly.
    """
    for key in ("quantity", "qty", "filled_quantity", "filled_qty"):
        if key in fill:
            v = _to_float(fill.get(key))
            if v is not None:
                return int(v)
    return 0


def _fill_price(fill: dict) -> float:
    for key in ("price", "average_price", "fill_price"):
        if key in fill:
            v = _to_float(fill.get(key))
            if v is not None:
                return v
    return 0.0


def _position_key(position: dict) -> str:
    symbol = position.get("tradingsymbol") or position.get("symbol") or "UNKNOWN"
    exchange = position.get("exchange") or "NSE"
    product = position.get("product") or "CNC"
    return f"{exchange}:{symbol}:{product}"


def _position_map(positions: Sequence[dict]) -> dict[str, dict]:
    """Merge duplicate rows for the same key rather than letting one win."""
    out: dict[str, dict] = {}
    for p in positions:
        key = _position_key(p)
        if key in out:
            merged = dict(out[key])
            q1, q2 = _int_of(merged, "quantity"), _int_of(p, "quantity")
            p1, p2 = _float_of(merged, "average_price"), _float_of(p, "average_price")
            total = q1 + q2
            merged["quantity"] = total
            merged["average_price"] = ((q1 * p1) + (q2 * p2)) / total if total else 0.0
            out[key] = merged
        else:
            out[key] = dict(p)
    return out


def _symbol_of(row: dict) -> str:
    return str(row.get("tradingsymbol") or row.get("symbol") or "")


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_of(row: dict, key: str) -> float:
    return _to_float(row.get(key)) or 0.0


def _int_of(row: dict, key: str) -> int:
    v = _to_float(row.get(key))
    return int(v) if v is not None else 0


def _price_differs(a: float, b: float, tol_pct: float) -> bool:
    if a <= 0 and b <= 0:
        return False
    if a <= 0 or b <= 0:
        return True
    return abs(a - b) / max(a, b) > tol_pct


def _broker_status_for(snap: ReconciliationSnapshot, order_id) -> Optional[str]:
    if not order_id:
        return None
    for bo in snap.broker_orders or []:
        if str(bo.get("order_id")) == str(order_id):
            return bo.get("status")
    return None


__all__ = [
    "AMBIGUOUS_STATUSES",
    "OPEN_STATUSES",
    "TERMINAL_STATUSES",
    "BrokerStateUnavailable",
    "Discrepancy",
    "DiscrepancyKind",
    "FillGroup",
    "KillSwitchActivationError",
    "LocalStateStore",
    "LocalStateUnavailable",
    "ReconciliationEngine",
    "ReconciliationError",
    "ReconciliationResult",
    "ReconciliationSnapshot",
    "ReconciliationStatus",
    "Severity",
    "SqlAlchemyLocalStateStore",
    "classify_order_status",
]
