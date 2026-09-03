"""
Explicit order lifecycle: states, legal transitions, durable records, stores.

Design rules (these are the whole point of this module):

1.  **Every** broker response and failure mode has an explicit state.  There is
    no state the system reads as "success by default".
2.  ``UNKNOWN`` is a real, reachable state.  It is entered whenever the broker
    outcome is genuinely ambiguous (timeout / connection reset / lost response).
    An order in ``UNKNOWN`` **blocks** further action until it has been
    reconciled against the broker.  It is never optimistically read as either
    "filled" or "not placed".
3.  Transitions are declared explicitly.  Illegal transitions raise.
4.  Every order carries a client-generated id from ``INTENT_CREATED`` onward.
    That id is also the broker tag, so an ambiguous submission can always be
    resolved by querying the broker's order book for the tag.
5.  Terminal states are terminal.

Nothing in this module talks to a network.  The ``OrderStore`` protocol is the
narrow persistence surface the order manager *requires*; two implementations
ship here (in-memory for tests/dev, Redis-backed for production).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Kite truncates order tags to 20 characters.  The client order id is the tag,
# so it must fit in 20 characters *exactly* or reconciliation-by-tag breaks.
TAG_MAX_LEN = 20
#: How many leading characters of the strategy name are embedded in the tag and
#: used by the duplicate-order gate.  Both sides use the same helper so tag
#: truncation can never defeat the duplicate check again.
STRATEGY_PREFIX_LEN = 8


# --------------------------------------------------------------------------- #
#  Exceptions                                                                  #
# --------------------------------------------------------------------------- #

class LifecycleError(RuntimeError):
    """Base class for order-lifecycle errors."""


class IllegalStateTransition(LifecycleError):
    """Raised when a transition that is not declared legal is attempted."""


class OrderBlockedError(LifecycleError):
    """Raised when an action is attempted on an order in an ambiguous state."""


class PersistenceError(LifecycleError):
    """Raised when the durable order store cannot be read or written."""


# --------------------------------------------------------------------------- #
#  States                                                                      #
# --------------------------------------------------------------------------- #

class OrderState(str, Enum):
    INTENT_CREATED = "INTENT_CREATED"
    RISK_CHECK_PENDING = "RISK_CHECK_PENDING"
    RISK_APPROVED = "RISK_APPROVED"
    RISK_REJECTED = "RISK_REJECTED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


#: States from which no further transition is legal.
TERMINAL_STATES: frozenset[OrderState] = frozenset({
    OrderState.RISK_REJECTED,
    OrderState.FILLED,
    OrderState.CANCELLED,
    OrderState.REJECTED,
    OrderState.EXPIRED,
})

#: States where the order may still be live at the broker.
OPEN_STATES: frozenset[OrderState] = frozenset({
    OrderState.INTENT_CREATED,
    OrderState.RISK_CHECK_PENDING,
    OrderState.RISK_APPROVED,
    OrderState.SUBMITTED,
    OrderState.ACKNOWLEDGED,
    OrderState.PARTIALLY_FILLED,
    OrderState.CANCEL_PENDING,
    OrderState.UNKNOWN,
})

#: States that require a broker reconciliation before anything else may happen.
BLOCKED_STATES: frozenset[OrderState] = frozenset({OrderState.UNKNOWN})

S = OrderState

#: Explicitly declared legal transitions.  Anything not listed raises.
LEGAL_TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    S.INTENT_CREATED: frozenset({S.RISK_CHECK_PENDING, S.RISK_REJECTED}),
    S.RISK_CHECK_PENDING: frozenset({S.RISK_APPROVED, S.RISK_REJECTED}),
    S.RISK_APPROVED: frozenset({S.SUBMITTED, S.RISK_REJECTED, S.CANCELLED}),
    # A submission whose outcome we never learned goes to UNKNOWN, never to a
    # optimistic "it probably worked" state.
    S.SUBMITTED: frozenset({
        S.ACKNOWLEDGED, S.PARTIALLY_FILLED, S.FILLED,
        S.REJECTED, S.CANCELLED, S.EXPIRED, S.UNKNOWN,
    }),
    S.ACKNOWLEDGED: frozenset({
        S.PARTIALLY_FILLED, S.FILLED, S.CANCEL_PENDING,
        S.CANCELLED, S.REJECTED, S.EXPIRED, S.UNKNOWN,
    }),
    S.PARTIALLY_FILLED: frozenset({
        S.PARTIALLY_FILLED,          # successive partial fills
        S.FILLED, S.CANCEL_PENDING, S.CANCELLED, S.EXPIRED, S.UNKNOWN,
    }),
    S.CANCEL_PENDING: frozenset({
        S.CANCELLED, S.FILLED, S.PARTIALLY_FILLED, S.REJECTED, S.UNKNOWN,
    }),
    # UNKNOWN may only be left by reconciliation (enforced separately).
    S.UNKNOWN: frozenset({
        S.ACKNOWLEDGED, S.PARTIALLY_FILLED, S.FILLED,
        S.CANCELLED, S.REJECTED, S.EXPIRED,
    }),
    # Terminal states are terminal.
    S.RISK_REJECTED: frozenset(),
    S.FILLED: frozenset(),
    S.CANCELLED: frozenset(),
    S.REJECTED: frozenset(),
    S.EXPIRED: frozenset(),
}


def assert_transition(
    current: OrderState,
    new: OrderState,
    *,
    reconciled: bool = False,
) -> None:
    """
    Raise :class:`IllegalStateTransition` unless ``current -> new`` is legal.

    ``UNKNOWN`` may only be left when ``reconciled=True`` — i.e. only after the
    broker has actually been queried about this order.
    """
    if current in TERMINAL_STATES:
        raise IllegalStateTransition(
            f"{current.value} is terminal; cannot transition to {new.value}."
        )
    allowed = LEGAL_TRANSITIONS.get(current, frozenset())
    if new not in allowed:
        raise IllegalStateTransition(
            f"Illegal transition {current.value} -> {new.value}. "
            f"Legal: {sorted(a.value for a in allowed)}"
        )
    if current is OrderState.UNKNOWN and not reconciled:
        raise IllegalStateTransition(
            "An order in UNKNOWN may only be resolved by broker reconciliation."
        )


# --------------------------------------------------------------------------- #
#  Broker status mapping                                                       #
# --------------------------------------------------------------------------- #

#: Kite order-book statuses -> lifecycle states.
_BROKER_STATUS_MAP: dict[str, OrderState] = {
    "COMPLETE": OrderState.FILLED,
    "CANCELLED": OrderState.CANCELLED,
    "REJECTED": OrderState.REJECTED,
    "OPEN": OrderState.ACKNOWLEDGED,
    "TRIGGER PENDING": OrderState.ACKNOWLEDGED,
    "AMO REQ RECEIVED": OrderState.ACKNOWLEDGED,
    "VALIDATION PENDING": OrderState.SUBMITTED,
    "PUT ORDER REQ RECEIVED": OrderState.SUBMITTED,
    "OPEN PENDING": OrderState.SUBMITTED,
    "MODIFY PENDING": OrderState.ACKNOWLEDGED,
    "CANCEL PENDING": OrderState.CANCEL_PENDING,
    "EXPIRED": OrderState.EXPIRED,
    # The paper venue reports partial fills as "PARTIAL". This map previously
    # covered Kite's vocabulary only, so "PARTIAL" fell through to UNKNOWN —
    # and UNKNOWN is a BLOCKING state. An ordinary partial fill therefore
    # halted trading until someone reconciled it by hand.
    #
    # Defaulting an unrecognised status to UNKNOWN is correct and stays that
    # way; the gap was that a venue we actually trade against was missing from
    # the vocabulary.
    "PARTIAL": OrderState.PARTIALLY_FILLED,
    "PARTIALLY FILLED": OrderState.PARTIALLY_FILLED,
    "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
}


def map_broker_status(
    status: Any,
    filled_qty: int = 0,
    order_qty: int = 0,
) -> OrderState:
    """
    Translate a broker status string into a lifecycle state.

    An unrecognised / missing status is ``UNKNOWN`` — never a success.
    A partially-filled open order is ``PARTIALLY_FILLED``, not ``ACKNOWLEDGED``.
    """
    if not isinstance(status, str) or not status.strip():
        return OrderState.UNKNOWN
    state = _BROKER_STATUS_MAP.get(status.strip().upper())
    if state is None:
        return OrderState.UNKNOWN
    if state is OrderState.ACKNOWLEDGED and filled_qty > 0:
        return OrderState.PARTIALLY_FILLED
    if state is OrderState.FILLED and 0 < order_qty and 0 < filled_qty < order_qty:
        # Broker claims COMPLETE but reports fewer shares than requested.
        return OrderState.PARTIALLY_FILLED
    return state


# --------------------------------------------------------------------------- #
#  Client order ids                                                            #
# --------------------------------------------------------------------------- #

def strategy_tag_prefix(strategy: str, length: int = STRATEGY_PREFIX_LEN) -> str:
    """Leading slice of the strategy name embedded in every tag."""
    return (strategy or "unknown")[:length]


def make_client_order_id(strategy: str = "") -> str:
    """
    Generate a unique, broker-tag-safe client order id (<= 20 chars).

    Layout: ``<strategy prefix (<=8)><uuid4 hex>`` padded/truncated to 20 chars
    so the whole id survives Kite's 20-character tag truncation intact.
    """
    prefix = strategy_tag_prefix(strategy)
    nonce = uuid.uuid4().hex
    return (prefix + nonce)[:TAG_MAX_LEN]


def deterministic_client_order_id(*parts: Any, prefix: str = "FLAT") -> str:
    """
    Deterministic client order id, used where the *same* logical action must
    produce the *same* id (emergency flatten), so a repeat is de-duplicated.
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return (prefix + digest)[:TAG_MAX_LEN]


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
#  Order record                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class OrderRecord:
    """
    The durable record of one order intent, from creation to a terminal state.

    This is written to the store *before* anything is sent to a broker, so a
    crash between "reserve" and "submit" is always recoverable.
    """

    client_order_id: str
    symbol: str
    exchange: str
    side: str                       # BUY / SELL
    qty: int
    order_type: str                 # MARKET / LIMIT / SL / SL-M
    product: str
    strategy: str = ""
    limit_price: float = 0.0
    trigger_price: float = 0.0
    reference_price: float = 0.0    # price used for the risk/capital arithmetic
    state: OrderState = OrderState.INTENT_CREATED
    broker_order_id: Optional[str] = None
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    reason: str = ""
    intent_kind: str = "entry"      # entry | exit | flatten
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    history: list[dict] = field(default_factory=list)
    applied_fill_ids: list[str] = field(default_factory=list)

    # -- state ---------------------------------------------------------- #

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def is_blocked(self) -> bool:
        """True while the broker outcome for this order is ambiguous."""
        return self.state in BLOCKED_STATES

    @property
    def tag(self) -> str:
        """The broker tag — identical to the client order id."""
        return self.client_order_id

    def transition(
        self,
        new_state: OrderState,
        *,
        reason: str = "",
        reconciled: bool = False,
    ) -> "OrderRecord":
        assert_transition(self.state, new_state, reconciled=reconciled)
        self.history.append({
            "from": self.state.value,
            "to": new_state.value,
            "reason": reason,
            "at": _utcnow(),
            "reconciled": reconciled,
        })
        self.state = new_state
        if reason:
            self.reason = reason
        self.updated_at = _utcnow()
        return self

    # -- fills ---------------------------------------------------------- #

    def apply_fill(self, qty: int, price: float, fill_id: str) -> bool:
        """
        Apply one fill event exactly once.

        Returns False (and changes nothing) when this ``fill_id`` was already
        applied — that is what makes duplicate fill messages harmless.
        """
        if fill_id in self.applied_fill_ids:
            return False
        if qty <= 0 or not math.isfinite(price):
            raise ValueError(f"Invalid fill qty={qty!r} price={price!r}")
        new_total = self.filled_qty + qty
        if new_total > self.qty:
            raise ValueError(
                f"Fill would over-fill order {self.client_order_id}: "
                f"{self.filled_qty}+{qty} > {self.qty}"
            )
        notional = self.avg_fill_price * self.filled_qty + price * qty
        self.filled_qty = new_total
        self.avg_fill_price = notional / new_total if new_total else 0.0
        self.applied_fill_ids.append(fill_id)
        self.updated_at = _utcnow()
        return True

    @property
    def remaining_qty(self) -> int:
        return max(0, self.qty - self.filled_qty)

    # -- serialisation --------------------------------------------------- #

    def to_dict(self) -> dict:
        d = {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "side": self.side,
            "qty": self.qty,
            "order_type": self.order_type,
            "product": self.product,
            "strategy": self.strategy,
            "limit_price": self.limit_price,
            "trigger_price": self.trigger_price,
            "reference_price": self.reference_price,
            "state": self.state.value,
            "broker_order_id": self.broker_order_id,
            "filled_qty": self.filled_qty,
            "avg_fill_price": self.avg_fill_price,
            "reason": self.reason,
            "intent_kind": self.intent_kind,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history": list(self.history),
            "applied_fill_ids": list(self.applied_fill_ids),
        }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "OrderRecord":
        data = dict(data)
        data["state"] = OrderState(data.get("state", OrderState.UNKNOWN.value))
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: Any) -> "OrderRecord":
        if isinstance(raw, bytes):
            raw = raw.decode()
        return cls.from_dict(json.loads(raw))


# --------------------------------------------------------------------------- #
#  Persistence protocol                                                        #
# --------------------------------------------------------------------------- #

@runtime_checkable
class OrderStore(Protocol):
    """
    The narrow durable surface the order manager requires.

    ``reserve`` MUST be atomic set-if-not-exists — that single property is what
    prevents a retry from placing a second live order.
    """

    async def reserve(self, record: OrderRecord) -> bool:
        """Persist ``record`` iff its client_order_id is unclaimed."""
        ...

    async def get(self, client_order_id: str) -> Optional[OrderRecord]:
        ...

    async def save(self, record: OrderRecord) -> None:
        """Durably overwrite an already-reserved record."""
        ...

    async def list_open(self) -> list[OrderRecord]:
        """Every record not in a terminal state (used for crash recovery)."""
        ...

    async def record_trade(self, client_order_id: str, fill: dict) -> bool:
        """Append a fill; returns False if this fill_id was already recorded."""
        ...

    async def apply_position_delta(
        self, symbol: str, exchange: str, product: str,
        qty_delta: int, price: float,
    ) -> dict:
        ...

    async def get_position(self, symbol: str, exchange: str, product: str) -> dict:
        ...


async def _maybe_await(value: Any) -> Any:
    """Support both sync (redis-py) and async (aioredis) clients."""
    if inspect.isawaitable(value):
        return await value
    return value


def _position_key(symbol: str, exchange: str, product: str) -> str:
    return f"{exchange}:{symbol}:{product}"


class InMemoryOrderStore:
    """
    Reference :class:`OrderStore` implementation.

    Durable only for the lifetime of the process — it exists so tests and
    single-process dev runs have a *real* store rather than a silent no-op.
    Anything that survives a restart must use :class:`RedisOrderStore` or an
    equivalent DB-backed implementation.
    """

    durable = False

    def __init__(self) -> None:
        self._records: dict[str, str] = {}
        self._trades: dict[str, list[dict]] = {}
        self._trade_ids: set[str] = set()
        self._positions: dict[str, dict] = {}

    async def reserve(self, record: OrderRecord) -> bool:
        if record.client_order_id in self._records:
            return False
        self._records[record.client_order_id] = record.to_json()
        return True

    async def get(self, client_order_id: str) -> Optional[OrderRecord]:
        raw = self._records.get(client_order_id)
        return OrderRecord.from_json(raw) if raw is not None else None

    async def save(self, record: OrderRecord) -> None:
        if record.client_order_id not in self._records:
            raise PersistenceError(
                f"save() before reserve() for {record.client_order_id}"
            )
        self._records[record.client_order_id] = record.to_json()

    async def list_open(self) -> list[OrderRecord]:
        out = []
        for raw in list(self._records.values()):
            rec = OrderRecord.from_json(raw)
            if not rec.is_terminal:
                out.append(rec)
        return out

    async def list_all(self) -> list[OrderRecord]:
        return [OrderRecord.from_json(r) for r in list(self._records.values())]

    async def record_trade(self, client_order_id: str, fill: dict) -> bool:
        fill_id = str(fill.get("fill_id") or "")
        if not fill_id:
            raise ValueError("fill must carry a fill_id")
        if fill_id in self._trade_ids:
            return False
        self._trade_ids.add(fill_id)
        self._trades.setdefault(client_order_id, []).append(dict(fill))
        return True

    async def list_trades(self, client_order_id: str) -> list[dict]:
        return list(self._trades.get(client_order_id, []))

    async def apply_position_delta(
        self, symbol: str, exchange: str, product: str,
        qty_delta: int, price: float,
    ) -> dict:
        key = _position_key(symbol, exchange, product)
        pos = self._positions.setdefault(
            key,
            {"symbol": symbol, "exchange": exchange, "product": product,
             "quantity": 0, "average_price": 0.0},
        )
        old_qty = pos["quantity"]
        new_qty = old_qty + qty_delta
        if old_qty >= 0 and qty_delta > 0:      # adding to a long
            notional = pos["average_price"] * old_qty + price * qty_delta
            pos["average_price"] = notional / new_qty if new_qty else 0.0
        elif new_qty == 0:
            pos["average_price"] = 0.0
        pos["quantity"] = new_qty
        return dict(pos)

    async def get_position(self, symbol: str, exchange: str, product: str) -> dict:
        key = _position_key(symbol, exchange, product)
        return dict(self._positions.get(
            key,
            {"symbol": symbol, "exchange": exchange, "product": product,
             "quantity": 0, "average_price": 0.0},
        ))


class RedisOrderStore:
    """
    Redis-backed :class:`OrderStore`.

    ``reserve`` uses ``SET key value NX EX ttl`` — a genuine atomic
    set-if-not-exists.  A client that does not support ``nx`` is rejected at
    construction time: silently degrading to a non-atomic check-then-set is
    exactly the race this class exists to prevent.

    Every store failure raises :class:`PersistenceError`.  Callers must treat
    that as "do not submit" — never as "carry on".
    """

    durable = True

    def __init__(self, client, *, namespace: str = "order", ttl_sec: int = 7 * 24 * 3600):
        if client is None:
            raise ValueError("RedisOrderStore requires a redis client.")
        for method in ("set", "get"):
            if not hasattr(client, method):
                raise ValueError(f"redis client lacks required method .{method}()")
        self._c = client
        self._ns = namespace
        self._ttl = ttl_sec

    # -- keys ------------------------------------------------------------ #

    def _key(self, cid: str) -> str:
        return f"{self._ns}:rec:{cid}"

    def _trade_key(self, fill_id: str) -> str:
        return f"{self._ns}:fill:{fill_id}"

    def _pos_key(self, symbol: str, exchange: str, product: str) -> str:
        return f"{self._ns}:pos:{_position_key(symbol, exchange, product)}"

    # -- protocol -------------------------------------------------------- #

    async def reserve(self, record: OrderRecord) -> bool:
        try:
            ok = await _maybe_await(
                self._c.set(self._key(record.client_order_id), record.to_json(),
                            nx=True, ex=self._ttl)
            )
        except TypeError as exc:
            raise PersistenceError(
                "redis client does not support atomic SET NX; refusing to "
                "fall back to a racy check-then-set."
            ) from exc
        except Exception as exc:
            raise PersistenceError(f"order reservation failed: {exc}") from exc
        return bool(ok)

    async def get(self, client_order_id: str) -> Optional[OrderRecord]:
        try:
            raw = await _maybe_await(self._c.get(self._key(client_order_id)))
        except Exception as exc:
            raise PersistenceError(f"order read failed: {exc}") from exc
        return OrderRecord.from_json(raw) if raw is not None else None

    async def save(self, record: OrderRecord) -> None:
        try:
            await _maybe_await(
                self._c.set(self._key(record.client_order_id), record.to_json(),
                            ex=self._ttl)
            )
        except Exception as exc:
            raise PersistenceError(f"order write failed: {exc}") from exc

    async def _iter_keys(self, pattern: str) -> list[str]:
        if hasattr(self._c, "scan_iter"):
            out = []
            it = self._c.scan_iter(match=pattern)
            if inspect.isasyncgen(it):
                async for k in it:                      # pragma: no cover
                    out.append(k)
            else:
                for k in it:
                    out.append(k)
            return out
        if hasattr(self._c, "keys"):
            return list(await _maybe_await(self._c.keys(pattern)) or [])
        raise PersistenceError(
            "redis client supports neither scan_iter nor keys; cannot recover "
            "open orders after a restart."
        )

    async def list_open(self) -> list[OrderRecord]:
        try:
            keys = await self._iter_keys(f"{self._ns}:rec:*")
        except PersistenceError:
            raise
        except Exception as exc:
            raise PersistenceError(f"order scan failed: {exc}") from exc
        out: list[OrderRecord] = []
        for k in keys:
            key = k.decode() if isinstance(k, bytes) else k
            try:
                raw = await _maybe_await(self._c.get(key))
            except Exception as exc:
                raise PersistenceError(f"order read failed: {exc}") from exc
            if raw is None:
                continue
            rec = OrderRecord.from_json(raw)
            if not rec.is_terminal:
                out.append(rec)
        return out

    async def record_trade(self, client_order_id: str, fill: dict) -> bool:
        fill_id = str(fill.get("fill_id") or "")
        if not fill_id:
            raise ValueError("fill must carry a fill_id")
        payload = json.dumps({**fill, "client_order_id": client_order_id},
                             separators=(",", ":"), default=str)
        try:
            ok = await _maybe_await(
                self._c.set(self._trade_key(fill_id), payload, nx=True, ex=self._ttl)
            )
        except Exception as exc:
            raise PersistenceError(f"trade write failed: {exc}") from exc
        return bool(ok)

    async def apply_position_delta(
        self, symbol: str, exchange: str, product: str,
        qty_delta: int, price: float,
    ) -> dict:
        key = self._pos_key(symbol, exchange, product)
        try:
            raw = await _maybe_await(self._c.get(key))
            pos = json.loads(raw) if raw else {
                "symbol": symbol, "exchange": exchange, "product": product,
                "quantity": 0, "average_price": 0.0,
            }
            old_qty = pos["quantity"]
            new_qty = old_qty + qty_delta
            if old_qty >= 0 and qty_delta > 0:
                notional = pos["average_price"] * old_qty + price * qty_delta
                pos["average_price"] = notional / new_qty if new_qty else 0.0
            elif new_qty == 0:
                pos["average_price"] = 0.0
            pos["quantity"] = new_qty
            await _maybe_await(
                self._c.set(key, json.dumps(pos, separators=(",", ":")), ex=self._ttl)
            )
        except Exception as exc:
            raise PersistenceError(f"position write failed: {exc}") from exc
        return dict(pos)

    async def get_position(self, symbol: str, exchange: str, product: str) -> dict:
        try:
            raw = await _maybe_await(self._c.get(self._pos_key(symbol, exchange, product)))
        except Exception as exc:
            raise PersistenceError(f"position read failed: {exc}") from exc
        if not raw:
            return {"symbol": symbol, "exchange": exchange, "product": product,
                    "quantity": 0, "average_price": 0.0}
        return json.loads(raw)


__all__ = [
    "BLOCKED_STATES",
    "IllegalStateTransition",
    "InMemoryOrderStore",
    "LEGAL_TRANSITIONS",
    "LifecycleError",
    "OPEN_STATES",
    "OrderBlockedError",
    "OrderRecord",
    "OrderState",
    "OrderStore",
    "PersistenceError",
    "RedisOrderStore",
    "STRATEGY_PREFIX_LEN",
    "TAG_MAX_LEN",
    "TERMINAL_STATES",
    "assert_transition",
    "deterministic_client_order_id",
    "make_client_order_id",
    "map_broker_status",
    "strategy_tag_prefix",
]
