"""
Durable record of everything the execution path does.

WHERE THE TRUTH LIVES
---------------------
Two different questions have two different authorities, and conflating them is
how trading systems lose money:

* **What actually happened at the venue** — the BROKER is authoritative. This
  module never invents a fill, never assumes an order filled because it was
  accepted, and never writes a quantity the broker did not report.
* **What we intended, and what we have observed** — THIS DATABASE is
  authoritative. It is the only thing that survives the process, and it is what
  reconciliation compares the broker against on the next start.

So the flow is always: submit → ask the broker what it did → persist that.
Never: submit → assume → persist the assumption.

WHY IT LIVES BEHIND ExecutionService
------------------------------------
``ExecutionService.submit_signal`` is the single authoritative order path. This
persistence is invoked from inside it, which means there is no way to place an
order that skips being recorded — a caller cannot forget, because a caller
cannot reach the broker any other way.

FAILURE POLICY
--------------
A persistence failure never silently drops a record and never fabricates
success. Which way it fails depends on when:

* failing to *claim an idempotency key* BLOCKS the order — if we cannot prove
  this order is not a duplicate, we do not send it;
* failing to *record an observed fill* does NOT un-happen the fill, so it is
  logged at ERROR and surfaced on the audit record. The money already moved;
  pretending otherwise would make the discrepancy invisible to reconciliation,
  which is precisely the thing that must find it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.database.models import (
    AccountCash,
    Order,
    OrderStateTransition,
    Position,
    ReconciliationRun,
    Trade,
    User,
)

logger = logging.getLogger(__name__)

#: Statuses that mean the broker is finished with the order.
TERMINAL_BROKER_STATUS = frozenset({"COMPLETE", "REJECTED", "CANCELLED", "EXPIRED"})


class PersistenceUnavailable(RuntimeError):
    """
    The durable record could not be read or written.

    Raised rather than returning a falsy value, because "I could not check"
    must never be mistaken for "I checked and it was fine".
    """


@dataclass
class SubmissionRecord:
    """What persistence knows about one order after the submission attempt."""

    order_row_id: Optional[int] = None
    client_order_id: Optional[str] = None
    #: True when this client_order_id was ALREADY present — i.e. this is a
    #: retry or a concurrent duplicate, and no second order may be sent.
    duplicate: bool = False
    error: Optional[str] = None


@dataclass
class FillSyncResult:
    """Outcome of syncing one order's broker-reported state into the database."""

    status: Optional[str] = None
    filled_quantity: int = 0
    average_fill_price: Optional[float] = None
    new_fills: int = 0
    realized_pnl: float = 0.0
    cash_after: Optional[float] = None
    #: The venue's OWN refusal reason, when it refused. Distinct from our
    #: pre-trade checks — those never ran for an order the broker turned down,
    #: so without this the audit trail cannot say why it was rejected.
    broker_reject_reason: Optional[str] = None
    errors: list[str] = field(default_factory=list)


class ExecutionPersistence(Protocol):
    """The narrow durable surface the execution boundary requires."""

    async def claim_idempotency_key(
        self, client_order_id: str, *, intent: Any, quantity: int, mode: str
    ) -> SubmissionRecord: ...

    async def attach_broker_order_id(
        self, order_row_id: int, broker_order_id: str
    ) -> None: ...

    async def sync_from_broker(
        self, order_row_id: int, broker: Any, broker_order_id: str, *, mode: str
    ) -> FillSyncResult: ...

    async def record_reconciliation(
        self, *, mode: str, state: str, trading_permitted: bool,
        discrepancies: Any = None, unavailable: Any = None, detail: str = "",
    ) -> None: ...


# --------------------------------------------------------------------------- #
#  SQLAlchemy implementation                                                    #
# --------------------------------------------------------------------------- #

class SqlAlchemyExecutionPersistence:
    """
    Production persistence, backed by the application database.

    ``session_factory`` is an async-session factory (not a session): every
    operation opens its own short transaction. A single long-lived session
    shared across concurrent submissions would serialise them and, worse, let
    one failed flush poison another order's transaction.
    """

    def __init__(self, session_factory: Any, *, user_id: int) -> None:
        if session_factory is None:
            raise ValueError("SqlAlchemyExecutionPersistence requires a session factory")
        self._sf = session_factory
        self._user_id = int(user_id)

    # -- idempotency ---------------------------------------------------- #

    async def claim_idempotency_key(
        self, client_order_id: str, *, intent: Any, quantity: int, mode: str
    ) -> SubmissionRecord:
        """
        Insert the order row, claiming ``client_order_id`` exclusively.

        ``quantity`` is passed separately because the order intent does not
        carry it — ``OrderManager.submit_order`` takes the size as its own
        argument, so reading it off the intent would silently record every
        order as zero.

        The claim is the UNIQUE constraint on (user_id, client_order_id), not a
        SELECT-then-INSERT. That distinction is the whole point: two workers
        racing on the same signal both pass a SELECT check, but only one can
        win the INSERT. The loser is told it is a duplicate and must not send
        a second order to the exchange.
        """
        if not client_order_id:
            raise PersistenceUnavailable(
                "refusing to record an order with no client_order_id: without "
                "one, an ambiguous submission cannot be resolved and a retry "
                "cannot be told apart from a new order."
            )
        try:
            async with self._sf() as session:
                row = Order(
                    user_id=self._user_id,
                    client_order_id=client_order_id,
                    order_id_broker=None,
                    symbol=str(getattr(intent, "symbol", "") or ""),
                    exchange=str(getattr(intent, "exchange", "NSE") or "NSE"),
                    transaction_type=_side(intent),
                    quantity=int(quantity),
                    price=_opt_float(getattr(intent, "price", None)),
                    order_type=_enum_value(getattr(intent, "order_type", "MARKET")),
                    product=_enum_value(getattr(intent, "product", None)),
                    status="INTENT_CREATED",
                    strategy=_opt_str(
                        getattr(intent, "strategy", None)
                        or getattr(intent, "strategy_name", None)
                    ),
                    filled_quantity=0,
                )
                session.add(row)
                try:
                    await session.flush()
                except IntegrityError:
                    await session.rollback()
                    existing = await self._find_by_client_order_id(session, client_order_id)
                    logger.warning(
                        "DUPLICATE submission suppressed: client_order_id=%s already "
                        "exists (row id=%s). No second order will be sent.",
                        client_order_id, getattr(existing, "id", None),
                    )
                    return SubmissionRecord(
                        order_row_id=getattr(existing, "id", None),
                        client_order_id=client_order_id,
                        duplicate=True,
                    )

                session.add(OrderStateTransition(
                    order_id=row.id, from_state=None, to_state="INTENT_CREATED",
                    source="submission", filled_quantity=0,
                ))
                await session.commit()
                return SubmissionRecord(order_row_id=row.id, client_order_id=client_order_id)
        except IntegrityError as exc:  # pragma: no cover - covered by the inner path
            raise PersistenceUnavailable(f"idempotency claim failed: {exc}") from exc
        except Exception as exc:
            # Cannot prove this is not a duplicate -> do not let the order out.
            raise PersistenceUnavailable(f"idempotency claim failed: {exc}") from exc

    async def _find_by_client_order_id(self, session: Any, key: str) -> Optional[Order]:
        res = await session.execute(
            select(Order).where(
                Order.user_id == self._user_id, Order.client_order_id == key
            )
        )
        return res.scalar_one_or_none()

    # -- submission ------------------------------------------------------ #

    async def attach_broker_order_id(self, order_row_id: int, broker_order_id: str) -> None:
        async with self._sf() as session:
            row = await session.get(Order, order_row_id)
            if row is None:
                raise PersistenceUnavailable(f"order row {order_row_id} vanished")
            prev = row.status
            row.order_id_broker = str(broker_order_id)
            row.status = "SUBMITTED"
            session.add(OrderStateTransition(
                order_id=order_row_id, from_state=prev, to_state="SUBMITTED",
                source="submission", filled_quantity=int(row.filled_quantity or 0),
            ))
            await session.commit()

    async def record_block(self, order_row_id: int, state: str, reason: str) -> None:
        """Record that an order never reached the broker, and why."""
        try:
            async with self._sf() as session:
                row = await session.get(Order, order_row_id)
                if row is None:
                    return
                prev = row.status
                row.status = state
                row.rejection_reason = reason[:2000] if reason else None
                session.add(OrderStateTransition(
                    order_id=order_row_id, from_state=prev, to_state=state,
                    reason=reason, source="submission",
                    filled_quantity=int(row.filled_quantity or 0),
                ))
                await session.commit()
        except Exception as exc:  # noqa: BLE001
            logger.error("could not record block for order %s: %s", order_row_id, exc)

    # -- fills ------------------------------------------------------------ #

    async def sync_from_broker(
        self, order_row_id: int, broker: Any, broker_order_id: str, *, mode: str
    ) -> FillSyncResult:
        """
        Ask the broker what happened, then persist exactly that.

        Every number written here comes from the broker's own order status and
        trade book. Nothing is inferred from the fact that submission returned
        without raising — an accepted order is not a filled order.
        """
        out = FillSyncResult()
        try:
            status = await broker.get_order_status(broker_order_id)
        except Exception as exc:  # noqa: BLE001
            # We do not know the outcome. UNKNOWN is a real state and blocks
            # further action on this order until reconciliation resolves it.
            out.errors.append(f"order status unavailable: {exc}")
            await self._record_unknown(order_row_id, str(exc))
            out.status = "UNKNOWN"
            return out

        try:
            all_trades = await broker.get_trades()
        except Exception as exc:  # noqa: BLE001
            out.errors.append(f"trade book unavailable: {exc}")
            all_trades = []

        fills = [t for t in (all_trades or []) if str(t.get("order_id")) == str(broker_order_id)]

        try:
            async with self._sf() as session:
                row = await session.get(Order, order_row_id)
                if row is None:
                    out.errors.append(f"order row {order_row_id} vanished")
                    return out

                already = await self._existing_fill_ids(session, broker_order_id)
                cash = await self._get_or_create_cash(session, mode)

                for seq, fill in enumerate(fills):
                    # The paper venue reports no per-fill id, so identity is
                    # derived deterministically from (order, sequence). Replaying
                    # the same trade book therefore cannot double-count.
                    fill_id = str(fill.get("trade_id") or f"{broker_order_id}:{seq}")
                    if fill_id in already:
                        continue
                    realized = await self._apply_fill(session, row, fill, fill_id, cash)
                    out.realized_pnl += realized or 0.0
                    out.new_fills += 1

                broker_status = str(status.get("status") or "").upper()
                filled_qty = int(status.get("filled_qty") or status.get("filled_quantity") or 0)
                avg_px = _opt_float(status.get("average_price"))

                prev = row.status
                row.filled_quantity = filled_qty
                row.average_fill_price = avg_px
                row.status = broker_status or prev
                reject = status.get("reject_reason") or status.get("message")
                if broker_status == "REJECTED" and reject:
                    row.rejection_reason = str(reject)[:2000]
                if reject:
                    out.broker_reject_reason = str(reject)[:2000]

                if row.status != prev:
                    session.add(OrderStateTransition(
                        order_id=order_row_id, from_state=prev, to_state=row.status,
                        reason=str(reject)[:2000] if reject else None,
                        source="fill", filled_quantity=filled_qty,
                    ))

                await session.commit()
                out.status = row.status
                out.filled_quantity = filled_qty
                out.average_fill_price = avg_px
                out.cash_after = float(cash.cash)
        except Exception as exc:  # noqa: BLE001
            # The fill already happened. Say so loudly rather than hiding it —
            # reconciliation is what must catch this, and it can only do that
            # if the failure is visible.
            logger.exception("FILL RECORDING FAILED for broker order %s", broker_order_id)
            out.errors.append(f"fill persistence failed: {exc}")
        return out

    async def _existing_fill_ids(self, session: Any, broker_order_id: str) -> set[str]:
        res = await session.execute(
            select(Trade.trade_id_broker).where(
                Trade.user_id == self._user_id,
                Trade.trade_id_broker.like(f"{broker_order_id}%"),
            )
        )
        return {r for (r,) in res.all() if r}

    async def _apply_fill(
        self, session: Any, order_row: Order, fill: dict, fill_id: str, cash: AccountCash
    ) -> Optional[float]:
        """
        Book one fill: trade row, position, cash, realised P&L.

        Cash and holdings move together in one transaction. They are the two
        halves of the same event, and a partial application of it would create
        or destroy value.
        """
        qty = int(fill.get("qty") or fill.get("quantity") or 0)
        price = float(fill.get("price") or 0.0)
        side = str(fill.get("txn_type") or fill.get("transaction_type") or "").upper()
        costs = fill.get("costs") or {}
        total_costs = float(costs.get("total") or 0.0)
        value = round(qty * price, 4)

        pos = await self._get_position(session, order_row.symbol, order_row.exchange)
        realized: Optional[float] = None

        if side == "BUY":
            # Opening or adding: nothing is realised, and the weighted average
            # cost moves toward the new price.
            cash.cash = float(cash.cash) - value - total_costs
            if pos is None:
                pos = Position(
                    user_id=self._user_id, symbol=order_row.symbol,
                    exchange=order_row.exchange, quantity=qty, average_price=price,
                    strategy=order_row.strategy or "unknown",
                    entry_date=_now(), is_open=True,
                )
                session.add(pos)
            else:
                prev_qty = int(pos.quantity)
                new_qty = prev_qty + qty
                pos.average_price = (
                    (prev_qty * float(pos.average_price) + qty * price) / new_qty
                    if new_qty else 0.0
                )
                pos.quantity = new_qty
                pos.is_open = new_qty != 0
        elif side == "SELL":
            cash.cash = float(cash.cash) + value - total_costs
            if pos is None or int(pos.quantity) <= 0:
                # The broker says shares were sold that our book does not have.
                # Recorded, not swallowed: this is a genuine divergence and it
                # is reconciliation's job to surface it, not ours to hide.
                logger.error(
                    "SELL fill for %s with no local position — recording the "
                    "trade and leaving the discrepancy for reconciliation.",
                    order_row.symbol,
                )
                realized = None
            else:
                avg = float(pos.average_price)
                realized = round(qty * (price - avg) - total_costs, 4)
                new_qty = int(pos.quantity) - qty
                pos.quantity = max(new_qty, 0)
                pos.is_open = pos.quantity != 0
                cash.realized_pnl = float(cash.realized_pnl) + realized
        else:
            raise PersistenceUnavailable(f"unrecognised fill side {side!r}")

        cash.total_costs = float(cash.total_costs) + total_costs

        session.add(Trade(
            user_id=self._user_id,
            order_id=order_row.id,
            trade_id_broker=fill_id,
            symbol=order_row.symbol,
            exchange=order_row.exchange,
            transaction_type=side,
            quantity=qty,
            price=price,
            value=value,
            brokerage=float(costs.get("brokerage") or 0.0),
            stt=float(costs.get("stt") or 0.0),
            exchange_charges=float(costs.get("exchange") or costs.get("exchange_charges") or 0.0),
            gst=float(costs.get("gst") or 0.0),
            stamp_duty=float(costs.get("stamp_duty") or 0.0),
            sebi_charges=float(costs.get("sebi") or costs.get("sebi_charges") or 0.0),
            total_costs=total_costs,
            net_value=round(value - total_costs if side == "SELL" else value + total_costs, 4),
            strategy=order_row.strategy,
            realized_pnl=realized,
        ))
        return realized

    async def _get_position(self, session: Any, symbol: str, exchange: str) -> Optional[Position]:
        res = await session.execute(
            select(Position).where(
                Position.user_id == self._user_id,
                Position.symbol == symbol,
                Position.exchange == exchange,
                Position.is_open.is_(True),
            )
        )
        return res.scalars().first()

    async def _get_or_create_cash(self, session: Any, mode: str) -> AccountCash:
        res = await session.execute(
            select(AccountCash).where(
                AccountCash.user_id == self._user_id, AccountCash.trading_mode == mode
            )
        )
        row = res.scalar_one_or_none()
        if row is None:
            raise PersistenceUnavailable(
                f"no AccountCash row for user={self._user_id} mode={mode}. "
                "Opening cash must be established explicitly — defaulting it to "
                "zero (or to anything else) would invent an account balance."
            )
        return row

    async def _record_unknown(self, order_row_id: int, reason: str) -> None:
        try:
            async with self._sf() as session:
                row = await session.get(Order, order_row_id)
                if row is None:
                    return
                prev = row.status
                row.status = "UNKNOWN"
                session.add(OrderStateTransition(
                    order_id=order_row_id, from_state=prev, to_state="UNKNOWN",
                    reason=f"broker outcome could not be determined: {reason}"[:2000],
                    source="fill", filled_quantity=int(row.filled_quantity or 0),
                ))
                await session.commit()
        except Exception as exc:  # noqa: BLE001
            logger.error("could not record UNKNOWN for order %s: %s", order_row_id, exc)

    # -- reconciliation --------------------------------------------------- #

    async def record_reconciliation(
        self, *, mode: str, state: str, trading_permitted: bool,
        discrepancies: Any = None, unavailable: Any = None, detail: str = "",
    ) -> None:
        try:
            async with self._sf() as session:
                session.add(ReconciliationRun(
                    user_id=self._user_id,
                    trading_mode=mode,
                    state=state,
                    trading_permitted=bool(trading_permitted),
                    discrepancy_count=len(discrepancies or []),
                    discrepancies=_jsonable(discrepancies),
                    unavailable=_jsonable(unavailable),
                    detail=(detail or "")[:4000] or None,
                ))
                await session.commit()
        except Exception as exc:  # noqa: BLE001
            logger.error("could not persist reconciliation run: %s", exc)


# --------------------------------------------------------------------------- #
#  Bootstrap helpers                                                            #
# --------------------------------------------------------------------------- #

async def get_or_create_system_user(session_factory: Any, *, email: str = "system@algodollar.local") -> int:
    """
    Return the id of the single account this deployment trades for.

    AlgoDollar is a personal platform: there is one trading account. A row
    still exists for it because every order, trade and position is scoped by
    user_id, and leaving that nullable would make the schema lie about a
    multi-account future it does not support today.
    """
    async with session_factory() as session:
        res = await session.execute(select(User).where(User.email == email))
        user = res.scalar_one_or_none()
        if user is not None:
            return int(user.id)
        user = User(
            email=email,
            # Not a login account. "!" is not a valid bcrypt hash, so
            # verify_password can never match it for any input — this row
            # cannot be authenticated into.
            hashed_password="!",
            is_active=True,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return int(user.id)


async def ensure_opening_cash(
    session_factory: Any, *, user_id: int, mode: str, opening_cash: float
) -> float:
    """
    Establish the opening balance for a mode, once.

    Returns the CURRENT balance, which on any run after the first is the
    balance the account actually has — not ``opening_cash``. Resetting it on
    every start would silently erase every loss the account had taken.
    """
    async with session_factory() as session:
        res = await session.execute(
            select(AccountCash).where(
                AccountCash.user_id == user_id, AccountCash.trading_mode == mode
            )
        )
        row = res.scalar_one_or_none()
        if row is not None:
            return float(row.cash)
        row = AccountCash(
            user_id=user_id, trading_mode=mode, cash=float(opening_cash),
            reserved=0.0, realized_pnl=0.0, total_costs=0.0,
        )
        session.add(row)
        await session.commit()
        return float(opening_cash)


# --------------------------------------------------------------------------- #
#  small helpers                                                                #
# --------------------------------------------------------------------------- #

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _enum_value(v: Any) -> Optional[str]:
    if v is None:
        return None
    return str(getattr(v, "value", v))


def _opt_float(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _opt_str(v: Any) -> Optional[str]:
    return None if v is None else str(v)


def _side(intent: Any) -> str:
    raw = _enum_value(getattr(intent, "txn_type", None) or getattr(intent, "direction", None))
    mapped = {"LONG": "BUY", "SHORT": "SELL", "EXIT": "SELL"}.get(str(raw).upper(), str(raw).upper())
    return mapped[:4] if mapped else "BUY"


def _jsonable(v: Any) -> Any:
    """Best-effort JSON coercion; never raises, never silently drops content."""
    if v is None:
        return None
    try:
        import json
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return [str(x) for x in v] if isinstance(v, (list, tuple, set)) else str(v)
