"""
paper.py — paper-trading broker with conservative, auditable accounting.

WHY THIS FILE WAS REWRITTEN
---------------------------
The previous implementation was unusable as a pre-live validation stage:

  D1  A SELL credited cash unconditionally but only touched a position
      ``if pos is not None``.  Selling stock never owned minted ~Rs 2.9 m of
      cash and left ``get_positions() == []``.  Overselling left a phantom
      negative quantity.
  D2  For every non-MARKET order ``market_price = price``, so a limit BUY at
      Rs 1 filled at Rs 1.0005 against a Rs 2,900 market.
  D3  ``fill_probability = min(1, volume*0.05/qty)`` is >= 1 at any realistic
      volume, so 100/100 orders filled in full; unknown volume also filled.
  D4  No square-off, no rejection path except cash, flat 5 bps slippage.
  D5  Positions were marked at cost, so ``return_pct`` ignored unrealised P&L.
  D6  Corrupt Redis state was swallowed and the account silently reset to
      full initial cash, erasing losses.

DESIGN RULES APPLIED HERE
-------------------------
1.  **Double-entry cash.**  Cash is held as an integer number of paise and is
    re-derived from the immutable trade ledger after every mutation
    (:meth:`PaperBroker._assert_invariants`).  If the running balance and the
    ledger ever disagree the broker raises rather than continuing.
2.  **No short selling.**  See :data:`SHORT_SELLING_SUPPORTED` below for the
    decision and its rationale.  A sell larger than the free holding is
    REJECTED; cash is never created out of nothing.
3.  **Never unrealistically favourable.**  Every modelled quantity (spread,
    impact, tick rounding, fill quantity, liquidity when unknown) is rounded
    against the trader.  Places where a guess was required are marked with a
    ``PESSIMISTIC:`` comment.
4.  **Deterministic fills.**  There is no RNG anywhere in the execution model.
    A fill is a pure function of (order, market snapshot, configuration), so a
    paper run is reproducible and can be diffed against a backtest.
5.  **One cost model.**  Costs come from
    :class:`app.backtesting.costs.ZerodhaCostModel` — the same object the
    backtester uses.  No second cost model is defined in this file.
6.  **Fail closed.**  Unreadable, corrupt or tampered persisted state raises
    :class:`PaperBrokerStateError` from :meth:`connect` instead of resetting
    the account.  A failed *save* degrades the broker to reject new orders.

KNOWN REMAINING UNREALISM (documented deliberately)
---------------------------------------------------
*  There is no order book queue.  A resting limit order is filled the first
   time :meth:`poll_open_orders` sees a marketable quote; real queue priority
   (you may be behind 50,000 shares at your price) is not modelled.  This is
   the one materially *favourable* simplification left.
*  Liquidity is taken from the quote's cumulative day ``volume``.  Intra-bar
   depth is not modelled.
*  The NSE trading-holiday calendar is a static list (2024-2025).  For a year
   with no calendar the broker falls back to weekday+session checks and logs a
   warning; supply ``holidays=`` to be exact.
*  Corporate actions, auction/circuit limits, and MIS leverage are not
   modelled.  MIS is treated as 1x (fully cash-funded), which is conservative.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from datetime import time as dtime
from pathlib import Path
from typing import Callable, Iterable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from ..backtesting.costs import ZerodhaCostModel
from .base import (
    BrokerInterface,
    OrderType,
    Product,
    TransactionType,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  THE SHORT-SELLING DECISION                                                  #
# --------------------------------------------------------------------------- #

#: Short selling is **deliberately not supported**.
#:
#: Rationale.  In the Indian cash segment a retail account cannot carry a naked
#: short: SEBI prohibits naked short selling, and an intraday short that is not
#: covered by 15:20 is auto-squared-off by the broker or lands in the exchange
#: auction settlement with a penalty that routinely exceeds the trade's edge.
#: Modelling that faithfully means modelling span/exposure margin, mark-to-
#: market margin calls, borrow availability under SLB, buy-to-cover, forced
#: square-off and auction losses.  A half-built version of that is exactly the
#: class of defect this rewrite removes, and it would make paper results look
#: better than live results.  So: a SELL that exceeds the free holding is
#: REJECTED with :data:`RejectReason.INSUFFICIENT_HOLDINGS`.
SHORT_SELLING_SUPPORTED = False


# --------------------------------------------------------------------------- #
#  Market session (IST) — naive datetimes are a hard error                     #
# --------------------------------------------------------------------------- #

IST = ZoneInfo("Asia/Kolkata")

MARKET_OPEN_IST = dtime(9, 15)
MARKET_CLOSE_IST = dtime(15, 30)
#: Zerodha auto-squares off open MIS positions from ~15:20.  Opening a *new*
#: intraday position after this time is rejected.
MIS_SQUAREOFF_IST = dtime(15, 20)

#: NSE trading holidays (equity segment).  Static, must be refreshed yearly.
#: A year absent from this map falls back to weekday+session checks only and
#: logs a warning — see ``KNOWN REMAINING UNREALISM`` in the module docstring.
NSE_TRADING_HOLIDAYS: dict[int, frozenset[date]] = {
    2024: frozenset({
        date(2024, 1, 22), date(2024, 1, 26), date(2024, 3, 8),
        date(2024, 3, 25), date(2024, 3, 29), date(2024, 4, 11),
        date(2024, 4, 17), date(2024, 5, 1), date(2024, 5, 20),
        date(2024, 6, 17), date(2024, 7, 17), date(2024, 8, 15),
        date(2024, 10, 2), date(2024, 11, 1), date(2024, 11, 15),
        date(2024, 12, 25),
    }),
    2025: frozenset({
        date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31),
        date(2025, 4, 10), date(2025, 4, 14), date(2025, 4, 18),
        date(2025, 5, 1), date(2025, 8, 15), date(2025, 8, 27),
        date(2025, 10, 2), date(2025, 10, 21), date(2025, 10, 22),
        date(2025, 11, 5), date(2025, 12, 25),
    }),
}


def ensure_aware(moment: datetime, *, what: str = "timestamp") -> datetime:
    """
    Return ``moment`` as an IST-localised aware datetime.

    Raises ``ValueError`` on a naive datetime.  The platform previously had a
    naive-datetime bug that, on a UTC server, would have judged 15:00 UTC
    (20:30 IST) to be inside the session and carried an intraday book
    overnight.  Naive input is therefore never silently interpreted.
    """
    if not isinstance(moment, datetime):
        raise TypeError(f"{what} must be a datetime, got {type(moment).__name__}")
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        raise ValueError(
            f"{what} is timezone-naive ({moment!r}). PaperBroker refuses naive "
            "datetimes: attach a tzinfo (UTC or Asia/Kolkata) explicitly."
        )
    return moment.astimezone(IST)


# --------------------------------------------------------------------------- #
#  Errors, statuses, reject reasons                                            #
# --------------------------------------------------------------------------- #

class PaperBrokerError(RuntimeError):
    """Base class for paper-broker failures."""


class PaperBrokerStateError(PaperBrokerError):
    """Persisted state is missing, unreadable, corrupt or tampered with."""


class AccountingInvariantError(PaperBrokerError):
    """An accounting invariant was violated — the book is not trustworthy."""


class OrderStatus:
    OPEN = "OPEN"            # accepted, resting, not (yet) filled
    PARTIAL = "PARTIAL"      # partly filled, remainder cancelled (IOC-style)
    COMPLETE = "COMPLETE"    # fully filled
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

    TERMINAL = frozenset({PARTIAL, COMPLETE, CANCELLED, REJECTED})


class RejectReason:
    NOT_CONNECTED = "NOT_CONNECTED"
    INVALID_QUANTITY = "INVALID_QUANTITY"
    INVALID_PRICE = "INVALID_PRICE"
    MARKET_CLOSED = "MARKET_CLOSED"
    SQUARE_OFF_WINDOW = "SQUARE_OFF_WINDOW"
    NO_PRICE = "NO_PRICE"
    STALE_PRICE = "STALE_PRICE"
    NO_LIQUIDITY_DATA = "NO_LIQUIDITY_DATA"
    NO_LIQUIDITY = "NO_LIQUIDITY"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    INSUFFICIENT_HOLDINGS = "INSUFFICIENT_HOLDINGS"
    SHORT_SELL_NOT_SUPPORTED = "SHORT_SELL_NOT_SUPPORTED"
    PERSISTENCE_DEGRADED = "PERSISTENCE_DEGRADED"
    UNSUPPORTED_PRODUCT = "UNSUPPORTED_PRODUCT"


# --------------------------------------------------------------------------- #
#  Money helpers — cash is integer paise, never a float                        #
# --------------------------------------------------------------------------- #

def to_paise(rupees: float) -> int:
    """Round rupees to whole paise (banker-free, half-away-from-zero)."""
    return int(math.floor(float(rupees) * 100 + 0.5)) if rupees >= 0 else -int(
        math.floor(-float(rupees) * 100 + 0.5)
    )


def to_rupees(paise: int) -> float:
    return round(paise / 100.0, 2)


def _floor_tick(price: float, tick: float) -> float:
    return round(math.floor(round(price / tick, 9)) * tick, 2)


def _ceil_tick(price: float, tick: float) -> float:
    return round(math.ceil(round(price / tick, 9)) * tick, 2)


# --------------------------------------------------------------------------- #
#  Cost model — the SAME object the backtester uses.  Do not add a second one. #
# --------------------------------------------------------------------------- #

_DEFAULT_COST_MODEL = ZerodhaCostModel()

#: ZerodhaCostModel understands MIS / CNC / NRML.  Cover and bracket orders are
#: intraday products, so they are charged as MIS.
_PRODUCT_TO_COST_PRODUCT = {
    Product.MIS: "MIS",
    Product.CNC: "CNC",
    Product.NRML: "NRML",
    Product.CO: "MIS",
    Product.BO: "MIS",
}

_INTRADAY_PRODUCTS = frozenset({Product.MIS, Product.CO, Product.BO})


def compute_transaction_costs(
    symbol: str,
    qty: int,
    price: float,
    txn_type: TransactionType,
    product: Product,
    exchange: str = "NSE",
    cost_model: ZerodhaCostModel | None = None,
) -> dict[str, float]:
    """
    Backwards-compatible shim that **delegates** to ``ZerodhaCostModel``.

    A previous audit found the backtester and the live path using divergent
    cost logic (a Rs 240 gap per trade).  This function exists only so old call
    sites keep working; it defines no rates of its own.
    """
    model = cost_model or _DEFAULT_COST_MODEL
    tx = txn_type.value if isinstance(txn_type, TransactionType) else str(txn_type)
    breakdown = model.calculate_costs(
        transaction_type=tx,
        qty=qty,
        price=price,
        exchange=exchange,
        product=_PRODUCT_TO_COST_PRODUCT.get(product, "MIS"),
    )
    return {
        "brokerage": breakdown.brokerage,
        "stt": breakdown.stt,
        "exchange_charges": breakdown.exchange_charge,
        "sebi": breakdown.sebi_charge,
        "gst": breakdown.gst,
        "stamp_duty": breakdown.stamp_duty,
        "dp_charges": breakdown.dp_charges,
        "total": breakdown.total,
    }


# --------------------------------------------------------------------------- #
#  Data structures                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class MarketSnapshot:
    """A point-in-time view of one instrument, with a modelled touch."""
    symbol: str
    exchange: str
    last_price: float
    bid: float
    ask: float
    volume: int                 # cumulative traded quantity for the session
    open: float
    high: float
    low: float
    prev_close: float
    half_spread_bps: float
    age_sec: Optional[float]    # None => the feed supplied no timestamp

    @property
    def liquidity_known(self) -> bool:
        return self.volume > 0


@dataclass
class PaperOrder:
    order_id: str
    symbol: str
    exchange: str
    txn_type: str
    qty: int
    price: float                       # limit price, or trigger for SL/SL-M
    order_type: str
    product: str
    tag: str
    status: str = OrderStatus.OPEN
    filled_qty: int = 0
    average_price: float = 0.0
    placed_at: str = ""
    filled_at: Optional[str] = None
    reject_reason: Optional[str] = None
    message: str = ""
    costs: dict = field(default_factory=dict)
    reserved_cash_paise: int = 0
    reserved_qty: int = 0


@dataclass
class PaperPosition:
    symbol: str
    exchange: str
    product: str
    quantity: int
    #: Volume-weighted execution price, excluding charges (Zerodha convention).
    average_price: float
    #: Full economic cost basis of the open quantity, in paise, INCLUDING the
    #: buy-side charges attributable to it.  Used for realised P&L.
    cost_basis_paise: int = 0
    last_price: float = 0.0
    realised: float = 0.0
    unrealised: float = 0.0
    pnl: float = 0.0

    @property
    def key(self) -> str:
        return position_key(self.symbol, self.exchange, self.product)


def position_key(symbol: str, exchange: str, product: str | Product) -> str:
    """
    Position identity.  The old code keyed on ``symbol:product`` only, so an
    NSE and a BSE line in the same scrip collided into one position.
    """
    prod = product.value if isinstance(product, Product) else str(product)
    return f"{exchange.upper()}:{symbol.upper()}:{prod}"


# --------------------------------------------------------------------------- #
#  PaperBroker                                                                 #
# --------------------------------------------------------------------------- #

class PaperBroker(BrokerInterface):
    """
    Paper-trading broker with conservative accounting and realistic execution.

    Parameters
    ----------
    data_broker
        A connected broker used only as a price feed.
    initial_cash
        Opening balance in rupees.
    slippage_pct
        *Floor* on modelled slippage (default 5 bps).  Retained for backwards
        compatibility; real slippage is ``max(slippage_pct, impact(size))`` so
        this broker is never cheaper than the old flat model.
    impact_coef
        Coefficient of the square-root market-impact law
        ``impact = impact_coef * sqrt(qty / session_volume)``.
    max_participation
        Fraction of the session's traded volume a single order may consume.
        Anything above this becomes a partial fill.
    tick_size
        Minimum price increment.  Fills are rounded *against* the trader.
    enforce_market_hours
        Reject orders outside the IST session / on holidays.
    clock
        Callable returning an aware ``datetime``.  Injected in tests.
    redis_client / state_path
        Where state is persisted.  Corrupt state fails closed.
    allow_state_reset
        Escape hatch: when True a corrupt state is logged and discarded instead
        of raising.  Requires an explicit human decision; default False.
    """

    REDIS_KEY_PREFIX = "paper_broker:"
    STATE_SCHEMA_VERSION = 2

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        data_broker: BrokerInterface,
        initial_cash: float = 1_000_000.0,
        slippage_pct: float = 0.0005,
        redis_client=None,
        account_id: str = "default",
        *,
        cost_model: ZerodhaCostModel | None = None,
        impact_coef: float = 0.02,
        max_participation: float = 0.10,
        tick_size: float = 0.05,
        max_quote_age_sec: float = 30.0,
        strict_quote_staleness: bool = False,
        enforce_market_hours: bool = True,
        enforce_squareoff_window: bool = True,
        holidays: Optional[Iterable[date]] = None,
        clock: Optional[Callable[[], datetime]] = None,
        state_path: Optional[str | Path] = None,
        allow_state_reset: bool = False,
    ) -> None:
        if initial_cash < 0:
            raise ValueError("initial_cash must be >= 0")
        if not 0 < max_participation <= 1:
            raise ValueError("max_participation must be in (0, 1]")
        if tick_size <= 0:
            raise ValueError("tick_size must be > 0")

        self._data_broker = data_broker
        self._initial_cash_paise = to_paise(initial_cash)
        self._slippage_pct = float(slippage_pct)
        self._impact_coef = float(impact_coef)
        self._max_participation = float(max_participation)
        self._tick = float(tick_size)
        self._max_quote_age_sec = float(max_quote_age_sec)
        self._strict_quote_staleness = bool(strict_quote_staleness)
        self._enforce_market_hours = bool(enforce_market_hours)
        self._enforce_squareoff = bool(enforce_squareoff_window)
        self._cost_model = cost_model or _DEFAULT_COST_MODEL
        self._redis = redis_client
        self._state_path = Path(state_path) if state_path else None
        self._allow_state_reset = bool(allow_state_reset)
        self._account_id = account_id
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._holidays: Optional[frozenset[date]] = (
            frozenset(holidays) if holidays is not None else None
        )

        self._connected = False
        self._persistence_degraded = False

        # --- the book -------------------------------------------------- #
        self._cash_paise: int = self._initial_cash_paise
        self._reserved_cash_paise: int = 0
        self._total_costs_paise: int = 0
        self._orders: dict[str, PaperOrder] = {}
        self._positions: dict[str, PaperPosition] = {}
        self._trades: list[dict] = []
        #: symbol -> monotonic-ish age bookkeeping for the safety layer
        self._last_quote_age: dict[str, Optional[float]] = {}

        self._assert_invariants()

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        """
        Restore persisted state.  **Fails closed**: corrupt or unreadable state
        raises :class:`PaperBrokerStateError` instead of resetting the account
        to full cash (defect D6).
        """
        self._load_state()
        self._assert_invariants()
        self._connected = True
        logger.info(
            "PaperBroker connected | cash=Rs %.2f | reserved=Rs %.2f | positions=%d "
            "| short_selling=%s",
            to_rupees(self._cash_paise),
            to_rupees(self._reserved_cash_paise),
            len(self._positions),
            SHORT_SELLING_SUPPORTED,
        )

    async def disconnect(self) -> None:
        self._save_state()
        self._connected = False
        logger.info("PaperBroker disconnected; state saved.")

    # ------------------------------------------------------------------ #
    #  Market session                                                      #
    # ------------------------------------------------------------------ #

    def now_ist(self) -> datetime:
        """Current time as an aware IST datetime.  Raises on a naive clock."""
        return ensure_aware(self._clock(), what="clock()")

    def _holidays_for(self, year: int) -> Optional[frozenset[date]]:
        if self._holidays is not None:
            return self._holidays
        return NSE_TRADING_HOLIDAYS.get(year)

    def is_trading_holiday(self, moment: Optional[datetime] = None) -> bool:
        now = ensure_aware(moment) if moment is not None else self.now_ist()
        cal = self._holidays_for(now.year)
        if cal is None:
            logger.warning(
                "No NSE holiday calendar for %d; falling back to weekday checks "
                "only. Pass holidays= for an exact calendar.", now.year,
            )
            return False
        return now.date() in cal

    def is_market_open(self, moment: Optional[datetime] = None) -> bool:
        """
        True when the NSE equity session is open, evaluated in IST.

        ``moment`` must be timezone-aware; a naive datetime raises ``ValueError``
        rather than being guessed at.  This is what makes the behaviour identical
        on a ``TZ=UTC`` server and a ``TZ=Asia/Kolkata`` one.
        """
        now = ensure_aware(moment) if moment is not None else self.now_ist()
        if now.weekday() >= 5:                      # Sat/Sun
            return False
        if self.is_trading_holiday(now):
            return False
        return MARKET_OPEN_IST <= now.timetz().replace(tzinfo=None) < MARKET_CLOSE_IST

    def is_squareoff_time(self, moment: Optional[datetime] = None) -> bool:
        """True inside the intraday auto-square-off window (>= 15:20 IST)."""
        now = ensure_aware(moment) if moment is not None else self.now_ist()
        return now.timetz().replace(tzinfo=None) >= MIS_SQUAREOFF_IST

    def is_stale_tick(self, symbol: str, max_age_seconds: float = 30.0) -> bool:
        """
        Staleness hook probed by ``ExecutionSafety.check_data_freshness``.

        PESSIMISTIC: a symbol that has never been quoted is stale.
        """
        if symbol not in self._last_quote_age:
            return True
        age = self._last_quote_age[symbol]
        if age is None:
            # Feed supplies no timestamp: unverifiable.  Strict mode calls it
            # stale; permissive mode allows it (documented in the docstring).
            return self._strict_quote_staleness
        return age > max_age_seconds

    # ------------------------------------------------------------------ #
    #  Persistence — fail closed                                           #
    # ------------------------------------------------------------------ #

    def _key(self, suffix: str) -> str:
        return f"{self.REDIS_KEY_PREFIX}{self._account_id}:{suffix}"

    def _serialise(self) -> str:
        body = json.dumps(
            {
                "initial_cash_paise": self._initial_cash_paise,
                "cash_paise": self._cash_paise,
                "reserved_cash_paise": self._reserved_cash_paise,
                "total_costs_paise": self._total_costs_paise,
                "orders": {k: asdict(v) for k, v in self._orders.items()},
                "positions": {k: asdict(v) for k, v in self._positions.items()},
                "trades": self._trades,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        envelope = {
            "schema": self.STATE_SCHEMA_VERSION,
            "checksum": hashlib.sha256(body.encode()).hexdigest(),
            "body": body,
        }
        return json.dumps(envelope, separators=(",", ":"))

    def _deserialise(self, raw: str | bytes) -> None:
        """Parse, verify and install persisted state, or raise."""
        if isinstance(raw, bytes):
            raw = raw.decode()
        envelope = json.loads(raw)                       # JSONDecodeError -> caller
        if not isinstance(envelope, dict):
            raise ValueError("state envelope is not an object")
        schema = envelope.get("schema")
        if schema != self.STATE_SCHEMA_VERSION:
            raise ValueError(
                f"state schema {schema!r} != expected {self.STATE_SCHEMA_VERSION}"
            )
        body = envelope.get("body")
        if not isinstance(body, str):
            raise ValueError("state body missing or not a string")
        digest = hashlib.sha256(body.encode()).hexdigest()
        if digest != envelope.get("checksum"):
            raise ValueError("state checksum mismatch — the book was modified")

        data = json.loads(body)
        cash = data["cash_paise"]
        reserved = data["reserved_cash_paise"]
        if not isinstance(cash, int) or not isinstance(reserved, int):
            raise ValueError("cash fields must be integer paise")

        orders = {k: PaperOrder(**v) for k, v in data["orders"].items()}
        positions = {k: PaperPosition(**v) for k, v in data["positions"].items()}

        self._initial_cash_paise = int(data["initial_cash_paise"])
        self._cash_paise = cash
        self._reserved_cash_paise = reserved
        self._total_costs_paise = int(data["total_costs_paise"])
        self._orders = orders
        self._positions = positions
        self._trades = list(data["trades"])

        # Re-derive cash from the ledger: catches a tampered or truncated book
        # even when the checksum was recomputed by whoever tampered with it.
        self._assert_invariants()

    def _read_raw_state(self) -> Optional[str]:
        """Read the persisted blob.  An I/O failure is *not* swallowed."""
        if self._redis is not None:
            return self._redis.get(self._key("state"))
        if self._state_path is not None:
            if not self._state_path.exists():
                return None
            return self._state_path.read_text()
        return None

    def _load_state(self) -> None:
        if self._redis is None and self._state_path is None:
            logger.warning(
                "PaperBroker has no redis_client and no state_path: the book "
                "will NOT survive a restart."
            )
            return
        try:
            raw = self._read_raw_state()
        except Exception as exc:                     # noqa: BLE001 - re-raised
            raise PaperBrokerStateError(
                f"Paper account {self._account_id!r}: state store is unreadable "
                f"({exc!r}). Refusing to start with a fabricated balance."
            ) from exc

        if raw is None:
            logger.info("No persisted paper state; starting fresh.")
            return

        try:
            self._deserialise(raw)
        except Exception as exc:                     # noqa: BLE001 - re-raised
            msg = (
                f"Paper account {self._account_id!r}: persisted state is corrupt "
                f"({exc}). Refusing to start — resetting to initial cash would "
                "erase realised losses and invalidate every result."
            )
            if self._allow_state_reset:
                logger.critical("%s  allow_state_reset=True: discarding state.", msg)
                self._reset_book()
                return
            raise PaperBrokerStateError(msg) from exc

        logger.info("Paper state restored: cash=Rs %.2f, positions=%d, trades=%d",
                    to_rupees(self._cash_paise), len(self._positions), len(self._trades))

    def _reset_book(self) -> None:
        self._cash_paise = self._initial_cash_paise
        self._reserved_cash_paise = 0
        self._total_costs_paise = 0
        self._orders = {}
        self._positions = {}
        self._trades = []

    def _save_state(self) -> None:
        if self._redis is None and self._state_path is None:
            return
        try:
            blob = self._serialise()
            if self._redis is not None:
                self._redis.set(self._key("state"), blob)
            else:
                self._state_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
                tmp.write_text(blob)
                tmp.replace(self._state_path)          # atomic
            self._persistence_degraded = False
        except Exception as exc:                       # noqa: BLE001
            # Do not fabricate a rollback of an already-applied fill; instead
            # stop accepting new orders so the divergence cannot grow.
            self._persistence_degraded = True
            logger.critical(
                "PaperBroker CANNOT PERSIST state (%r). New orders will be "
                "rejected until persistence recovers.", exc,
            )

    # ------------------------------------------------------------------ #
    #  Accounting invariants — enforced, not merely asserted in tests      #
    # ------------------------------------------------------------------ #

    def _ledger_cash_paise(self) -> int:
        """Re-derive the cash balance independently, from the trade ledger."""
        cash = self._initial_cash_paise
        for t in self._trades:
            notional = int(t["notional_paise"])
            costs = int(t["costs_paise"])
            if t["txn_type"] == TransactionType.BUY.value:
                cash -= notional
            else:
                cash += notional
            cash -= costs
        return cash

    def _assert_invariants(self) -> None:
        """
        Raise :class:`AccountingInvariantError` if the book is not coherent.

        Called after every mutation.  These are the properties that make paper
        results meaningful; if one breaks, continuing would produce fiction.
        """
        if self._cash_paise < 0:
            raise AccountingInvariantError(
                f"cash went negative: {to_rupees(self._cash_paise)}"
            )
        if self._reserved_cash_paise < 0:
            raise AccountingInvariantError("reserved cash went negative")
        if self._reserved_cash_paise > self._cash_paise:
            raise AccountingInvariantError(
                f"reserved Rs {to_rupees(self._reserved_cash_paise)} exceeds cash "
                f"Rs {to_rupees(self._cash_paise)}"
            )
        for key, pos in self._positions.items():
            if pos.quantity < 0 and not SHORT_SELLING_SUPPORTED:
                raise AccountingInvariantError(
                    f"position {key} went short ({pos.quantity}) but short "
                    "selling is not supported"
                )
            if pos.quantity < 0:
                raise AccountingInvariantError(f"position {key} is negative")
            if pos.cost_basis_paise < 0:
                raise AccountingInvariantError(f"position {key} has negative basis")
        derived = self._ledger_cash_paise()
        if derived != self._cash_paise:
            raise AccountingInvariantError(
                f"cash {self._cash_paise} paise disagrees with the trade ledger "
                f"{derived} paise (delta {self._cash_paise - derived} paise)"
            )

    # ------------------------------------------------------------------ #
    #  Reservations (so resting orders cannot double-spend)                #
    # ------------------------------------------------------------------ #

    def _available_cash_paise(self) -> int:
        return self._cash_paise - self._reserved_cash_paise

    def _reserved_qty(self, key: str) -> int:
        return sum(
            o.reserved_qty for o in self._orders.values()
            if o.status == OrderStatus.OPEN
            and position_key(o.symbol, o.exchange, o.product) == key
        )

    def _free_qty(self, symbol: str, exchange: str, product: Product) -> int:
        key = position_key(symbol, exchange, product)
        pos = self._positions.get(key)
        held = pos.quantity if pos else 0
        return held - self._reserved_qty(key)

    # ------------------------------------------------------------------ #
    #  Market data                                                         #
    # ------------------------------------------------------------------ #

    def _record_quote_age(self, symbol: str, quote: dict) -> Optional[float]:
        """
        Record how old a quote was, for `is_stale_tick`, and return that age.

        A quote observation IS a freshness observation, so anything that reads
        a price updates the staleness cache. `None` means the feed supplied no
        timestamp, which strict mode treats as stale.
        """
        age: Optional[float] = None
        ts = (quote or {}).get("timestamp") or (quote or {}).get("last_trade_time")
        if ts is not None:
            try:
                stamp = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
                stamp = ensure_aware(stamp, what="quote timestamp")
                age = (self.now_ist() - stamp).total_seconds()
            except (ValueError, TypeError) as exc:
                logger.warning("Unparseable quote timestamp %r for %s: %s", ts, symbol, exc)
                age = None
        self._last_quote_age[symbol] = age
        return age

    async def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        """
        Fetch quotes AND record their freshness.

        Recording here is not incidental — it is what makes any symbol
        tradeable at all. `is_stale_tick` reads `_last_quote_age`, and that
        cache used to be written only inside `_snapshot`, which is reachable
        only from `place_order`. Since `ExecutionSafety.check_data_freshness`
        probes staleness BEFORE every order, the result was a deadlock:

            order -> symbol never quoted -> stale -> refused
                  -> place_order never runs -> cache never written
                  -> next order refused identically

        Every symbol was permanently un-tradeable. It failed closed, so no
        money was ever at risk, but nothing could trade either. Priming the
        cache from a plain quote read breaks the circularity, because a feed
        (or a caller checking a price) naturally calls this first.
        """
        quotes = await self._data_broker.get_quote(symbols)
        for key, quote in (quotes or {}).items():
            # Callers may pass bare symbols or "EXCHANGE:SYMBOL"; the
            # staleness cache is keyed on the bare symbol.
            self._record_quote_age(str(key).split(":")[-1], quote)
        return quotes

    async def get_historical_data(
        self, symbol: str, exchange: str, interval: str, from_date: str, to_date: str,
    ) -> pd.DataFrame:
        return await self._data_broker.get_historical_data(
            symbol, exchange, interval, from_date, to_date
        )

    def _model_half_spread_bps(self, price: float, volume: int) -> float:
        """
        Modelled half-spread in basis points, wider for less liquid names.

        PESSIMISTIC: tiering is by session turnover, and the result is floored
        at half a tick — a spread can never be narrower than the tick grid.
        """
        turnover = price * max(volume, 0)
        if turnover >= 1e9:            # >= Rs 100 cr/day: index heavyweights
            bps = 2.0
        elif turnover >= 1e8:          # Rs 10-100 cr: liquid mid caps
            bps = 5.0
        elif turnover >= 1e7:          # Rs 1-10 cr: small caps
            bps = 15.0
        else:                          # illiquid / unknown
            bps = 40.0
        half_tick_bps = (self._tick / 2.0) / max(price, 1e-9) * 10_000.0
        return max(bps, half_tick_bps)

    async def _snapshot(self, symbol: str, exchange: str) -> MarketSnapshot:
        key = f"{exchange}:{symbol}"
        quotes = await self._data_broker.get_quote([key])
        q = (quotes or {}).get(key) or {}

        last = q.get("last_price")
        if last is None or float(last) <= 0:
            self._last_quote_age[symbol] = None
            raise PaperBrokerError(f"no usable price for {key}")
        last = float(last)

        volume = int(q.get("volume") or 0)
        ohlc = q.get("ohlc") or {}
        op = float(ohlc.get("open") or last)
        hi = float(ohlc.get("high") or last)
        lo = float(ohlc.get("low") or last)
        prev_close = float(ohlc.get("close") or last)

        age = self._record_quote_age(symbol, q)

        model_half_bps = self._model_half_spread_bps(last, volume)
        bid_q, ask_q = q.get("bid"), q.get("ask")
        if bid_q and ask_q and float(ask_q) > float(bid_q) > 0:
            real_half_bps = (float(ask_q) - float(bid_q)) / 2.0 / last * 10_000.0
            # PESSIMISTIC: take the WIDER of the quoted and the modelled spread
            # so a momentarily tight book cannot flatter the simulation.
            half_bps = max(real_half_bps, model_half_bps)
        else:
            half_bps = model_half_bps

        half = last * half_bps / 10_000.0
        return MarketSnapshot(
            symbol=symbol, exchange=exchange, last_price=last,
            bid=max(_floor_tick(last - half, self._tick), self._tick),
            ask=_ceil_tick(last + half, self._tick),
            volume=volume, open=op, high=hi, low=lo, prev_close=prev_close,
            half_spread_bps=half_bps, age_sec=age,
        )

    # ------------------------------------------------------------------ #
    #  Execution model                                                     #
    # ------------------------------------------------------------------ #

    def _impact_frac(self, qty: int, volume: int) -> float:
        """
        Square-root market impact, floored at ``slippage_pct``.

        ``impact = max(slippage_pct, impact_coef * sqrt(qty / volume))``

        Strictly increasing in ``qty``, and never below the old flat 5 bps, so
        this model is never cheaper than the one it replaces (defect D4).
        """
        if volume <= 0:
            return 1.0                        # unusable; caller rejects first
        participation = qty / volume
        return max(self._slippage_pct, self._impact_coef * math.sqrt(participation))

    def _max_qty_within_limit(
        self, limit: float, touch: float, volume: int, txn: TransactionType
    ) -> int:
        """
        Largest quantity whose impacted price still respects ``limit``.

        Inverts the square-root law.  A big limit order therefore fills only
        partially rather than pretending the whole size clears at the touch.
        """
        allowed = (limit / touch - 1.0) if txn == TransactionType.BUY else (1.0 - limit / touch)
        if allowed < self._slippage_pct:
            return 0
        if self._impact_coef <= 0:
            return 10**12
        participation = (allowed / self._impact_coef) ** 2
        return int(participation * volume)

    def _plan_fill(
        self, order: PaperOrder, snap: MarketSnapshot
    ) -> tuple[int, float, Optional[str]]:
        """
        Decide (fill_qty, fill_price, reject_reason) for ``order`` against
        ``snap``.  ``fill_qty == 0`` with no reason means "rest, not marketable".
        """
        txn = TransactionType(order.txn_type)
        otype = OrderType(order.order_type)
        remaining = order.qty - order.filled_qty
        if remaining <= 0:
            return 0, 0.0, None

        if not snap.liquidity_known:
            # PESSIMISTIC: with no volume we cannot size impact, so we refuse
            # rather than assume infinite liquidity (the old code filled 100%).
            return 0, 0.0, RejectReason.NO_LIQUIDITY_DATA

        # Liquidity budget for a single order.
        capacity = int(snap.volume * self._max_participation)
        if capacity < 1:
            return 0, 0.0, RejectReason.NO_LIQUIDITY

        touch = snap.ask if txn == TransactionType.BUY else snap.bid

        # ---- marketability / trigger --------------------------------- #
        if otype == OrderType.MARKET:
            qty = min(remaining, capacity)
            raw = touch * (1 + self._impact_frac(qty, snap.volume)) if txn == TransactionType.BUY \
                else touch * (1 - self._impact_frac(qty, snap.volume))
            price = _ceil_tick(raw, self._tick) if txn == TransactionType.BUY \
                else _floor_tick(raw, self._tick)
            return qty, price, None

        if otype == OrderType.LIMIT:
            limit = self._snap_limit(order.price, txn)
            # A limit BUY fills only if the market trades at or below the limit;
            # a limit SELL only at or above.  Judged against the *touch* we
            # would actually have to cross, not the last traded price.
            if txn == TransactionType.BUY and touch > limit:
                return 0, 0.0, None            # rest
            if txn == TransactionType.SELL and touch < limit:
                return 0, 0.0, None            # rest

            qty = min(remaining, capacity,
                      self._max_qty_within_limit(limit, touch, snap.volume, txn))
            if qty < 1:
                return 0, 0.0, None            # size does not fit inside limit
            impact = self._impact_frac(qty, snap.volume)
            raw = touch * (1 + impact) if txn == TransactionType.BUY else touch * (1 - impact)
            if txn == TransactionType.BUY:
                price = min(limit, _ceil_tick(raw, self._tick))
            else:
                price = max(limit, _floor_tick(raw, self._tick))
            return qty, price, None

        if otype in (OrderType.SL, OrderType.SL_M):
            trigger = order.price
            # A stop BUY triggers when the market rises to/through the trigger;
            # a stop SELL when it falls to/through it.
            triggered = (touch >= trigger) if txn == TransactionType.BUY else (touch <= trigger)
            if not triggered:
                return 0, 0.0, None            # rest
            qty = min(remaining, capacity)
            impact = self._impact_frac(qty, snap.volume)
            raw = touch * (1 + impact) if txn == TransactionType.BUY else touch * (1 - impact)
            # GAP HANDLING: the fill is the prevailing market, which may be far
            # through the trigger.  Never better than the trigger.
            if txn == TransactionType.BUY:
                price = max(trigger, _ceil_tick(raw, self._tick))
            else:
                price = min(trigger, _floor_tick(raw, self._tick))
            return qty, price, None

        return 0, 0.0, RejectReason.UNSUPPORTED_PRODUCT

    def _snap_limit(self, price: float, txn: TransactionType) -> float:
        """
        Align a limit price to the tick grid *against* the trader: a BUY limit
        rounds down (stricter), a SELL limit rounds up (stricter).
        """
        return _floor_tick(price, self._tick) if txn == TransactionType.BUY \
            else _ceil_tick(price, self._tick)

    def _costs_paise(
        self, qty: int, price: float, txn: TransactionType, product: Product, exchange: str
    ) -> tuple[int, dict]:
        breakdown = self._cost_model.calculate_costs(
            transaction_type=txn.value, qty=qty, price=price,
            exchange=exchange, product=_PRODUCT_TO_COST_PRODUCT.get(product, "MIS"),
        )
        return to_paise(breakdown.total), asdict(breakdown)

    # ------------------------------------------------------------------ #
    #  Order placement                                                     #
    # ------------------------------------------------------------------ #

    def _reject(self, order: PaperOrder, reason: str, message: str = "") -> str:
        order.status = OrderStatus.REJECTED
        order.reject_reason = reason
        order.message = message
        order.reserved_cash_paise = 0
        order.reserved_qty = 0
        self._orders[order.order_id] = order
        logger.warning(
            "Paper order REJECTED [%s] %s %s %s x%d: %s",
            reason, order.txn_type, order.exchange, order.symbol, order.qty, message,
        )
        return order.order_id

    async def place_order(
        self,
        symbol: str,
        exchange: str,
        txn_type: TransactionType,
        qty: int,
        price: float,
        order_type: OrderType,
        product: Product,
        tag: str = "",
        trigger_price: Optional[float] = None,
    ) -> str:
        now = self.now_ist()

        # SL/SL-M carry a trigger distinct from the limit price. This broker
        # stores the trigger in PaperOrder.price (its internal convention), so
        # an explicit trigger_price takes precedence when supplied. Without
        # this, a stop tested here would use a different trigger than the same
        # stop sent to Kite — a mismatch that only surfaces when the stop
        # actually fires.
        effective_price = float(price)
        if order_type in (OrderType.SL, OrderType.SL_M) and trigger_price is not None:
            effective_price = float(trigger_price)

        order = PaperOrder(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            exchange=exchange,
            txn_type=txn_type.value,
            qty=int(qty) if isinstance(qty, int) else 0,
            price=effective_price,
            order_type=order_type.value,
            product=product.value,
            tag=tag,
            placed_at=now.isoformat(),
        )

        # ---- static validation --------------------------------------- #
        if self._persistence_degraded:
            return self._reject(order, RejectReason.PERSISTENCE_DEGRADED,
                                "state store is unavailable; refusing new orders")
        if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
            return self._reject(order, RejectReason.INVALID_QUANTITY,
                                f"quantity must be a positive int, got {qty!r}")
        if order_type in (OrderType.LIMIT, OrderType.SL, OrderType.SL_M) and (
            not math.isfinite(price) or price <= 0
        ):
            return self._reject(order, RejectReason.INVALID_PRICE,
                                f"{order_type.value} needs a positive price, got {price!r}")

        # ---- session gates ------------------------------------------- #
        if self._enforce_market_hours and not self.is_market_open(now):
            return self._reject(
                order, RejectReason.MARKET_CLOSED,
                f"NSE session closed at {now.isoformat()} (IST)",
            )
        if (
            self._enforce_squareoff
            and product in _INTRADAY_PRODUCTS
            and self.is_squareoff_time(now)
            and txn_type == TransactionType.BUY
        ):
            return self._reject(
                order, RejectReason.SQUARE_OFF_WINDOW,
                f"cannot open a new {product.value} position after "
                f"{MIS_SQUAREOFF_IST.isoformat()} IST (auto square-off)",
            )

        # ---- D1: holdings check BEFORE anything touches cash ---------- #
        if txn_type == TransactionType.SELL:
            free = self._free_qty(symbol, exchange, product)
            if qty > free:
                reason = (
                    RejectReason.SHORT_SELL_NOT_SUPPORTED if free <= 0
                    else RejectReason.INSUFFICIENT_HOLDINGS
                )
                return self._reject(
                    order, reason,
                    f"sell {qty} but only {free} free {symbol} held; short "
                    "selling is not supported by this broker",
                )

        # ---- price ---------------------------------------------------- #
        try:
            snap = await self._snapshot(symbol, exchange)
        except PaperBrokerError as exc:
            return self._reject(order, RejectReason.NO_PRICE, str(exc))
        except Exception as exc:                                    # noqa: BLE001
            return self._reject(order, RejectReason.NO_PRICE, f"quote failed: {exc!r}")

        if (
            snap.age_sec is not None and snap.age_sec > self._max_quote_age_sec
        ) or (self._strict_quote_staleness and snap.age_sec is None):
            return self._reject(
                order, RejectReason.STALE_PRICE,
                f"quote age {snap.age_sec}s exceeds {self._max_quote_age_sec}s",
            )

        if not snap.liquidity_known:
            # PESSIMISTIC: with no volume the impact model cannot be sized, so
            # the order is refused rather than assumed to clear (defect D3).
            return self._reject(order, RejectReason.NO_LIQUIDITY_DATA,
                                f"feed reported no traded volume for {symbol}")

        # ---- reserve buying power / stock ----------------------------- #
        if txn_type == TransactionType.BUY:
            need = self._worst_case_outflow_paise(
                qty, price, order_type, product, exchange, snap
            )
            if need > self._available_cash_paise():
                return self._reject(
                    order, RejectReason.INSUFFICIENT_CASH,
                    f"need Rs {to_rupees(need)}, available Rs "
                    f"{to_rupees(self._available_cash_paise())}",
                )
            order.reserved_cash_paise = need
            self._reserved_cash_paise += need
        else:
            order.reserved_qty = qty

        self._orders[order.order_id] = order

        # ---- attempt an immediate execution --------------------------- #
        reason = self._execute_against(order, snap)
        if reason is not None and order.filled_qty == 0:
            self._release(order)
            return self._reject(order, reason, f"{reason} for {symbol}")

        self._finalise(order)
        self._assert_invariants()
        self._save_state()
        return order.order_id

    def _release(self, order: PaperOrder) -> None:
        """Return any unused reservation to the free pool."""
        if order.reserved_cash_paise:
            self._reserved_cash_paise -= order.reserved_cash_paise
            order.reserved_cash_paise = 0
        order.reserved_qty = 0

    def _finalise(self, order: PaperOrder) -> None:
        """
        Settle an order after a fill attempt.

        There is no order-book queue, so a *partially* filled order does not
        rest: the remainder is cancelled (IOC-style) and the reservation freed.
        A wholly unfilled marketable-later order stays OPEN for
        :meth:`poll_open_orders`.
        """
        if order.filled_qty >= order.qty:
            order.status = OrderStatus.COMPLETE
            self._release(order)
        elif order.filled_qty > 0:
            order.status = OrderStatus.PARTIAL
            order.message = (
                f"filled {order.filled_qty}/{order.qty}; remainder cancelled "
                "(insufficient displayed liquidity)"
            )
            self._release(order)
        else:
            order.status = OrderStatus.OPEN
            if order.reserved_qty:
                order.reserved_qty = order.qty

    def _worst_case_outflow_paise(
        self,
        qty: int,
        price: float,
        order_type: OrderType,
        product: Product,
        exchange: str,
        snap: MarketSnapshot,
    ) -> int:
        """
        The most this BUY could possibly cost, used to earmark buying power.

        PESSIMISTIC: a LIMIT can never fill above its limit, but a MARKET or a
        stop can, so those are sized at the ask *plus* the impact the full
        quantity would cause.  Under-reserving is what let a market buy overdraw
        the account in the randomized suite.
        """
        if order_type == OrderType.LIMIT:
            ref = self._snap_limit(price, TransactionType.BUY)
        else:
            base = snap.ask if order_type == OrderType.MARKET else max(price, snap.ask)
            ref = _ceil_tick(
                base * (1 + self._impact_frac(qty, max(snap.volume, 1))), self._tick
            )
        est_costs, _ = self._costs_paise(qty, ref, TransactionType.BUY, product, exchange)
        return qty * to_paise(ref) + est_costs

    def _affordable_qty(
        self, order: PaperOrder, qty: int, price: float, product: Product
    ) -> int:
        """
        Shrink ``qty`` until the fill fits the cash actually available.

        The budget is free cash plus whatever this order already earmarked.
        This is the hard cash constraint: it is what guarantees cash can never
        go negative, no matter how far the market moved after the order rested.
        """
        budget = (
            self._cash_paise - self._reserved_cash_paise + order.reserved_cash_paise
        )
        px = to_paise(price)
        if budget <= 0 or px <= 0:
            return 0
        while qty > 0:
            costs, _ = self._costs_paise(
                qty, price, TransactionType.BUY, product, order.exchange
            )
            need = qty * px + costs
            if need <= budget:
                return qty
            qty = min(int(budget * qty // need), qty - 1)
        return 0

    def _execute_against(self, order: PaperOrder, snap: MarketSnapshot) -> Optional[str]:
        """Apply whatever fill ``snap`` supports.  Returns a reject reason or None."""
        qty, price, reason = self._plan_fill(order, snap)
        if reason is not None:
            return reason
        if qty <= 0:
            return None

        txn = TransactionType(order.txn_type)
        product = Product(order.product)
        if txn == TransactionType.BUY:
            # Cash constraint, re-checked at the *fill* price rather than the
            # estimate made when the order was accepted.
            qty = self._affordable_qty(order, qty, price, product)
            if qty <= 0:
                return RejectReason.INSUFFICIENT_CASH
        else:
            # Stock constraint, re-checked at the mutation site.  Reservations
            # should already guarantee this; the clamp makes it structural.
            pos = self._positions.get(position_key(order.symbol, order.exchange, product))
            qty = min(qty, pos.quantity if pos else 0)
            if qty <= 0:
                return RejectReason.INSUFFICIENT_HOLDINGS

        self._apply_fill(order, qty, price, snap)
        return None

    # ------------------------------------------------------------------ #
    #  The single place where cash and positions change                    #
    # ------------------------------------------------------------------ #

    def _apply_fill(
        self, order: PaperOrder, qty: int, price: float, snap: MarketSnapshot
    ) -> None:
        txn = TransactionType(order.txn_type)
        product = Product(order.product)
        price_paise = to_paise(price)
        notional_paise = qty * price_paise
        costs_paise, cost_breakdown = self._costs_paise(
            qty, price, txn, product, order.exchange
        )
        key = position_key(order.symbol, order.exchange, product)
        pos = self._positions.get(key)

        if txn == TransactionType.BUY:
            outflow = notional_paise + costs_paise
            # Release this order's reservation first so the cash test is against
            # genuinely free money and can never leave cash negative.
            if order.reserved_cash_paise:
                take = min(order.reserved_cash_paise, outflow)
                self._reserved_cash_paise -= take
                order.reserved_cash_paise -= take
            if outflow > self._cash_paise:
                raise AccountingInvariantError(
                    f"BUY would overdraw: need {outflow}p, have {self._cash_paise}p"
                )
            self._cash_paise -= outflow
            if pos is None:
                pos = PaperPosition(
                    symbol=order.symbol, exchange=order.exchange,
                    product=product.value, quantity=qty,
                    average_price=price, cost_basis_paise=outflow,
                    last_price=snap.last_price,
                )
                self._positions[key] = pos
            else:
                new_qty = pos.quantity + qty
                pos.average_price = round(
                    (pos.average_price * pos.quantity + price * qty) / new_qty, 4
                )
                pos.quantity = new_qty
                pos.cost_basis_paise += outflow
                pos.last_price = snap.last_price
        else:
            # D1: a sell is only reachable with sufficient free stock, but the
            # guard is repeated here because this is the mutation site.
            if pos is None or pos.quantity < qty:
                raise AccountingInvariantError(
                    f"SELL {qty} {order.symbol} without the shares "
                    f"(held {0 if pos is None else pos.quantity})"
                )
            basis_out = round(pos.cost_basis_paise * qty / pos.quantity)
            self._cash_paise += notional_paise - costs_paise
            pos.quantity -= qty
            pos.cost_basis_paise -= basis_out
            pos.realised = round(
                pos.realised + to_rupees(notional_paise - costs_paise - basis_out), 2
            )
            pos.last_price = snap.last_price
            if order.reserved_qty:
                order.reserved_qty = max(0, order.reserved_qty - qty)
            if pos.quantity == 0:
                pos.cost_basis_paise = 0
                pos.average_price = 0.0

        self._total_costs_paise += costs_paise

        filled_at = self.now_ist().isoformat()
        prev_filled = order.filled_qty
        order.filled_qty += qty
        order.average_price = round(
            (order.average_price * prev_filled + price * qty) / order.filled_qty, 4
        )
        order.filled_at = filled_at
        merged = dict(order.costs or {})
        for k, v in cost_breakdown.items():
            if k == "total_pct":
                continue
            merged[k] = round(float(merged.get(k, 0.0)) + float(v), 4)
        traded = order.average_price * order.filled_qty
        merged["total_pct"] = round(merged["total"] / traded, 8) if traded else 0.0
        order.costs = merged

        self._trades.append({
            "order_id": order.order_id,
            "symbol": order.symbol,
            "exchange": order.exchange,
            "txn_type": txn.value,
            "qty": qty,
            "price": price,
            "notional_paise": notional_paise,
            "costs_paise": costs_paise,
            "product": product.value,
            "costs": cost_breakdown,
            "timestamp": filled_at,
            "tag": order.tag,
        })

        logger.info(
            "Paper FILL %s %s:%s x%d @ Rs %.2f (costs Rs %.2f) cash=Rs %.2f",
            txn.value, order.exchange, order.symbol, qty, price,
            to_rupees(costs_paise), to_rupees(self._cash_paise),
        )

    # ------------------------------------------------------------------ #
    #  Resting orders                                                      #
    # ------------------------------------------------------------------ #

    async def poll_open_orders(self) -> int:
        """
        Re-evaluate every resting order against a fresh quote.

        This is how a limit or stop order eventually trades, and how a stop that
        the market gapped through fills at the gapped price rather than at the
        requested trigger.  Returns the number of orders that filled.
        """
        filled = 0
        for order in list(self._orders.values()):
            if order.status != OrderStatus.OPEN:
                continue
            try:
                snap = await self._snapshot(order.symbol, order.exchange)
            except Exception as exc:                                # noqa: BLE001
                logger.warning("poll: no quote for %s (%s)", order.symbol, exc)
                continue
            before = order.filled_qty
            reason = self._execute_against(order, snap)
            if reason is not None and order.filled_qty == 0:
                continue                       # keep resting; not a rejection
            if order.filled_qty > before:
                filled += 1
                self._finalise(order)
        if filled:
            self._assert_invariants()
            self._save_state()
        return filled

    async def square_off_intraday(self, tag: str = "auto-squareoff") -> list[str]:
        """
        Close every open intraday (MIS/CO/BO) position at market.

        The old broker had no square-off at all, so an intraday book silently
        became an overnight one.
        """
        order_ids: list[str] = []
        for pos in list(self._positions.values()):
            if Product(pos.product) not in _INTRADAY_PRODUCTS or pos.quantity <= 0:
                continue
            for o in list(self._orders.values()):
                if o.status == OrderStatus.OPEN and o.symbol == pos.symbol:
                    await self.cancel_order(o.order_id)
            order_ids.append(await self.place_order(
                pos.symbol, pos.exchange, TransactionType.SELL, pos.quantity,
                0.0, OrderType.MARKET, Product(pos.product), tag=tag,
            ))
        return order_ids

    # ------------------------------------------------------------------ #
    #  Order management                                                    #
    # ------------------------------------------------------------------ #

    async def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status != OrderStatus.OPEN:
            return False
        self._release(order)
        order.status = OrderStatus.CANCELLED
        self._assert_invariants()
        self._save_state()
        return True

    async def modify_order(
        self, order_id: str, qty: Optional[int] = None, price: Optional[float] = None,
    ) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status != OrderStatus.OPEN:
            return False
        if qty is not None:
            if not isinstance(qty, int) or qty <= order.filled_qty:
                return False
            order.qty = qty
            if order.reserved_qty:
                free = self._free_qty(order.symbol, order.exchange, Product(order.product))
                if qty - order.filled_qty > free + order.reserved_qty:
                    return False
                order.reserved_qty = qty - order.filled_qty
        if price is not None:
            if not math.isfinite(price) or price <= 0:
                return False
            order.price = float(price)
        self._assert_invariants()
        self._save_state()
        return True

    async def get_order_status(self, order_id: str) -> dict:
        order = self._orders.get(order_id)
        return asdict(order) if order is not None else {}

    async def get_orders(self) -> list[dict]:
        return [asdict(o) for o in self._orders.values()]

    async def get_trades(self) -> list[dict]:
        return list(self._trades)

    # ------------------------------------------------------------------ #
    #  Account views — MARKED TO MARKET                                    #
    # ------------------------------------------------------------------ #

    async def get_profile(self) -> dict:
        return {
            "user_name": f"PaperTrader_{self._account_id}",
            "user_type": "paper",
            "email": "",
            "broker": "PAPER",
        }

    async def _mark(self, pos: PaperPosition) -> PaperPosition:
        """Refresh ``last_price``/``unrealised`` from the live feed."""
        try:
            snap = await self._snapshot(pos.symbol, pos.exchange)
            pos.last_price = snap.last_price
        except Exception as exc:                                    # noqa: BLE001
            # PESSIMISTIC: keep the last known mark and say so; never fall back
            # to average cost, which is what made return_pct fiction (D5).
            logger.warning("Cannot mark %s to market (%s); using last known Rs %.2f",
                           pos.symbol, exc, pos.last_price)
        basis_per_share = (
            to_rupees(pos.cost_basis_paise) / pos.quantity if pos.quantity else 0.0
        )
        pos.unrealised = round((pos.last_price - basis_per_share) * pos.quantity, 2)
        pos.pnl = round(pos.unrealised + pos.realised, 2)
        return pos

    async def get_positions(self) -> list[dict]:
        out = []
        for pos in self._positions.values():
            if pos.quantity == 0:
                continue
            out.append(asdict(await self._mark(pos)))
        return out

    async def get_holdings(self) -> list[dict]:
        out = []
        for pos in self._positions.values():
            if pos.product == Product.CNC.value and pos.quantity > 0:
                out.append(asdict(await self._mark(pos)))
        return out

    async def get_funds(self) -> dict:
        return {
            "cash": to_rupees(self._available_cash_paise()),
            "total_cash": to_rupees(self._cash_paise),
            "margin_available": to_rupees(self._available_cash_paise()),
            "margin_used": to_rupees(self._reserved_cash_paise),
        }

    async def get_paper_performance(self) -> dict:
        """
        P&L for the paper account, with positions **marked to market**.

        ``portfolio_value = free+reserved cash + Σ(qty × last traded price)``.
        The old version used ``average_price``, so a position that doubled
        showed a negative return (defect D5).
        """
        market_value_paise = 0
        unrealised = 0.0
        realised = 0.0
        for pos in self._positions.values():
            realised += pos.realised
            if pos.quantity == 0:
                continue
            marked = await self._mark(pos)
            market_value_paise += marked.quantity * to_paise(marked.last_price)
            unrealised += marked.unrealised

        portfolio_paise = self._cash_paise + market_value_paise
        initial = self._initial_cash_paise
        return {
            "initial_cash": to_rupees(initial),
            "current_cash": to_rupees(self._cash_paise),
            "available_cash": to_rupees(self._available_cash_paise()),
            "market_value": to_rupees(market_value_paise),
            "portfolio_value": to_rupees(portfolio_paise),
            "total_trades": len(self._trades),
            "total_transaction_costs": to_rupees(self._total_costs_paise),
            "realised_pnl": round(realised, 2),
            "unrealised_pnl": round(unrealised, 2),
            "return_pct": (
                round((portfolio_paise - initial) / initial * 100, 4) if initial else 0.0
            ),
            "short_selling_supported": SHORT_SELLING_SUPPORTED,
        }

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def trading_mode(self) -> str:
        return "paper"

    def instrument_token(self, symbol: str, exchange: str) -> int:
        return self._data_broker.instrument_token(symbol, exchange)
