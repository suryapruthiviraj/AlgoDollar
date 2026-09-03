"""
Execution safety gates — every order must pass all checks before submission.

Two rules govern this module:

*   **Every gate fails closed.**  A gate that raises *anything at all* — an
    expected safety error, an ``AttributeError`` from a malformed broker
    payload, a Redis ``ConnectionError`` — rejects the order.  There is no
    code path in which an errored gate leaves ``passed=True``.
*   **Unreadable state is dangerous state.**  If the kill-switch backend cannot
    be read, the kill switch is treated as ACTIVE.
"""

from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from ..broker.base import BrokerInterface
from ..core.exceptions import KillSwitchActiveError as _CoreKillSwitchActiveError
from .lifecycle import STRATEGY_PREFIX_LEN, strategy_tag_prefix

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Custom exceptions                                                           #
# --------------------------------------------------------------------------- #

class SafetyCheckError(RuntimeError):
    """Base class for every safety-gate failure.

    ALL gate exceptions derive from this.  ``validate_order`` treats any
    ``SafetyCheckError`` as a hard rejection; anything else that escapes a gate
    is *also* a hard rejection (see the generic handler).  Never make a gate
    exception that is not a subclass of this — but if you do, it still fails
    closed.
    """


class KillSwitchActiveError(SafetyCheckError, _CoreKillSwitchActiveError):
    """
    Raised when the global kill switch is engaged (or cannot be read).

    Inherits from BOTH this module's `SafetyCheckError` and the shared
    `core.exceptions.KillSwitchActiveError`. Two independent classes with the
    same name existed for the same concept, and `ExecutionService` caught only
    the shared one — so its kill-switch handler was dead code, and a
    kill-switch rejection would have been misclassified by the generic handler.

    This is the same defect that made the original safety gates fail open: an
    exception that does not subclass what the handler catches. One concept gets
    one catchable type.
    """


class MarketClosedError(SafetyCheckError):
    """Raised when an order is attempted outside market hours."""


class StaleDataError(SafetyCheckError):
    """Raised when market data is too old (or has never ticked)."""


class DailyLossLimitError(SafetyCheckError):
    """Raised when realised P&L for the day has breached its floor."""


# --------------------------------------------------------------------------- #
#  Result dataclass                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class OrderValidationResult:
    passed: bool
    failed_checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  ExecutionSafety                                                             #
# --------------------------------------------------------------------------- #

class ExecutionSafety:
    """
    Collection of safety gates.  A kill-switch backend is mandatory.

    Parameters
    ----------
    kill_switch_store
        Any object with a ``.get(key)`` method (e.g. a Redis client).  The kill
        switch is active when ``store.get("kill_switch")`` is truthy **or when
        the store cannot be read at all**.  Required — constructing an
        ``ExecutionSafety`` with no kill-switch backend raises, because a
        kill switch nobody can read is not a kill switch.
    require_market_hours_support
        When True (default) a broker that cannot report market hours / tick
        freshness fails those gates instead of silently skipping them.
    """

    KILL_SWITCH_KEY = "kill_switch"
    _FALSEY = (b"0", "0", b"false", "false", b"False", "False", b"", "")

    def __init__(
        self,
        kill_switch_store,
        *,
        require_market_hours_support: bool = True,
    ) -> None:
        if kill_switch_store is None:
            raise ValueError(
                "ExecutionSafety requires a kill_switch_store. A kill switch "
                "with no backend is a no-op gate and must never be built."
            )
        if not hasattr(kill_switch_store, "get"):
            raise ValueError(
                "kill_switch_store must expose .get(key); got "
                f"{type(kill_switch_store).__name__}"
            )
        self._store = kill_switch_store
        self._require_market_hours_support = require_market_hours_support

    # ------------------------------------------------------------------ #
    #  Individual gates                                                    #
    # ------------------------------------------------------------------ #

    async def check_kill_switch(self) -> None:
        """
        Raise :class:`KillSwitchActiveError` if the kill switch is on **or if
        the store cannot be read**.

        A Redis blip must never silently disable the halt mechanism.
        """
        try:
            val = self._store.get(self.KILL_SWITCH_KEY)
            if inspect.isawaitable(val):
                val = await val
        except Exception as exc:
            logger.critical(
                "Kill-switch store unreadable (%s) — treating kill switch as "
                "ACTIVE and blocking the order.", exc,
            )
            raise KillSwitchActiveError(
                f"Kill-switch store unreadable ({exc}); failing closed."
            ) from exc

        active = val is not None and val not in self._FALSEY and bool(val)
        if active:
            logger.critical("KILL SWITCH IS ACTIVE — blocking order.")
            raise KillSwitchActiveError("Global kill switch is active.")
        logger.debug("Kill switch: OFF")

    async def is_kill_switch_active(self) -> bool:
        """Non-raising probe. Unreadable store -> True (fail closed)."""
        try:
            await self.check_kill_switch()
        except KillSwitchActiveError:
            return True
        return False

    async def engage_kill_switch(self, reason: str = "") -> None:
        """
        Set the kill switch and verify the write landed.

        Raises if the store cannot be written or the value cannot be read back:
        a kill switch you *believe* you set but did not is worse than none.
        """
        if not hasattr(self._store, "set"):
            raise SafetyCheckError("kill-switch store cannot be written (.set missing)")
        try:
            res = self._store.set(self.KILL_SWITCH_KEY, "1")
            if inspect.isawaitable(res):
                await res
        except Exception as exc:
            raise SafetyCheckError(f"failed to engage kill switch: {exc}") from exc
        try:
            val = self._store.get(self.KILL_SWITCH_KEY)
            if inspect.isawaitable(val):
                val = await val
        except Exception as exc:
            raise SafetyCheckError(
                f"kill switch write could not be verified: {exc}"
            ) from exc
        if val is None or val in self._FALSEY:
            raise SafetyCheckError(
                "kill switch write did not persist — refusing to report success"
            )
        logger.critical("KILL SWITCH ENGAGED. reason=%s", reason or "unspecified")

    async def check_market_status(self, broker: BrokerInterface) -> None:
        """Raise MarketClosedError if the market is not open, or unknowable."""
        if hasattr(broker, "is_market_open"):
            if not broker.is_market_open():
                raise MarketClosedError("Market is currently closed.")
            logger.debug("Market status: OPEN")
            return
        if self._require_market_hours_support:
            raise MarketClosedError(
                f"{type(broker).__name__} cannot report market hours; failing "
                "closed. Implement is_market_open() or construct "
                "ExecutionSafety(require_market_hours_support=False) knowingly."
            )
        logger.warning("Market-hours gate skipped: broker cannot report it.")

    async def check_data_freshness(
        self,
        broker: BrokerInterface,
        symbol: str,
        max_age_sec: float = 30.0,
    ) -> None:
        """Raise StaleDataError if the last tick is too old, or unknowable."""
        if hasattr(broker, "is_stale_tick"):
            if broker.is_stale_tick(symbol, max_age_seconds=max_age_sec):
                raise StaleDataError(
                    f"Market data for {symbol} is older than {max_age_sec}s "
                    f"(or has never ticked)."
                )
            logger.debug("Data freshness for %s: OK", symbol)
            return
        if self._require_market_hours_support:
            raise StaleDataError(
                f"{type(broker).__name__} cannot report tick freshness; "
                "failing closed."
            )
        logger.warning("Staleness gate skipped: broker cannot report it.")

    async def check_instrument_validity(self, symbol: str, exchange: str) -> None:
        """Basic sanity checks on symbol / exchange."""
        if not symbol or not isinstance(symbol, str) or not symbol.isascii():
            raise SafetyCheckError(f"Invalid symbol: {symbol!r}")
        if not exchange or not isinstance(exchange, str):
            raise SafetyCheckError(f"Invalid exchange: {exchange!r}")
        valid_exchanges = {"NSE", "BSE", "NFO", "BFO", "MCX", "CDS"}
        if exchange.upper() not in valid_exchanges:
            raise SafetyCheckError(
                f"Unknown exchange: {exchange!r}. Valid: {valid_exchanges}"
            )
        logger.debug("Instrument validity %s/%s: OK", symbol, exchange)

    async def check_position_limit(
        self, current_positions: list[dict], max_positions: int
    ) -> None:
        count = len([p for p in current_positions if p.get("quantity", 0) != 0])
        if count >= max_positions:
            raise SafetyCheckError(
                f"Position limit reached: {count}/{max_positions} open positions."
            )
        logger.debug("Position count %d/%d: OK", count, max_positions)

    async def check_single_stock_exposure(
        self,
        symbol: str,
        trade_value: float,
        total_portfolio: float,
        max_pct: float = 0.10,
    ) -> None:
        _require_finite(trade_value=trade_value, total_portfolio=total_portfolio)
        if total_portfolio <= 0:
            # Cannot size an exposure limit against an unknown portfolio.
            raise SafetyCheckError(
                f"Portfolio value unknown ({total_portfolio}); cannot evaluate "
                f"single-stock exposure for {symbol}. Failing closed."
            )
        pct = trade_value / total_portfolio
        if pct > max_pct:
            raise SafetyCheckError(
                f"Single-stock exposure {pct:.1%} > limit {max_pct:.1%} for {symbol}."
            )
        logger.debug("Single-stock exposure %.2f%%: OK", pct * 100)

    async def check_sector_exposure(
        self,
        sector: str,
        sector_value: float,
        total_portfolio: float,
        max_pct: float = 0.30,
    ) -> None:
        _require_finite(sector_value=sector_value, total_portfolio=total_portfolio)
        if total_portfolio <= 0:
            raise SafetyCheckError(
                f"Portfolio value unknown ({total_portfolio}); cannot evaluate "
                f"sector exposure for {sector}. Failing closed."
            )
        pct = sector_value / total_portfolio
        if pct > max_pct:
            raise SafetyCheckError(
                f"Sector exposure {sector} {pct:.1%} > limit {max_pct:.1%}."
            )
        logger.debug("Sector exposure %s %.2f%%: OK", sector, pct * 100)

    async def check_risk_limit(
        self,
        trade_risk: float,
        daily_risk_used: float,
        max_daily_risk: float,
    ) -> None:
        """
        Per-trade risk budget.

        ``trade_risk`` is *capital at risk* — ``qty * |entry - stop|``, or the
        full notional when no stop is defined.  It is emphatically NOT the
        commission; see ``OrderManager._compute_trade_risk``.
        """
        _require_finite(
            trade_risk=trade_risk,
            daily_risk_used=daily_risk_used,
            max_daily_risk=max_daily_risk,
        )
        if trade_risk < 0:
            raise SafetyCheckError(f"Negative trade risk {trade_risk}.")
        if daily_risk_used + trade_risk > max_daily_risk:
            raise SafetyCheckError(
                f"Daily risk limit would be breached: "
                f"used Rs {daily_risk_used:.0f} + this Rs {trade_risk:.0f} "
                f"> max Rs {max_daily_risk:.0f}."
            )
        logger.debug(
            "Risk limit: used Rs %.0f / Rs %.0f: OK", daily_risk_used, max_daily_risk
        )

    async def check_daily_loss_limit(
        self,
        realised_pnl_today: float,
        max_daily_loss: float,
    ) -> None:
        """
        Realised-P&L circuit breaker, gated *separately* from per-trade risk.

        ``max_daily_loss`` is a positive rupee amount; a realised loss at or
        beyond it halts new orders.
        """
        _require_finite(
            realised_pnl_today=realised_pnl_today, max_daily_loss=max_daily_loss
        )
        if max_daily_loss <= 0:
            raise DailyLossLimitError(
                f"max_daily_loss must be a positive amount, got {max_daily_loss}."
            )
        if realised_pnl_today <= -abs(max_daily_loss):
            raise DailyLossLimitError(
                f"Daily loss limit breached: realised Rs {realised_pnl_today:.0f} "
                f"<= -Rs {abs(max_daily_loss):.0f}."
            )
        logger.debug("Daily realised P&L Rs %.0f: OK", realised_pnl_today)

    async def check_capital_availability(
        self,
        required_capital: float,
        available_cash: float,
    ) -> None:
        _require_finite(
            required_capital=required_capital, available_cash=available_cash
        )
        if required_capital <= 0:
            raise SafetyCheckError(
                f"Required capital {required_capital} is not positive — the "
                f"order value could not be established."
            )
        if required_capital > available_cash:
            raise SafetyCheckError(
                f"Insufficient capital: need Rs {required_capital:.2f}, "
                f"available Rs {available_cash:.2f}."
            )
        logger.debug(
            "Capital availability Rs %.0f/Rs %.0f: OK", required_capital, available_cash
        )

    async def check_duplicate_order(
        self,
        symbol: str,
        strategy: str,
        open_orders: list[dict],
        client_order_id: Optional[str] = None,
    ) -> None:
        """
        Reject when this strategy already has a live order in this symbol.

        ``order.get("tag")`` can legitimately be ``None`` (Kite returns that for
        untagged orders) — hence ``or ""``.  Matching is done on the first
        ``STRATEGY_PREFIX_LEN`` characters, which both this gate and the tag
        builder use, so a 20-character tag truncation can no longer defeat it.
        """
        prefix = strategy_tag_prefix(strategy)
        live = ("OPEN", "TRIGGER PENDING", "AMO REQ RECEIVED", "OPEN PENDING",
                "VALIDATION PENDING", "MODIFY PENDING", "TRIGGER PENDING")
        for order in open_orders:
            if not isinstance(order, dict):
                raise SafetyCheckError(f"Malformed open order entry: {order!r}")
            tag = order.get("tag") or ""
            if not isinstance(tag, str):
                tag = str(tag)
            status = order.get("status") or ""
            if client_order_id and tag == client_order_id:
                raise SafetyCheckError(
                    f"Order {client_order_id} is already live at the broker."
                )
            if (
                order.get("symbol") == symbol
                and tag[:STRATEGY_PREFIX_LEN].startswith(prefix)
                and status in live
            ):
                raise SafetyCheckError(
                    f"Duplicate order detected for {symbol} / strategy {strategy}."
                )
        logger.debug("Duplicate check %s/%s: OK", symbol, strategy)

    async def check_broker_connectivity(self, broker: BrokerInterface) -> None:
        if not broker.is_connected:
            raise SafetyCheckError("Broker is not connected.")
        logger.debug("Broker connectivity: OK")

    async def check_order_validity(
        self,
        qty,
        price,
        min_qty: int = 1,
        max_qty: Optional[int] = None,
        max_price: Optional[float] = None,
        max_notional: Optional[float] = None,
    ) -> None:
        """
        Numeric sanity on quantity and price.

        NaN/inf are rejected *explicitly and first*: every ordering comparison
        against NaN is False, so a corrupt quote otherwise sails through every
        downstream gate untouched.
        """
        if isinstance(qty, bool) or not isinstance(qty, (int, float)):
            raise SafetyCheckError(f"Order qty {qty!r} is not numeric.")
        if isinstance(price, bool) or not isinstance(price, (int, float)):
            raise SafetyCheckError(f"Order price {price!r} is not numeric.")
        if not math.isfinite(qty):
            raise SafetyCheckError(f"Order qty {qty!r} is not finite.")
        if not math.isfinite(price):
            raise SafetyCheckError(f"Order price {price!r} is not finite.")
        if float(qty) != int(qty):
            raise SafetyCheckError(f"Order qty {qty!r} is not a whole number.")
        qty = int(qty)
        if qty < min_qty:
            raise SafetyCheckError(f"Order qty {qty} < minimum {min_qty}.")
        if price <= 0:
            raise SafetyCheckError(
                f"Order price {price} is not positive — a reference price is "
                f"required for every order, including MARKET orders."
            )
        if max_qty is not None and qty > max_qty:
            raise SafetyCheckError(f"Fat-finger: qty {qty} > max {max_qty}.")
        if max_price is not None and price > max_price:
            raise SafetyCheckError(f"Fat-finger: price {price} > max {max_price}.")
        if max_notional is not None and qty * price > max_notional:
            raise SafetyCheckError(
                f"Fat-finger: notional {qty * price:.0f} > max {max_notional:.0f}."
            )
        logger.debug("Order validity qty=%d price=%.4f: OK", qty, price)

    # ------------------------------------------------------------------ #
    #  Master validator                                                    #
    # ------------------------------------------------------------------ #

    async def validate_order(
        self,
        *,
        broker: BrokerInterface,
        symbol: str,
        exchange: str,
        qty: int,
        price: float,
        trade_value: float,
        trade_risk: float,
        strategy: str,
        current_positions: list[dict],
        open_orders: list[dict],
        daily_risk_used: float,
        max_daily_risk: float,
        available_cash: float,
        total_portfolio: float,
        max_positions: int = 20,
        max_single_stock_pct: float = 0.10,
        sector: Optional[str] = None,
        sector_value: float = 0.0,
        max_sector_pct: float = 0.30,
        realised_pnl_today: float = 0.0,
        max_daily_loss: float = 50_000.0,
        client_order_id: Optional[str] = None,
        max_qty: Optional[int] = None,
        max_price: Optional[float] = None,
        max_notional: Optional[float] = None,
    ) -> OrderValidationResult:
        """
        Run every safety gate and return an :class:`OrderValidationResult`.

        FAIL CLOSED: any exception escaping any gate — expected or not — sets
        ``passed=False`` and is recorded in ``failed_checks``.  A gate that
        errors never permits the order.
        """
        result = OrderValidationResult(passed=True)

        checks = [
            ("kill_switch", self.check_kill_switch()),
            ("broker_connectivity", self.check_broker_connectivity(broker)),
            ("market_status", self.check_market_status(broker)),
            ("data_freshness", self.check_data_freshness(broker, symbol)),
            ("instrument_validity", self.check_instrument_validity(symbol, exchange)),
            ("order_validity", self.check_order_validity(
                qty, price, max_qty=max_qty, max_price=max_price,
                max_notional=max_notional,
            )),
            ("capital_availability", self.check_capital_availability(
                trade_value, available_cash)),
            ("position_limit", self.check_position_limit(
                current_positions, max_positions)),
            ("single_stock_exposure", self.check_single_stock_exposure(
                symbol, trade_value, total_portfolio, max_single_stock_pct
            )),
            ("risk_limit", self.check_risk_limit(
                trade_risk, daily_risk_used, max_daily_risk)),
            ("daily_loss_limit", self.check_daily_loss_limit(
                realised_pnl_today, max_daily_loss)),
            ("duplicate_order", self.check_duplicate_order(
                symbol, strategy, open_orders, client_order_id)),
        ]
        if sector:
            checks.append((
                "sector_exposure",
                self.check_sector_exposure(
                    sector, sector_value, total_portfolio, max_sector_pct),
            ))

        for name, coro in checks:
            try:
                await coro
                logger.debug("Safety check [%s]: PASSED", name)
            except SafetyCheckError as exc:
                logger.warning("Safety check [%s]: FAILED — %s", name, exc)
                result.passed = False
                result.failed_checks.append(f"{name}: {exc}")
            except Exception as exc:
                # FAIL CLOSED. An unexpected error inside a gate means the gate
                # did not run; an un-run gate must never permit an order.
                logger.error(
                    "Safety check [%s]: ERRORED (%s: %s) — failing closed.",
                    name, type(exc).__name__, exc,
                )
                result.passed = False
                result.failed_checks.append(
                    f"{name}: unexpected error, failing closed "
                    f"({type(exc).__name__}: {exc})"
                )

        if result.passed:
            logger.info(
                "All safety checks PASSED for %s/%s qty=%s", symbol, exchange, qty)
        else:
            logger.warning(
                "Safety checks FAILED for %s/%s: %s",
                symbol, exchange, result.failed_checks,
            )
        return result


def _require_finite(**values: float) -> None:
    """Reject NaN / inf / non-numeric inputs before any comparison happens."""
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SafetyCheckError(f"{name}={value!r} is not numeric.")
        if not math.isfinite(value):
            raise SafetyCheckError(f"{name}={value!r} is not finite (NaN/inf).")
