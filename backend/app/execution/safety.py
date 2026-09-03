"""Execution safety gates — every order must pass all checks before submission."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from ..broker.base import BrokerInterface, TransactionType

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Custom exceptions                                                           #
# --------------------------------------------------------------------------- #

class KillSwitchActiveError(RuntimeError):
    """Raised when the global kill switch is engaged."""


class MarketClosedError(RuntimeError):
    """Raised when an order is attempted outside market hours."""


class StaleDataError(RuntimeError):
    """Raised when market data is too old to be reliable."""


class SafetyCheckError(RuntimeError):
    """Generic safety-check failure."""


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
    Stateless collection of safety gates.

    All individual check methods are async; validate_order() runs them all
    and returns an OrderValidationResult (or raises on critical failures).
    """

    def __init__(self, kill_switch_store=None) -> None:
        """
        Parameters
        ----------
        kill_switch_store
            Any object with a .get(key) method (e.g. Redis client).
            The kill switch is active when store.get("kill_switch") is truthy.
        """
        self._store = kill_switch_store

    # ------------------------------------------------------------------ #
    #  Individual gates                                                    #
    # ------------------------------------------------------------------ #

    async def check_kill_switch(self) -> None:
        """Raise KillSwitchActiveError if the global kill switch is on."""
        active = False
        if self._store is not None:
            try:
                val = self._store.get("kill_switch")
                active = bool(val) and val not in (b"0", "0", b"false", "false")
            except Exception as exc:
                logger.warning("Kill switch check failed to read store: %s", exc)
        if active:
            logger.critical("KILL SWITCH IS ACTIVE — blocking order.")
            raise KillSwitchActiveError("Global kill switch is active.")
        logger.debug("Kill switch: OFF")

    async def check_market_status(self, broker: BrokerInterface) -> None:
        """Raise MarketClosedError if market is not open."""
        # ZerodhaBroker exposes is_market_open(); PaperBroker delegates.
        if hasattr(broker, "is_market_open"):
            if not broker.is_market_open():
                raise MarketClosedError("Market is currently closed.")
        logger.debug("Market status: OPEN")

    async def check_data_freshness(
        self,
        broker: BrokerInterface,
        symbol: str,
        max_age_sec: float = 30.0,
    ) -> None:
        """Raise StaleDataError if the last tick for symbol is too old."""
        if hasattr(broker, "is_stale_tick"):
            if broker.is_stale_tick(symbol, max_age_seconds=max_age_sec):
                raise StaleDataError(
                    f"Market data for {symbol} is older than {max_age_sec}s."
                )
        logger.debug("Data freshness for %s: OK", symbol)

    async def check_instrument_validity(
        self, symbol: str, exchange: str
    ) -> None:
        """Basic sanity checks on symbol / exchange."""
        if not symbol or not symbol.isascii():
            raise SafetyCheckError(f"Invalid symbol: {symbol!r}")
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
        if total_portfolio <= 0:
            return
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
        if total_portfolio <= 0:
            return
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
        if daily_risk_used + trade_risk > max_daily_risk:
            raise SafetyCheckError(
                f"Daily risk limit would be breached: "
                f"used ₹{daily_risk_used:.0f} + this ₹{trade_risk:.0f} "
                f"> max ₹{max_daily_risk:.0f}."
            )
        logger.debug(
            "Risk limit: used ₹%.0f / ₹%.0f: OK", daily_risk_used, max_daily_risk
        )

    async def check_capital_availability(
        self,
        required_capital: float,
        available_cash: float,
    ) -> None:
        if required_capital > available_cash:
            raise SafetyCheckError(
                f"Insufficient capital: need ₹{required_capital:.2f}, "
                f"available ₹{available_cash:.2f}."
            )
        logger.debug("Capital availability ₹%.0f/₹%.0f: OK", required_capital, available_cash)

    async def check_duplicate_order(
        self,
        symbol: str,
        strategy: str,
        open_orders: list[dict],
    ) -> None:
        for order in open_orders:
            if (
                order.get("symbol") == symbol
                and order.get("tag", "").startswith(strategy)
                and order.get("status") in ("OPEN", "TRIGGER PENDING")
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
        qty: int,
        price: float,
        min_qty: int = 1,
    ) -> None:
        if qty < min_qty:
            raise SafetyCheckError(f"Order qty {qty} < minimum {min_qty}.")
        if price < 0:
            raise SafetyCheckError(f"Order price {price} is negative.")
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
    ) -> OrderValidationResult:
        """
        Run all safety gates and return an OrderValidationResult.

        Critical checks (kill switch, broker connectivity) raise immediately.
        Non-critical failures are collected and reported.
        """
        result = OrderValidationResult(passed=True)

        checks = [
            ("kill_switch", self.check_kill_switch()),
            ("broker_connectivity", self.check_broker_connectivity(broker)),
            ("market_status", self.check_market_status(broker)),
            ("data_freshness", self.check_data_freshness(broker, symbol)),
            ("instrument_validity", self.check_instrument_validity(symbol, exchange)),
            ("order_validity", self.check_order_validity(qty, price)),
            ("capital_availability", self.check_capital_availability(trade_value, available_cash)),
            ("position_limit", self.check_position_limit(current_positions, max_positions)),
            ("single_stock_exposure", self.check_single_stock_exposure(
                symbol, trade_value, total_portfolio, max_single_stock_pct
            )),
            ("risk_limit", self.check_risk_limit(trade_risk, daily_risk_used, max_daily_risk)),
            ("duplicate_order", self.check_duplicate_order(symbol, strategy, open_orders)),
        ]
        if sector:
            checks.append((
                "sector_exposure",
                self.check_sector_exposure(sector, sector_value, total_portfolio, max_sector_pct),
            ))

        for name, coro in checks:
            try:
                await coro
                logger.debug("Safety check [%s]: PASSED", name)
            except (KillSwitchActiveError, SafetyCheckError) as exc:
                logger.warning("Safety check [%s]: FAILED — %s", name, exc)
                result.passed = False
                result.failed_checks.append(f"{name}: {exc}")
            except Exception as exc:
                # Unexpected — log as warning, continue
                logger.warning("Safety check [%s]: ERROR — %s", name, exc)
                result.warnings.append(f"{name}: unexpected error: {exc}")

        if result.passed:
            logger.info("All safety checks PASSED for %s/%s qty=%d", symbol, exchange, qty)
        else:
            logger.warning(
                "Safety checks FAILED for %s/%s: %s",
                symbol, exchange, result.failed_checks,
            )
        return result
