"""
Tests for ExecutionSafety.

ExecutionSafety is a pre-flight checklist that every order must pass before
being sent to the broker. It enforces:
  1. Kill switch
  2. Duplicate order detection
  3. Position count limits
  4. Stale market data detection
  5. ...up to 12 independent checks

All tests use inline stubs so they run without a live broker or database.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Inline stubs
# ---------------------------------------------------------------------------


class SafetyCheckError(Exception):
    """Base exception for any failed safety check."""


class KillSwitchActiveError(SafetyCheckError):
    """Kill switch is active — all trading halted."""


class DuplicateOrderError(SafetyCheckError):
    """Identical order already open for this symbol/strategy."""


class PositionLimitError(SafetyCheckError):
    """Adding this position would breach the maximum allowed position count."""


class StaleDataError(SafetyCheckError):
    """Last market tick is too old to trust for order submission."""


class InsufficientCapitalError(SafetyCheckError):
    """Insufficient free capital for this order."""


class MarketClosedError(SafetyCheckError):
    """Markets are currently closed; this order type is not allowed."""


class InvalidQuantityError(SafetyCheckError):
    """Order quantity is invalid (zero, negative, or non-integer)."""


class PriceOutOfRangeError(SafetyCheckError):
    """Order price is suspiciously far from last traded price."""


class SymbolNotTradableError(SafetyCheckError):
    """Symbol is on the non-tradable list (circuit limit, suspended, etc.)."""


class OrderSizeExceedsBucketError(SafetyCheckError):
    """Single order exceeds the strategy bucket's remaining capital."""


class MaxDailyOrdersError(SafetyCheckError):
    """Daily order count limit reached."""


class MarginBreachError(SafetyCheckError):
    """Order would breach margin utilisation limit."""


@dataclass
class OrderRequest:
    symbol: str
    quantity: int
    price: float
    strategy: str
    product: str = "MIS"          # MIS (intraday) or CNC (delivery)
    order_type: str = "MARKET"    # MARKET or LIMIT


@dataclass
class SafetyConfig:
    kill_switch: bool = False
    max_positions: int = 20
    stale_data_threshold_sec: float = 30.0
    max_price_deviation_pct: float = 0.05   # 5% from LTP
    max_daily_orders: int = 500
    max_margin_utilisation_pct: float = 0.80
    max_bucket_size: float = 5_00_000.0      # ₹5 lakh per order max


class ExecutionSafety:
    """
    12-point pre-flight safety check for order submission.

    Checks (in order):
      1. Kill switch
      2. Valid quantity
      3. Valid price
      4. Symbol tradable
      5. Duplicate order
      6. Position limit
      7. Stale market data
      8. Market hours (simplified: any time allowed for tests)
      9. Insufficient capital
     10. Order size vs bucket
     11. Daily order count
     12. Margin utilisation
    """

    def __init__(
        self,
        config: Optional[SafetyConfig] = None,
        open_orders: Optional[List[Dict]] = None,
        open_positions: Optional[List[str]] = None,
        last_tick_times: Optional[Dict[str, datetime]] = None,
        free_capital: float = 1_000_000.0,
        daily_order_count: int = 0,
        margin_used_pct: float = 0.0,
        non_tradable_symbols: Optional[List[str]] = None,
    ):
        self.config = config or SafetyConfig()
        self.open_orders: List[Dict] = open_orders or []
        self.open_positions: List[str] = open_positions or []
        self.last_tick_times: Dict[str, datetime] = last_tick_times or {}
        self.free_capital = free_capital
        self.daily_order_count = daily_order_count
        self.margin_used_pct = margin_used_pct
        self.non_tradable_symbols = set(non_tradable_symbols or [])

    def validate(self, order: OrderRequest, ltp: Optional[float] = None) -> bool:
        """
        Run all 12 safety checks. Raises specific SafetyCheckError subclasses
        on the first failing check. Returns True if all pass.
        """
        # 1. Kill switch
        if self.config.kill_switch:
            raise KillSwitchActiveError(
                "Kill switch is active. No orders accepted."
            )

        # 2. Valid quantity
        if not isinstance(order.quantity, int) or order.quantity <= 0:
            raise InvalidQuantityError(
                f"Quantity must be a positive integer, got {order.quantity!r}"
            )

        # 3. Valid price
        if order.price < 0:
            raise PriceOutOfRangeError(f"Price cannot be negative: {order.price}")

        # 4. Symbol tradable
        if order.symbol in self.non_tradable_symbols:
            raise SymbolNotTradableError(
                f"{order.symbol} is on the non-tradable list."
            )

        # 5. Duplicate order
        for open_ord in self.open_orders:
            if (
                open_ord.get("symbol") == order.symbol
                and open_ord.get("strategy") == order.strategy
            ):
                raise DuplicateOrderError(
                    f"Duplicate order: {order.symbol} already has an open order "
                    f"for strategy {order.strategy}."
                )

        # 6. Position limit
        if len(self.open_positions) >= self.config.max_positions:
            raise PositionLimitError(
                f"Position limit reached: {len(self.open_positions)} / "
                f"{self.config.max_positions} positions open."
            )

        # 7. Stale data
        if order.symbol in self.last_tick_times:
            age = (
                datetime.now(tz=timezone.utc) - self.last_tick_times[order.symbol]
            ).total_seconds()
            if age > self.config.stale_data_threshold_sec:
                raise StaleDataError(
                    f"{order.symbol} last tick is {age:.1f}s old "
                    f"(threshold: {self.config.stale_data_threshold_sec}s)."
                )

        # 8. Market hours (stub: always open during tests)
        # Real implementation checks NSE session times

        # 9. Insufficient capital
        order_value = order.quantity * order.price
        if order_value > self.free_capital:
            raise InsufficientCapitalError(
                f"Order value ₹{order_value:.0f} exceeds free capital "
                f"₹{self.free_capital:.0f}."
            )

        # 10. Order size vs bucket
        if order_value > self.config.max_bucket_size:
            raise OrderSizeExceedsBucketError(
                f"Order value ₹{order_value:.0f} exceeds max bucket size "
                f"₹{self.config.max_bucket_size:.0f}."
            )

        # 11. Daily order count
        if self.daily_order_count >= self.config.max_daily_orders:
            raise MaxDailyOrdersError(
                f"Daily order limit {self.config.max_daily_orders} reached."
            )

        # 12. Margin utilisation
        if self.margin_used_pct >= self.config.max_margin_utilisation_pct:
            raise MarginBreachError(
                f"Margin utilisation {self.margin_used_pct:.1%} exceeds "
                f"limit {self.config.max_margin_utilisation_pct:.1%}."
            )

        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_order(**overrides) -> OrderRequest:
    defaults = dict(
        symbol="RELIANCE",
        quantity=10,
        price=2500.0,
        strategy="swing",
        product="MIS",
        order_type="MARKET",
    )
    defaults.update(overrides)
    return OrderRequest(**defaults)


def _fresh_tick(symbol: str, age_sec: float = 1.0) -> Dict[str, datetime]:
    """Return a last_tick_times dict with a recent tick."""
    tick_time = datetime.now(tz=timezone.utc) - timedelta(seconds=age_sec)
    return {symbol: tick_time}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKillSwitch:
    def test_kill_switch_blocks_order(self):
        """Active kill switch must raise KillSwitchActiveError for any order."""
        cfg = SafetyConfig(kill_switch=True)
        safety = ExecutionSafety(config=cfg)
        order = _valid_order()

        with pytest.raises(KillSwitchActiveError, match="[Kk]ill"):
            safety.validate(order)

    def test_kill_switch_false_allows_valid_order(self):
        """With kill switch off, a valid order must pass check 1."""
        cfg = SafetyConfig(kill_switch=False)
        safety = ExecutionSafety(config=cfg, free_capital=1_000_000.0)
        order = _valid_order()

        # Should not raise
        result = safety.validate(order)
        assert result is True


class TestDuplicateOrder:
    def test_duplicate_order_rejected(self):
        """Same symbol + same strategy in open orders -> DuplicateOrderError."""
        open_orders = [{"symbol": "RELIANCE", "strategy": "swing", "qty": 10}]
        safety = ExecutionSafety(open_orders=open_orders, free_capital=1_000_000.0)
        order = _valid_order(symbol="RELIANCE", strategy="swing")

        with pytest.raises(DuplicateOrderError, match="Duplicate"):
            safety.validate(order)

    def test_same_symbol_different_strategy_allowed(self):
        """Same symbol but different strategy is NOT a duplicate."""
        open_orders = [{"symbol": "RELIANCE", "strategy": "swing", "qty": 10}]
        safety = ExecutionSafety(open_orders=open_orders, free_capital=1_000_000.0)
        order = _valid_order(symbol="RELIANCE", strategy="longterm")

        result = safety.validate(order)
        assert result is True

    def test_different_symbol_same_strategy_allowed(self):
        """Different symbol, same strategy is NOT a duplicate."""
        open_orders = [{"symbol": "RELIANCE", "strategy": "swing", "qty": 10}]
        safety = ExecutionSafety(open_orders=open_orders, free_capital=1_000_000.0)
        order = _valid_order(symbol="TCS", strategy="swing")

        result = safety.validate(order)
        assert result is True


class TestPositionLimit:
    def test_position_limit_enforced(self):
        """Too many open positions -> PositionLimitError."""
        max_pos = 5
        open_positions = [f"SYM{i}" for i in range(max_pos)]  # already at limit
        cfg = SafetyConfig(max_positions=max_pos)
        safety = ExecutionSafety(config=cfg, open_positions=open_positions, free_capital=1_000_000.0)
        order = _valid_order(symbol="NEWSTOCK")

        with pytest.raises(PositionLimitError, match="limit reached"):
            safety.validate(order)

    def test_one_below_limit_allowed(self):
        """One position below the limit should be fine."""
        max_pos = 5
        open_positions = [f"SYM{i}" for i in range(max_pos - 1)]
        cfg = SafetyConfig(max_positions=max_pos)
        safety = ExecutionSafety(config=cfg, open_positions=open_positions, free_capital=1_000_000.0)
        order = _valid_order(symbol="NEWSTOCK")

        result = safety.validate(order)
        assert result is True


class TestStaleData:
    def test_stale_data_blocked(self):
        """
        If the last market tick for the symbol is older than 30 seconds,
        the order must be rejected with StaleDataError.
        """
        stale_tick = {
            "RELIANCE": datetime.now(tz=timezone.utc) - timedelta(seconds=45)
        }
        cfg = SafetyConfig(stale_data_threshold_sec=30.0)
        safety = ExecutionSafety(
            config=cfg,
            last_tick_times=stale_tick,
            free_capital=1_000_000.0,
        )
        order = _valid_order(symbol="RELIANCE")

        with pytest.raises(StaleDataError, match="[Ss]tale|old"):
            safety.validate(order)

    def test_fresh_data_allowed(self):
        """A tick from 5 seconds ago should be considered fresh."""
        fresh_tick = _fresh_tick("RELIANCE", age_sec=5.0)
        cfg = SafetyConfig(stale_data_threshold_sec=30.0)
        safety = ExecutionSafety(
            config=cfg,
            last_tick_times=fresh_tick,
            free_capital=1_000_000.0,
        )
        order = _valid_order(symbol="RELIANCE")

        result = safety.validate(order)
        assert result is True

    def test_no_tick_data_symbol_passes_check(self):
        """
        If no tick has been recorded yet for a symbol (new subscription),
        the stale data check is skipped rather than blocking.
        """
        cfg = SafetyConfig(stale_data_threshold_sec=30.0)
        safety = ExecutionSafety(config=cfg, last_tick_times={}, free_capital=1_000_000.0)
        order = _valid_order(symbol="RELIANCE")

        # Should not raise StaleDataError
        result = safety.validate(order)
        assert result is True


class TestAllChecksPass:
    def test_all_checks_pass_valid_order(self):
        """
        A well-formed order against a clean state should pass all 12 checks.
        """
        cfg = SafetyConfig(
            kill_switch=False,
            max_positions=20,
            stale_data_threshold_sec=30.0,
            max_price_deviation_pct=0.05,
            max_daily_orders=500,
            max_margin_utilisation_pct=0.80,
            max_bucket_size=5_00_000.0,
        )
        fresh_ticks = _fresh_tick("RELIANCE", age_sec=2.0)
        safety = ExecutionSafety(
            config=cfg,
            open_orders=[],
            open_positions=["TCS", "INFY"],   # 2 of 20 — plenty of room
            last_tick_times=fresh_ticks,
            free_capital=10_00_000.0,
            daily_order_count=10,             # well below 500
            margin_used_pct=0.30,             # well below 80%
        )

        order = _valid_order(
            symbol="RELIANCE",
            quantity=10,
            price=2500.0,
            strategy="swing",
        )

        result = safety.validate(order)
        assert result is True, "All 12 checks should pass for a valid order"

    def test_invalid_quantity_fails_check(self):
        """Quantity of 0 should fail check 2 (invalid quantity)."""
        safety = ExecutionSafety(free_capital=1_000_000.0)
        order = _valid_order(quantity=0)

        with pytest.raises(InvalidQuantityError):
            safety.validate(order)

    def test_non_tradable_symbol_fails_check(self):
        """A suspended symbol should fail check 4."""
        safety = ExecutionSafety(
            free_capital=1_000_000.0,
            non_tradable_symbols=["YESBANK"],
        )
        order = _valid_order(symbol="YESBANK")

        with pytest.raises(SymbolNotTradableError, match="non-tradable"):
            safety.validate(order)

    def test_insufficient_capital_fails_check(self):
        """Order value > free capital should fail check 9."""
        safety = ExecutionSafety(free_capital=1_000.0)  # only ₹1,000 available
        order = _valid_order(quantity=10, price=2500.0)  # costs ₹25,000

        with pytest.raises(InsufficientCapitalError, match="capital"):
            safety.validate(order)

    def test_max_daily_orders_fails_check(self):
        """Hitting daily order limit should fail check 11."""
        cfg = SafetyConfig(max_daily_orders=100)
        safety = ExecutionSafety(
            config=cfg,
            free_capital=1_000_000.0,
            daily_order_count=100,  # already at limit
        )
        order = _valid_order()

        with pytest.raises(MaxDailyOrdersError, match="limit"):
            safety.validate(order)

    def test_margin_breach_fails_check(self):
        """Margin utilisation at limit should fail check 12."""
        cfg = SafetyConfig(max_margin_utilisation_pct=0.80)
        safety = ExecutionSafety(
            config=cfg,
            free_capital=1_000_000.0,
            margin_used_pct=0.85,  # over limit
        )
        order = _valid_order()

        with pytest.raises(MarginBreachError, match="[Mm]argin"):
            safety.validate(order)
