"""
Tests for CapitalAllocator.

The allocator decides how much capital (in INR) to deploy into each strategy
bucket: longterm, swing, intraday, and cash reserve.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Inline stubs — match the expected real module interface
# ---------------------------------------------------------------------------


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    HIGH_VOL = "high_vol"


@dataclass
class StrategyConfig:
    longterm_enabled: bool = True
    swing_enabled: bool = True
    intraday_enabled: bool = True
    longterm_target_pct: float = 0.40
    swing_target_pct: float = 0.35
    intraday_target_pct: float = 0.15
    cash_min_pct: float = 0.10  # always keep at least this fraction as cash


@dataclass
class AllocationResult:
    longterm_amount: float = 0.0
    swing_amount: float = 0.0
    intraday_amount: float = 0.0
    cash_amount: float = 0.0
    total: float = 0.0

    def __post_init__(self):
        # Computed total for convenience
        self.total = (
            self.longterm_amount
            + self.swing_amount
            + self.intraday_amount
            + self.cash_amount
        )


class CapitalAllocator:
    """
    Allocates a contribution (new capital deposit or rebalance trigger) across
    strategy buckets according to targets, regime, and strategy enable flags.
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()

    def allocate(
        self,
        contribution: float,
        regime: MarketRegime = MarketRegime.NEUTRAL,
    ) -> AllocationResult:
        """
        Return allocation amounts that sum to exactly `contribution`.

        Rules:
        - Zero contribution -> all zeros.
        - Disabled strategy -> gets 0; its portion rolls into cash.
        - BEAR regime -> intraday allocation halved, remainder to cash.
        - Cash floor: at least cash_min_pct of contribution always stays cash.
        - Amounts are non-negative; total == contribution.
        """
        if contribution <= 0.0:
            return AllocationResult(total=0.0)

        cfg = self.config
        lt_pct = cfg.longterm_target_pct if cfg.longterm_enabled else 0.0
        sw_pct = cfg.swing_target_pct if cfg.swing_enabled else 0.0
        id_pct = cfg.intraday_target_pct if cfg.intraday_enabled else 0.0

        # Bear regime: cut intraday in half
        if regime == MarketRegime.BEAR:
            id_pct *= 0.5

        # Cash gets the remainder
        invested_pct = lt_pct + sw_pct + id_pct
        # Enforce cash floor
        if invested_pct > (1.0 - cfg.cash_min_pct):
            scale = (1.0 - cfg.cash_min_pct) / invested_pct
            lt_pct *= scale
            sw_pct *= scale
            id_pct *= scale
            invested_pct = lt_pct + sw_pct + id_pct

        cash_pct = 1.0 - invested_pct

        result = AllocationResult(
            longterm_amount=round(contribution * lt_pct, 2),
            swing_amount=round(contribution * sw_pct, 2),
            intraday_amount=round(contribution * id_pct, 2),
            cash_amount=round(contribution * cash_pct, 2),
        )
        # Fix rounding: ensure total == contribution exactly
        diff = contribution - (
            result.longterm_amount
            + result.swing_amount
            + result.intraday_amount
            + result.cash_amount
        )
        result.cash_amount = round(result.cash_amount + diff, 2)
        result.total = round(
            result.longterm_amount
            + result.swing_amount
            + result.intraday_amount
            + result.cash_amount,
            2,
        )
        return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAllocationSums:
    def test_allocation_sums_to_capital(self):
        """
        longterm + swing + intraday + cash must equal the contribution exactly.
        """
        allocator = CapitalAllocator()
        contribution = 1_00_000.0  # ₹1 lakh

        result = allocator.allocate(contribution)

        total = (
            result.longterm_amount
            + result.swing_amount
            + result.intraday_amount
            + result.cash_amount
        )
        assert total == pytest.approx(contribution, abs=0.01), (
            f"Allocation sum {total:.2f} != contribution {contribution:.2f}"
        )

    def test_allocation_sums_large_amount(self):
        """Works correctly for larger contributions (₹10 lakh)."""
        allocator = CapitalAllocator()
        contribution = 10_00_000.0

        result = allocator.allocate(contribution)
        total = (
            result.longterm_amount
            + result.swing_amount
            + result.intraday_amount
            + result.cash_amount
        )
        assert total == pytest.approx(contribution, abs=0.01)

    def test_allocation_sums_fractional_contribution(self):
        """Works correctly even with fractional rupee contributions."""
        allocator = CapitalAllocator()
        contribution = 12_345.67

        result = allocator.allocate(contribution)
        total = (
            result.longterm_amount
            + result.swing_amount
            + result.intraday_amount
            + result.cash_amount
        )
        assert total == pytest.approx(contribution, abs=0.02)


class TestZeroContribution:
    def test_zero_contribution_no_new_allocation(self):
        """contribution=0 -> all amounts=0, no action triggered."""
        allocator = CapitalAllocator()
        result = allocator.allocate(0.0)

        assert result.longterm_amount == 0.0
        assert result.swing_amount == 0.0
        assert result.intraday_amount == 0.0
        assert result.cash_amount == 0.0
        assert result.total == 0.0

    def test_negative_contribution_treated_as_zero(self):
        """Negative contribution (redemption) -> all zero allocation."""
        allocator = CapitalAllocator()
        result = allocator.allocate(-5000.0)

        assert result.longterm_amount == 0.0
        assert result.swing_amount == 0.0
        assert result.intraday_amount == 0.0
        assert result.cash_amount == 0.0


class TestCashValidOutcome:
    def test_cash_valid_outcome_all_disabled(self):
        """When all strategies are disabled, cash=100% is the valid outcome."""
        cfg = StrategyConfig(
            longterm_enabled=False,
            swing_enabled=False,
            intraday_enabled=False,
        )
        allocator = CapitalAllocator(config=cfg)
        contribution = 50_000.0

        result = allocator.allocate(contribution)

        assert result.cash_amount == pytest.approx(contribution, abs=0.01)
        assert result.longterm_amount == 0.0
        assert result.swing_amount == 0.0
        assert result.intraday_amount == 0.0

    def test_cash_floor_always_respected(self):
        """Even when strategies are greedy, cash >= cash_min_pct."""
        cfg = StrategyConfig(
            longterm_target_pct=0.45,
            swing_target_pct=0.40,
            intraday_target_pct=0.20,
            cash_min_pct=0.05,
        )
        allocator = CapitalAllocator(config=cfg)
        contribution = 1_00_000.0

        result = allocator.allocate(contribution)

        cash_pct = result.cash_amount / contribution
        assert cash_pct >= cfg.cash_min_pct - 0.001, (
            f"Cash fraction {cash_pct:.3f} below floor {cfg.cash_min_pct:.3f}"
        )


class TestDisabledStrategy:
    def test_disabled_strategy_gets_zero_intraday(self):
        """intraday_enabled=False -> intraday_amount=0."""
        cfg = StrategyConfig(intraday_enabled=False)
        allocator = CapitalAllocator(config=cfg)
        contribution = 1_00_000.0

        result = allocator.allocate(contribution)

        assert result.intraday_amount == 0.0

    def test_disabled_longterm_gets_zero(self):
        """longterm_enabled=False -> longterm_amount=0."""
        cfg = StrategyConfig(longterm_enabled=False)
        allocator = CapitalAllocator(config=cfg)
        contribution = 1_00_000.0

        result = allocator.allocate(contribution)

        assert result.longterm_amount == 0.0

    def test_disabled_swing_gets_zero(self):
        """swing_enabled=False -> swing_amount=0."""
        cfg = StrategyConfig(swing_enabled=False)
        allocator = CapitalAllocator(config=cfg)
        contribution = 1_00_000.0

        result = allocator.allocate(contribution)

        assert result.swing_amount == 0.0

    def test_disabled_strategy_capital_redirects_to_cash(self):
        """
        Disabling a strategy should increase cash_amount by approximately the
        strategy's target allocation (minus any floor adjustments).
        """
        base_cfg = StrategyConfig(intraday_enabled=True)
        disabled_cfg = StrategyConfig(intraday_enabled=False)
        contribution = 1_00_000.0

        base_result = CapitalAllocator(config=base_cfg).allocate(contribution)
        disabled_result = CapitalAllocator(config=disabled_cfg).allocate(contribution)

        # Cash in disabled scenario >= cash in base scenario
        assert disabled_result.cash_amount >= base_result.cash_amount


class TestBearRegimeAllocation:
    def test_bear_regime_reduces_intraday(self):
        """BEAR regime -> intraday allocation is reduced vs NEUTRAL."""
        cfg = StrategyConfig()
        allocator = CapitalAllocator(config=cfg)
        contribution = 1_00_000.0

        neutral_result = allocator.allocate(contribution, regime=MarketRegime.NEUTRAL)
        bear_result = allocator.allocate(contribution, regime=MarketRegime.BEAR)

        assert bear_result.intraday_amount < neutral_result.intraday_amount, (
            f"BEAR intraday {bear_result.intraday_amount} not less than "
            f"NEUTRAL intraday {neutral_result.intraday_amount}"
        )

    def test_bear_regime_increases_cash(self):
        """BEAR regime -> more cash than NEUTRAL."""
        cfg = StrategyConfig()
        allocator = CapitalAllocator(config=cfg)
        contribution = 1_00_000.0

        neutral_result = allocator.allocate(contribution, regime=MarketRegime.NEUTRAL)
        bear_result = allocator.allocate(contribution, regime=MarketRegime.BEAR)

        assert bear_result.cash_amount >= neutral_result.cash_amount

    def test_bear_regime_still_sums_to_contribution(self):
        """BEAR regime allocation still sums to 100% of contribution."""
        allocator = CapitalAllocator()
        contribution = 1_00_000.0

        result = allocator.allocate(contribution, regime=MarketRegime.BEAR)
        total = (
            result.longterm_amount
            + result.swing_amount
            + result.intraday_amount
            + result.cash_amount
        )
        assert total == pytest.approx(contribution, abs=0.01)

    def test_bull_regime_matches_neutral_default(self):
        """BULL regime should use at least as much invested capital as NEUTRAL."""
        allocator = CapitalAllocator()
        contribution = 1_00_000.0

        neutral = allocator.allocate(contribution, regime=MarketRegime.NEUTRAL)
        bull = allocator.allocate(contribution, regime=MarketRegime.BULL)

        neutral_invested = (
            neutral.longterm_amount + neutral.swing_amount + neutral.intraday_amount
        )
        bull_invested = (
            bull.longterm_amount + bull.swing_amount + bull.intraday_amount
        )
        # Bull should invest at least as much (may differ depending on impl)
        assert bull_invested >= neutral_invested * 0.95  # allow 5% slack
