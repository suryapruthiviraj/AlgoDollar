"""
Regression tests for the REAL CapitalAllocator (app.portfolio.allocator).

Every test here exercises the production class — not a stub.  Each one pins a
defect that was empirically reproduced before the fix:

  D1  regime detection had zero effect on allocation (string-keyed table that
      shared no keys with the MarketRegime enum, looked up with .get(default))
  D2  re-normalisation after capping voided every user hard cap
  D3  score normalisation to a simplex turned deployment into a binary cliff
      at 95%, so an 80% collapse in expected Sharpe moved zero rupees
  D4  health vocabulary mismatch: 'ACTIVE' scored 0.0 → healthy strategy got
      zero capital
  D5  bare `assert` for the sum invariant, dead residual branch, negative net
      capital silently clamped to 0, unused `market_data`
"""
from __future__ import annotations

import ast
import inspect
import random

import numpy as np
import pandas as pd
import pytest

from app.portfolio import allocator as allocator_module
from app.portfolio.allocator import (
    AllocationInvariantError,
    AllocationResult,
    CapitalAllocator,
)
from app.risk.regime import (
    REGIME_SLEEVE_MULTIPLIERS,
    SLEEVES,
    MarketRegime,
    RegimeDetector,
    regime_sleeve_multipliers,
)
from app.strategies.base import StrategyHealth

ALL_HEALTHY = {s: StrategyHealth.HEALTHY for s in SLEEVES}
CAP = 1_000_000.0


def _allocate(alloc: CapitalAllocator, capital=CAP, *, regime, settings=None,
              health=None, market_data=None, existing_cash=0.0):
    return alloc.allocate(
        capital,
        {"cash": existing_cash},
        market_data,
        health or ALL_HEALTHY,
        settings,
        regime=regime,
    )


def _deployed(result: AllocationResult) -> float:
    return result.longterm_amount + result.swing_amount + result.intraday_amount


# ---------------------------------------------------------------------------
# Invariant: the allocation is an exact partition of available capital
# ---------------------------------------------------------------------------

class TestSumInvariant:

    @pytest.mark.parametrize(
        "contribution,existing_cash",
        [(0.0, 0.0), (0.01, 0.0), (1e7, 0.0), (1e12, 0.0),
         (0.0, 1234.567), (1e7, 1234.567), (1e12, 0.005)],
    )
    def test_exact_sum_edge_amounts(self, contribution, existing_cash):
        alloc = CapitalAllocator()
        r = alloc.allocate(contribution, {"cash": existing_cash}, None,
                           ALL_HEALTHY, None, regime=MarketRegime.STRONG_BULL)
        total = (r.longterm_amount + r.swing_amount
                 + r.intraday_amount + r.cash_amount)
        assert total == pytest.approx(r.available_capital, abs=1e-6, rel=1e-12)
        assert r.validate()

    def test_sum_invariant_property(self):
        """Property test: sum == available_capital over many random inputs."""
        rng = random.Random(20240917)
        regimes = list(MarketRegime)
        healths = list(StrategyHealth)

        for _ in range(500):
            contribution = rng.choice([
                0.0, rng.uniform(0, 1e3), rng.uniform(0, 1e6), rng.uniform(0, 1e10)
            ])
            existing_cash = rng.choice([0.0, rng.uniform(-1e3, 1e6)])
            settings = {
                "longterm_enabled": rng.random() > 0.2,
                "swing_enabled": rng.random() > 0.2,
                "intraday_enabled": rng.random() > 0.2,
                "max_longterm_pct": rng.uniform(0.0, 1.0),
                "max_swing_pct": rng.uniform(0.0, 1.0),
                "max_intraday_pct": rng.uniform(0.0, 1.0),
                "min_cash_pct": rng.uniform(0.0, 0.5),
            }
            health = {s: rng.choice(healths) for s in SLEEVES}
            sharpe = {s: rng.uniform(0.0, 1.0) for s in SLEEVES}
            alloc = CapitalAllocator(sharpe_normalized=sharpe)
            r = alloc.allocate(contribution, {"cash": existing_cash}, None,
                               health, settings, regime=rng.choice(regimes))

            total = (r.longterm_amount + r.swing_amount
                     + r.intraday_amount + r.cash_amount)
            scale = max(1.0, abs(r.available_capital))
            assert total == pytest.approx(r.available_capital, rel=1e-12, abs=1e-6), (
                f"residual {total - r.available_capital} on {contribution=} "
                f"{existing_cash=}"
            )
            assert r.validate()

            if r.available_capital > 0:
                # Hard constraints hold on the RETURNED rupee amounts.
                amounts = {"longterm": r.longterm_amount,
                           "swing": r.swing_amount,
                           "intraday": r.intraday_amount}
                for sleeve, amount in amounts.items():
                    assert amount >= 0.0
                    cap_pct = settings[f"max_{sleeve}_pct"]
                    assert amount <= cap_pct * r.available_capital + 1e-9 * scale
                    if not settings[f"{sleeve}_enabled"]:
                        assert amount == 0.0
                assert (r.cash_amount
                        >= settings["min_cash_pct"] * r.available_capital
                        - 1e-9 * scale)


# ---------------------------------------------------------------------------
# D1 — regime detection must actually move capital
# ---------------------------------------------------------------------------

class TestRegimeSensitivity:

    def test_strong_bull_and_panic_differ_materially(self):
        alloc = CapitalAllocator()
        bull = _allocate(alloc, regime=MarketRegime.STRONG_BULL)
        panic = _allocate(alloc, regime=MarketRegime.PANIC)

        # Before the fix these were byte-identical:
        #   STRONG_BULL -> LT=488,851 SW=353,059 ID=108,090 CASH=50,000
        #   PANIC       -> LT=488,851 SW=353,059 ID=108,090 CASH=50,000
        assert _deployed(bull) - _deployed(panic) > 0.5 * CAP
        assert panic.cash_amount > bull.cash_amount
        assert panic.cash_amount == pytest.approx(CAP)      # panic → 100% cash
        assert _deployed(bull) > 0.5 * CAP

    def test_every_regime_produces_a_distinct_allocation(self):
        alloc = CapitalAllocator()
        deployed = {}
        for regime in MarketRegime:
            r = _allocate(alloc, regime=regime)
            deployed[regime] = round(_deployed(r), 2)
        # 8 regimes, 8 different answers (the bug produced exactly 1).
        assert len(set(deployed.values())) == len(MarketRegime), deployed

    def test_deployment_is_monotone_in_regime_risk_appetite(self):
        alloc = CapitalAllocator()
        ladder = [
            MarketRegime.STRONG_BULL,
            MarketRegime.WEAK_BULL,
            MarketRegime.RECOVERY,
            MarketRegime.SIDEWAYS,
            MarketRegime.HIGH_VOL,
            MarketRegime.WEAK_BEAR,
            MarketRegime.STRONG_BEAR,
            MarketRegime.PANIC,
        ]
        deployed = [_deployed(_allocate(alloc, regime=r)) for r in ladder]
        assert deployed == sorted(deployed, reverse=True), dict(
            zip([r.value for r in ladder], deployed)
        )

    def test_panic_does_not_deploy_most_of_the_book(self):
        """The headline consequence: 95% deployed into a panic."""
        r = _allocate(CapitalAllocator(), regime=MarketRegime.PANIC)
        assert _deployed(r) <= 0.10 * CAP

    def test_unknown_regime_raises_instead_of_defaulting(self):
        alloc = CapitalAllocator()
        for bogus in ["BULL_LOW_VOL", "PANIC_HIGH_VOL", "NONSENSE", "bull", 42]:
            with pytest.raises(KeyError):
                _allocate(alloc, regime=bogus)

    def test_missing_regime_raises(self):
        with pytest.raises(ValueError):
            _allocate(CapitalAllocator(), regime=None)

    def test_combined_regime_object_is_accepted(self):
        class FakeCombined:
            price_regime = MarketRegime.STRONG_BEAR

        r = _allocate(CapitalAllocator(), regime=FakeCombined())
        assert r.regime_label == "STRONG_BEAR"
        direct = _allocate(CapitalAllocator(), regime=MarketRegime.STRONG_BEAR)
        assert _deployed(r) == pytest.approx(_deployed(direct))

    def test_canonical_table_covers_the_enum_and_has_no_cash_key(self):
        assert set(REGIME_SLEEVE_MULTIPLIERS) == set(MarketRegime)
        for regime, row in REGIME_SLEEVE_MULTIPLIERS.items():
            assert set(row) == set(SLEEVES), regime
            assert all(0.0 <= v <= 1.0 for v in row.values()), regime

    def test_regime_multiplier_lookup_raises_on_unknown(self):
        with pytest.raises(KeyError):
            regime_sleeve_multipliers("STRONG_BULL")   # string, not enum
        with pytest.raises(KeyError):
            regime_sleeve_multipliers(None)


# ---------------------------------------------------------------------------
# D2 — user hard caps are hard
# ---------------------------------------------------------------------------

class TestUserCaps:

    def test_two_percent_intraday_cap_is_never_exceeded(self):
        """Reproduced before the fix: intraday took 95.00% against a 2% cap."""
        alloc = CapitalAllocator()
        settings = {
            "max_intraday_pct": 0.02,
            "longterm_enabled": False,
            "swing_enabled": False,
            "intraday_enabled": True,
        }
        for regime in MarketRegime:
            r = _allocate(alloc, regime=regime, settings=settings)
            assert r.intraday_amount <= 0.02 * r.available_capital + 1e-9

        # ...and across a wide sweep of capital sizes and edge strengths.
        rng = random.Random(11)
        for _ in range(200):
            capital = rng.uniform(1.0, 5e9)
            sharpe = {s: rng.uniform(0.0, 1.0) for s in SLEEVES}
            r = _allocate(
                CapitalAllocator(sharpe_normalized=sharpe),
                capital=capital,
                regime=rng.choice(list(MarketRegime)),
                settings={"max_intraday_pct": 0.02},
            )
            assert r.intraday_amount <= 0.02 * capital + 1e-9 * max(1.0, capital)

    def test_default_caps_hold_in_every_regime(self):
        """Before: intraday 11.61% vs a 10% cap; longterm 95% vs an 80% cap."""
        alloc = CapitalAllocator()
        for regime in MarketRegime:
            r = _allocate(alloc, regime=regime)
            assert r.longterm_amount <= 0.80 * CAP + 1e-9
            assert r.swing_amount <= 0.40 * CAP + 1e-9
            assert r.intraday_amount <= 0.10 * CAP + 1e-9

    def test_capped_remainder_goes_to_cash_not_to_another_sleeve(self):
        alloc = CapitalAllocator()
        loose = _allocate(alloc, regime=MarketRegime.STRONG_BULL,
                          settings={"max_intraday_pct": 0.50})
        tight = _allocate(alloc, regime=MarketRegime.STRONG_BULL,
                          settings={"max_intraday_pct": 0.01})
        assert tight.intraday_amount < loose.intraday_amount
        # No re-normalisation: the other sleeves are untouched, cash absorbs it.
        assert tight.longterm_amount == pytest.approx(loose.longterm_amount)
        assert tight.swing_amount == pytest.approx(loose.swing_amount)
        assert tight.cash_amount > loose.cash_amount

    def test_zero_cap_means_zero_rupees(self):
        r = _allocate(CapitalAllocator(), regime=MarketRegime.STRONG_BULL,
                      settings={"max_intraday_pct": 0.0, "max_swing_pct": 0.0})
        assert r.intraday_amount == 0.0
        assert r.swing_amount == 0.0

    def test_min_cash_floor_is_respected(self):
        for floor in (0.0, 0.05, 0.25, 0.9):
            r = _allocate(CapitalAllocator(), regime=MarketRegime.STRONG_BULL,
                          settings={"min_cash_pct": floor})
            assert r.cash_amount >= floor * CAP - 1e-9

    def test_disabled_sleeve_gets_nothing(self):
        r = _allocate(CapitalAllocator(), regime=MarketRegime.STRONG_BULL,
                      settings={"longterm_enabled": False})
        assert r.longterm_amount == 0.0

    def test_typo_in_settings_key_raises(self):
        """A mistyped cap key must not silently fail to apply."""
        with pytest.raises(ValueError, match="Unknown user_settings key"):
            _allocate(CapitalAllocator(), regime=MarketRegime.STRONG_BULL,
                      settings={"max_intraday": 0.02})

    @pytest.mark.parametrize("bad", [{"max_swing_pct": 1.5},
                                     {"min_cash_pct": -0.1},
                                     {"max_longterm_pct": float("nan")}])
    def test_out_of_range_settings_raise(self, bad):
        with pytest.raises(ValueError):
            _allocate(CapitalAllocator(), regime=MarketRegime.STRONG_BULL,
                      settings=bad)


# ---------------------------------------------------------------------------
# D3 — deployment scales continuously with edge; "no trade" is reachable
# ---------------------------------------------------------------------------

class TestDeploymentIntensity:

    @staticmethod
    def _uncapped(scale: float):
        """Allocator with caps/floors removed, so only the edge drives size."""
        base = {"longterm": 0.75, "swing": 0.65, "intraday": 0.55}
        alloc = CapitalAllocator(
            sharpe_normalized={k: v * scale for k, v in base.items()},
            min_opportunity_score=0.0,
            min_total_score_for_deployment=0.0,
        )
        return _allocate(
            alloc,
            regime=MarketRegime.STRONG_BULL,
            settings={"max_longterm_pct": 1.0, "max_swing_pct": 1.0,
                      "max_intraday_pct": 1.0, "min_cash_pct": 0.0},
        )

    def test_deployment_is_proportional_to_edge_not_a_cliff(self):
        """Before: score 1.95 and score 0.39 both deployed 95.000% — identical
        to the rupee.  Now deployment tracks the edge linearly."""
        for scale in (1.0, 0.8, 0.5, 0.2, 0.05):
            r = self._uncapped(scale)
            assert _deployed(r) == pytest.approx(scale * CAP, abs=1.0)

    def test_deployment_takes_many_distinct_values(self):
        scales = [i / 100.0 for i in range(1, 101)]
        deployed = [round(_deployed(self._uncapped(s)), 2) for s in scales]
        assert len(set(deployed)) == len(scales)        # no binary cliff
        assert deployed == sorted(deployed)             # monotone in edge
        assert all(b > a for a, b in zip(deployed, deployed[1:]))

    def test_small_change_in_edge_makes_a_small_change_in_rupees(self):
        a = _deployed(self._uncapped(0.50))
        b = _deployed(self._uncapped(0.51))
        delta = b - a
        assert 0 < delta
        assert delta == pytest.approx(0.01 * CAP, abs=1.0)

    def test_eighty_percent_edge_collapse_moves_real_money(self):
        full = _deployed(self._uncapped(1.0))
        collapsed = _deployed(self._uncapped(0.2))
        assert full - collapsed == pytest.approx(0.8 * CAP, abs=2.0)

    def test_hundred_percent_cash_when_scores_near_zero(self):
        alloc = CapitalAllocator(
            sharpe_normalized={s: 1e-4 for s in SLEEVES}
        )
        r = _allocate(alloc, regime=MarketRegime.STRONG_BULL)
        assert r.cash_amount == pytest.approx(CAP)
        assert _deployed(r) == 0.0
        assert r.deployed_fraction == 0.0

    def test_hundred_percent_cash_when_all_sleeves_paused(self):
        r = _allocate(CapitalAllocator(), regime=MarketRegime.STRONG_BULL,
                      health={s: StrategyHealth.PAUSED for s in SLEEVES})
        assert r.cash_amount == pytest.approx(CAP)

    def test_deployed_fraction_reported_and_bounded(self):
        for scale in (0.0, 0.25, 0.5, 1.0, 4.0):
            r = self._uncapped(scale)
            assert 0.0 <= r.deployed_fraction <= 1.0
            assert r.deployed_fraction == pytest.approx(min(scale, 1.0), abs=1e-9)


# ---------------------------------------------------------------------------
# D4 — health vocabulary
# ---------------------------------------------------------------------------

class TestStrategyHealth:

    def test_enum_members_are_accepted(self):
        alloc = CapitalAllocator()
        healthy = _allocate(alloc, regime=MarketRegime.STRONG_BULL,
                            health={s: StrategyHealth.HEALTHY for s in SLEEVES})
        reduced = _allocate(alloc, regime=MarketRegime.STRONG_BULL,
                            health={s: StrategyHealth.REDUCED for s in SLEEVES})
        assert _deployed(healthy) > _deployed(reduced) > 0.0

    def test_legacy_active_is_treated_as_healthy_not_as_zero(self):
        """Before: health='ACTIVE' -> LT=0 SW=0 ID=0 CASH=1,000,000."""
        alloc = CapitalAllocator()
        active = _allocate(alloc, regime=MarketRegime.STRONG_BULL,
                           health={s: "ACTIVE" for s in SLEEVES})
        healthy = _allocate(alloc, regime=MarketRegime.STRONG_BULL,
                            health={s: "HEALTHY" for s in SLEEVES})
        assert _deployed(active) > 0.0
        assert active.as_dict() == healthy.as_dict()

    def test_legacy_stopped_is_treated_as_disabled(self):
        r = _allocate(CapitalAllocator(), regime=MarketRegime.STRONG_BULL,
                      health={s: "STOPPED" for s in SLEEVES})
        assert _deployed(r) == 0.0

    @pytest.mark.parametrize("bogus", ["BOGUS", "healthy-ish", "", 1, None])
    def test_unknown_health_raises_instead_of_silently_zeroing(self, bogus):
        with pytest.raises(KeyError):
            _allocate(CapitalAllocator(), regime=MarketRegime.STRONG_BULL,
                      health={"longterm": bogus, "swing": "HEALTHY",
                              "intraday": "HEALTHY"})

    def test_paused_sleeve_only_zeroes_that_sleeve(self):
        r = _allocate(CapitalAllocator(), regime=MarketRegime.STRONG_BULL,
                      health={"longterm": StrategyHealth.HEALTHY,
                              "swing": StrategyHealth.PAUSED,
                              "intraday": StrategyHealth.HEALTHY})
        assert r.swing_amount == 0.0
        assert r.longterm_amount > 0.0
        assert r.intraday_amount > 0.0


# ---------------------------------------------------------------------------
# D5 — leftovers
# ---------------------------------------------------------------------------

class TestHardening:

    def test_no_bare_assert_statements_in_allocator(self):
        """`assert` is stripped by `python -O`; invariants must be raises."""
        source = inspect.getsource(allocator_module)
        tree = ast.parse(source)
        asserts = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
        assert not asserts, (
            f"bare assert(s) at line(s) "
            f"{[n.lineno for n in asserts]} in allocator.py"
        )

    def test_invariant_violation_raises_a_real_exception(self):
        bad = AllocationResult(
            longterm_amount=900_000.0, swing_amount=0.0, intraday_amount=0.0,
            cash_amount=100_000.0, available_capital=1_000_000.0,
            longterm_risk_pct=0.0, swing_risk_pct=0.0, intraday_risk_pct=0.0,
            regime_label="STRONG_BULL", opportunity_scores={}, explanation="",
            confidence=1.0,
        )
        settings = CapitalAllocator._resolve_settings(None)
        with pytest.raises(AllocationInvariantError, match="exceeds the user cap"):
            CapitalAllocator._verify(bad, settings)

        broken_sum = AllocationResult(
            longterm_amount=1.0, swing_amount=0.0, intraday_amount=0.0,
            cash_amount=0.0, available_capital=1_000_000.0,
            longterm_risk_pct=0.0, swing_risk_pct=0.0, intraday_risk_pct=0.0,
            regime_label="STRONG_BULL", opportunity_scores={}, explanation="",
            confidence=1.0,
        )
        with pytest.raises(AllocationInvariantError, match="does not sum"):
            CapitalAllocator._verify(broken_sum, settings)

    def test_negative_net_capital_is_reported_honestly(self):
        """Before: available=0.0, validate()==True, true figure -40,000."""
        r = CapitalAllocator().allocate(
            10_000.0, {"cash": -50_000.0}, None, ALL_HEALTHY, None,
            regime=MarketRegime.STRONG_BULL,
        )
        assert r.available_capital == pytest.approx(-40_000.0)
        assert r.cash_amount == pytest.approx(-40_000.0)
        assert _deployed(r) == 0.0
        assert r.validate()
        assert r.as_dict()["available_capital"] == pytest.approx(-40_000.0)

    def test_zero_capital_is_all_cash(self):
        r = CapitalAllocator().allocate(0.0, {"cash": 0.0}, None, ALL_HEALTHY,
                                        None, regime=MarketRegime.STRONG_BULL)
        assert r.available_capital == 0.0
        assert r.cash_amount == 0.0
        assert r.validate()

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), "abc"])
    def test_non_finite_inputs_raise(self, bad):
        with pytest.raises(ValueError):
            CapitalAllocator().allocate(bad, {"cash": 0.0}, None, ALL_HEALTHY,
                                        None, regime=MarketRegime.STRONG_BULL)

    def test_market_data_is_actually_read(self):
        """`market_data` used to be accepted and never read."""
        alloc = CapitalAllocator()
        base = _allocate(alloc, regime=MarketRegime.STRONG_BULL)
        live = _allocate(
            alloc, regime=MarketRegime.STRONG_BULL,
            market_data={"sharpe_normalized": {s: 0.2 for s in SLEEVES}},
        )
        assert _deployed(live) < _deployed(base)

        direct = _allocate(
            alloc, regime=MarketRegime.STRONG_BULL,
            market_data={"opportunity_scores": {s: 0.0 for s in SLEEVES}},
        )
        assert direct.cash_amount == pytest.approx(CAP)

    def test_no_unused_imports_in_allocator(self):
        tree = ast.parse(inspect.getsource(allocator_module))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name != "annotations":
                        imported.add((alias.asname or alias.name).split(".")[0])
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        used |= {
            n.value.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.arg) and node.annotation is not None:
                used |= {
                    x.id for x in ast.walk(node.annotation)
                    if isinstance(x, ast.Name)
                }
        assert not (imported - used), f"unused imports: {sorted(imported - used)}"

    def test_explanation_is_produced_and_mentions_the_regime(self):
        r = _allocate(CapitalAllocator(), regime=MarketRegime.WEAK_BEAR)
        assert "WEAK_BEAR" in r.explanation
        assert "Deployed" in r.explanation


# ---------------------------------------------------------------------------
# Regime detector: PANIC reachability, symmetric bands, no look-ahead
# ---------------------------------------------------------------------------

class TestRegimeDetector:

    @staticmethod
    def _series(values):
        idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
        return pd.Series(values, index=idx, dtype=float)

    def test_panic_is_reachable_without_vix(self):
        """0/2000 simulated paths reached PANIC before; a crash must now."""
        prices = [100.0] * 40 + [100.0 * (0.985 ** i) for i in range(1, 41)]
        regime = RegimeDetector().detect_regime(self._series(prices))
        assert regime is MarketRegime.PANIC

    def test_sharp_crash_over_ten_sessions_is_panic(self):
        prices = [100.0] * 60 + [100.0 * (0.97 ** i) for i in range(1, 11)]
        assert RegimeDetector().detect_regime(self._series(prices)) is MarketRegime.PANIC

    def test_short_history_crash_is_not_sideways(self):
        """A -22.9% slide over 50 bars used to classify as SIDEWAYS."""
        prices = [100.0 * (1 - 0.229) ** (i / 49) for i in range(50)]
        regime = RegimeDetector().detect_regime(self._series(prices))
        assert regime in {MarketRegime.PANIC, MarketRegime.STRONG_BEAR,
                          MarketRegime.WEAK_BEAR}

    def test_panic_does_not_fire_on_a_rising_market(self):
        """Reachable is not the same as trigger-happy."""
        det = RegimeDetector()
        rising = self._series(100 * np.exp(np.linspace(0, 0.4, 300)))
        assert det.detect_regime(rising) is not MarketRegime.PANIC

        rng = np.random.default_rng(5)
        panics = sum(
            det.detect_regime(
                self._series(100 * np.exp(np.cumsum(
                    rng.normal(0.0004, 0.009, 300))))
            ) is MarketRegime.PANIC
            for _ in range(100)
        )
        assert panics < 20, f"PANIC fired on {panics}/100 benign paths"

    def test_recovered_market_is_not_panic_forever(self):
        """A 25% crash followed by a long recovery must leave PANIC."""
        crash = list(100 * np.linspace(1.0, 0.75, 60))
        recovery = list(crash[-1] * np.linspace(1.0, 1.25, 120))
        regime = RegimeDetector().detect_regime(self._series(crash + recovery))
        assert regime is not MarketRegime.PANIC

    def test_bands_are_symmetric(self):
        det = RegimeDetector()
        for net in range(-6, 7):
            bull = max(net, 0)
            bear = max(-net, 0)
            up = det._scores_to_regime(
                {"bull": bull, "bear": bear, "vol": 0, "recovery": 0}, None)
            down = det._scores_to_regime(
                {"bull": bear, "bear": bull, "vol": 0, "recovery": 0}, None)
            mirror = {
                MarketRegime.STRONG_BULL: MarketRegime.STRONG_BEAR,
                MarketRegime.WEAK_BULL: MarketRegime.WEAK_BEAR,
                MarketRegime.SIDEWAYS: MarketRegime.SIDEWAYS,
                MarketRegime.STRONG_BEAR: MarketRegime.STRONG_BULL,
                MarketRegime.WEAK_BEAR: MarketRegime.WEAK_BULL,
            }
            assert mirror[up] is down, f"net={net}: {up} vs {down}"
        # The specific asymmetry that was reported: net=+1 -> SIDEWAYS but
        # net=-1 -> WEAK_BEAR.
        assert det._scores_to_regime(
            {"bull": 1, "bear": 0, "vol": 0, "recovery": 0}, None
        ) is MarketRegime.SIDEWAYS
        assert det._scores_to_regime(
            {"bull": 0, "bear": 1, "vol": 0, "recovery": 0}, None
        ) is MarketRegime.SIDEWAYS

    def test_no_lookahead_future_bars_do_not_change_the_past(self):
        """
        Classifications for bars 260..300 must be identical whether or not
        another 100 bars of (violently different) future data exist in the
        series.  This is what catches a whole-series .rolling() or a forward
        slice sneaking into a helper.
        """
        rng = np.random.default_rng(3)
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 300)))
        series = self._series(prices)
        crash_future = series.iloc[-1] * np.array(
            [0.9 ** (i / 10) for i in range(1, 101)]
        )
        extended = self._series(np.concatenate([prices, crash_future]))

        det = RegimeDetector()
        for i in range(260, 300):
            assert (det.detect_regime(series.iloc[: i + 1])
                    is det.detect_regime(extended.iloc[: i + 1]))

    def test_allocation_multipliers_raise_on_unknown_regime(self):
        with pytest.raises(KeyError):
            RegimeDetector().get_regime_allocation_multipliers("STRONG_BULL")


# ---------------------------------------------------------------------------
# End-to-end: detector → allocator with one vocabulary
# ---------------------------------------------------------------------------

def test_detector_output_feeds_the_allocator_directly():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    crash = pd.Series(
        [100.0] * 250 + [100.0 * (0.98 ** i) for i in range(1, 51)], index=idx
    )
    boom = pd.Series(100 * np.exp(np.linspace(0, 0.5, 300)), index=idx)

    det = RegimeDetector()
    alloc = CapitalAllocator()
    crash_result = _allocate(alloc, regime=det.detect_regime(crash))
    boom_result = _allocate(alloc, regime=det.detect_regime(boom))

    assert _deployed(crash_result) < _deployed(boom_result)
    assert crash_result.validate() and boom_result.validate()
