"""
The allocation engine: every scenario must end in a valid target or in CASH.

The organising principle of these tests is that **capital existing is never a
reason to deploy it**. Most of them assert a refusal, and the ones that assert a
deployment also assert which constraint shaped it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.portfolio.allocation import (
    AllocationInputs,
    PositionInput,
    RiskLimits,
    SignalInput,
    StrategyBucket,
)
from app.portfolio.engine import PortfolioAllocationEngine, allocate_and_snapshot

NOW = datetime(2025, 6, 3, 11, 0, tzinfo=timezone.utc)


def signal(
    symbol: str = "RELIANCE", strategy: str = "swing", *, price: float = 1000.0,
    edge: float = 0.02, vol: float = 0.25, traded_value: float = 500_000_000.0,
    sector: str = "ENERGY", quote_age: float = 5.0, mu: float = 0.03,
    sd: float = 0.10,
) -> SignalInput:
    return SignalInput(
        symbol=symbol, strategy=strategy, direction="LONG", edge_score=edge,
        expected_return=mu, expected_return_std=sd, price=price, sector=sector,
        volatility=vol, median_traded_value=traded_value, quote_age_sec=quote_age,
    )


def universe(n: int = 12, strategy: str = "swing", **kw) -> list[SignalInput]:
    """A spread of names across sectors, so sector caps do not bind by default."""
    return [
        signal(f"SYM{i:02d}", strategy, sector=f"SEC{i % 6}", **kw)
        for i in range(n)
    ]


def inputs(**kw) -> AllocationInputs:
    base = dict(
        as_of=NOW, total_capital=1_000_000.0, cash=1_000_000.0,
        signals=universe(), strategy_health={}, limits=RiskLimits(),
    )
    base.update(kw)
    return AllocationInputs(**base)  # type: ignore[arg-type]


def allocate(**kw):
    return PortfolioAllocationEngine().allocate(inputs(**kw))


# =========================================================================== #
#  Refusals — capital existing is not a reason to deploy it                   #
# =========================================================================== #

class TestRefusals:

    def test_zero_capital_produces_cash_not_an_error(self):
        t = allocate(total_capital=0.0, cash=0.0)
        assert t.is_no_trade
        assert not t.positions
        assert any("nothing to allocate" in r for r in t.reasons)

    def test_negative_capital_produces_cash(self):
        t = allocate(total_capital=-5000.0, cash=-5000.0)
        assert t.is_no_trade and not t.positions

    def test_kill_switch_blocks_all_allocation(self):
        t = allocate(kill_switch_active=True)
        assert t.is_no_trade
        assert not t.positions
        assert any("KILL SWITCH" in r for r in t.reasons)

    def test_kill_switch_does_not_force_liquidation(self):
        """
        A halt means "place no new orders", not "sell everything now".

        An unwind is itself a large, costly trade and must be an explicit
        decision, never a side effect of a gate firing.
        """
        held = [PositionInput("RELIANCE", 100, 900.0, 1000.0, "swing", "ENERGY")]
        t = allocate(kill_switch_active=True, positions=held)
        assert not t.positions, "the gate emitted trades"
        assert t.is_no_trade

    def test_drawdown_breach_blocks_new_capital(self):
        t = allocate(current_drawdown_pct=-0.18)
        assert t.is_no_trade
        assert any("DRAWDOWN BREACH" in r for r in t.reasons)
        assert any(c.name == "max_portfolio_drawdown" for c in t.binding_constraints)

    def test_drawdown_just_inside_the_limit_still_allocates(self):
        """The limit must bind AT the threshold, not before it."""
        t = allocate(current_drawdown_pct=-0.14)
        assert t.positions, "a 14% drawdown against a 15% limit blocked trading"

    def test_daily_loss_limit_blocks_new_positions(self):
        t = allocate(daily_pnl_pct=-0.025)
        assert t.is_no_trade
        assert any("DAILY LOSS LIMIT" in r for r in t.reasons)

    def test_a_daily_GAIN_never_blocks(self):
        t = allocate(daily_pnl_pct=+0.05)
        assert t.positions

    def test_stale_market_data_blocks_allocation(self):
        t = allocate(market_data_stale=True)
        assert t.is_no_trade
        assert any("stale" in r.lower() for r in t.reasons)

    def test_blocked_trading_gate_blocks_allocation(self):
        t = allocate(trading_permitted=False)
        assert t.is_no_trade
        assert any("not permitted" in r for r in t.reasons)

    def test_every_failing_gate_is_reported_not_just_the_first(self):
        """An operator needs the whole picture, not the earliest problem."""
        t = allocate(
            kill_switch_active=True, market_data_stale=True,
            current_drawdown_pct=-0.20,
        )
        joined = " ".join(t.reasons)
        assert "KILL SWITCH" in joined
        assert "stale" in joined.lower()
        assert "DRAWDOWN" in joined

    def test_no_signals_produces_cash(self):
        t = allocate(signals=[])
        assert t.is_no_trade
        assert t.cash_reserve == pytest.approx(1_000_000.0)

    def test_all_strategies_disabled_produces_cash(self):
        t = allocate(strategy_health={"swing": "DISABLED"})
        assert t.is_no_trade
        assert any("DISABLED" in w for w in t.warnings)


class TestSignalScreening:
    """A signal that cannot be sized safely is dropped, with a stated reason."""

    def test_a_signal_without_volatility_is_dropped(self):
        s = signal()
        s.volatility = None
        t = allocate(signals=[s])
        assert not t.positions
        assert any("no volatility estimate" in w for w in t.warnings)

    def test_a_signal_without_traded_value_is_dropped(self):
        s = signal()
        s.median_traded_value = None
        t = allocate(signals=[s])
        assert not t.positions
        assert any("liquidity limit cannot be applied" in w for w in t.warnings)

    def test_a_stale_quote_is_dropped(self):
        t = allocate(signals=[signal(quote_age=9999.0)])
        assert not t.positions
        assert any("old" in w for w in t.warnings)

    def test_a_non_positive_edge_is_dropped(self):
        t = allocate(signals=[signal(edge=0.0)])
        assert not t.positions

    def test_a_zero_price_is_dropped(self):
        t = allocate(signals=[signal(price=0.0)])
        assert not t.positions


# =========================================================================== #
#  Capital vs risk — the central distinction                                  #
# =========================================================================== #

class TestCapitalAndRiskAreDifferent:

    def test_both_are_reported_per_strategy(self):
        t = allocate()
        for s in t.strategies:
            assert s.capital_pct is not None
            assert s.risk_budget_pct is not None

    def test_risk_budget_is_not_copied_from_the_capital_split(self):
        """
        A sleeve holding all the capital must not automatically hold all the
        risk budget — they are computed from different things.
        """
        t = allocate(signals=universe(8, "swing"))
        swing = next(s for s in t.strategies if s.bucket is StrategyBucket.SWING)
        assert swing.capital_pct > 0
        # Risk budget comes from the fixed baseline shares, renormalised over
        # the ACTIVE sleeves — not from the capital fraction.
        assert swing.risk_budget_pct == pytest.approx(1.0, abs=1e-9)
        assert swing.capital_pct != pytest.approx(swing.risk_budget_pct)

    def test_positions_carry_a_risk_contribution_distinct_from_weight(self):
        t = allocate()
        live = [p for p in t.positions if p.target_value > 0]
        assert live
        assert all(p.risk_contribution_pct is not None for p in live)
        assert sum(p.risk_contribution_pct for p in live) == pytest.approx(1.0, abs=1e-6)

    def test_a_high_vol_name_consumes_more_risk_than_its_weight(self):
        """The whole reason the two numbers are kept apart."""
        sigs = [
            signal("CALM", vol=0.10, sector="A"),
            signal("WILD", vol=0.60, sector="B"),
        ]
        t = allocate(signals=sigs)
        by = {p.symbol: p for p in t.positions if p.target_value > 0}
        if len(by) == 2:
            wild, calm = by["WILD"], by["CALM"]
            assert wild.risk_contribution_pct > calm.risk_contribution_pct, (
                "the volatile name does not dominate risk, so risk is being "
                "measured as if it were capital"
            )

    def test_realised_vol_above_target_scales_the_risk_budget_down(self):
        t = allocate(realised_vol=0.30)
        assert any(c.name == "target_portfolio_vol" for c in t.binding_constraints)
        assert any("risk budget is scaled" in r for r in t.reasons)


# =========================================================================== #
#  Constraints                                                                #
# =========================================================================== #

class TestConstraints:

    def test_no_position_exceeds_the_position_cap(self):
        t = allocate(signals=universe(3))
        for p in t.positions:
            assert p.target_weight <= 0.10 + 1e-9, f"{p.symbol} at {p.target_weight}"

    def test_the_position_cap_is_recorded_when_it_binds(self):
        t = allocate(signals=universe(2))
        assert any(c.name == "max_position_pct" for c in t.binding_constraints)

    def test_sector_exposure_is_capped_cumulatively(self):
        """Ten names in one sector must not become 100% of that sector."""
        sigs = [signal(f"S{i}", sector="BANKS") for i in range(10)]
        t = allocate(signals=sigs)
        by_sector: dict[str, float] = {}
        for p in t.positions:
            by_sector[p.sector] = by_sector.get(p.sector, 0.0) + p.target_weight
        assert by_sector.get("BANKS", 0.0) <= 0.25 + 1e-9

    def test_an_illiquid_name_is_capped_by_traded_value(self):
        t = allocate(signals=[signal("THIN", traded_value=100_000.0)])
        assert any(
            c.name == "max_liquidity_participation" for c in t.binding_constraints
        )
        for p in t.positions:
            assert p.target_value <= 100_000.0 * 0.05 + 1.0

    def test_an_utterly_illiquid_name_is_not_held_at_all(self):
        t = allocate(signals=[signal("DEAD", traded_value=1_000.0, price=1000.0)])
        assert not [p for p in t.positions if p.target_value > 0]

    def test_the_cash_floor_is_respected(self):
        t = allocate()
        assert t.cash_reserve_pct >= 0.05 - 1e-9, (
            f"cash {t.cash_reserve_pct:.4f} is below the 5% minimum"
        )

    def test_gross_deployment_never_exceeds_capital(self):
        t = allocate()
        assert t.deployed_value <= 1_000_000.0 + 1.0

    def test_strategy_exposure_is_capped(self):
        limits = RiskLimits(max_strategy_pct=0.20)
        t = PortfolioAllocationEngine().allocate(
            inputs(signals=universe(10, "swing"), limits=limits)
        )
        swing_value = sum(
            p.target_value for p in t.positions if p.strategy == "swing"
        )
        assert swing_value <= 1_000_000.0 * 0.20 + 1.0

    def test_kelly_is_fractional_and_capped(self):
        """A huge apparent edge must not produce a huge position."""
        t = allocate(signals=[signal("HOT", mu=0.50, sd=0.05, sector="X")])
        for p in t.positions:
            assert p.target_weight <= 0.15 + 1e-9

    def test_full_kelly_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="Kelly"):
            RiskLimits(kelly_fraction=1.0)

    def test_binding_constraints_record_what_was_wanted(self):
        t = allocate(signals=universe(2))
        for c in t.binding_constraints:
            assert c.requested >= c.applied - 1e-9
            assert str(c)


# =========================================================================== #
#  Existing portfolio                                                          #
# =========================================================================== #

class TestExistingPortfolio:

    def test_a_contribution_does_not_rebuild_the_book(self):
        """
        The delta must be smaller than the target when the book already holds it.

        Blindly overwriting would sell and re-buy the same names, paying twice.
        """
        held = [
            PositionInput(f"SYM{i:02d}", 40, 950.0, 1000.0, "swing", f"SEC{i % 6}")
            for i in range(6)
        ]
        fresh = allocate(positions=[], contribution=100_000.0)
        topped = allocate(positions=held, contribution=100_000.0)

        fresh_turnover = sum(abs(p.delta_value) for p in fresh.positions)
        topped_turnover = sum(abs(p.delta_value) for p in topped.positions)
        assert topped_turnover < fresh_turnover, (
            "topping up an existing book traded as much as building one from cash"
        )

    def test_an_already_correct_book_produces_no_trade(self):
        """
        A book at the FULL target must not trade.

        The turnover cap is lifted here on purpose. With it in place a single
        rebalance only moves partway, so the first allocation's positions are
        *en route* to the target rather than at it — and a second pass correctly
        wants to continue. That behaviour is asserted separately in
        test_turnover_scaling_moves_partway_rather_than_completing_a_few.
        """
        loose = RiskLimits(max_turnover_pct=10.0)
        t1 = PortfolioAllocationEngine().allocate(inputs(limits=loose))
        held = [
            PositionInput(p.symbol, p.target_quantity, p.price, p.price,
                          p.strategy, p.sector)
            for p in t1.positions if p.target_quantity > 0
        ]
        t2 = PortfolioAllocationEngine().allocate(
            inputs(positions=held, limits=loose)
        )
        assert t2.is_no_trade, (
            f"a book already at target still traded: "
            f"{[(p.symbol, p.delta_quantity) for p in t2.positions if p.delta_quantity]}"
        )

    def test_a_partially_built_book_keeps_moving_toward_the_target(self):
        """The other half of the same behaviour: progress continues next time."""
        t1 = allocate()
        held = [
            PositionInput(p.symbol, p.target_quantity, p.price, p.price,
                          p.strategy, p.sector)
            for p in t1.positions if p.target_quantity > 0
        ]
        t2 = allocate(positions=held)
        assert not t2.is_no_trade
        assert all(p.delta_quantity >= 0 for p in t2.positions), (
            "the second pass reversed a trade the first pass made"
        )

    def test_a_held_name_with_no_signal_is_targeted_to_zero(self):
        """
        Leaving it out of the target would silently mean 'hold forever'.

        The turnover cap is lifted so the exit completes in one pass; under the
        default cap it is scaled like any other trade and completes over
        several rebalances, which is asserted below.
        """
        loose = RiskLimits(max_turnover_pct=10.0)
        held = [PositionInput("ORPHAN", 50, 900.0, 1000.0, "swing", "OLD")]
        t = PortfolioAllocationEngine().allocate(
            inputs(positions=held, limits=loose)
        )
        orphan = next(p for p in t.positions if p.symbol == "ORPHAN")
        assert orphan.target_quantity == 0
        assert orphan.delta_quantity == -50
        assert "no current signal" in orphan.reason

    def test_an_orphan_exit_is_turnover_capped_but_still_reduces(self):
        """
        Under the turnover cap the exit is scaled, never reversed.

        Scaling an exit is defensible — selling costs money too — but it must
        always move the position DOWN. A cap that let an unwanted holding grow
        would be a bug, not a cost control.
        """
        held = [PositionInput("ORPHAN", 50, 900.0, 1000.0, "swing", "OLD")]
        t = allocate(positions=held)
        orphan = next(p for p in t.positions if p.symbol == "ORPHAN")
        assert orphan.delta_quantity < 0, "the exit was not begun"
        assert orphan.target_quantity < 50, "the unwanted holding did not shrink"

    def test_deltas_are_computed_against_current_quantity(self):
        held = [PositionInput("SYM00", 10, 900.0, 1000.0, "swing", "SEC0")]
        t = allocate(positions=held)
        p = next(p for p in t.positions if p.symbol == "SYM00")
        assert p.delta_quantity == p.target_quantity - 10


class TestTurnover:

    def test_the_turnover_limit_binds_and_scales_every_trade(self):
        limits = RiskLimits(max_turnover_pct=0.05)
        t = PortfolioAllocationEngine().allocate(inputs(limits=limits))
        assert t.expected_turnover_pct <= 0.05 + 1e-6
        assert any(c.name == "max_turnover_pct" for c in t.binding_constraints)

    def test_turnover_scaling_moves_partway_rather_than_completing_a_few(self):
        limits = RiskLimits(max_turnover_pct=0.05)
        t = PortfolioAllocationEngine().allocate(inputs(limits=limits))
        traded = [p for p in t.positions if p.delta_quantity != 0]
        assert len(traded) > 1, "the limit completed a few names instead of scaling all"

    def test_expected_turnover_and_cost_are_reported(self):
        t = allocate()
        assert t.expected_turnover_pct > 0
        assert t.estimated_cost > 0
        expected = sum(abs(p.delta_value) for p in t.positions) * 25.0 / 10_000.0
        assert t.estimated_cost == pytest.approx(expected, rel=1e-9)


# =========================================================================== #
#  Correlation and expected risk                                              #
# =========================================================================== #

class TestCorrelation:

    def test_correlated_positions_raise_expected_portfolio_vol(self):
        """
        Assuming independence understates the risk of a correlated book.

        The same weights and the same per-name volatilities must produce a
        HIGHER portfolio volatility when the names move together.
        """
        syms = [f"SYM{i:02d}" for i in range(4)]
        sigs = [signal(s, sector=f"SEC{i}") for i, s in enumerate(syms)]
        indep = allocate(signals=sigs)

        corr = [[1.0 if i == j else 0.9 for j in range(4)] for i in range(4)]
        correlated = allocate(
            signals=sigs, correlation_symbols=syms, correlation_matrix=corr
        )
        assert correlated.expected_portfolio_vol > indep.expected_portfolio_vol, (
            "a 0.9-correlated book reported no more risk than an independent one"
        )

    def test_a_malformed_correlation_matrix_falls_back_to_independence(self):
        t = allocate(
            correlation_symbols=["SYM00"], correlation_matrix=[[float("nan")]]
        )
        assert t.expected_portfolio_vol is not None

    def test_expected_vol_is_reported(self):
        t = allocate()
        assert t.expected_portfolio_vol is not None and t.expected_portfolio_vol >= 0


# =========================================================================== #
#  Output completeness and reproducibility                                    #
# =========================================================================== #

class TestOutputCompleteness:

    def test_every_required_output_is_present(self):
        t = allocate()
        assert t.strategies and t.positions
        assert t.cash_reserve >= 0
        assert t.expected_portfolio_vol is not None
        assert t.expected_turnover_pct >= 0
        assert t.estimated_cost >= 0
        assert t.reasons

    def test_every_position_carries_a_reason(self):
        t = allocate()
        assert all(p.reason for p in t.positions)

    def test_every_strategy_carries_a_reason(self):
        t = allocate()
        assert all(s.reason for s in t.strategies)

    def test_all_four_buckets_are_reported(self):
        t = allocate()
        buckets = {s.bucket for s in t.strategies}
        assert StrategyBucket.CASH in buckets
        assert len(buckets) == 4

    def test_the_summary_states_no_trade_when_it_is_one(self):
        t = allocate(kill_switch_active=True)
        assert t.summary().startswith("NO TRADE / CASH")


class TestReproducibility:

    def test_identical_inputs_produce_an_identical_fingerprint(self):
        assert inputs().fingerprint() == inputs().fingerprint()

    def test_a_changed_input_changes_the_fingerprint(self):
        assert inputs().fingerprint() != inputs(total_capital=2_000_000.0).fingerprint()

    def test_the_same_inputs_produce_the_same_target(self):
        i = inputs()
        a = PortfolioAllocationEngine().allocate(i)
        b = PortfolioAllocationEngine().allocate(i)
        assert [(p.symbol, p.target_quantity) for p in a.positions] == \
               [(p.symbol, p.target_quantity) for p in b.positions]

    def test_a_snapshot_captures_inputs_outputs_and_fingerprint(self):
        target, snap = allocate_and_snapshot(inputs())
        assert snap.fingerprint == target.input_fingerprint
        assert snap.inputs and snap.target
        assert isinstance(snap.to_json(), str)

    def test_a_snapshot_round_trips_through_json(self):
        import json

        _, snap = allocate_and_snapshot(inputs())
        loaded = json.loads(snap.to_json())
        assert loaded["fingerprint"] == snap.fingerprint
        assert "positions" in loaded["target"]
