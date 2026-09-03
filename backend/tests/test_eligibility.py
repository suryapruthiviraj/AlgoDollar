"""
Tests for the live-trading eligibility gate.

This component decides whether real money may move. The tests below are
therefore adversarial rather than illustrative: each one tries to obtain a
permissive verdict by a route a careless change might open — an empty report, a
trimmed gate list, a substituted gate, a crashing gate, a forged JSON payload, a
NaN, a stale timestamp — and asserts that the route stays closed.

The sections marked REGRESSION each correspond to a hole that was actually open
in this module and was found by audit. They are not hypothetical. Read the
docstring on each before weakening it.

The one test that is not adversarial is `test_current_repo_state_is_data_blocked`,
which asserts what this repository actually evaluates to today.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone

import pytest

from app.governance import eligibility as elig
from app.governance.eligibility import (
    ALL_GATES,
    MAX_APPROVAL_AGE_DAYS,
    MAX_EVIDENCE_AGE_HOURS,
    MAX_REPORT_AGE_MINUTES,
    EligibilityReport,
    EligibilityState,
    Evidence,
    Gate,
    GateCategory,
    GateResult,
    LiveTradingBlocked,
    OrderIntent,
    ReportProvenance,
    assess_live_trading_eligibility,
    assess_repo_live_trading_eligibility,
    gather_repo_evidence,
    live_trading_eligibility,
    require_live_eligible,
)

BLOCKED_STATES = tuple(s for s in EligibilityState if s is not EligibilityState.LIVE_ELIGIBLE)

# Values that are not measurements. Every one of these must fail every gate
# that reads it, in both comparison directions.
POISON_NUMBERS = (
    float("nan"),
    float("inf"),
    float("-inf"),
    -1e300,
    1e300,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@contextmanager
def frozen_clock(when: datetime):
    """
    Redirect the module's clock. Used to build reports that are genuinely
    COMPUTED but genuinely old, which is the only way to exercise the
    report-staleness branch of require_live_eligible.
    """
    original = elig._now
    elig._now = lambda: when
    try:
        yield when
    finally:
        elig._now = original


def _compliant_evidence(now: datetime | None = None) -> Evidence:
    """
    The only Evidence in this file that clears every gate.

    It describes a system that does not exist. Nothing in this repository can
    produce these values today; they are written by hand so that the tests can
    prove LIVE_ELIGIBLE is reachable in principle and unreachable in practice
    without each of them.

    Note that it reads the clock through the module, so `frozen_clock` moves
    the evidence and the assessment together.
    """
    now = now or elig._now()
    return Evidence(
        evidence_gathered_at=now,
        price_history_symbols=100,
        price_history_start=now.date() - timedelta(days=365 * 19),
        price_history_end=now.date(),
        price_history_is_real_market_data=True,
        point_in_time_index_constituents_available=True,
        point_in_time_fundamentals_available=True,
        intraday_history_days=750,
        data_quality_audit_passed=True,
        real_ohlc_available=True,
        corporate_actions_adjusted=True,
        market_data_feed_healthy=True,
        purged_walk_forward_completed=True,
        label_embargo_applied=True,
        deflated_sharpe_ratio=0.99,
        probability_of_backtest_overfitting=0.20,
        benchmark_name="NIFTY 50 total return",
        net_annual_excess_return_vs_benchmark=0.04,
        n_trials_recorded=37,
        n_trials_used_in_dsr=37,
        oos_sharpe=1.10,
        max_drawdown=0.12,
        slippage_multiple_survived=2.0,
        oos_sharpe_at_stressed_costs=0.70,
        max_single_name_pnl_share=0.09,
        max_sector_pnl_share=0.22,
        max_single_year_pnl_share=0.30,
        top_5_trades_pnl_share=0.18,
        broker_auth_verified=True,
        order_lifecycle_verified=True,
        reconciliation_verified=True,
        duplicate_order_protection_verified=True,
        kill_switch_verified=True,
        timezone_squareoff_verified=True,
        broker_reachable_at=now,
        broker_session_valid=True,
        broker_api_degraded=False,
        last_reconciliation_at=now,
        last_reconciliation_succeeded=True,
        open_reconciliation_breaks=0,
        risk_limits_loaded=True,
        active_risk_limit_breaches=0,
        current_drawdown=0.03,
        kill_switch_engaged=False,
        eligibility_enforced_at_order_path=True,
        paper_trading_days=120,
        paper_sharpe=0.95,
        backtest_expected_sharpe=1.10,
        trading_mode="live",
        live_enable_approved_by="A. Human",
        live_enable_approved_at=now.date() - timedelta(days=3),
    )


def _substitute_gate(name: str, predicate) -> tuple[Gate, ...]:
    """Replace one gate with another of the same name, keeping the rest intact."""
    return tuple(
        Gate(g.name, g.category, g.blocking_state, g.requirement, predicate)
        if g.name == name else g
        for g in ALL_GATES
    )


def _patched(**kwargs) -> Evidence:
    return dataclasses.replace(_compliant_evidence(), **kwargs)


# ===========================================================================
# The default is blocked
# ===========================================================================

def test_enum_default_is_a_blocked_state():
    assert EligibilityState.default().is_blocked
    assert not EligibilityState.default().permits_live_trading
    # The first declared member is blocked, so any code that reaches for
    # "the first state" lands somewhere safe.
    assert list(EligibilityState)[0].is_blocked


def test_only_live_eligible_permits_trading():
    for state in BLOCKED_STATES:
        assert not state.permits_live_trading, state
        assert state.is_blocked, state
    assert EligibilityState.LIVE_ELIGIBLE.permits_live_trading


def test_paper_only_does_not_permit_live_trading():
    """PAPER_ONLY is the easiest state to misread as permission. It is not."""
    assert not EligibilityState.PAPER_ONLY.permits_live_trading


def test_assessment_with_no_evidence_is_blocked():
    report = assess_live_trading_eligibility()
    assert not report.permits_live_trading
    assert report.state is EligibilityState.BLOCKED_INSUFFICIENT_DATA
    assert len(report.failed_gates) == len(ALL_GATES)


def test_default_evidence_fails_every_single_gate():
    """Unknown evidence is failure, not absence of failure."""
    report = assess_live_trading_eligibility(Evidence())
    assert report.passed_gates == ()
    for result in report.results:
        assert not result.passed, f"{result.name} passed on empty evidence"


def test_empty_report_is_blocked_not_eligible():
    """A report constructed with no results at all must not be permissive."""
    report = EligibilityReport(results=())
    assert not report.permits_live_trading
    assert len(report.results) == len(ALL_GATES)
    assert all(not r.passed for r in report.results)


def test_state_cannot_be_assigned():
    """The verdict is derived, so there is no attribute to overwrite."""
    report = assess_live_trading_eligibility(Evidence())
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        report.state = EligibilityState.LIVE_ELIGIBLE  # type: ignore[misc]


def test_evidence_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _compliant_evidence().oos_sharpe = 99.0  # type: ignore[misc]


def test_live_trading_eligibility_helper_reports_the_repo_state():
    assert live_trading_eligibility().is_blocked


# ===========================================================================
# A gate that raises must fail, never pass
# ===========================================================================

def _explode(_evidence: Evidence) -> tuple[bool, str]:
    raise RuntimeError("kaboom: the evidence store was unreachable")


def test_raising_gate_is_recorded_as_failed():
    gate = Gate(
        name="explodes",
        category=GateCategory.DATA,
        blocking_state=EligibilityState.BLOCKED_INSUFFICIENT_DATA,
        requirement="never satisfied",
        predicate=_explode,
    )
    result = gate.check(Evidence())
    assert result.passed is False
    assert "RuntimeError" in result.reason
    assert "kaboom" in result.reason


def test_raising_gate_does_not_permit_trading_even_when_all_else_passes():
    """
    The critical test. Perfect evidence everywhere, one gate that crashes.
    The crash must block, and must block via that gate's own blocking state.
    """
    gates = _substitute_gate("kill_switch", _explode)
    report = assess_live_trading_eligibility(_compliant_evidence(), gates=gates)

    assert not report.permits_live_trading
    assert report.state is EligibilityState.BLOCKED_EXECUTION_NOT_VALIDATED

    failed = report.failed_gates
    assert [r.name for r in failed] == ["kill_switch"]
    assert "RuntimeError" in failed[0].reason


@pytest.mark.parametrize(
    "predicate",
    [
        lambda e: "not a tuple",
        lambda e: None,
        lambda e: (True,),
        lambda e: (True, "ok", "extra"),
        lambda e: 1 / 0,
    ],
    ids=["string", "none", "short-tuple", "long-tuple", "zero-division"],
)
def test_gate_with_a_broken_predicate_fails_closed(predicate):
    """A predicate with the wrong shape is a broken gate, hence a failed gate."""
    gates = _substitute_gate("kill_switch", predicate)
    report = assess_live_trading_eligibility(_compliant_evidence(), gates=gates)
    assert not report.permits_live_trading


def test_gate_returning_a_truthy_non_bool_does_not_pass():
    """`passed` is `is True`, not truthiness. "yes" is not a verified fact."""
    for truthy in ("yes", 1, [1], object()):
        gates = _substitute_gate("kill_switch", lambda e, v=truthy: (v, "sure"))
        report = assess_live_trading_eligibility(_compliant_evidence(), gates=gates)
        assert not report.permits_live_trading, truthy


# ===========================================================================
# Per-gate blocking: every gate blocks on realistic bad evidence
#
# Each patch below is chosen to break EXACTLY one gate, so the test proves the
# named gate is what did the blocking rather than something incidental.
# ===========================================================================

GATE_BREAKING_EVIDENCE: dict[str, dict] = {
    # DATA
    "evidence_freshness": {"evidence_gathered_at": None},
    "evidence_well_formed": {"current_drawdown": -0.5},
    "real_price_history": {"price_history_symbols": 10},
    "market_data_current": {"market_data_feed_healthy": False},
    "point_in_time_index_constituents": {"point_in_time_index_constituents_available": False},
    "point_in_time_fundamentals": {"point_in_time_fundamentals_available": False},
    "intraday_history_span": {"intraday_history_days": 60},
    "data_quality_audit": {"real_ohlc_available": False},
    # STATISTICAL
    "purged_walk_forward": {"label_embargo_applied": False},
    "deflated_sharpe_ratio": {"deflated_sharpe_ratio": 0.50},
    "probability_of_backtest_overfitting": {"probability_of_backtest_overfitting": 0.70},
    "beats_passive_benchmark": {"net_annual_excess_return_vs_benchmark": -0.01},
    "honest_trial_count": {"n_trials_used_in_dsr": 1},
    # PERFORMANCE
    "oos_sharpe_floor": {"oos_sharpe": 0.10},
    "max_drawdown_limit": {"max_drawdown": 0.40},
    "cost_and_slippage_stress": {"slippage_multiple_survived": 1.0},
    "return_concentration": {"max_sector_pnl_share": 0.90},
    # EXECUTION
    "broker_auth": {"broker_auth_verified": False},
    "broker_connectivity": {"broker_api_degraded": True},
    "order_lifecycle": {"order_lifecycle_verified": False},
    "reconciliation": {"reconciliation_verified": False},
    "reconciliation_current": {"last_reconciliation_succeeded": False},
    "duplicate_order_protection": {"duplicate_order_protection_verified": False},
    "kill_switch": {"kill_switch_verified": False},
    "timezone_and_squareoff": {"timezone_squareoff_verified": False},
    "enforcement_wired_into_order_path": {"eligibility_enforced_at_order_path": False},
    # RISK
    "risk_limits_enforced": {"active_risk_limit_breaches": 3},
    # OPERATIONAL
    "paper_trading_duration": {"paper_trading_days": 10},
    "paper_matches_backtest": {"paper_sharpe": 0.10},
    "human_enabled_live_mode": {"trading_mode": "paper"},
    "approval_is_current": {"live_enable_approved_at": date(2020, 1, 1)},
}


def test_the_breaking_table_covers_every_registered_gate():
    """If a gate is added without a blocking test, this fails."""
    assert set(GATE_BREAKING_EVIDENCE) == {g.name for g in ALL_GATES}


@pytest.mark.parametrize("gate_name", sorted(GATE_BREAKING_EVIDENCE))
def test_every_gate_blocks_on_realistic_bad_evidence(gate_name):
    report = assess_live_trading_eligibility(_patched(**GATE_BREAKING_EVIDENCE[gate_name]))
    assert not report.permits_live_trading, gate_name
    assert [r.name for r in report.failed_gates] == [gate_name], report.checklist()


@pytest.mark.parametrize("gate_name", [g.name for g in ALL_GATES])
def test_every_gate_can_block_on_its_own(gate_name):
    gates = _substitute_gate(gate_name, lambda e: (False, "forced failure"))
    report = assess_live_trading_eligibility(_compliant_evidence(), gates=gates)
    assert not report.permits_live_trading
    assert [r.name for r in report.failed_gates] == [gate_name]


@pytest.mark.parametrize("dropped", [g.name for g in ALL_GATES])
def test_dropping_a_gate_cannot_buy_eligibility(dropped):
    """No gate may be silently skipped. An absent gate is a failed gate."""
    gates = tuple(g for g in ALL_GATES if g.name != dropped)
    report = assess_live_trading_eligibility(_compliant_evidence(), gates=gates)

    assert not report.permits_live_trading
    injected = next(r for r in report.results if r.name == dropped)
    assert injected.passed is False
    assert "NOT EVALUATED" in injected.reason


def test_report_always_covers_every_category():
    report = assess_live_trading_eligibility(_compliant_evidence(), gates=())
    assert {r.category for r in report.results} == set(GateCategory)
    assert not report.permits_live_trading
    assert len(report.results) == len(ALL_GATES)


def test_fully_compliant_evidence_is_live_eligible():
    """LIVE_ELIGIBLE must be reachable, or the gate is a wall rather than a gate."""
    report = assess_live_trading_eligibility(_compliant_evidence())
    assert report.failed_gates == (), report.checklist()
    assert report.state is EligibilityState.LIVE_ELIGIBLE
    assert report.permits_live_trading
    assert report.provenance is ReportProvenance.COMPUTED


_EVIDENCE_FIELDS = [f.name for f in dataclasses.fields(Evidence) if f.name != "notes"]


@pytest.mark.parametrize("field_name", _EVIDENCE_FIELDS)
def test_every_piece_of_evidence_is_load_bearing(field_name):
    """
    Erase one fact from an otherwise perfect record and eligibility must be
    lost. This proves two things at once: unknown evidence fails, and no
    declared evidence field is decorative.
    """
    report = assess_live_trading_eligibility(_patched(**{field_name: None}))
    assert not report.permits_live_trading, f"{field_name} did not gate anything"


# ===========================================================================
# REGRESSION: NaN and infinity
#
# `_paper_matches_backtest` used to compare `if paper < floor: return FAIL`.
# NaN fails every comparison, so a NaN paper Sharpe fell through to a PASS —
# and `-inf` satisfied every `<=` cap in the module while `+inf` satisfied
# every `>=` floor. A poisoned evidence field granted LIVE_ELIGIBLE.
# ===========================================================================

_NUMERIC_FIELDS = sorted(elig._NUMERIC_DOMAINS)


@pytest.mark.parametrize("field_name", _NUMERIC_FIELDS)
@pytest.mark.parametrize("poison", POISON_NUMBERS, ids=lambda v: f"{v:g}")
def test_no_poisoned_number_can_reach_live_eligible(field_name, poison):
    report = assess_live_trading_eligibility(_patched(**{field_name: poison}))
    assert not report.permits_live_trading, (
        f"{field_name}={poison} reached LIVE_ELIGIBLE"
    )


@pytest.mark.parametrize(
    "paper,backtest",
    [
        (float("nan"), 1.10),
        (0.95, float("nan")),
        (float("nan"), float("nan")),
        (float("inf"), float("inf")),
        (float("inf"), 0.01),
        (0.01, float("-inf")),
        (float("-inf"), 1.10),
    ],
    ids=["nan-paper", "nan-backtest", "both-nan", "both-inf", "inf-paper",
         "neg-inf-backtest", "neg-inf-paper"],
)
def test_paper_vs_backtest_never_passes_on_a_non_number(paper, backtest):
    """The exact hole: `not (nan < floor)` is True, and used to mean "pass"."""
    report = assess_live_trading_eligibility(
        _patched(paper_sharpe=paper, backtest_expected_sharpe=backtest)
    )
    gate = next(r for r in report.results if r.name == "paper_matches_backtest")
    assert gate.passed is False, gate.reason
    assert not report.permits_live_trading


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("max_drawdown", -5.0),          # a negative drawdown is not a drawdown
        ("probability_of_backtest_overfitting", -3.0),   # p < 0
        ("probability_of_backtest_overfitting", 1.5),    # p > 1
        ("max_single_name_pnl_share", -10.0),
        ("max_sector_pnl_share", -10.0),
        ("max_single_year_pnl_share", -10.0),
        ("top_5_trades_pnl_share", -10.0),
        ("current_drawdown", -1.0),
        ("oos_sharpe", 1e6),             # a Sharpe of a million is a bug
        ("deflated_sharpe_ratio", 1e6),
        ("slippage_multiple_survived", 1e12),
        ("intraday_history_days", 10**9),
        ("paper_trading_days", 10**9),
        ("price_history_symbols", 10**12),
    ],
)
def test_physically_impossible_values_are_rejected(field_name, value):
    """
    Out-of-domain numbers used to sail through: `-inf <= 0.15` is True, so a
    negative drawdown "met" the drawdown cap.
    """
    report = assess_live_trading_eligibility(_patched(**{field_name: value}))
    assert not report.permits_live_trading
    schema = next(r for r in report.results if r.name == "evidence_well_formed")
    assert schema.passed is False
    assert field_name in schema.reason


# ===========================================================================
# REGRESSION: staleness
#
# There was no notion of "current" anywhere. Evidence gathered in 1999, price
# history ending in 1920, and a human approval dated 2099 all passed.
# ===========================================================================

@pytest.mark.parametrize(
    "patch,why",
    [
        ({"evidence_gathered_at": None}, "no timestamp at all"),
        ({"evidence_gathered_at": datetime(1999, 1, 1, tzinfo=timezone.utc)}, "from 1999"),
        ({"evidence_gathered_at": datetime(2026, 1, 1, 12, 0)}, "timezone-naive"),
    ],
    ids=["absent", "ancient", "naive"],
)
def test_stale_or_unusable_evidence_timestamp_blocks(patch, why):
    report = assess_live_trading_eligibility(_patched(**patch))
    assert not report.permits_live_trading, why


def test_evidence_older_than_the_limit_blocks():
    now = elig._now()
    just_over = now - timedelta(hours=MAX_EVIDENCE_AGE_HOURS + 1)
    report = assess_live_trading_eligibility(
        dataclasses.replace(_compliant_evidence(now), evidence_gathered_at=just_over)
    )
    assert not report.permits_live_trading
    gate = next(r for r in report.results if r.name == "evidence_freshness")
    assert "STALE" in gate.reason


def test_future_dated_evidence_blocks():
    """Evidence dated tomorrow is fabricated or the clock is wrong. Either way."""
    now = elig._now()
    report = assess_live_trading_eligibility(
        dataclasses.replace(
            _compliant_evidence(now), evidence_gathered_at=now + timedelta(days=1)
        )
    )
    assert not report.permits_live_trading
    gate = next(r for r in report.results if r.name == "evidence_freshness")
    assert "FUTURE" in gate.reason


@pytest.mark.parametrize(
    "patch,why",
    [
        ({"price_history_end": date(2019, 1, 1)}, "price data stops in 2019"),
        ({"price_history_start": date(2200, 1, 1),
          "price_history_end": date(2300, 1, 1)}, "price data from the year 2200"),
        ({"price_history_start": date(1900, 1, 1),
          "price_history_end": date(1920, 1, 1)}, "price data from 1900"),
    ],
    ids=["stale", "future", "antique"],
)
def test_stale_or_impossible_price_history_blocks(patch, why):
    assert not assess_live_trading_eligibility(_patched(**patch)).permits_live_trading, why


def test_approval_that_has_gone_stale_blocks():
    """A sign-off from last year is a historical fact, not a current decision."""
    old = elig._now().date() - timedelta(days=MAX_APPROVAL_AGE_DAYS + 10)
    report = assess_live_trading_eligibility(_patched(live_enable_approved_at=old))
    assert not report.permits_live_trading
    assert [r.name for r in report.failed_gates] == ["approval_is_current"]


def test_future_dated_approval_blocks():
    ahead = elig._now().date() + timedelta(days=30)
    assert not assess_live_trading_eligibility(
        _patched(live_enable_approved_at=ahead)
    ).permits_live_trading


@pytest.mark.parametrize(
    "patch",
    [
        {"broker_reachable_at": None},
        {"broker_reachable_at": datetime(2020, 1, 1, tzinfo=timezone.utc)},
        {"last_reconciliation_at": None},
        {"last_reconciliation_at": datetime(2020, 1, 1, tzinfo=timezone.utc)},
    ],
    ids=["no-broker-heartbeat", "old-broker-heartbeat",
         "no-reconciliation", "old-reconciliation"],
)
def test_stale_operational_timestamps_block(patch):
    assert not assess_live_trading_eligibility(_patched(**patch)).permits_live_trading


# ===========================================================================
# REGRESSION: broker unavailability, failed reconciliation, risk breaches
# ===========================================================================

@pytest.mark.parametrize(
    "patch,why",
    [
        ({"broker_session_valid": False}, "broker session invalid"),
        ({"broker_api_degraded": True}, "broker API degraded"),
        ({"broker_session_valid": None}, "broker session unknown"),
        ({"broker_api_degraded": None}, "broker degradation unknown"),
    ],
)
def test_broker_unavailability_blocks(patch, why):
    report = assess_live_trading_eligibility(_patched(**patch))
    assert not report.permits_live_trading, why
    assert report.state is EligibilityState.BLOCKED_EXECUTION_NOT_VALIDATED


@pytest.mark.parametrize(
    "patch,why",
    [
        ({"last_reconciliation_succeeded": False}, "reconciliation failed"),
        ({"last_reconciliation_succeeded": None}, "reconciliation outcome unknown"),
        ({"open_reconciliation_breaks": 1}, "one unexplained position break"),
        ({"open_reconciliation_breaks": 10_000}, "many breaks"),
    ],
)
def test_failed_reconciliation_blocks(patch, why):
    report = assess_live_trading_eligibility(_patched(**patch))
    assert not report.permits_live_trading, why


@pytest.mark.parametrize(
    "patch,why",
    [
        ({"active_risk_limit_breaches": 1}, "one live limit breach"),
        ({"active_risk_limit_breaches": None}, "breach count unknown"),
        ({"current_drawdown": 0.30}, "30% live drawdown"),
        ({"current_drawdown": None}, "drawdown unknown"),
        ({"risk_limits_loaded": False}, "limits not loaded"),
        ({"kill_switch_engaged": True}, "kill switch already engaged"),
        ({"kill_switch_engaged": None}, "kill switch state unknown"),
    ],
)
def test_risk_limit_violations_block(patch, why):
    report = assess_live_trading_eligibility(_patched(**patch))
    assert not report.permits_live_trading, why
    assert report.state is EligibilityState.BLOCKED_RISK_LIMIT_BREACH


def test_an_engaged_kill_switch_is_never_countermanded():
    """
    If something has already decided to halt trading, this module must agree.
    """
    report = assess_live_trading_eligibility(_patched(kill_switch_engaged=True))
    assert not report.permits_live_trading
    with pytest.raises(LiveTradingBlocked):
        require_live_eligible(report, action="trade despite the kill switch")


# ===========================================================================
# REGRESSION: gate substitution
#
# `assess_live_trading_eligibility(evidence, gates=...)` accepted any gate
# objects at all. One same-named gate with a trivially-passing predicate —
# four lines, the exact shape of the test helper directly above — produced
# LIVE_ELIGIBLE from a completely empty Evidence.
# ===========================================================================

def test_substituting_every_gate_cannot_buy_eligibility():
    spoofed = tuple(
        Gate(g.name, g.category, g.blocking_state, g.requirement, lambda e: (True, "ok"))
        for g in ALL_GATES
    )
    report = assess_live_trading_eligibility(Evidence(), gates=spoofed)
    assert not report.permits_live_trading
    assert report.state is EligibilityState.BLOCKED_INSUFFICIENT_DATA
    assert len(report.failed_gates) == len(ALL_GATES)


def test_substituting_one_gate_cannot_hide_a_real_failure():
    evidence = _patched(kill_switch_verified=False)
    spoofed = _substitute_gate("kill_switch", lambda e: (True, "looks fine to me"))
    report = assess_live_trading_eligibility(evidence, gates=spoofed)

    assert not report.permits_live_trading
    gate = next(r for r in report.results if r.name == "kill_switch")
    assert gate.passed is False
    assert "SUBSTITUTED" in gate.reason


def test_a_substituted_gate_may_still_report_a_failure():
    """
    Failure injection must keep working, or the rest of this file cannot test
    anything. Only a substituted gate's *passes* are discarded.
    """
    gates = _substitute_gate("kill_switch", lambda e: (False, "injected failure"))
    report = assess_live_trading_eligibility(_compliant_evidence(), gates=gates)
    gate = next(r for r in report.results if r.name == "kill_switch")
    assert gate.passed is False
    assert gate.reason == "injected failure"


def test_an_extra_unregistered_gate_cannot_grant_anything():
    extra = ALL_GATES + (
        Gate("i_invented_this", GateCategory.DATA, EligibilityState.PAPER_ONLY,
             "invented", lambda e: (True, "sure")),
    )
    report = assess_live_trading_eligibility(_patched(kill_switch_verified=False),
                                            gates=extra)
    assert not report.permits_live_trading


# ===========================================================================
# REGRESSION: monkeypatching the registry
#
# `ALL_GATES` is a rebindable module global and `__post_init__` used to read
# it. Setting `eligibility.ALL_GATES = ()` made an empty report LIVE_ELIGIBLE.
# ===========================================================================

def test_rebinding_all_gates_cannot_widen_the_verdict(monkeypatch):
    monkeypatch.setattr(elig, "ALL_GATES", ())
    report = EligibilityReport(results=())
    assert not report.permits_live_trading
    assert len(report.results) == len(elig._CANONICAL_GATES)


def test_rebinding_all_gates_to_a_trivial_gate_cannot_widen_the_verdict(monkeypatch):
    trivial = (Gate("anything", GateCategory.DATA, EligibilityState.PAPER_ONLY,
                    "nothing", lambda e: (True, "ok")),)
    monkeypatch.setattr(elig, "ALL_GATES", trivial)
    assert not assess_live_trading_eligibility(Evidence()).permits_live_trading


# ===========================================================================
# REGRESSION: forged and deserialized reports
#
# `from_dict` trusted the `passed` booleans in the payload. A hand-written JSON
# file with 23 `"passed": true` entries produced a report whose checklist read
# "LIVE PERMITTED : YES".
# ===========================================================================

def _forged_payload(passed: bool = True) -> dict:
    return {
        "evaluated_at": elig._now().isoformat(),
        "evidence_notes": ["forged"],
        "results": [
            {"name": g.name, "category": g.category.value,
             "blocking_state": g.blocking_state.value, "requirement": g.requirement,
             "passed": passed, "reason": "trust me"}
            for g in ALL_GATES
        ],
    }


def test_a_forged_payload_claiming_every_gate_passed_authorizes_nothing():
    forged = EligibilityReport.from_json(json.dumps(_forged_payload()))
    assert forged.provenance is ReportProvenance.UNTRUSTED
    assert not forged.permits_live_trading
    assert "LIVE PERMITTED : NO" in forged.checklist()
    assert "untrusted" in forged.checklist()
    with pytest.raises(LiveTradingBlocked, match="not computed in-process"):
        require_live_eligible(forged, action="forged payload")


def test_tampered_json_cannot_declare_eligibility():
    """
    Flipping the serialized verdict string must not survive deserialization:
    the state is recomputed from the gate results, not read from the payload.
    """
    payload = json.loads(assess_repo_live_trading_eligibility().to_json())
    payload["state"] = "live_eligible"
    payload["permits_live_trading"] = True
    payload["provenance"] = "computed"

    restored = EligibilityReport.from_dict(payload)
    assert restored.state is EligibilityState.BLOCKED_INSUFFICIENT_DATA
    assert not restored.permits_live_trading
    assert restored.provenance is ReportProvenance.UNTRUSTED


def test_report_with_results_stripped_from_json_is_blocked():
    payload = json.loads(assess_live_trading_eligibility(_compliant_evidence()).to_json())
    payload["results"] = []
    assert not EligibilityReport.from_dict(payload).permits_live_trading


def test_a_forged_payload_cannot_downgrade_the_severity_of_a_failure():
    """
    Relabelling every failure as the mildest blocking state used to work, and
    would have told an operator the system was nearly ready when its data layer
    was broken.
    """
    real = assess_live_trading_eligibility(
        _patched(point_in_time_fundamentals_available=False)
    )
    payload = json.loads(real.to_json())
    for entry in payload["results"]:
        entry["blocking_state"] = "paper_only"
        entry["category"] = "operational"
    restored = EligibilityReport.from_dict(payload)
    assert restored.state is real.state
    assert restored.state is EligibilityState.BLOCKED_INSUFFICIENT_DATA


def test_a_hand_built_report_is_untrusted_even_when_every_result_passes():
    results = tuple(
        GateResult(g.name, g.category, g.blocking_state, g.requirement, True, "fine")
        for g in ALL_GATES
    )
    report = EligibilityReport(results=results)
    assert report.state is EligibilityState.LIVE_ELIGIBLE   # it says so...
    assert not report.permits_live_trading                  # ...and it authorizes nothing
    with pytest.raises(LiveTradingBlocked):
        require_live_eligible(report, action="hand-built")


def test_mutating_a_computed_report_downgrades_it_to_untrusted():
    good = assess_live_trading_eligibility(_compliant_evidence())
    assert good.permits_live_trading
    mutated = dataclasses.replace(good, evidence_notes=("edited",))
    assert mutated.provenance is ReportProvenance.UNTRUSTED
    assert not mutated.permits_live_trading


# ===========================================================================
# REGRESSION: subclassing
# ===========================================================================

def test_a_subclass_that_lies_about_its_state_is_refused_by_the_enforcer():
    class Liar(EligibilityReport):
        @property
        def state(self):
            return EligibilityState.LIVE_ELIGIBLE

        @property
        def permits_live_trading(self):
            return True

    liar = Liar(results=())
    assert liar.permits_live_trading          # the subclass can say what it likes
    with pytest.raises(LiveTradingBlocked, match="only the exact type"):
        require_live_eligible(liar, action="subclass attack")


# ===========================================================================
# REGRESSION: blank, invisible and mistyped strings
#
# `str.strip()` does not remove U+200B, so an approver recorded as a single
# zero-width space satisfied "a named human took responsibility".
# ===========================================================================

@pytest.mark.parametrize(
    "approver",
    ["", "   ", "\t\n", "​", "​ ", "﻿", " ", "X"],
    ids=["empty", "spaces", "tabs", "zero-width", "zw+nbsp", "bom", "line-sep",
         "single-char"],
)
def test_an_invisible_or_trivial_approver_is_not_a_human(approver):
    report = assess_live_trading_eligibility(_patched(live_enable_approved_by=approver))
    assert not report.permits_live_trading, repr(approver)


@pytest.mark.parametrize(
    "benchmark", ["", "   ", "\t", "​"],
    ids=["empty", "spaces", "tab", "zero-width"],
)
def test_a_blank_benchmark_name_is_not_a_benchmark(benchmark):
    assert not assess_live_trading_eligibility(
        _patched(benchmark_name=benchmark)
    ).permits_live_trading


@pytest.mark.parametrize("mode", ["paper", "live ", " live", "LIVE", "Live", "", None])
def test_only_the_exact_string_live_enables_live_mode(mode):
    assert not assess_live_trading_eligibility(
        _patched(trading_mode=mode)
    ).permits_live_trading


def test_live_mode_without_a_human_approver_is_not_eligible():
    """A config flag alone must not open the gate."""
    for patch in (
        {"live_enable_approved_by": ""},
        {"live_enable_approved_by": "   "},
        {"live_enable_approved_at": None},
    ):
        report = assess_live_trading_eligibility(_patched(**patch))
        assert not report.permits_live_trading
        assert report.state is EligibilityState.PAPER_ONLY


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("kill_switch_verified", 1),
        ("kill_switch_verified", "true"),
        ("broker_session_valid", "yes"),
        ("price_history_symbols", "100"),
        ("oos_sharpe", "1.5"),
        ("evidence_gathered_at", "2026-09-03T00:00:00Z"),
        ("price_history_end", "2026-09-03"),
    ],
)
def test_mistyped_evidence_blocks(field_name, value):
    """A string that looks like a number is not a measurement."""
    assert not assess_live_trading_eligibility(
        _patched(**{field_name: value})
    ).permits_live_trading


# ===========================================================================
# ENFORCEMENT — require_live_eligible
# ===========================================================================

def test_require_live_eligible_blocks_on_the_real_repo_state():
    """The whole point. Today, the live order path must not be permitted."""
    with pytest.raises(LiveTradingBlocked) as exc:
        require_live_eligible(action="submit a live order")
    assert exc.value.state.is_blocked
    assert "gate(s) not passing" in exc.value.reason


def test_require_live_eligible_passes_on_a_fresh_compliant_report():
    report = assess_live_trading_eligibility(_compliant_evidence())
    returned = require_live_eligible(report, action="submit a live order")
    assert returned is report


def test_require_live_eligible_rejects_a_stale_but_genuinely_computed_report():
    """
    A LIVE_ELIGIBLE report computed three hours ago is a statement about three
    hours ago. Positions, broker sessions and risk usage all move in between.
    """
    now = elig._now()
    with frozen_clock(now - timedelta(hours=3)):
        old = assess_live_trading_eligibility(_compliant_evidence())
    assert old.provenance is ReportProvenance.COMPUTED
    assert old.state is EligibilityState.LIVE_ELIGIBLE
    assert old.is_stale
    with pytest.raises(LiveTradingBlocked, match="minutes old"):
        require_live_eligible(old, action="trade on a stale verdict")


def test_a_report_just_inside_the_age_limit_is_accepted():
    now = elig._now()
    with frozen_clock(now - timedelta(minutes=MAX_REPORT_AGE_MINUTES / 2)):
        recent = assess_live_trading_eligibility(_compliant_evidence())
    assert require_live_eligible(recent, action="trade") is recent


@pytest.mark.parametrize(
    "not_a_report", [None if False else 0, "live_eligible", {"state": "live_eligible"},
                     object()],
    ids=["int", "string", "dict", "object"],
)
def test_require_live_eligible_rejects_anything_that_is_not_a_report(not_a_report):
    with pytest.raises(LiveTradingBlocked):
        require_live_eligible(not_a_report, action="type confusion")


def test_require_live_eligible_blocks_when_the_assessment_itself_raises(monkeypatch):
    """An unevaluable gate blocks. An unevaluable *assessment* blocks harder."""
    def boom():
        raise OSError("the evidence store is on fire")

    monkeypatch.setattr(elig, "gather_repo_evidence", boom)
    with pytest.raises(LiveTradingBlocked, match="could not be assessed"):
        require_live_eligible(action="submit a live order")


def test_require_live_eligible_does_not_trust_the_reports_own_state(monkeypatch):
    """
    Enforcement re-derives from the results rather than reading `.state`, so a
    bug in the state property cannot open the gate.
    """
    report = assess_live_trading_eligibility(_patched(kill_switch_verified=False))
    monkeypatch.setattr(
        type(report), "state",
        property(lambda self: EligibilityState.LIVE_ELIGIBLE),
    )
    assert report.state is EligibilityState.LIVE_ELIGIBLE  # the lie is in place
    with pytest.raises(LiveTradingBlocked):
        require_live_eligible(report, action="trust the state property")


def test_risk_reducing_actions_are_permitted_while_blocked():
    """
    A blocked system must still be able to get flat. This carve-out is scoped
    to flatten/square-off/cancel and is logged every time.
    """
    returned = require_live_eligible(
        action="emergency flatten", intent=OrderIntent.REDUCE_RISK
    )
    # It permitted the ACTION; it did not hand back an eligibility grant.
    assert not returned.permits_live_trading


def test_risk_reducing_carve_out_is_logged_loudly(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="app.governance.eligibility"):
        require_live_eligible(action="flatten NIFTY", intent=OrderIntent.REDUCE_RISK)
    assert any("RISK-REDUCING ACTION PERMITTED" in r.message for r in caplog.records)
    assert any("flatten NIFTY" in str(r.args) or "flatten NIFTY" in r.getMessage()
               for r in caplog.records)


def test_a_block_is_logged_at_critical(caplog):
    import logging
    with caplog.at_level(logging.CRITICAL, logger="app.governance.eligibility"):
        with pytest.raises(LiveTradingBlocked):
            require_live_eligible(action="submit a live order")
    assert any(r.levelno >= logging.CRITICAL for r in caplog.records)
    assert any("LIVE TRADING BLOCKED" in r.getMessage() for r in caplog.records)


def test_an_unrecognized_intent_blocks():
    with pytest.raises(LiveTradingBlocked):
        require_live_eligible(action="sneaky", intent="increase_risk")  # type: ignore[arg-type]


def test_no_bypass_mechanism_exists():
    """
    There is deliberately no override: no force flag, no environment variable,
    no config switch. If someone adds one, this test fails and they must argue
    for it in review rather than land it quietly.
    """
    import inspect

    params = set(inspect.signature(require_live_eligible).parameters)
    assert not (params & {"force", "override", "bypass", "skip", "allow", "unsafe"})

    source = inspect.getsource(elig)
    code = "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("#")
    )
    for forbidden in ("os.environ", "os.getenv", "getenv("):
        assert forbidden not in code, f"{forbidden} appeared in the eligibility module"


def test_the_enforcement_gate_detects_that_the_order_path_is_not_wired():
    """
    The gate that audits its own teeth. When someone wires
    require_live_eligible() into OrderManager.submit_order, this test's
    assertion flips — update it then, deliberately.
    """
    wired = elig._enforcement_is_wired()
    evidence = gather_repo_evidence()
    assert evidence.eligibility_enforced_at_order_path is wired
    gate = next(
        r for r in assess_repo_live_trading_eligibility().results
        if r.name == "enforcement_wired_into_order_path"
    )
    assert gate.passed is bool(wired)
    if not wired:
        assert "decoration" in gate.requirement or "require_live_eligible" in gate.reason


def test_unreadable_order_path_is_treated_as_unwired(monkeypatch):
    monkeypatch.setattr(elig, "_ORDER_PATH", elig.Path("/nonexistent/order_manager.py"))
    assert elig._enforcement_is_wired() is None
    assert gather_repo_evidence().eligibility_enforced_at_order_path is None


# ===========================================================================
# Severity: the most severe blocking state wins
# ===========================================================================

def test_severity_ordering_is_strict_and_total():
    severities = [s.severity for s in EligibilityState]
    assert severities == sorted(severities)
    assert len(set(severities)) == len(severities)
    assert EligibilityState.LIVE_ELIGIBLE.severity == max(severities)


def test_most_severe_state_wins_over_a_less_severe_one():
    """Data and execution both broken: the report must say DATA."""
    report = assess_live_trading_eligibility(_patched(
        point_in_time_fundamentals_available=False,   # data
        kill_switch_verified=False,                   # execution
        paper_trading_days=1,                         # operational
    ))
    assert report.state is EligibilityState.BLOCKED_INSUFFICIENT_DATA
    assert {r.name for r in report.failed_gates} == {
        "point_in_time_fundamentals", "kill_switch", "paper_trading_duration",
    }
    assert "point_in_time_fundamentals" in report.blocking_reason


def test_severity_cascade_across_every_layer():
    """Fix the layers one at a time; the reported state must climb in step."""
    evidence = _patched(
        point_in_time_fundamentals_available=False,
        purged_walk_forward_completed=False,
        deflated_sharpe_ratio=0.10,
        oos_sharpe=0.01,
        max_drawdown=0.40,
        kill_switch_verified=False,
        active_risk_limit_breaches=2,
        paper_trading_days=0,
    )
    expected = [
        ({}, EligibilityState.BLOCKED_INSUFFICIENT_DATA),
        ({"point_in_time_fundamentals_available": True},
         EligibilityState.BLOCKED_VALIDATION_INCOMPLETE),
        ({"purged_walk_forward_completed": True},
         EligibilityState.BLOCKED_INSUFFICIENT_STATISTICAL_EVIDENCE),
        ({"deflated_sharpe_ratio": 0.99},
         EligibilityState.BLOCKED_POOR_OOS_PERFORMANCE),
        ({"oos_sharpe": 1.10}, EligibilityState.BLOCKED_EXCESSIVE_DRAWDOWN),
        ({"max_drawdown": 0.12}, EligibilityState.BLOCKED_EXECUTION_NOT_VALIDATED),
        ({"kill_switch_verified": True}, EligibilityState.BLOCKED_RISK_LIMIT_BREACH),
        ({"active_risk_limit_breaches": 0}, EligibilityState.PAPER_ONLY),
        ({"paper_trading_days": 120}, EligibilityState.LIVE_ELIGIBLE),
    ]
    for patch, want in expected:
        evidence = dataclasses.replace(evidence, **patch)
        got = assess_live_trading_eligibility(evidence).state
        assert got is want, f"after {patch}: expected {want.name}, got {got.name}"


def test_operational_failure_alone_yields_paper_only():
    report = assess_live_trading_eligibility(_patched(trading_mode="paper"))
    assert report.state is EligibilityState.PAPER_ONLY
    assert not report.permits_live_trading


def test_dishonest_trial_count_is_rejected():
    """37 trials run, DSR deflated for 1 — the DSR is against a lie."""
    report = assess_live_trading_eligibility(
        _patched(n_trials_recorded=37, n_trials_used_in_dsr=1)
    )
    assert report.state is EligibilityState.BLOCKED_INSUFFICIENT_STATISTICAL_EVIDENCE
    assert "not honest" in report.blocking_reason


@pytest.mark.parametrize(
    "patch",
    [
        {"deflated_sharpe_ratio": 0.95},                 # boundary: must be strictly >
        {"probability_of_backtest_overfitting": 0.50},   # boundary: must be strictly <
        {"net_annual_excess_return_vs_benchmark": 0.0},  # must strictly beat the benchmark
        {"oos_sharpe": 0.49999},
        {"max_drawdown": 0.15000001},
        {"paper_trading_days": 89},
        {"intraday_history_days": 503},
        {"price_history_symbols": 49},
    ],
)
def test_boundary_values_are_rejected(patch):
    assert not assess_live_trading_eligibility(_patched(**patch)).permits_live_trading


@pytest.mark.parametrize(
    "patch",
    [
        {"deflated_sharpe_ratio": 0.9500001},
        {"probability_of_backtest_overfitting": 0.4999999},
        {"oos_sharpe": 0.50},
        {"max_drawdown": 0.15},
        {"paper_trading_days": 90},
        {"intraday_history_days": 504},
        {"price_history_symbols": 50},
    ],
)
def test_values_exactly_at_the_permitted_edge_are_accepted(patch):
    """The bars must be reachable, or they are not bars but walls."""
    assert assess_live_trading_eligibility(_patched(**patch)).permits_live_trading


# ===========================================================================
# The real-world assertion
# ===========================================================================

def test_current_repo_state_is_data_blocked():
    """
    The honest status of this repository today.

    Real daily NSE OHLCV now exists, so the price-history gate passes, and the
    verdict is now actually ENFORCED on the order path. Nothing else passes:
    the data is not current, there are no point-in-time index constituents, no
    point-in-time fundamentals and essentially no intraday history, and no
    statistical, performance, execution, risk or operational result has ever
    been persisted.

    `enforcement_wired_into_order_path` was previously among the FAILING gates,
    which was the most serious finding of the eligibility audit: the verdict
    was purely advisory, so the system could have placed live orders while
    reporting BLOCKED. `OrderManager.submit_order` now calls
    `require_live_eligible()` before touching a broker, keyed on the broker's
    own declared mode, so that gate passes. Everything it gates remains
    blocked — being enforced is not the same as being eligible.
    """
    report = assess_repo_live_trading_eligibility()

    assert not report.permits_live_trading
    assert report.state is EligibilityState.BLOCKED_INSUFFICIENT_DATA

    failed = {r.name for r in report.failed_gates}
    passed = {r.name for r in report.results if r.passed}

    assert "point_in_time_index_constituents" in failed
    assert "point_in_time_fundamentals" in failed
    assert "intraday_history_span" in failed
    assert "risk_limits_enforced" in failed
    assert "broker_connectivity" in failed
    assert "reconciliation_current" in failed

    # Regression: enforcement must STAY wired. If this flips back to failing,
    # the eligibility verdict has become advisory again and stops nothing.
    assert "enforcement_wired_into_order_path" in passed, (
        "eligibility is no longer enforced on the order path"
    )

    # No research result exists, so every non-data category is unproven — with
    # exactly one exception. `enforcement_wired_into_order_path` is an EXECUTION
    # gate that asserts a property of THIS CODEBASE (does the order path call
    # require_live_eligible?) rather than a property of a broker session, so it
    # can be satisfied by writing code. Everything else in these categories
    # needs evidence from a real broker, a real backtest or a real paper run,
    # none of which exists.
    proven_by_code_alone = {"enforcement_wired_into_order_path"}

    for category in (GateCategory.STATISTICAL, GateCategory.PERFORMANCE,
                     GateCategory.EXECUTION, GateCategory.OPERATIONAL):
        in_cat = [r for r in report.results if r.category is category]
        assert in_cat, category
        unexpectedly_passing = [
            r.name for r in in_cat
            if r.passed and r.name not in proven_by_code_alone
        ]
        assert not unexpectedly_passing, (
            f"{category} gate(s) passing without evidence: {unexpectedly_passing}"
        )


def test_repo_evidence_reports_paper_mode():
    evidence = gather_repo_evidence()
    assert evidence.trading_mode == "paper"
    assert evidence.point_in_time_index_constituents_available is False


def test_repo_evidence_leaves_unmeasured_fields_unknown():
    """Nothing may be invented. Unmeasured means None, and None fails."""
    evidence = gather_repo_evidence()
    for name in (
        "deflated_sharpe_ratio", "probability_of_backtest_overfitting", "oos_sharpe",
        "max_drawdown", "n_trials_recorded", "purged_walk_forward_completed",
        "broker_auth_verified", "kill_switch_verified", "paper_trading_days",
        "live_enable_approved_by", "broker_reachable_at", "broker_session_valid",
        "last_reconciliation_at", "last_reconciliation_succeeded",
        "risk_limits_loaded", "active_risk_limit_breaches", "current_drawdown",
        "market_data_feed_healthy",
    ):
        assert getattr(evidence, name) is None, f"{name} was invented"


def test_repo_evidence_is_freshly_stamped():
    evidence = gather_repo_evidence()
    assert evidence.evidence_gathered_at is not None
    assert evidence.evidence_gathered_at.tzinfo is not None
    assert abs((elig._now() - evidence.evidence_gathered_at).total_seconds()) < 60


def test_checklist_names_the_blocking_gates():
    text = assess_repo_live_trading_eligibility().checklist()
    assert "BLOCKED_INSUFFICIENT_DATA" in text
    assert "LIVE PERMITTED : NO" in text
    assert "point_in_time_index_constituents" in text
    for category in GateCategory:
        assert category.value.upper() in text


def test_there_are_at_least_as_many_gates_as_the_audit_required():
    """23 gates were the original bar; gates may be added but not quietly lost."""
    assert len(ALL_GATES) >= 23
    assert len({g.name for g in ALL_GATES}) == len(ALL_GATES)


# ===========================================================================
# Serialization
# ===========================================================================

def test_json_round_trips_the_checklist_content():
    original = assess_repo_live_trading_eligibility()
    restored = EligibilityReport.from_json(original.to_json())

    assert restored.state is original.state
    assert restored.results == original.results
    assert restored.evaluated_at == original.evaluated_at
    assert restored.evidence_notes == original.evidence_notes
    # ...but the restored copy is a record, not an authorization.
    assert restored.provenance is ReportProvenance.UNTRUSTED


def test_json_round_trip_preserves_the_state_but_not_the_authority():
    original = assess_live_trading_eligibility(_compliant_evidence())
    restored = EligibilityReport.from_json(original.to_json())
    assert restored.state is EligibilityState.LIVE_ELIGIBLE
    assert original.permits_live_trading
    assert not restored.permits_live_trading


def test_json_payload_carries_the_verdict_and_the_failures():
    payload = json.loads(assess_repo_live_trading_eligibility().to_json())
    assert payload["state"] == "blocked_insufficient_data"
    assert payload["permits_live_trading"] is False
    assert payload["provenance"] == "computed"
    assert payload["n_failed"] == payload["n_gates"] - payload["n_passed"]
    assert len(payload["failed_gates"]) == payload["n_failed"]
    assert len(payload["results"]) == len(ALL_GATES)
    for entry in payload["failed_gates"]:
        assert entry["reason"]
        assert entry["requirement"]


def test_gate_result_round_trips():
    result = GateResult(
        name="kill_switch",
        category=GateCategory.EXECUTION,
        blocking_state=EligibilityState.BLOCKED_EXECUTION_NOT_VALIDATED,
        requirement=next(g.requirement for g in ALL_GATES if g.name == "kill_switch"),
        passed=False,
        reason="not verified",
    )
    assert GateResult.from_dict(result.to_dict()) == result


def test_evaluated_at_is_timezone_aware():
    report = EligibilityReport(results=(), evaluated_at=elig._now())
    assert report.evaluated_at.tzinfo is not None
    assert assess_live_trading_eligibility().evaluated_at.tzinfo is not None


# ===========================================================================
# Randomized property test
#
# Generate many Evidence records from a pool of good, bad, missing and poisoned
# values and assert the central property: LIVE_ELIGIBLE is reached ONLY when
# every canonical gate genuinely passes, and never on unknown or impossible
# input.
# ===========================================================================

def _value_pool(now: datetime) -> dict[str, tuple]:
    """For each field: (good, *bad). `None` is always in the bad set."""
    good = _compliant_evidence(now)
    pool: dict[str, tuple] = {}
    for f in dataclasses.fields(Evidence):
        if f.name == "notes":
            continue
        g = getattr(good, f.name)
        bad: list = [None]
        if f.name in elig._NUMERIC_DOMAINS:
            lo, hi = elig._NUMERIC_DOMAINS[f.name]
            bad += [float("nan"), float("inf"), float("-inf"), lo - 1.0, hi + 1.0,
                    -1.0, 1e300]
        elif f.name in elig._BOOL_FIELDS:
            bad += [not g, 1, "yes"]
        elif f.name in elig._DATETIME_FIELDS:
            bad += [now - timedelta(days=400), now + timedelta(days=5),
                    now.replace(tzinfo=None), "2026-01-01"]
        elif f.name in elig._DATE_FIELDS:
            bad += [date(1900, 1, 1), now.date() + timedelta(days=400), "2026-01-01"]
        elif f.name == "trading_mode":
            bad += ["paper", "LIVE", "live ", ""]
        elif f.name == "benchmark_name":
            bad += ["", "   ", "​"]
        elif f.name == "live_enable_approved_by":
            bad += ["", "  ", "​", "X"]
        else:  # pragma: no cover - the field table above is exhaustive today
            bad += [""]
        pool[f.name] = (g, tuple(bad))
    return pool


def _independently_compliant(evidence: Evidence, good: Evidence) -> bool:
    """
    A deliberately dumb oracle: this evidence is compliant iff every field is
    exactly the known-good value. It cannot be gamed by a bug in the gate logic
    because it does not consult the gate logic at all.
    """
    for f in dataclasses.fields(Evidence):
        if f.name == "notes":
            continue
        a, b = getattr(evidence, f.name), getattr(good, f.name)
        if isinstance(a, float) and math.isnan(a):
            return False
        if a != b or type(a) is not type(b):
            return False
    return True


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_randomized_evidence_reaches_live_eligible_only_when_everything_passes(seed):
    """
    The headline property. 400 scenarios per seed, 2000 in total.

    Asserted for every generated scenario:
      P1  permits_live_trading  =>  all 31 canonical gates present and passed
      P2  permits_live_trading  =>  no field is None
      P3  permits_live_trading  =>  no numeric field is NaN or infinite
      P4  permits_live_trading  <=> the independent oracle says fully compliant
      P5  permits_live_trading  <=> require_live_eligible does not raise
    """
    rng = random.Random(seed)
    now = elig._now()
    good = _compliant_evidence(now)
    pool = _value_pool(now)
    names = list(pool)
    canonical = {g.name for g in ALL_GATES}

    scenarios = 0
    violations: list[str] = []
    eligible_seen = 0

    for _ in range(400):
        # Corrupt between 0 and 4 fields. 0 must yield LIVE_ELIGIBLE; anything
        # else must not.
        n_bad = rng.choices([0, 1, 2, 3, 4], weights=[8, 40, 25, 15, 12])[0]
        patch = {}
        for name in rng.sample(names, n_bad):
            patch[name] = rng.choice(pool[name][1])

        evidence = dataclasses.replace(good, **patch)
        report = assess_live_trading_eligibility(evidence)
        permitted = report.permits_live_trading
        scenarios += 1

        if permitted:
            eligible_seen += 1
            passed = {r.name for r in report.results if r.passed is True}
            if not canonical.issubset(passed):                       # P1
                violations.append(f"P1 {patch}: missing {sorted(canonical - passed)}")
            for f in dataclasses.fields(Evidence):                   # P2 / P3
                if f.name == "notes":
                    continue
                v = getattr(evidence, f.name)
                if v is None:
                    violations.append(f"P2 {patch}: {f.name} is None")
                if isinstance(v, float) and not math.isfinite(v):
                    violations.append(f"P3 {patch}: {f.name} is {v}")

        oracle = _independently_compliant(evidence, good)             # P4
        if oracle != permitted:
            violations.append(
                f"P4 {patch}: oracle={oracle} module={permitted} state={report.state.name}"
            )

        try:                                                          # P5
            require_live_eligible(report, action="property test")
            enforced = True
        except LiveTradingBlocked:
            enforced = False
        if enforced != permitted:
            violations.append(f"P5 {patch}: permits={permitted} enforced={enforced}")

    assert scenarios == 400
    assert eligible_seen > 0, "the generator never produced a compliant record"
    assert not violations, (
        f"{len(violations)} violation(s) in {scenarios} scenarios:\n"
        + "\n".join(violations[:20])
    )


def test_randomized_single_field_corruption_always_blocks():
    """
    Exhaustive rather than random: every field x every bad value, one at a
    time. If any single corruption still yields LIVE_ELIGIBLE, that field is
    not gated.
    """
    now = elig._now()
    good = _compliant_evidence(now)
    pool = _value_pool(now)

    scenarios = 0
    violations = []
    for name, (_, bads) in pool.items():
        for bad in bads:
            scenarios += 1
            report = assess_live_trading_eligibility(
                dataclasses.replace(good, **{name: bad})
            )
            if report.permits_live_trading:
                violations.append(f"{name}={bad!r}")

    assert scenarios > 200
    assert not violations, (
        f"{len(violations)}/{scenarios} single-field corruptions reached "
        f"LIVE_ELIGIBLE: {violations[:20]}"
    )
