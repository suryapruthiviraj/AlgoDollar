"""
Tests for the live-trading eligibility gate.

This component decides whether real money may move. The tests below are
therefore adversarial rather than illustrative: each one tries to obtain a
permissive verdict by a route a careless change might open — an empty report, a
trimmed gate list, a crashing gate, a missing piece of evidence — and asserts
that the route stays closed.

The one test that is not adversarial is `test_current_repo_state_is_data_blocked`,
which asserts what this repository actually evaluates to today.
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime, timezone

import pytest

from app.governance.eligibility import (
    ALL_GATES,
    EligibilityReport,
    EligibilityState,
    Evidence,
    Gate,
    GateCategory,
    GateResult,
    assess_live_trading_eligibility,
    assess_repo_live_trading_eligibility,
    gather_repo_evidence,
)

BLOCKED_STATES = tuple(s for s in EligibilityState if s is not EligibilityState.LIVE_ELIGIBLE)


# ===========================================================================
# Fixtures
# ===========================================================================

def _compliant_evidence() -> Evidence:
    """
    The only Evidence in this file that clears every gate.

    It describes a system that does not exist. Nothing in this repository can
    produce these values today; they are written by hand so that the tests can
    prove LIVE_ELIGIBLE is reachable in principle and unreachable in practice
    without each of them.
    """
    return Evidence(
        price_history_symbols=100,
        price_history_start=date(2006, 1, 1),
        price_history_end=date(2025, 1, 1),
        price_history_is_real_market_data=True,
        point_in_time_index_constituents_available=True,
        point_in_time_fundamentals_available=True,
        intraday_history_days=750,
        data_quality_audit_passed=True,
        real_ohlc_available=True,
        corporate_actions_adjusted=True,
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
        paper_trading_days=120,
        paper_sharpe=0.95,
        backtest_expected_sharpe=1.10,
        trading_mode="live",
        live_enable_approved_by="A. Human",
        live_enable_approved_at=date(2026, 1, 15),
    )


def _substitute_gate(name: str, predicate) -> tuple[Gate, ...]:
    """Replace one gate with another of the same name, keeping the rest intact."""
    return tuple(
        Gate(g.name, g.category, g.blocking_state, g.requirement, predicate)
        if g.name == name else g
        for g in ALL_GATES
    )


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


def test_gate_returning_garbage_does_not_crash_the_assessment():
    """A predicate with the wrong shape is a broken gate, hence a failed gate."""
    gates = _substitute_gate("kill_switch", lambda e: "not a tuple")
    report = assess_live_trading_eligibility(_compliant_evidence(), gates=gates)
    assert not report.permits_live_trading


# ===========================================================================
# LIVE_ELIGIBLE requires every gate
# ===========================================================================

def test_fully_compliant_evidence_is_live_eligible():
    """LIVE_ELIGIBLE must be reachable, or the gate is a wall rather than a gate."""
    report = assess_live_trading_eligibility(_compliant_evidence())
    assert report.failed_gates == (), report.checklist()
    assert report.state is EligibilityState.LIVE_ELIGIBLE
    assert report.permits_live_trading


_EVIDENCE_FIELDS = [f.name for f in dataclasses.fields(Evidence) if f.name != "notes"]


@pytest.mark.parametrize("field_name", _EVIDENCE_FIELDS)
def test_every_piece_of_evidence_is_load_bearing(field_name):
    """
    Erase one fact from an otherwise perfect record and eligibility must be
    lost. This proves two things at once: unknown evidence fails, and no
    declared evidence field is decorative.
    """
    evidence = dataclasses.replace(_compliant_evidence(), **{field_name: None})
    report = assess_live_trading_eligibility(evidence)
    assert not report.permits_live_trading, f"{field_name} did not gate anything"


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
    evidence = dataclasses.replace(
        _compliant_evidence(),
        point_in_time_fundamentals_available=False,   # data
        kill_switch_verified=False,                   # execution
        paper_trading_days=1,                         # operational
    )
    report = assess_live_trading_eligibility(evidence)
    assert report.state is EligibilityState.BLOCKED_INSUFFICIENT_DATA
    assert {r.name for r in report.failed_gates} == {
        "point_in_time_fundamentals", "kill_switch", "paper_trading_duration",
    }
    assert "point_in_time_fundamentals" in report.blocking_reason


def test_severity_cascade_across_every_layer():
    """Fix the layers one at a time; the reported state must climb in step."""
    evidence = dataclasses.replace(
        _compliant_evidence(),
        point_in_time_fundamentals_available=False,
        purged_walk_forward_completed=False,
        deflated_sharpe_ratio=0.10,
        oos_sharpe=0.01,
        max_drawdown=0.40,
        kill_switch_verified=False,
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
        ({"kill_switch_verified": True}, EligibilityState.PAPER_ONLY),
        ({"paper_trading_days": 120}, EligibilityState.LIVE_ELIGIBLE),
    ]
    for patch, want in expected:
        evidence = dataclasses.replace(evidence, **patch)
        got = assess_live_trading_eligibility(evidence).state
        assert got is want, f"after {patch}: expected {want.name}, got {got.name}"


def test_operational_failure_alone_yields_paper_only():
    evidence = dataclasses.replace(_compliant_evidence(), trading_mode="paper")
    report = assess_live_trading_eligibility(evidence)
    assert report.state is EligibilityState.PAPER_ONLY
    assert not report.permits_live_trading


def test_live_mode_without_a_human_approver_is_not_eligible():
    """A config flag alone must not open the gate."""
    for patch in (
        {"live_enable_approved_by": ""},
        {"live_enable_approved_by": "   "},
        {"live_enable_approved_at": None},
    ):
        evidence = dataclasses.replace(_compliant_evidence(), **patch)
        assert assess_live_trading_eligibility(evidence).state is EligibilityState.PAPER_ONLY


def test_dishonest_trial_count_is_rejected():
    """37 trials run, DSR deflated for 1 — the DSR is against a lie."""
    evidence = dataclasses.replace(
        _compliant_evidence(), n_trials_recorded=37, n_trials_used_in_dsr=1
    )
    report = assess_live_trading_eligibility(evidence)
    assert report.state is EligibilityState.BLOCKED_INSUFFICIENT_STATISTICAL_EVIDENCE
    assert "not honest" in report.blocking_reason


@pytest.mark.parametrize(
    "patch",
    [
        {"deflated_sharpe_ratio": 0.95},                 # boundary: must be strictly >
        {"probability_of_backtest_overfitting": 0.50},   # boundary: must be strictly <
        {"net_annual_excess_return_vs_benchmark": 0.0},  # must strictly beat the benchmark
    ],
)
def test_boundary_values_are_rejected(patch):
    evidence = dataclasses.replace(_compliant_evidence(), **patch)
    assert not assess_live_trading_eligibility(evidence).permits_live_trading


# ===========================================================================
# The real-world assertion
# ===========================================================================

def test_current_repo_state_is_data_blocked():
    """
    The honest status of this repository today.

    Real daily NSE OHLCV now exists, so the price-history gate passes. Nothing
    else does: there are no point-in-time index constituents, no point-in-time
    fundamentals, and essentially no intraday history, and no statistical,
    performance, execution or operational result has ever been persisted.
    """
    report = assess_repo_live_trading_eligibility()

    assert not report.permits_live_trading
    assert report.state is EligibilityState.BLOCKED_INSUFFICIENT_DATA

    failed = {r.name for r in report.failed_gates}
    assert "point_in_time_index_constituents" in failed
    assert "point_in_time_fundamentals" in failed
    assert "intraday_history_span" in failed

    # No research result exists, so every non-data category is entirely unproven.
    for category in (GateCategory.STATISTICAL, GateCategory.PERFORMANCE,
                     GateCategory.EXECUTION, GateCategory.OPERATIONAL):
        in_cat = [r for r in report.results if r.category is category]
        assert in_cat and all(not r.passed for r in in_cat), category


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
        "live_enable_approved_by",
    ):
        assert getattr(evidence, name) is None, f"{name} was invented"


def test_checklist_names_the_blocking_gates():
    text = assess_repo_live_trading_eligibility().checklist()
    assert "BLOCKED_INSUFFICIENT_DATA" in text
    assert "LIVE PERMITTED : NO" in text
    assert "point_in_time_index_constituents" in text
    for category in GateCategory:
        assert category.value.upper() in text


# ===========================================================================
# Serialization
# ===========================================================================

def test_json_round_trips():
    original = assess_repo_live_trading_eligibility()
    restored = EligibilityReport.from_json(original.to_json())

    assert restored.to_json() == original.to_json()
    assert restored.state is original.state
    assert restored.results == original.results
    assert restored.evaluated_at == original.evaluated_at
    assert restored.evidence_notes == original.evidence_notes


def test_json_round_trips_for_the_eligible_case():
    original = assess_live_trading_eligibility(_compliant_evidence())
    restored = EligibilityReport.from_json(original.to_json())
    assert restored.state is EligibilityState.LIVE_ELIGIBLE
    assert restored.to_json() == original.to_json()


def test_json_payload_carries_the_verdict_and_the_failures():
    import json

    payload = json.loads(assess_repo_live_trading_eligibility().to_json())
    assert payload["state"] == "blocked_insufficient_data"
    assert payload["permits_live_trading"] is False
    assert payload["n_failed"] == payload["n_gates"] - payload["n_passed"]
    assert len(payload["failed_gates"]) == payload["n_failed"]
    assert len(payload["results"]) == len(ALL_GATES)
    for entry in payload["failed_gates"]:
        assert entry["reason"]
        assert entry["requirement"]


def test_tampered_json_cannot_declare_eligibility():
    """
    Flipping the serialized verdict string must not survive deserialization:
    the state is recomputed from the gate results, not read from the payload.
    """
    import json

    payload = json.loads(assess_repo_live_trading_eligibility().to_json())
    payload["state"] = "live_eligible"
    payload["permits_live_trading"] = True

    restored = EligibilityReport.from_dict(payload)
    assert restored.state is EligibilityState.BLOCKED_INSUFFICIENT_DATA
    assert not restored.permits_live_trading


def test_report_with_results_stripped_from_json_is_blocked():
    import json

    payload = json.loads(assess_live_trading_eligibility(_compliant_evidence()).to_json())
    payload["results"] = []
    restored = EligibilityReport.from_dict(payload)
    assert not restored.permits_live_trading


def test_gate_result_round_trips():
    result = GateResult(
        name="kill_switch",
        category=GateCategory.EXECUTION,
        blocking_state=EligibilityState.BLOCKED_EXECUTION_NOT_VALIDATED,
        requirement="verified",
        passed=False,
        reason="not verified",
    )
    assert GateResult.from_dict(result.to_dict()) == result


def test_evaluated_at_is_timezone_aware():
    report = EligibilityReport(results=(), evaluated_at=datetime.now(timezone.utc))
    assert report.evaluated_at.tzinfo is not None
    assert assess_live_trading_eligibility().evaluated_at.tzinfo is not None
