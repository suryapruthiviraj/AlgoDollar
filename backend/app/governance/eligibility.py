"""
LIVE_TRADING_ELIGIBILITY — the gate between research and real money.

WHY THIS EXISTS
---------------
The adversarial audit (docs/AUDIT_REPORT.md) found ~30 defects, several of
which would have lost money in production: regime detection that had no effect
on allocation, risk caps that were undone by re-normalization, an intraday book
that never squared off. Every one of them was invisible because nothing in the
system was required to state, in one place, whether it was fit to trade.

This module is that one place. It answers a single question — *may this system
place real orders?* — and it is built so that the answer is NO unless every
piece of supporting evidence has been positively supplied.

THE THREE STRUCTURAL PROPERTIES THAT MAKE THIS SAFE
---------------------------------------------------
1. FAIL CLOSED ON SILENCE. Every field of :class:`Evidence` defaults to
   ``None``, and ``None`` is a FAILURE with the reason "no evidence recorded",
   not a skipped check. Absence of evidence is never evidence of absence.

2. FAIL CLOSED ON ERROR. A gate that raises is recorded as FAILED, with the
   exception text as its reason. A crash can never be mistaken for a pass.

3. THE STATE IS DERIVED, NEVER ASSIGNED. :attr:`EligibilityReport.state` is a
   read-only property computed from the gate results. There is no code path,
   accidental or deliberate, that writes ``LIVE_ELIGIBLE`` into a report. It is
   returned only when every gate in :data:`ALL_GATES` produced a passing result
   — and any registered gate missing from a report is injected as a failure, so
   trimming the gate list cannot buy a permissive answer either.

READING THE OUTPUT
------------------
:meth:`EligibilityReport.checklist` prints the failures grouped by category,
worst first. It is meant to be worked top to bottom. ``to_json``/``from_json``
carry the same content to an API or dashboard without loss.

THRESHOLDS ARE PRIORS, NOT ESTIMATES
------------------------------------
The numeric bars below are hand-chosen, defensible defaults — the same honesty
caveat the audit applies to the regime multipliers (§9.8). They are deliberately
conservative. They are not calibrated from data, because there is no validated
strategy from which to calibrate them.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)


# ===========================================================================
# Thresholds
# ===========================================================================

# Data
MIN_PRICE_HISTORY_YEARS = 10.0        # enough to span more than one regime
MIN_PRICE_HISTORY_SYMBOLS = 50        # cross-sectional work needs a cross-section
MIN_INTRADAY_HISTORY_DAYS = 504       # ~2 trading years; 60 days is one regime

# Statistical
MIN_DEFLATED_SHARPE = 0.95            # Bailey & Lopez de Prado; audit §5.3
MAX_PBO = 0.50                        # above 0.5 selection is worse than useless

# Performance
MIN_OOS_SHARPE = 0.50                 # net of costs, out of sample
MAX_DRAWDOWN = 0.15                   # mirrors settings.max_portfolio_drawdown_pct
MIN_SLIPPAGE_STRESS_MULTIPLE = 1.5    # audit §12.9: reject anything dying above 1.5x
MAX_SINGLE_NAME_PNL_SHARE = 0.25
MAX_SECTOR_PNL_SHARE = 0.40
MAX_SINGLE_YEAR_PNL_SHARE = 0.50
MAX_TOP5_TRADES_PNL_SHARE = 0.50

# Operational
MIN_PAPER_TRADING_DAYS = 90           # audit §12.15: 3-6 months minimum
MIN_PAPER_TO_BACKTEST_SHARPE_RATIO = 0.5  # paper must not be half the backtest


# ===========================================================================
# States
# ===========================================================================

class EligibilityState(str, Enum):
    """
    The verdict. Ordered by remediation priority: the first member is the most
    severe and the default.

    Only ``LIVE_ELIGIBLE`` permits real orders. ``PAPER_ONLY`` is a BLOCKED
    state with respect to live trading — it means the research bars are cleared
    but the operational ones are not.
    """

    BLOCKED_INSUFFICIENT_DATA = "blocked_insufficient_data"
    BLOCKED_VALIDATION_INCOMPLETE = "blocked_validation_incomplete"
    BLOCKED_INSUFFICIENT_STATISTICAL_EVIDENCE = "blocked_insufficient_statistical_evidence"
    BLOCKED_POOR_OOS_PERFORMANCE = "blocked_poor_oos_performance"
    BLOCKED_EXCESSIVE_DRAWDOWN = "blocked_excessive_drawdown"
    BLOCKED_EXECUTION_NOT_VALIDATED = "blocked_execution_not_validated"
    PAPER_ONLY = "paper_only"
    LIVE_ELIGIBLE = "live_eligible"

    @classmethod
    def default(cls) -> "EligibilityState":
        """The state of a system about which nothing is known."""
        return cls.BLOCKED_INSUFFICIENT_DATA

    @property
    def severity(self) -> int:
        """Lower is more severe. Follows the remediation order of AUDIT §12."""
        return _SEVERITY[self]

    @property
    def is_blocked(self) -> bool:
        return self is not EligibilityState.LIVE_ELIGIBLE

    @property
    def permits_live_trading(self) -> bool:
        return self is EligibilityState.LIVE_ELIGIBLE


_SEVERITY: dict[EligibilityState, int] = {
    EligibilityState.BLOCKED_INSUFFICIENT_DATA: 0,
    EligibilityState.BLOCKED_VALIDATION_INCOMPLETE: 1,
    EligibilityState.BLOCKED_INSUFFICIENT_STATISTICAL_EVIDENCE: 2,
    EligibilityState.BLOCKED_POOR_OOS_PERFORMANCE: 3,
    EligibilityState.BLOCKED_EXCESSIVE_DRAWDOWN: 4,
    EligibilityState.BLOCKED_EXECUTION_NOT_VALIDATED: 5,
    EligibilityState.PAPER_ONLY: 6,
    EligibilityState.LIVE_ELIGIBLE: 7,
}


class GateCategory(str, Enum):
    DATA = "data"
    STATISTICAL = "statistical"
    PERFORMANCE = "performance"
    EXECUTION = "execution"
    OPERATIONAL = "operational"


# The state a whole category collapses to when it is missing from a report.
_CATEGORY_FALLBACK: dict[GateCategory, EligibilityState] = {
    GateCategory.DATA: EligibilityState.BLOCKED_INSUFFICIENT_DATA,
    GateCategory.STATISTICAL: EligibilityState.BLOCKED_VALIDATION_INCOMPLETE,
    GateCategory.PERFORMANCE: EligibilityState.BLOCKED_POOR_OOS_PERFORMANCE,
    GateCategory.EXECUTION: EligibilityState.BLOCKED_EXECUTION_NOT_VALIDATED,
    GateCategory.OPERATIONAL: EligibilityState.PAPER_ONLY,
}


# ===========================================================================
# Evidence
# ===========================================================================

@dataclass(frozen=True)
class Evidence:
    """
    Everything the gates are allowed to consult.

    Every field defaults to ``None``, meaning "not established". A default
    ``Evidence()`` therefore fails every gate, which is the correct verdict for
    a system that has told us nothing about itself.

    Nothing here is inferred. A value is present only because something
    measured it and recorded it.
    """

    # ── Data ──────────────────────────────────────────────────────────────
    price_history_symbols: Optional[int] = None
    price_history_start: Optional[date] = None
    price_history_end: Optional[date] = None
    price_history_is_real_market_data: Optional[bool] = None
    point_in_time_index_constituents_available: Optional[bool] = None
    point_in_time_fundamentals_available: Optional[bool] = None
    intraday_history_days: Optional[int] = None
    data_quality_audit_passed: Optional[bool] = None
    real_ohlc_available: Optional[bool] = None
    corporate_actions_adjusted: Optional[bool] = None

    # ── Statistical ───────────────────────────────────────────────────────
    purged_walk_forward_completed: Optional[bool] = None
    label_embargo_applied: Optional[bool] = None
    deflated_sharpe_ratio: Optional[float] = None
    probability_of_backtest_overfitting: Optional[float] = None
    benchmark_name: Optional[str] = None
    net_annual_excess_return_vs_benchmark: Optional[float] = None
    n_trials_recorded: Optional[int] = None
    n_trials_used_in_dsr: Optional[int] = None

    # ── Performance (all net of costs, all out of sample) ─────────────────
    oos_sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None            # positive fraction: 0.22 == 22%
    slippage_multiple_survived: Optional[float] = None
    oos_sharpe_at_stressed_costs: Optional[float] = None
    max_single_name_pnl_share: Optional[float] = None
    max_sector_pnl_share: Optional[float] = None
    max_single_year_pnl_share: Optional[float] = None
    top_5_trades_pnl_share: Optional[float] = None

    # ── Execution (verified against a real broker session) ────────────────
    broker_auth_verified: Optional[bool] = None
    order_lifecycle_verified: Optional[bool] = None
    reconciliation_verified: Optional[bool] = None
    duplicate_order_protection_verified: Optional[bool] = None
    kill_switch_verified: Optional[bool] = None
    timezone_squareoff_verified: Optional[bool] = None

    # ── Operational ───────────────────────────────────────────────────────
    paper_trading_days: Optional[int] = None
    paper_sharpe: Optional[float] = None
    backtest_expected_sharpe: Optional[float] = None
    trading_mode: Optional[str] = None
    live_enable_approved_by: Optional[str] = None
    live_enable_approved_at: Optional[date] = None

    # Free-text provenance notes, surfaced in the report but never checked.
    notes: tuple[str, ...] = ()


# ===========================================================================
# Check helpers — every one of them treats None as a failure
# ===========================================================================

def _flag(value: Optional[bool], label: str) -> tuple[bool, str]:
    if value is None:
        return False, f"NO EVIDENCE RECORDED: {label}."
    if value is True:
        return True, f"{label}: verified."
    return False, f"{label}: NOT verified."


def _min_value(
    value: Optional[float], floor: float, label: str, strict: bool = False
) -> tuple[bool, str]:
    rel = ">" if strict else ">="
    if value is None:
        return False, f"NO EVIDENCE RECORDED: {label} (required {rel} {floor:g})."
    ok = value > floor if strict else value >= floor
    verdict = "meets" if ok else "FAILS"
    return ok, f"{label} = {value:.4g} {verdict} the required {rel} {floor:g}."


def _max_value(
    value: Optional[float], cap: float, label: str, strict: bool = False
) -> tuple[bool, str]:
    rel = "<" if strict else "<="
    if value is None:
        return False, f"NO EVIDENCE RECORDED: {label} (required {rel} {cap:g})."
    ok = value < cap if strict else value <= cap
    verdict = "meets" if ok else "FAILS"
    return ok, f"{label} = {value:.4g} {verdict} the required {rel} {cap:g}."


def _all_of(*checks: tuple[bool, str]) -> tuple[bool, str]:
    """Pass only if every sub-check passes; report the failures, not the passes."""
    failures = [reason for ok, reason in checks if not ok]
    if failures:
        return False, " ".join(failures)
    return True, " ".join(reason for _, reason in checks)


# ===========================================================================
# Gate
# ===========================================================================

@dataclass(frozen=True)
class GateResult:
    """One line of the checklist. Denormalized so it serializes on its own."""

    name: str
    category: GateCategory
    blocking_state: EligibilityState
    requirement: str
    passed: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category.value,
            "blocking_state": self.blocking_state.value,
            "requirement": self.requirement,
            "passed": self.passed,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GateResult":
        return cls(
            name=str(d["name"]),
            category=GateCategory(d["category"]),
            blocking_state=EligibilityState(d["blocking_state"]),
            requirement=str(d["requirement"]),
            passed=bool(d["passed"]),
            reason=str(d["reason"]),
        )


@dataclass(frozen=True)
class Gate:
    """
    One precondition for live trading.

    ``predicate`` returns ``(passed, reason)``. It is never called directly —
    :meth:`check` wraps it so that an exception becomes a FAILED result rather
    than propagating and aborting the assessment.
    """

    name: str
    category: GateCategory
    blocking_state: EligibilityState
    requirement: str
    predicate: Callable[[Evidence], tuple[bool, str]] = field(repr=False, compare=False)

    def check(self, evidence: Evidence) -> GateResult:
        try:
            passed, reason = self.predicate(evidence)
            passed = bool(passed)
            reason = str(reason)
        except Exception as exc:  # noqa: BLE001 — a broken gate is a failed gate
            logger.exception("Eligibility gate %r raised; recording as FAILED.", self.name)
            passed, reason = False, (
                f"GATE COULD NOT BE EVALUATED — {type(exc).__name__}: {exc}. "
                f"An unevaluable gate fails."
            )
        return GateResult(
            name=self.name,
            category=self.category,
            blocking_state=self.blocking_state,
            requirement=self.requirement,
            passed=passed,
            reason=reason,
        )


# ===========================================================================
# The gates
# ===========================================================================

def _price_history(e: Evidence) -> tuple[bool, str]:
    if e.price_history_start is None or e.price_history_end is None:
        span_check = (False, "NO EVIDENCE RECORDED: price history date range.")
    else:
        years = (e.price_history_end - e.price_history_start).days / 365.25
        span_check = _min_value(years, MIN_PRICE_HISTORY_YEARS, "price history span (years)")
    return _all_of(
        _flag(e.price_history_is_real_market_data, "price history is real market data"),
        _min_value(e.price_history_symbols, MIN_PRICE_HISTORY_SYMBOLS, "symbols with history"),
        span_check,
    )


def _data_quality(e: Evidence) -> tuple[bool, str]:
    return _all_of(
        _flag(e.data_quality_audit_passed, "data quality audit"),
        _flag(e.real_ohlc_available, "real OHLC (not synthesized from close)"),
        _flag(e.corporate_actions_adjusted, "corporate-action adjustment"),
    )


def _purged_walk_forward(e: Evidence) -> tuple[bool, str]:
    return _all_of(
        _flag(e.purged_walk_forward_completed, "purged walk-forward validation"),
        _flag(e.label_embargo_applied, "label embargo between train and test"),
    )


def _beats_benchmark(e: Evidence) -> tuple[bool, str]:
    if not e.benchmark_name:
        return False, "NO EVIDENCE RECORDED: no passive benchmark was named."
    ok, reason = _min_value(
        e.net_annual_excess_return_vs_benchmark,
        0.0,
        f"annual excess return vs {e.benchmark_name}, net of costs",
        strict=True,
    )
    return ok, reason


def _trial_count(e: Evidence) -> tuple[bool, str]:
    """
    An honest trial count includes every discarded variant. A DSR computed
    against a smaller count than the one recorded is a DSR computed against a
    lie, so the two numbers must agree (AUDIT §1.2 F4, §12.7).
    """
    ok, reason = _min_value(e.n_trials_recorded, 1, "recorded trial count")
    if not ok:
        return False, reason
    if e.n_trials_used_in_dsr is None:
        return False, "NO EVIDENCE RECORDED: trial count used in the DSR computation."
    if e.n_trials_used_in_dsr != e.n_trials_recorded:
        return False, (
            f"Trial count is not honest: {e.n_trials_recorded} trials were run but the "
            f"DSR was deflated for only {e.n_trials_used_in_dsr}."
        )
    return True, f"{e.n_trials_recorded} trials recorded and fully reflected in the DSR."


def _concentration(e: Evidence) -> tuple[bool, str]:
    return _all_of(
        _max_value(e.max_single_name_pnl_share, MAX_SINGLE_NAME_PNL_SHARE,
                   "largest single-name share of PnL"),
        _max_value(e.max_sector_pnl_share, MAX_SECTOR_PNL_SHARE,
                   "largest sector share of PnL"),
        _max_value(e.max_single_year_pnl_share, MAX_SINGLE_YEAR_PNL_SHARE,
                   "largest single-year share of PnL"),
        _max_value(e.top_5_trades_pnl_share, MAX_TOP5_TRADES_PNL_SHARE,
                   "top-5-trades share of PnL"),
    )


def _cost_stress(e: Evidence) -> tuple[bool, str]:
    return _all_of(
        _min_value(e.slippage_multiple_survived, MIN_SLIPPAGE_STRESS_MULTIPLE,
                   "slippage multiple survived"),
        _min_value(e.oos_sharpe_at_stressed_costs, MIN_OOS_SHARPE,
                   "OOS Sharpe under stressed costs"),
    )


def _paper_matches_backtest(e: Evidence) -> tuple[bool, str]:
    if e.paper_sharpe is None or e.backtest_expected_sharpe is None:
        return False, "NO EVIDENCE RECORDED: paper and backtest Sharpe for comparison."
    if e.backtest_expected_sharpe <= 0:
        return False, (
            f"Backtest expected Sharpe is {e.backtest_expected_sharpe:.4g}; there is no "
            f"positive expectation for paper trading to confirm."
        )
    floor = MIN_PAPER_TO_BACKTEST_SHARPE_RATIO * e.backtest_expected_sharpe
    if e.paper_sharpe < floor:
        return False, (
            f"Paper Sharpe {e.paper_sharpe:.4g} is below {floor:.4g} "
            f"({MIN_PAPER_TO_BACKTEST_SHARPE_RATIO:g}x the backtest's "
            f"{e.backtest_expected_sharpe:.4g}). A material divergence means the backtest "
            f"is wrong — investigate rather than proceed."
        )
    return True, (
        f"Paper Sharpe {e.paper_sharpe:.4g} is consistent with the backtest's "
        f"{e.backtest_expected_sharpe:.4g}."
    )


def _human_enabled_live(e: Evidence) -> tuple[bool, str]:
    """
    Two independent things must be true: the config says live, and a named
    human took responsibility for it. Neither alone is sufficient — a config
    flag can be flipped by a deploy, and an approval can go stale.
    """
    if e.trading_mode is None:
        return False, "NO EVIDENCE RECORDED: TRADING_MODE."
    if e.trading_mode != "live":
        return False, f"TRADING_MODE is {e.trading_mode!r}, not 'live'."
    if not (e.live_enable_approved_by or "").strip():
        return False, "TRADING_MODE is 'live' but no human approver is recorded."
    if e.live_enable_approved_at is None:
        return False, (
            f"TRADING_MODE is 'live', approved by {e.live_enable_approved_by!r}, "
            f"but with no approval date."
        )
    return True, (
        f"TRADING_MODE explicitly set to 'live' by {e.live_enable_approved_by} "
        f"on {e.live_enable_approved_at.isoformat()}."
    )


_D = GateCategory.DATA
_S = GateCategory.STATISTICAL
_P = GateCategory.PERFORMANCE
_X = GateCategory.EXECUTION
_O = GateCategory.OPERATIONAL

_INSUFFICIENT_DATA = EligibilityState.BLOCKED_INSUFFICIENT_DATA
_VALIDATION_INCOMPLETE = EligibilityState.BLOCKED_VALIDATION_INCOMPLETE
_WEAK_EVIDENCE = EligibilityState.BLOCKED_INSUFFICIENT_STATISTICAL_EVIDENCE
_POOR_OOS = EligibilityState.BLOCKED_POOR_OOS_PERFORMANCE
_DRAWDOWN = EligibilityState.BLOCKED_EXCESSIVE_DRAWDOWN
_EXECUTION = EligibilityState.BLOCKED_EXECUTION_NOT_VALIDATED
_PAPER = EligibilityState.PAPER_ONLY


ALL_GATES: tuple[Gate, ...] = (
    # ── DATA ──────────────────────────────────────────────────────────────
    Gate("real_price_history", _D, _INSUFFICIENT_DATA,
         f"Real daily OHLCV for >= {MIN_PRICE_HISTORY_SYMBOLS} symbols spanning "
         f">= {MIN_PRICE_HISTORY_YEARS:g} years.",
         _price_history),
    Gate("point_in_time_index_constituents", _D, _INSUFFICIENT_DATA,
         "Point-in-time index membership, without which every long-horizon backtest is "
         "biased upward by survivorship (AUDIT §9.4).",
         lambda e: _flag(e.point_in_time_index_constituents_available,
                         "point-in-time index constituents")),
    Gate("point_in_time_fundamentals", _D, _INSUFFICIENT_DATA,
         "Point-in-time fundamentals with publication dates, without which fundamental "
         "features cannot be built causally (AUDIT §9.3).",
         lambda e: _flag(e.point_in_time_fundamentals_available,
                         "point-in-time fundamentals")),
    Gate("intraday_history_span", _D, _INSUFFICIENT_DATA,
         f"At least {MIN_INTRADAY_HISTORY_DAYS} days of intraday history; a 60-day window "
         f"spans one regime at best.",
         lambda e: _min_value(e.intraday_history_days, MIN_INTRADAY_HISTORY_DAYS,
                              "intraday history (days)")),
    Gate("data_quality_audit", _D, _INSUFFICIENT_DATA,
         "Data quality audited: real OHLC (not synthesized from close) and corporate "
         "actions adjusted (AUDIT §9.5).",
         _data_quality),

    # ── STATISTICAL ───────────────────────────────────────────────────────
    Gate("purged_walk_forward", _S, _VALIDATION_INCOMPLETE,
         "Purged, embargoed walk-forward validation completed (AUDIT §1.2 F5).",
         _purged_walk_forward),
    Gate("deflated_sharpe_ratio", _S, _WEAK_EVIDENCE,
         f"Deflated Sharpe Ratio > {MIN_DEFLATED_SHARPE:g}, i.e. the result is "
         f"distinguishable from the luckiest of many trials.",
         lambda e: _min_value(e.deflated_sharpe_ratio, MIN_DEFLATED_SHARPE,
                              "Deflated Sharpe Ratio", strict=True)),
    Gate("probability_of_backtest_overfitting", _S, _WEAK_EVIDENCE,
         f"PBO < {MAX_PBO:g}; above that, in-sample selection is worse than useless.",
         lambda e: _max_value(e.probability_of_backtest_overfitting, MAX_PBO,
                              "Probability of Backtest Overfitting", strict=True)),
    Gate("beats_passive_benchmark", _S, _POOR_OOS,
         "Positive excess return over a named passive benchmark, net of costs "
         "(AUDIT §12.6).",
         _beats_benchmark),
    Gate("honest_trial_count", _S, _WEAK_EVIDENCE,
         "The number of trials — including every discarded variant — is recorded and is "
         "the number the DSR was deflated for (AUDIT §12.7).",
         _trial_count),

    # ── PERFORMANCE ───────────────────────────────────────────────────────
    Gate("oos_sharpe_floor", _P, _POOR_OOS,
         f"Out-of-sample Sharpe >= {MIN_OOS_SHARPE:g}, net of costs.",
         lambda e: _min_value(e.oos_sharpe, MIN_OOS_SHARPE, "OOS Sharpe")),
    Gate("max_drawdown_limit", _P, _DRAWDOWN,
         f"Out-of-sample max drawdown <= {MAX_DRAWDOWN:.0%}.",
         lambda e: _max_value(e.max_drawdown, MAX_DRAWDOWN, "OOS max drawdown")),
    Gate("cost_and_slippage_stress", _P, _POOR_OOS,
         f"The edge survives slippage stress to >= {MIN_SLIPPAGE_STRESS_MULTIPLE:g}x "
         f"(AUDIT §12.9).",
         _cost_stress),
    Gate("return_concentration", _P, _WEAK_EVIDENCE,
         "PnL is not concentrated in one stock, one sector, one year, or a handful of "
         "trades.",
         _concentration),

    # ── EXECUTION ─────────────────────────────────────────────────────────
    Gate("broker_auth", _X, _EXECUTION,
         "Broker authentication and token refresh verified against a live session "
         "(AUDIT §9.10).",
         lambda e: _flag(e.broker_auth_verified, "broker auth and token refresh")),
    Gate("order_lifecycle", _X, _EXECUTION,
         "Full order lifecycle verified: place, modify, cancel, partial fill, reject.",
         lambda e: _flag(e.order_lifecycle_verified, "order lifecycle")),
    Gate("reconciliation", _X, _EXECUTION,
         "Startup reconciliation against real broker positions verified.",
         lambda e: _flag(e.reconciliation_verified, "startup reconciliation")),
    Gate("duplicate_order_protection", _X, _EXECUTION,
         "Duplicate-order protection verified against a real broker session.",
         lambda e: _flag(e.duplicate_order_protection_verified,
                         "duplicate-order protection")),
    Gate("kill_switch", _X, _EXECUTION,
         "Kill switch verified to halt and flatten against a real broker session.",
         lambda e: _flag(e.kill_switch_verified, "kill switch")),
    Gate("timezone_and_squareoff", _X, _EXECUTION,
         "IST handling and intraday square-off verified on a UTC host (AUDIT §1.1 C4).",
         lambda e: _flag(e.timezone_squareoff_verified, "timezone and square-off")),

    # ── OPERATIONAL ───────────────────────────────────────────────────────
    Gate("paper_trading_duration", _O, _PAPER,
         f"Paper trading on live data for >= {MIN_PAPER_TRADING_DAYS} days "
         f"(AUDIT §12.15).",
         lambda e: _min_value(e.paper_trading_days, MIN_PAPER_TRADING_DAYS,
                              "paper trading (days)")),
    Gate("paper_matches_backtest", _O, _PAPER,
         "Paper results are consistent with backtest expectations (AUDIT §12.16).",
         _paper_matches_backtest),
    Gate("human_enabled_live_mode", _O, _PAPER,
         "TRADING_MODE explicitly set to 'live' by a named human on a recorded date.",
         _human_enabled_live),
)

_ALL_GATE_NAMES: tuple[str, ...] = tuple(g.name for g in ALL_GATES)


# ===========================================================================
# Report
# ===========================================================================

@dataclass(frozen=True)
class EligibilityReport:
    """
    The verdict plus the checklist that produced it.

    ``state`` is a property, not a field. There is no way to hand this class a
    permissive answer; it can only compute one, and it computes ``LIVE_ELIGIBLE``
    only when every gate in :data:`ALL_GATES` is present and passing.
    """

    results: tuple[GateResult, ...]
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    evidence_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """No gate may be silently skipped: absent ones are injected as failures."""
        seen = {r.name for r in self.results}
        missing = [g for g in ALL_GATES if g.name not in seen]
        if missing:
            injected = tuple(
                GateResult(
                    name=g.name,
                    category=g.category,
                    blocking_state=g.blocking_state,
                    requirement=g.requirement,
                    passed=False,
                    reason="GATE WAS NOT EVALUATED. An unevaluated gate fails.",
                )
                for g in missing
            )
            object.__setattr__(self, "results", tuple(self.results) + injected)
        else:
            object.__setattr__(self, "results", tuple(self.results))

    # ── Verdict ───────────────────────────────────────────────────────────

    @property
    def state(self) -> EligibilityState:
        failed = self.failed_gates
        if not failed:
            return EligibilityState.LIVE_ELIGIBLE
        return min((r.blocking_state for r in failed), key=lambda s: s.severity)

    @property
    def permits_live_trading(self) -> bool:
        return self.state.permits_live_trading

    @property
    def failed_gates(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if not r.passed)

    @property
    def passed_gates(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if r.passed)

    @property
    def blocking_reason(self) -> str:
        failed = self.failed_gates
        if not failed:
            return "All gates passed."
        state = self.state
        first = next(r for r in failed if r.blocking_state is state)
        return f"{first.name}: {first.reason}"

    # ── Human output ──────────────────────────────────────────────────────

    def checklist(self) -> str:
        """A checklist to be worked top to bottom."""
        state = self.state
        lines = [
            "LIVE TRADING ELIGIBILITY",
            "=" * 72,
            f"STATE          : {state.name}",
            f"LIVE PERMITTED : {'YES' if state.permits_live_trading else 'NO'}",
            f"GATES          : {len(self.passed_gates)}/{len(self.results)} passed",
            f"EVALUATED AT   : {self.evaluated_at.isoformat()}",
        ]
        if state.is_blocked:
            lines.append(f"BLOCKED BY     : {self.blocking_reason}")
        lines.append("")
        for category in GateCategory:
            in_cat = [r for r in self.results if r.category is category]
            if not in_cat:
                continue
            n_ok = sum(1 for r in in_cat if r.passed)
            lines.append(f"{category.value.upper()}  ({n_ok}/{len(in_cat)} passed)")
            for r in in_cat:
                mark = "PASS" if r.passed else "FAIL"
                lines.append(f"  [{mark}] {r.name}")
                if not r.passed:
                    lines.append(f"         need: {r.requirement}")
                    lines.append(f"         why : {r.reason}")
            lines.append("")
        for note in self.evidence_notes:
            lines.append(f"note: {note}")
        return "\n".join(lines).rstrip() + "\n"

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        state = self.state
        return {
            "state": state.value,
            "permits_live_trading": state.permits_live_trading,
            "severity": state.severity,
            "blocking_reason": self.blocking_reason,
            "evaluated_at": self.evaluated_at.isoformat(),
            "n_gates": len(self.results),
            "n_passed": len(self.passed_gates),
            "n_failed": len(self.failed_gates),
            "failed_gates": [r.to_dict() for r in self.failed_gates],
            "results": [r.to_dict() for r in self.results],
            "evidence_notes": list(self.evidence_notes),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "EligibilityReport":
        return cls(
            results=tuple(GateResult.from_dict(r) for r in d["results"]),
            evaluated_at=datetime.fromisoformat(d["evaluated_at"]),
            evidence_notes=tuple(d.get("evidence_notes", ())),
        )

    @classmethod
    def from_json(cls, payload: str) -> "EligibilityReport":
        return cls.from_dict(json.loads(payload))


# ===========================================================================
# Entry point
# ===========================================================================

def assess_live_trading_eligibility(
    evidence: Optional[Evidence] = None,
    gates: Sequence[Gate] = ALL_GATES,
) -> EligibilityReport:
    """
    Evaluate every gate and return the verdict.

    Passing no evidence means "nothing is known", which blocks. It does not
    mean "assume the best".

    ``gates`` exists for testing. Narrowing it cannot widen the verdict: any
    gate in :data:`ALL_GATES` missing from the result set is injected as a
    failure by :meth:`EligibilityReport.__post_init__`.
    """
    if evidence is None:
        evidence = Evidence(notes=("No evidence was supplied to the assessor.",))
    return EligibilityReport(
        results=tuple(g.check(evidence) for g in gates),
        evidence_notes=tuple(evidence.notes),
    )


# ===========================================================================
# Evidence gathered from the repository as it actually stands
# ===========================================================================

_CACHE_FILENAME = re.compile(r"^(?P<symbol>.+)__(?P<start>[\d-]+)__(?P<end>[\d-]+)\.parquet$")


def _price_cache_dir() -> Path:
    """Where app.data.providers writes its parquet cache."""
    try:
        from app.data.providers import _CACHE_DIR  # noqa: PLC0415
        return Path(_CACHE_DIR)
    except Exception:  # noqa: BLE001 — the module may not exist; fall back to the layout
        return Path(__file__).resolve().parents[3] / "data_cache"


def gather_repo_evidence() -> Evidence:
    """
    Build an :class:`Evidence` record from what this repository can actually
    demonstrate right now.

    This function is deliberately stingy. It sets a field only when it has
    directly observed the fact. Everything it cannot observe — every statistical
    result, every performance number, every execution verification — is left
    ``None``, because no such result has ever been persisted anywhere in this
    repository. That is a finding, not a gap in this function.
    """
    notes: list[str] = []
    symbols: Optional[int] = None
    start: Optional[date] = None
    end: Optional[date] = None
    is_real: Optional[bool] = None

    cache = _price_cache_dir()
    try:
        files = sorted(cache.glob("*.parquet")) if cache.is_dir() else []
        parsed = [
            (m.group("symbol"), m.group("start"), m.group("end"))
            for m in (_CACHE_FILENAME.match(f.name) for f in files)
            if m is not None
        ]
        if parsed:
            symbols = len({s for s, _, _ in parsed if not s.startswith("BENCH_")})
            start = min(date.fromisoformat(s) for _, s, _ in parsed)
            end = max(date.fromisoformat(x) for _, _, x in parsed)
            is_real = True
            notes.append(f"{len(files)} parquet files of real daily OHLCV in {cache}.")
        else:
            symbols, is_real = 0, False
            notes.append(f"No cached price data found in {cache}.")
    except Exception as exc:  # noqa: BLE001 — an unreadable cache is not evidence
        notes.append(f"Price cache at {cache} could not be inspected: {exc!r}.")

    trading_mode: Optional[str] = None
    try:
        from app.core.config import settings  # noqa: PLC0415
        trading_mode = str(settings.trading_mode)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"TRADING_MODE could not be read: {exc!r}.")

    fundamentals_real: Optional[bool] = None
    try:
        from app.strategies.longterm import _MockFundamentalProvider  # noqa: PLC0415
        fundamentals_real = not bool(_MockFundamentalProvider.IS_MOCK)
        notes.append(
            "Fundamentals provider is "
            f"{'MOCK' if _MockFundamentalProvider.IS_MOCK else 'real'}."
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Fundamentals provider could not be inspected: {exc!r}.")

    notes.append(
        "The price source (Yahoo Finance) supplies no point-in-time index membership "
        "and no point-in-time fundamentals, and its intraday history is ~60 days — "
        "see app/data/providers.py."
    )
    notes.append(
        "No model, backtest result, validation record or paper-trading record has ever "
        "been persisted by this repository, so every statistical, performance, "
        "execution and operational field below is unknown."
    )

    return Evidence(
        # Data — observed.
        price_history_symbols=symbols,
        price_history_start=start,
        price_history_end=end,
        price_history_is_real_market_data=is_real,
        point_in_time_index_constituents_available=False,
        point_in_time_fundamentals_available=fundamentals_real,
        intraday_history_days=0,
        # Statistical / performance / execution / operational — unobserved,
        # therefore left as None, therefore failing.
        trading_mode=trading_mode,
        notes=tuple(notes),
    )


def assess_repo_live_trading_eligibility() -> EligibilityReport:
    """Assess this repository as it actually stands."""
    return assess_live_trading_eligibility(gather_repo_evidence())


__all__ = [
    "ALL_GATES",
    "Evidence",
    "EligibilityReport",
    "EligibilityState",
    "Gate",
    "GateCategory",
    "GateResult",
    "assess_live_trading_eligibility",
    "assess_repo_live_trading_eligibility",
    "gather_repo_evidence",
]
