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

THE FIVE STRUCTURAL PROPERTIES THAT MAKE THIS SAFE
--------------------------------------------------
1. FAIL CLOSED ON SILENCE. Every field of :class:`Evidence` defaults to
   ``None``, and ``None`` is a FAILURE with the reason "no evidence recorded",
   not a skipped check. Absence of evidence is never evidence of absence.

2. FAIL CLOSED ON ERROR. A gate that raises is recorded as FAILED, with the
   exception text as its reason. A crash can never be mistaken for a pass.

3. FAIL CLOSED ON NONSENSE. A measurement that is NaN, infinite, or outside
   its physically possible range (a negative drawdown, a probability of -3, a
   Sharpe of one million) is not a measurement. It fails, loudly, and it fails
   in *both* comparison directions — see :func:`_as_finite`.

4. THE STATE IS DERIVED, NEVER ASSIGNED, AND ONLY A COMPUTED REPORT
   AUTHORIZES ANYTHING. :attr:`EligibilityReport.state` is a read-only property
   computed from the gate results against the canonical gate registry captured
   at import. A report parsed from JSON, or hand-constructed, carries
   :attr:`ReportProvenance.UNTRUSTED` and can never authorize an order however
   its ``passed`` flags read. A gate substituted for one of the canonical gates
   cannot report a pass.

5. THE VERDICT IS ENFORCED, NOT MERELY REPORTED. :func:`require_live_eligible`
   raises :class:`LiveTradingBlocked` unless a freshly computed report clears
   every canonical gate. See ENFORCEMENT below — the call site still has to be
   wired in, and gate ``enforcement_wired_into_order_path`` fails until it is.

ENFORCEMENT — READ THIS
-----------------------
A verdict that nothing consults is decoration. The audit of this module found
that **no code path in the execution, broker, api or main modules of the
`app` package imported this module at all**: the system could have placed live orders while
this file reported BLOCKED, and nothing would have stopped it.

:func:`require_live_eligible` is the assertion that closes that hole. It must be
called at the top of the live order path — concretely, inside
``OrderManager.submit_order`` (execution/order_manager.py) before the broker
is touched. That file is owned elsewhere; until the call exists, the gate
``enforcement_wired_into_order_path`` detects its own absence by reading that
module's source and FAILS, so LIVE_ELIGIBLE is unreachable while the system is
unenforced. That is deliberate.

Risk-*reducing* actions (emergency flatten, square-off, cancel) are exempt via
``intent=OrderIntent.REDUCE_RISK``, because a blocked system must still be able
to get flat. That exemption is logged at WARNING every time it is used.

THERE IS NO OVERRIDE
--------------------
There is no environment variable, no keyword argument, no config flag and no
"force" path that turns a blocked verdict into a permissive one. This was a
deliberate choice, not an omission: this system has never placed a live order,
so there is no operational need that an override would serve, and an override
is the single most likely thing to be reached for at the exact moment judgement
is worst. ``test_no_bypass_mechanism_exists`` asserts that this stays true.

READING THE OUTPUT
------------------
:meth:`EligibilityReport.checklist` prints the failures grouped by category,
worst first. It is meant to be worked top to bottom. ``to_json``/``from_json``
carry the same content to an API or dashboard — but a deserialized report is
*information about* an assessment, never an authorization to act.

THRESHOLDS ARE PRIORS, NOT ESTIMATES
------------------------------------
The numeric bars below are hand-chosen, defensible defaults — the same honesty
caveat the audit applies to the regime multipliers (§9.8). They are deliberately
conservative. They are not calibrated from data, because there is no validated
strategy from which to calibrate them. docs/LIVE_TRADING_GATES.md lists which
ones are pure priors.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field, replace
from dataclasses import fields as _dataclass_fields
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)


# ===========================================================================
# Thresholds
#
# Every constant here is a hand-chosen prior unless its comment cites a source.
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

# Freshness. Evidence gathered months ago is a historical record, not a
# statement about the system that is about to trade. Every one of these is a
# hand-chosen prior.
MAX_EVIDENCE_AGE_HOURS = 24.0             # the Evidence record itself
MAX_PRICE_DATA_STALENESS_DAYS = 7.0       # last bar in the price history
MAX_BROKER_HEARTBEAT_AGE_MINUTES = 15.0   # last successful broker round-trip
MAX_RECONCILIATION_AGE_HOURS = 24.0       # last position reconciliation
MAX_APPROVAL_AGE_DAYS = 90.0              # a human sign-off goes stale
MAX_REPORT_AGE_MINUTES = 15.0             # a verdict used at an order gate
CLOCK_SKEW_TOLERANCE_SECONDS = 120.0      # how far ahead of "now" is not a lie

# Risk
MAX_ACTIVE_RISK_LIMIT_BREACHES = 0        # any live breach blocks
MAX_OPEN_RECONCILIATION_BREAKS = 0        # any unexplained position break blocks


def _now() -> datetime:
    """Current UTC time. Indirected so tests can reason about freshness."""
    return datetime.now(timezone.utc)


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
    BLOCKED_RISK_LIMIT_BREACH = "blocked_risk_limit_breach"
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
    EligibilityState.BLOCKED_RISK_LIMIT_BREACH: 6,
    EligibilityState.PAPER_ONLY: 7,
    EligibilityState.LIVE_ELIGIBLE: 8,
}


class GateCategory(str, Enum):
    DATA = "data"
    STATISTICAL = "statistical"
    PERFORMANCE = "performance"
    EXECUTION = "execution"
    OPERATIONAL = "operational"


class ReportProvenance(str, Enum):
    """
    Where a report came from, and therefore whether it may authorize anything.

    ``COMPUTED`` is set only by :func:`assess_live_trading_eligibility`, which
    holds the only reference to the private trust token. Everything else —
    reports parsed from JSON, reports built by hand in a test or a REPL, reports
    reconstructed by a dashboard — is ``UNTRUSTED``. An untrusted report may
    describe a passing assessment; it may never *be* one.
    """

    COMPUTED = "computed"
    UNTRUSTED = "untrusted"


class OrderIntent(str, Enum):
    """
    Why the order path is asking.

    ``INCREASE_RISK`` covers every order that opens or adds to exposure. It
    requires LIVE_ELIGIBLE.

    ``REDUCE_RISK`` covers emergency flatten, intraday square-off and order
    cancellation. A blocked system must still be able to get flat, so these are
    permitted while blocked — and logged at WARNING every single time.
    """

    INCREASE_RISK = "increase_risk"
    REDUCE_RISK = "reduce_risk"


class LiveTradingBlocked(RuntimeError):
    """
    Raised by :func:`require_live_eligible` when live trading is not permitted.

    This is deliberately an error and not a return value: a caller that ignores
    a returned ``False`` places the order anyway, whereas a caller that ignores
    an exception does not place anything.
    """

    def __init__(
        self,
        state: "EligibilityState",
        reason: str,
        report: Optional["EligibilityReport"] = None,
    ) -> None:
        super().__init__(
            f"LIVE TRADING BLOCKED [{state.name}]: {reason}"
        )
        self.state = state
        self.reason = reason
        self.report = report


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
    measured it and recorded it — and ``evidence_gathered_at`` records *when*,
    because a measurement from last quarter is not a statement about today.
    """

    # ── Provenance and freshness ──────────────────────────────────────────
    evidence_gathered_at: Optional[datetime] = None

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
    market_data_feed_healthy: Optional[bool] = None

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

    # ── Execution, right now (not "was verified once") ────────────────────
    broker_reachable_at: Optional[datetime] = None
    broker_session_valid: Optional[bool] = None
    broker_api_degraded: Optional[bool] = None
    last_reconciliation_at: Optional[datetime] = None
    last_reconciliation_succeeded: Optional[bool] = None
    open_reconciliation_breaks: Optional[int] = None

    # ── Risk, right now ───────────────────────────────────────────────────
    risk_limits_loaded: Optional[bool] = None
    active_risk_limit_breaches: Optional[int] = None
    current_drawdown: Optional[float] = None
    kill_switch_engaged: Optional[bool] = None

    # ── Enforcement wiring ────────────────────────────────────────────────
    eligibility_enforced_at_order_path: Optional[bool] = None

    # ── Operational ───────────────────────────────────────────────────────
    paper_trading_days: Optional[int] = None
    paper_sharpe: Optional[float] = None
    backtest_expected_sharpe: Optional[float] = None
    trading_mode: Optional[str] = None
    live_enable_approved_by: Optional[str] = None
    live_enable_approved_at: Optional[date] = None

    # Free-text provenance notes, surfaced in the report but never checked.
    notes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# The physically possible range of every numeric field.
#
# These are not thresholds — the gates hold the thresholds. These are the
# bounds outside which a number cannot be a real measurement, only a bug: a
# negative drawdown, a probability of -3, a Sharpe of a million, a NaN from a
# 0/0. Before this table existed, ``-inf`` satisfied every ``<=`` cap in the
# module and ``+inf`` satisfied every ``>=`` floor.
# ---------------------------------------------------------------------------

_NUMERIC_DOMAINS: dict[str, tuple[float, float]] = {
    "price_history_symbols": (0.0, 100_000.0),
    "intraday_history_days": (0.0, 40_000.0),
    "deflated_sharpe_ratio": (-20.0, 20.0),
    "probability_of_backtest_overfitting": (0.0, 1.0),
    "net_annual_excess_return_vs_benchmark": (-1.0, 10.0),
    "n_trials_recorded": (0.0, 10_000_000.0),
    "n_trials_used_in_dsr": (0.0, 10_000_000.0),
    "oos_sharpe": (-20.0, 20.0),
    "max_drawdown": (0.0, 1.0),
    "slippage_multiple_survived": (0.0, 1_000.0),
    "oos_sharpe_at_stressed_costs": (-20.0, 20.0),
    "max_single_name_pnl_share": (0.0, 100.0),
    "max_sector_pnl_share": (0.0, 100.0),
    "max_single_year_pnl_share": (0.0, 100.0),
    "top_5_trades_pnl_share": (0.0, 100.0),
    "open_reconciliation_breaks": (0.0, 1_000_000.0),
    "active_risk_limit_breaches": (0.0, 1_000_000.0),
    "current_drawdown": (0.0, 1.0),
    "paper_trading_days": (0.0, 36_500.0),
    "paper_sharpe": (-20.0, 20.0),
    "backtest_expected_sharpe": (-20.0, 20.0),
}

_BOOL_FIELDS: tuple[str, ...] = (
    "price_history_is_real_market_data",
    "point_in_time_index_constituents_available",
    "point_in_time_fundamentals_available",
    "data_quality_audit_passed",
    "real_ohlc_available",
    "corporate_actions_adjusted",
    "market_data_feed_healthy",
    "purged_walk_forward_completed",
    "label_embargo_applied",
    "broker_auth_verified",
    "order_lifecycle_verified",
    "reconciliation_verified",
    "duplicate_order_protection_verified",
    "kill_switch_verified",
    "timezone_squareoff_verified",
    "broker_session_valid",
    "broker_api_degraded",
    "last_reconciliation_succeeded",
    "risk_limits_loaded",
    "kill_switch_engaged",
    "eligibility_enforced_at_order_path",
)

_DATE_FIELDS: tuple[str, ...] = (
    "price_history_start",
    "price_history_end",
    "live_enable_approved_at",
)

_DATETIME_FIELDS: tuple[str, ...] = (
    "evidence_gathered_at",
    "broker_reachable_at",
    "last_reconciliation_at",
)

# Every field except the free-text notes. Used to detect a wholly empty record.
_EVIDENCE_FIELD_NAMES: tuple[str, ...] = tuple(
    f.name for f in _dataclass_fields(Evidence) if f.name != "notes"
)

# A hand-chosen prior with a specific job: catching epoch-fallback bugs.
# A failed date parse very often yields 1900-01-01 (the Excel epoch) or
# 1970-01-01 (the Unix epoch), and either one silently satisfied the
# "at least 10 years of history" span check with data that cannot exist —
# NSE opened in 1994. Any market-data date before this is a parsing bug.
MIN_PLAUSIBLE_DATA_DATE = date(1990, 1, 1)


# ===========================================================================
# Check helpers — every one of them treats None, NaN and inf as failures
# ===========================================================================

# str.strip() does not remove U+200B and friends, so an "approver" of a single
# zero-width space used to satisfy a `.strip()` truthiness test.
_BLANK_RE = re.compile(
    "^["
    "\\s"                      # ASCII whitespace
    "\u00a0\u1680\u180e"        # no-break / ogham / mongolian vowel separator
    "\u2000-\u200f"             # en..hair spaces, ZWSP, ZWNJ, ZWJ, LRM, RLM
    "\u2028\u2029"              # line / paragraph separator
    "\u202f\u205f\u2060"        # narrow no-break, medium math, word joiner
    "\u3000\ufeff"              # ideographic space, BOM / zero-width no-break
    "]*$"
)


def _is_blank(text: Optional[str]) -> bool:
    """True for None, "", whitespace, and invisible-character-only strings."""
    if text is None:
        return True
    if not isinstance(text, str):
        return True
    return _BLANK_RE.match(text) is not None


def _as_finite(value: Any, label: str) -> tuple[Optional[float], Optional[str]]:
    """
    Coerce a measurement to a finite float, or explain why it is not one.

    This is the single choke point that closes the NaN family of holes. NaN
    fails *every* comparison, so a hand-written ``if x < floor: return FAIL``
    silently falls through to a pass when ``x`` is NaN — which is exactly how
    ``paper_matches_backtest`` used to award a pass to a strategy whose paper
    Sharpe was NaN. Nothing downstream compares a number that has not been
    through here.
    """
    if value is None:
        return None, f"NO EVIDENCE RECORDED: {label}."
    if isinstance(value, bool):
        return None, f"{label}: a boolean is not a measurement."
    if not isinstance(value, (int, float)):
        return None, f"{label}: {value!r} is not a number."
    v = float(value)
    if math.isnan(v):
        return None, (
            f"{label} is NaN. A NaN satisfies no comparison in either direction and "
            f"must never be read as a pass."
        )
    if math.isinf(v):
        return None, (
            f"{label} is {'+' if v > 0 else '-'}infinity. An infinite measurement is "
            f"not a measurement."
        )
    return v, None


def _flag(value: Optional[bool], label: str) -> tuple[bool, str]:
    if value is None:
        return False, f"NO EVIDENCE RECORDED: {label}."
    if value is True:
        return True, f"{label}: verified."
    if value is False:
        return False, f"{label}: NOT verified."
    return False, (
        f"{label}: recorded as {value!r}, which is not a boolean. Only literal True "
        f"counts as verified."
    )


def _not_flag(value: Optional[bool], label: str) -> tuple[bool, str]:
    """The inverse of :func:`_flag`: the fact must be positively recorded FALSE."""
    if value is None:
        return False, f"NO EVIDENCE RECORDED: {label}."
    if value is False:
        return True, f"{label}: confirmed absent."
    if value is True:
        return False, f"{label}: PRESENT."
    return False, (
        f"{label}: recorded as {value!r}, which is not a boolean. Only literal False "
        f"counts as confirmed absent."
    )


def _min_value(
    value: Any, floor: float, label: str, strict: bool = False
) -> tuple[bool, str]:
    rel = ">" if strict else ">="
    v, problem = _as_finite(value, label)
    if problem is not None:
        return False, f"{problem} (required {rel} {floor:g})."
    assert v is not None
    ok = v > floor if strict else v >= floor
    verdict = "meets" if ok else "FAILS"
    return ok, f"{label} = {v:.4g} {verdict} the required {rel} {floor:g}."


def _max_value(
    value: Any, cap: float, label: str, strict: bool = False
) -> tuple[bool, str]:
    rel = "<" if strict else "<="
    v, problem = _as_finite(value, label)
    if problem is not None:
        return False, f"{problem} (required {rel} {cap:g})."
    assert v is not None
    ok = v < cap if strict else v <= cap
    verdict = "meets" if ok else "FAILS"
    return ok, f"{label} = {v:.4g} {verdict} the required {rel} {cap:g}."


def _fresh(
    stamp: Optional[datetime], max_age: timedelta, label: str
) -> tuple[bool, str]:
    """
    A timestamp must exist, be timezone-aware, be recent, and not be in the
    future. Evidence dated tomorrow is as untrustworthy as evidence dated 2019.
    """
    if stamp is None:
        return False, f"NO EVIDENCE RECORDED: {label} timestamp."
    if not isinstance(stamp, datetime):
        return False, f"{label} timestamp is {stamp!r}, not a datetime."
    if stamp.tzinfo is None or stamp.tzinfo.utcoffset(stamp) is None:
        return False, (
            f"{label} timestamp {stamp.isoformat()} is timezone-naive; its age cannot "
            f"be established."
        )
    now = _now()
    age = now - stamp
    if age < -timedelta(seconds=CLOCK_SKEW_TOLERANCE_SECONDS):
        return False, (
            f"{label} timestamp {stamp.isoformat()} is in the FUTURE by "
            f"{-age.total_seconds() / 60:.1f} min. Future-dated evidence is fabricated "
            f"or the clock is wrong; either way it is not proof."
        )
    if age > max_age:
        return False, (
            f"{label} is STALE: gathered {stamp.isoformat()}, "
            f"{age.total_seconds() / 3600:.1f} h ago, limit "
            f"{max_age.total_seconds() / 3600:.1f} h. Old evidence describes an old "
            f"system."
        )
    return True, (
        f"{label} is current ({age.total_seconds() / 3600:.2f} h old)."
    )


def _fresh_date(
    day: Optional[date], max_age: timedelta, label: str
) -> tuple[bool, str]:
    """As :func:`_fresh`, for a calendar date."""
    if day is None:
        return False, f"NO EVIDENCE RECORDED: {label} date."
    if not isinstance(day, date) or isinstance(day, datetime):
        return False, f"{label} date is {day!r}, not a date."
    today = _now().date()
    age_days = (today - day).days
    if age_days < -1:
        return False, (
            f"{label} is dated {day.isoformat()}, {-age_days} days in the FUTURE. "
            f"Future-dated evidence is not proof."
        )
    limit_days = max_age.total_seconds() / 86400.0
    if age_days > limit_days:
        return False, (
            f"{label} is STALE: {day.isoformat()} is {age_days} days ago, limit "
            f"{limit_days:.0f} days."
        )
    return True, f"{label} is current ({day.isoformat()}, {max(age_days, 0)} days ago)."


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
            passed = passed is True
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

def _evidence_freshness(e: Evidence) -> tuple[bool, str]:
    """
    The Evidence record must state when it was gathered, and that must be
    recent. Without this, a JSON file of perfect evidence written in 2019 is
    indistinguishable from a measurement taken this morning.
    """
    return _fresh(
        e.evidence_gathered_at,
        timedelta(hours=MAX_EVIDENCE_AGE_HOURS),
        "evidence record",
    )


def _evidence_well_formed(e: Evidence) -> tuple[bool, str]:
    """
    Every supplied value must be a possible measurement.

    This is the schema gate. It rejects NaN, +/-inf, out-of-domain numbers
    (negative drawdowns, probabilities outside [0,1], Sharpes of a million),
    non-boolean booleans, non-date dates, timezone-naive datetimes and
    future-dated anything. Individual gates re-check the values they use, so
    removing this gate would not open the holes it closes — it exists so that a
    poisoned field is reported *as* a poisoned field rather than as an unrelated
    threshold failure.

    An empty record fails. "Nothing is recorded" is not "nothing is wrong" —
    a schema check that passes on silence is the exact shape of gate this
    module exists to forbid.
    """
    problems: list[str] = []

    if not any(getattr(e, n) is not None for n in _EVIDENCE_FIELD_NAMES):
        return False, (
            "NO EVIDENCE RECORDED AT ALL: the Evidence record is empty. An empty "
            "record is not a well-formed one."
        )

    for name, (lo, hi) in _NUMERIC_DOMAINS.items():
        value = getattr(e, name)
        if value is None:
            continue
        v, problem = _as_finite(value, name)
        if problem is not None:
            problems.append(problem)
            continue
        assert v is not None
        if not (lo <= v <= hi):
            problems.append(
                f"{name} = {v:.6g} is outside its physically possible range "
                f"[{lo:g}, {hi:g}]; that is a bug, not a measurement."
            )

    for name in _BOOL_FIELDS:
        value = getattr(e, name)
        if value is not None and value is not True and value is not False:
            problems.append(f"{name} = {value!r} is not a boolean.")

    for name in ("benchmark_name", "trading_mode", "live_enable_approved_by"):
        value = getattr(e, name)
        if value is not None and not isinstance(value, str):
            problems.append(f"{name} = {value!r} is not a string.")

    today = _now().date()
    for name in _DATE_FIELDS:
        value = getattr(e, name)
        if value is None:
            continue
        if not isinstance(value, date) or isinstance(value, datetime):
            problems.append(f"{name} = {value!r} is not a date.")
            continue
        if (value - today).days > 1:
            problems.append(
                f"{name} = {value.isoformat()} is in the future; evidence cannot be "
                f"dated after the day it is read."
            )
        elif value < MIN_PLAUSIBLE_DATA_DATE:
            problems.append(
                f"{name} = {value.isoformat()} predates "
                f"{MIN_PLAUSIBLE_DATA_DATE.isoformat()}. No market data for this "
                f"venue exists that early; this is an epoch-fallback from a failed "
                f"date parse, not a measurement."
            )

    now = _now()
    for name in _DATETIME_FIELDS:
        value = getattr(e, name)
        if value is None:
            continue
        if not isinstance(value, datetime):
            problems.append(f"{name} = {value!r} is not a datetime.")
            continue
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            problems.append(f"{name} = {value.isoformat()} is timezone-naive.")
            continue
        if (value - now).total_seconds() > CLOCK_SKEW_TOLERANCE_SECONDS:
            problems.append(f"{name} = {value.isoformat()} is in the future.")

    if (
        e.price_history_start is not None
        and e.price_history_end is not None
        and isinstance(e.price_history_start, date)
        and isinstance(e.price_history_end, date)
        and e.price_history_end < e.price_history_start
    ):
        problems.append(
            f"price history ends ({e.price_history_end.isoformat()}) before it starts "
            f"({e.price_history_start.isoformat()})."
        )

    if problems:
        return False, "IMPOSSIBLE EVIDENCE: " + " ".join(problems)
    return True, "Every supplied value is finite, in-domain and correctly typed."


def _price_history(e: Evidence) -> tuple[bool, str]:
    if e.price_history_start is None or e.price_history_end is None:
        span_check = (False, "NO EVIDENCE RECORDED: price history date range.")
    elif not (isinstance(e.price_history_start, date) and isinstance(e.price_history_end, date)):
        span_check = (False, "Price history date range is not made of dates.")
    else:
        years = (e.price_history_end - e.price_history_start).days / 365.25
        span_check = _min_value(years, MIN_PRICE_HISTORY_YEARS, "price history span (years)")
    return _all_of(
        _flag(e.price_history_is_real_market_data, "price history is real market data"),
        _min_value(e.price_history_symbols, MIN_PRICE_HISTORY_SYMBOLS, "symbols with history"),
        span_check,
    )


def _market_data_current(e: Evidence) -> tuple[bool, str]:
    """
    A ten-year history that stops in 2019 is a museum piece. The last bar must
    be recent and the feed must currently be healthy.
    """
    return _all_of(
        _fresh_date(
            e.price_history_end,
            timedelta(days=MAX_PRICE_DATA_STALENESS_DAYS),
            "last bar of price history",
        ),
        _flag(e.market_data_feed_healthy, "market data feed health"),
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
    if _is_blank(e.benchmark_name):
        return False, (
            f"NO EVIDENCE RECORDED: no passive benchmark was named "
            f"(got {e.benchmark_name!r})."
        )
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
    used, problem = _as_finite(e.n_trials_used_in_dsr, "trial count used in the DSR")
    if problem is not None:
        return False, problem
    recorded, _ = _as_finite(e.n_trials_recorded, "recorded trial count")
    assert recorded is not None and used is not None
    if used != recorded:
        return False, (
            f"Trial count is not honest: {recorded:g} trials were run but the "
            f"DSR was deflated for only {used:g}."
        )
    return True, f"{recorded:g} trials recorded and fully reflected in the DSR."


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


def _broker_connectivity(e: Evidence) -> tuple[bool, str]:
    """
    Broker unavailability blocks. ``broker_auth_verified`` says the plumbing
    worked once; this says it works now.
    """
    return _all_of(
        _fresh(
            e.broker_reachable_at,
            timedelta(minutes=MAX_BROKER_HEARTBEAT_AGE_MINUTES),
            "broker heartbeat",
        ),
        _flag(e.broker_session_valid, "broker session validity"),
        _not_flag(e.broker_api_degraded, "broker API degradation"),
    )


def _reconciliation_current(e: Evidence) -> tuple[bool, str]:
    """
    A failed or stale reconciliation blocks. If the book we think we hold and
    the book the broker thinks we hold have diverged, every downstream size,
    risk number and square-off is computed against a fiction.
    """
    return _all_of(
        _fresh(
            e.last_reconciliation_at,
            timedelta(hours=MAX_RECONCILIATION_AGE_HOURS),
            "position reconciliation",
        ),
        _flag(e.last_reconciliation_succeeded, "last reconciliation outcome"),
        _max_value(
            e.open_reconciliation_breaks,
            MAX_OPEN_RECONCILIATION_BREAKS,
            "open reconciliation breaks",
        ),
    )


def _risk_limits(e: Evidence) -> tuple[bool, str]:
    """
    Live risk-limit violations block. So does an engaged kill switch: if
    something has already decided to halt trading, this gate must not
    countermand it.
    """
    return _all_of(
        _flag(e.risk_limits_loaded, "risk limits loaded and active"),
        _max_value(
            e.active_risk_limit_breaches,
            MAX_ACTIVE_RISK_LIMIT_BREACHES,
            "active risk-limit breaches",
        ),
        _max_value(e.current_drawdown, MAX_DRAWDOWN, "current live drawdown"),
        _not_flag(e.kill_switch_engaged, "kill switch engagement"),
    )


def _enforcement_wired(e: Evidence) -> tuple[bool, str]:
    """
    The gate that audits its own teeth.

    A verdict nothing consults is decoration. This fails until the live order
    path actually calls :func:`require_live_eligible`.
    """
    ok, reason = _flag(
        e.eligibility_enforced_at_order_path,
        "eligibility enforcement wired into the live order path",
    )
    if ok:
        return True, reason
    return False, (
        f"{reason} Until OrderManager.submit_order (execution/order_manager.py) "
        f"calls require_live_eligible(), this module reports a verdict that nothing "
        f"obeys and live orders can be placed while BLOCKED."
    )


def _paper_matches_backtest(e: Evidence) -> tuple[bool, str]:
    paper, problem_p = _as_finite(e.paper_sharpe, "paper Sharpe")
    if problem_p is not None:
        return False, problem_p
    backtest, problem_b = _as_finite(e.backtest_expected_sharpe, "backtest expected Sharpe")
    if problem_b is not None:
        return False, problem_b
    assert paper is not None and backtest is not None

    if backtest <= 0:
        return False, (
            f"Backtest expected Sharpe is {backtest:.4g}; there is no "
            f"positive expectation for paper trading to confirm."
        )
    floor = MIN_PAPER_TO_BACKTEST_SHARPE_RATIO * backtest
    if not (paper >= floor):
        return False, (
            f"Paper Sharpe {paper:.4g} is below {floor:.4g} "
            f"({MIN_PAPER_TO_BACKTEST_SHARPE_RATIO:g}x the backtest's "
            f"{backtest:.4g}). A material divergence means the backtest "
            f"is wrong — investigate rather than proceed."
        )
    return True, (
        f"Paper Sharpe {paper:.4g} is consistent with the backtest's "
        f"{backtest:.4g}."
    )


def _human_enabled_live(e: Evidence) -> tuple[bool, str]:
    """
    Two independent things must be true: the config says live, and a named
    human took responsibility for it. Neither alone is sufficient — a config
    flag can be flipped by a deploy, and an approval can go stale.
    """
    if e.trading_mode is None:
        return False, "NO EVIDENCE RECORDED: TRADING_MODE."
    if not isinstance(e.trading_mode, str) or e.trading_mode != "live":
        return False, f"TRADING_MODE is {e.trading_mode!r}, not 'live'."
    if _is_blank(e.live_enable_approved_by):
        return False, (
            f"TRADING_MODE is 'live' but no human approver is recorded "
            f"(got {e.live_enable_approved_by!r}). Whitespace and invisible "
            f"characters are not a name."
        )
    approver = str(e.live_enable_approved_by).strip()
    if len(approver) < 2:
        return False, (
            f"Recorded approver {e.live_enable_approved_by!r} is not a name."
        )
    if e.live_enable_approved_at is None:
        return False, (
            f"TRADING_MODE is 'live', approved by {approver!r}, "
            f"but with no approval date."
        )
    if not isinstance(e.live_enable_approved_at, date) or isinstance(
        e.live_enable_approved_at, datetime
    ):
        return False, (
            f"Approval date is {e.live_enable_approved_at!r}, not a date. "
            f"An approval that cannot be dated cannot be aged."
        )
    return True, (
        f"TRADING_MODE explicitly set to 'live' by {approver} "
        f"on {e.live_enable_approved_at.isoformat()}."
    )


def _approval_is_current(e: Evidence) -> tuple[bool, str]:
    """A sign-off from last year is a historical fact, not a current decision."""
    return _fresh_date(
        e.live_enable_approved_at,
        timedelta(days=MAX_APPROVAL_AGE_DAYS),
        "human live-trading approval",
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
_RISK = EligibilityState.BLOCKED_RISK_LIMIT_BREACH
_PAPER = EligibilityState.PAPER_ONLY


ALL_GATES: tuple[Gate, ...] = (
    # ── DATA ──────────────────────────────────────────────────────────────
    Gate("evidence_freshness", _D, _INSUFFICIENT_DATA,
         f"The Evidence record states when it was gathered, and that is within "
         f"{MAX_EVIDENCE_AGE_HOURS:g}h and not in the future.",
         _evidence_freshness),
    Gate("evidence_well_formed", _D, _INSUFFICIENT_DATA,
         "Every supplied value is finite, correctly typed and inside its physically "
         "possible range. NaN, +/-inf, negative drawdowns and future dates are bugs, "
         "not measurements.",
         _evidence_well_formed),
    Gate("real_price_history", _D, _INSUFFICIENT_DATA,
         f"Real daily OHLCV for >= {MIN_PRICE_HISTORY_SYMBOLS} symbols spanning "
         f">= {MIN_PRICE_HISTORY_YEARS:g} years.",
         _price_history),
    Gate("market_data_current", _D, _INSUFFICIENT_DATA,
         f"The last bar of price history is within {MAX_PRICE_DATA_STALENESS_DAYS:g} "
         f"days and the feed is currently healthy.",
         _market_data_current),
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
    Gate("broker_connectivity", _X, _EXECUTION,
         f"The broker answered within the last {MAX_BROKER_HEARTBEAT_AGE_MINUTES:g} "
         f"minutes, the session is valid, and the API is not degraded.",
         _broker_connectivity),
    Gate("order_lifecycle", _X, _EXECUTION,
         "Full order lifecycle verified: place, modify, cancel, partial fill, reject.",
         lambda e: _flag(e.order_lifecycle_verified, "order lifecycle")),
    Gate("reconciliation", _X, _EXECUTION,
         "Startup reconciliation against real broker positions verified.",
         lambda e: _flag(e.reconciliation_verified, "startup reconciliation")),
    Gate("reconciliation_current", _X, _EXECUTION,
         f"The most recent reconciliation ran within {MAX_RECONCILIATION_AGE_HOURS:g}h, "
         f"succeeded, and left no open position breaks.",
         _reconciliation_current),
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
    Gate("enforcement_wired_into_order_path", _X, _EXECUTION,
         "The live order path calls require_live_eligible() before touching the broker. "
         "A verdict nothing consults is decoration.",
         _enforcement_wired),

    # ── RISK ──────────────────────────────────────────────────────────────
    Gate("risk_limits_enforced", _O, _RISK,
         f"Risk limits are loaded, no limit is currently breached, live drawdown is "
         f"<= {MAX_DRAWDOWN:.0%}, and the kill switch is not engaged.",
         _risk_limits),

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
    Gate("approval_is_current", _O, _PAPER,
         f"That human approval is less than {MAX_APPROVAL_AGE_DAYS:g} days old.",
         _approval_is_current),
)

# ---------------------------------------------------------------------------
# The canonical registry.
#
# ``ALL_GATES`` is a module-level name and can be rebound by anything in the
# process — including, historically, to ``()``, at which point a report with no
# results had no failures and therefore reported LIVE_ELIGIBLE. Everything that
# decides a verdict reads the private copies below, which are captured here at
# import and never consulted through the public name.
# ---------------------------------------------------------------------------

_CANONICAL_GATES: tuple[Gate, ...] = ALL_GATES
_CANONICAL_BY_NAME: dict[str, Gate] = {g.name: g for g in _CANONICAL_GATES}
_CANONICAL_GATE_NAMES: frozenset[str] = frozenset(_CANONICAL_BY_NAME)

if len(_CANONICAL_BY_NAME) != len(_CANONICAL_GATES):  # pragma: no cover - import guard
    raise RuntimeError("Duplicate gate names in ALL_GATES; the registry is ambiguous.")

# Only assess_live_trading_eligibility holds this. A report that cannot present
# it is UNTRUSTED and cannot authorize an order.
_TRUST_TOKEN = object()


# ===========================================================================
# Report
# ===========================================================================

@dataclass(frozen=True)
class EligibilityReport:
    """
    The verdict plus the checklist that produced it.

    ``state`` is a property, not a field. There is no way to hand this class a
    permissive answer; it can only compute one, and it computes ``LIVE_ELIGIBLE``
    only when every canonical gate is present and passing.

    ``provenance`` is the second lock. Deriving LIVE_ELIGIBLE from a set of
    ``passed: true`` flags is easy for anyone who can write a JSON file, so a
    report that was not computed in-process by
    :func:`assess_live_trading_eligibility` is ``UNTRUSTED`` — it may *say*
    LIVE_ELIGIBLE, ``permits_live_trading`` is still False, and
    :func:`require_live_eligible` refuses it.
    """

    results: tuple[GateResult, ...]
    # The lambda matters: `default_factory=_now` would bind the function object
    # at class-definition time, so a test that redirects the module clock would
    # silently keep receiving the real one and the staleness path would never
    # actually be exercised.
    evaluated_at: datetime = field(default_factory=lambda: _now())
    evidence_notes: tuple[str, ...] = ()
    provenance: ReportProvenance = ReportProvenance.UNTRUSTED
    trust: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """
        Normalize the result set against the canonical registry.

        Three things happen here, all of them one-directional (they can only
        make a report less permissive):

        1. Any canonical gate absent from ``results`` is injected as a failure.
        2. Any result carrying a canonical gate's name has its category,
           blocking state and requirement re-bound to the canonical values, so a
           forged payload cannot relabel a DATA failure as a mild PAPER_ONLY one
           and mislead whoever is reading the checklist.
        3. ``provenance`` is forced to UNTRUSTED unless the caller presented the
           private trust token.
        """
        if self.trust is not _TRUST_TOKEN:
            object.__setattr__(self, "provenance", ReportProvenance.UNTRUSTED)
        object.__setattr__(self, "trust", None)

        normalized: list[GateResult] = []
        for r in self.results:
            canonical = _CANONICAL_BY_NAME.get(r.name)
            if canonical is not None and (
                r.category is not canonical.category
                or r.blocking_state is not canonical.blocking_state
                or r.requirement != canonical.requirement
            ):
                r = replace(
                    r,
                    category=canonical.category,
                    blocking_state=canonical.blocking_state,
                    requirement=canonical.requirement,
                )
            normalized.append(r)

        seen = {r.name for r in normalized}
        for g in _CANONICAL_GATES:
            if g.name not in seen:
                normalized.append(
                    GateResult(
                        name=g.name,
                        category=g.category,
                        blocking_state=g.blocking_state,
                        requirement=g.requirement,
                        passed=False,
                        reason="GATE WAS NOT EVALUATED. An unevaluated gate fails.",
                    )
                )

        object.__setattr__(self, "results", tuple(normalized))

    # ── Verdict ───────────────────────────────────────────────────────────

    @property
    def state(self) -> EligibilityState:
        """
        The derived verdict.

        LIVE_ELIGIBLE requires that every canonical gate is present *and*
        passing — not merely that nothing in ``results`` failed, which an empty
        or filtered result set would also satisfy.
        """
        failed = self.failed_gates
        if failed:
            return min((r.blocking_state for r in failed), key=lambda s: s.severity)
        passed_names = {r.name for r in self.results if r.passed}
        if not _CANONICAL_GATE_NAMES.issubset(passed_names):
            return EligibilityState.default()
        return EligibilityState.LIVE_ELIGIBLE

    @property
    def permits_live_trading(self) -> bool:
        """
        The only property any caller should branch on — and even then, prefer
        :func:`require_live_eligible`, which also checks that this report is
        fresh enough to be acted on.
        """
        return (
            self.state is EligibilityState.LIVE_ELIGIBLE
            and self.provenance is ReportProvenance.COMPUTED
        )

    @property
    def is_stale(self) -> bool:
        return self.age > timedelta(minutes=MAX_REPORT_AGE_MINUTES)

    @property
    def age(self) -> timedelta:
        stamp = self.evaluated_at
        if not isinstance(stamp, datetime):
            return timedelta.max
        if stamp.tzinfo is None or stamp.tzinfo.utcoffset(stamp) is None:
            return timedelta.max
        return _now() - stamp

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
            if self.state is not EligibilityState.LIVE_ELIGIBLE:
                return "Not every canonical gate is present in this report."
            if self.provenance is not ReportProvenance.COMPUTED:
                return (
                    "This report was not computed in-process; an untrusted report "
                    "cannot authorize live trading."
                )
            return "All gates passed."
        # `next(...)` with no default would raise StopIteration if `state` and
        # the failures ever disagreed — and this property is called from inside
        # require_live_eligible's block path, where raising anything other than
        # LiveTradingBlocked would turn a clean refusal into an unhandled crash.
        state = self.state
        first = next((r for r in failed if r.blocking_state is state), failed[0])
        return f"{first.name}: {first.reason}"

    # ── Human output ──────────────────────────────────────────────────────

    def checklist(self) -> str:
        """A checklist to be worked top to bottom."""
        state = self.state
        permitted = self.permits_live_trading
        lines = [
            "LIVE TRADING ELIGIBILITY",
            "=" * 72,
            f"STATE          : {state.name}",
            f"LIVE PERMITTED : {'YES' if permitted else 'NO'}",
            f"GATES          : {len(self.passed_gates)}/{len(self.results)} passed",
            f"EVALUATED AT   : {self.evaluated_at.isoformat()}",
            f"PROVENANCE     : {self.provenance.name}",
        ]
        if self.provenance is not ReportProvenance.COMPUTED:
            lines.append(
                "                 (untrusted — informational only, authorizes nothing)"
            )
        if not permitted:
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
            "permits_live_trading": self.permits_live_trading,
            "severity": state.severity,
            "provenance": self.provenance.value,
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
        """
        Rebuild a report from a payload.

        The result is always UNTRUSTED, whatever the payload claims. It is a
        record of an assessment someone else performed; it is not permission.
        """
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
    gates: Optional[Sequence[Gate]] = None,
) -> EligibilityReport:
    """
    Evaluate every gate and return the verdict.

    Passing no evidence means "nothing is known", which blocks. It does not
    mean "assume the best".

    ``gates`` exists for testing and defaults to the canonical registry.
    Neither narrowing nor substituting it can widen the verdict:

    * a canonical gate missing from the result set is injected as a failure by
      :meth:`EligibilityReport.__post_init__`;
    * a *substituted* gate — one carrying a canonical gate's name but not the
      canonical gate object — may report a failure (so tests can inject one)
      but its passes are discarded. Before this check existed, one line of
      test-helper-shaped code produced LIVE_ELIGIBLE from empty evidence.
    """
    if evidence is None:
        evidence = Evidence(notes=("No evidence was supplied to the assessor.",))
    if not isinstance(evidence, Evidence):
        raise TypeError(f"evidence must be an Evidence, got {type(evidence).__name__}")

    # None means "the canonical registry". An explicitly empty sequence means
    # exactly that — evaluate nothing — and every canonical gate is then
    # injected as a failure, which is the point of allowing it.
    chosen = _CANONICAL_GATES if gates is None else tuple(gates)

    results: list[GateResult] = []
    for g in chosen:
        result = g.check(evidence)
        canonical = _CANONICAL_BY_NAME.get(g.name)
        if canonical is not None and g is not canonical and result.passed:
            logger.error(
                "Gate %r was SUBSTITUTED and reported a pass; discarding the pass.",
                g.name,
            )
            result = replace(
                result,
                passed=False,
                reason=(
                    f"GATE WAS SUBSTITUTED. A gate object carrying the canonical name "
                    f"{g.name!r} but which is not the canonical gate cannot report a "
                    f"pass. Original reason: {result.reason}"
                ),
            )
        results.append(result)

    return EligibilityReport(
        results=tuple(results),
        evidence_notes=tuple(evidence.notes),
        provenance=ReportProvenance.COMPUTED,
        trust=_TRUST_TOKEN,
    )


def live_trading_eligibility(evidence: Optional[Evidence] = None) -> EligibilityState:
    """
    LIVE_TRADING_ELIGIBILITY: the current derived state.

    A function rather than a constant, because a constant computed at import
    would be a claim about the past.
    """
    if evidence is None:
        evidence = gather_repo_evidence()
    return assess_live_trading_eligibility(evidence).state


# ===========================================================================
# ENFORCEMENT
# ===========================================================================

def require_live_eligible(
    report: Optional[EligibilityReport] = None,
    *,
    action: str = "place a live order",
    intent: OrderIntent = OrderIntent.INCREASE_RISK,
    max_report_age: Optional[timedelta] = None,
) -> EligibilityReport:
    """
    Assert that live trading is permitted, or raise :class:`LiveTradingBlocked`.

    **THIS IS THE ENFORCEMENT POINT AND IT MUST BE CALLED.** Wire it into
    ``OrderManager.submit_order`` (execution/order_manager.py), at the top,
    before any broker call::

        from app.governance.eligibility import require_live_eligible, OrderIntent

        async def submit_order(self, signal, position_size, broker, ...):
            require_live_eligible(action=f"submit {signal.symbol} x{position_size}")
            ...

    and use ``intent=OrderIntent.REDUCE_RISK`` in ``emergency_flatten_all`` and
    ``cancel_order``, which must keep working while blocked.

    Until that call exists, gate ``enforcement_wired_into_order_path`` fails and
    LIVE_ELIGIBLE is unreachable, so the system cannot quietly become
    "eligible but unenforced".

    Parameters
    ----------
    report
        A report to check. If omitted, one is computed now from
        :func:`gather_repo_evidence` — the safe default, because a caller who
        passes nothing gets a *fresh* verdict rather than an assumed one.
    action
        What is being attempted. Appears in the block log; make it specific.
    intent
        ``INCREASE_RISK`` (default) requires eligibility. ``REDUCE_RISK``
        is permitted while blocked and logged at WARNING.
    max_report_age
        How old a passed-in report may be. Defaults to
        ``MAX_REPORT_AGE_MINUTES``. A stale LIVE_ELIGIBLE is not a current one.

    Raises
    ------
    LiveTradingBlocked
        Whenever live trading is not permitted — including when the report is
        the wrong type, untrusted, stale, or missing a canonical gate.

    Notes
    -----
    There is no override parameter, environment variable or force flag, by
    design. See the module docstring.
    """
    if intent is OrderIntent.REDUCE_RISK:
        logger.warning(
            "RISK-REDUCING ACTION PERMITTED WITHOUT AN ELIGIBILITY CHECK: %s. "
            "A blocked system must still be able to get flat.",
            action,
        )
        return report if isinstance(report, EligibilityReport) else EligibilityReport(
            results=()
        )

    if intent is not OrderIntent.INCREASE_RISK:
        raise LiveTradingBlocked(
            EligibilityState.default(),
            f"Unrecognized OrderIntent {intent!r}; an unrecognized intent is blocked.",
        )

    if report is None:
        try:
            report = assess_live_trading_eligibility(gather_repo_evidence())
        except Exception as exc:  # noqa: BLE001 — an unevaluable gate blocks
            logger.critical(
                "LIVE TRADING BLOCKED: eligibility could not be assessed for %r: %r",
                action, exc,
            )
            raise LiveTradingBlocked(
                EligibilityState.default(),
                f"Eligibility could not be assessed ({type(exc).__name__}: {exc}). "
                f"An unevaluable gate blocks.",
            ) from exc

    # Exactly EligibilityReport — not a subclass. A subclass can override
    # `state` and `permits_live_trading` to return anything it likes.
    if type(report) is not EligibilityReport:
        logger.critical(
            "LIVE TRADING BLOCKED: %r was offered a %s, not an EligibilityReport.",
            action, type(report).__name__,
        )
        raise LiveTradingBlocked(
            EligibilityState.default(),
            f"Expected an EligibilityReport, got {type(report).__name__}. A subclass "
            f"can override the verdict, so only the exact type is accepted.",
        )

    if report.provenance is not ReportProvenance.COMPUTED:
        logger.critical(
            "LIVE TRADING BLOCKED: %r was offered an %s report.",
            action, report.provenance.name,
        )
        raise LiveTradingBlocked(
            EligibilityState.default(),
            "This report was not computed in-process. A deserialized or "
            "hand-constructed report describes an assessment; it is not one.",
            report,
        )

    limit = max_report_age or timedelta(minutes=MAX_REPORT_AGE_MINUTES)
    age = report.age
    if age > limit:
        logger.critical(
            "LIVE TRADING BLOCKED: %r used a report %.1f min old (limit %.1f min).",
            action, age.total_seconds() / 60, limit.total_seconds() / 60,
        )
        raise LiveTradingBlocked(
            EligibilityState.default(),
            f"Eligibility report is {age.total_seconds() / 60:.1f} minutes old "
            f"(limit {limit.total_seconds() / 60:.1f}). Re-assess before trading.",
            report,
        )

    # Re-derive independently. Do not trust report.state — walk the results.
    passed = {r.name for r in report.results if r.passed is True}
    missing = sorted(_CANONICAL_GATE_NAMES - passed)
    if missing:
        # Everything from here to the raise is best-effort presentation. A
        # failure to *describe* the block must never become a failure to
        # *perform* it, so the exact refusal is built from `missing` alone and
        # the nicer detail is added only if it can be obtained.
        try:
            state = report.state
            detail = f" {report.blocking_reason}"
        except Exception:  # noqa: BLE001 — a broken report still blocks
            state, detail = EligibilityState.default(), ""
        if not isinstance(state, EligibilityState) or state.permits_live_trading:
            # A report whose own state disagrees with its results is corrupt.
            state = EligibilityState.default()
        logger.critical(
            "LIVE TRADING BLOCKED [%s]: %r refused. %d gate(s) not passing: %s",
            state.name, action, len(missing), ", ".join(missing[:8]),
        )
        raise LiveTradingBlocked(
            state,
            f"{len(missing)} gate(s) not passing: {', '.join(missing)}.{detail}",
            report,
        )

    logger.info(
        "LIVE TRADING PERMITTED: %s (all %d gates passing, report %.1fs old).",
        action, len(_CANONICAL_GATES), age.total_seconds(),
    )
    return report


# ===========================================================================
# Evidence gathered from the repository as it actually stands
# ===========================================================================

_CACHE_FILENAME = re.compile(r"^(?P<symbol>.+)__(?P<start>[\d-]+)__(?P<end>[\d-]+)\.parquet$")

_ORDER_PATH = Path(__file__).resolve().parents[1] / "execution" / "order_manager.py"


def _price_cache_dir() -> Path:
    """Where app.data.providers writes its parquet cache."""
    try:
        from app.data.providers import _CACHE_DIR  # noqa: PLC0415
        return Path(_CACHE_DIR)
    except Exception:  # noqa: BLE001 — the module may not exist; fall back to the layout
        return Path(__file__).resolve().parents[3] / "data_cache"


def _enforcement_is_wired() -> Optional[bool]:
    """
    Does the live order path actually call :func:`require_live_eligible`?

    Read the source and look. Returning ``None`` (unknown, therefore failing)
    when the file cannot be read is deliberate: an unreadable order path is not
    a wired one.
    """
    try:
        source = _ORDER_PATH.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s to check enforcement: %r", _ORDER_PATH, exc)
        return None
    return "require_live_eligible" in source


def gather_repo_evidence() -> Evidence:
    """
    Build an :class:`Evidence` record from what this repository can actually
    demonstrate right now.

    This function is deliberately stingy. It sets a field only when it has
    directly observed the fact. Everything it cannot observe — every statistical
    result, every performance number, every execution verification, every live
    risk reading — is left ``None``, because no such result has ever been
    persisted anywhere in this repository. That is a finding, not a gap in this
    function.
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

    enforced = _enforcement_is_wired()
    if enforced is False:
        notes.append(
            "CRITICAL: execution/order_manager.py does NOT call "
            "require_live_eligible(). The eligibility verdict is currently advisory "
            "— nothing in the live order path consults it."
        )
    elif enforced is None:
        notes.append(
            "execution/order_manager.py could not be read; enforcement wiring is "
            "unknown and therefore treated as absent."
        )
    else:
        notes.append(
            "execution/order_manager.py references require_live_eligible()."
        )

    notes.append(
        "The price source (Yahoo Finance) supplies no point-in-time index membership "
        "and no point-in-time fundamentals, and its intraday history is ~60 days — "
        "see app/data/providers.py."
    )
    notes.append(
        "No model, backtest result, validation record, paper-trading record, broker "
        "session, reconciliation run or live risk reading has ever been persisted by "
        "this repository, so every statistical, performance, execution, risk and "
        "operational field below is unknown."
    )

    return Evidence(
        # Provenance — observed: this record is being built right now.
        evidence_gathered_at=_now(),
        # Data — observed.
        price_history_symbols=symbols,
        price_history_start=start,
        price_history_end=end,
        price_history_is_real_market_data=is_real,
        point_in_time_index_constituents_available=False,
        point_in_time_fundamentals_available=fundamentals_real,
        intraday_history_days=0,
        # Enforcement — observed by reading the order path's source.
        eligibility_enforced_at_order_path=enforced,
        # Statistical / performance / execution / risk / operational —
        # unobserved, therefore left as None, therefore failing.
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
    "LiveTradingBlocked",
    "OrderIntent",
    "ReportProvenance",
    "assess_live_trading_eligibility",
    "assess_repo_live_trading_eligibility",
    "gather_repo_evidence",
    "live_trading_eligibility",
    "require_live_eligible",
]
