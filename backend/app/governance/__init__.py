"""
Governance package — the gate between research and real money.

The only question this package answers is whether the system may place live
orders, and its default answer is no.

The name to reach for is :func:`require_live_eligible`. It raises
:class:`LiveTradingBlocked` unless a freshly computed assessment clears every
gate, and it is what must be called from the live order path — reading
``permits_live_trading`` and branching on it is weaker, because a caller who
forgets to branch places the order anyway. See docs/LIVE_TRADING_GATES.md.
"""

from .eligibility import (
    ALL_GATES,
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

__all__ = [
    "ALL_GATES",
    "EligibilityReport",
    "EligibilityState",
    "Evidence",
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
