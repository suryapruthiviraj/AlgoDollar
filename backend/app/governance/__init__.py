"""
Governance package — the gate between research and real money.

The only question this package answers is whether the system may place live
orders, and its default answer is no.
"""

from .eligibility import (
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

__all__ = [
    "ALL_GATES",
    "EligibilityReport",
    "EligibilityState",
    "Evidence",
    "Gate",
    "GateCategory",
    "GateResult",
    "assess_live_trading_eligibility",
    "assess_repo_live_trading_eligibility",
    "gather_repo_evidence",
]
