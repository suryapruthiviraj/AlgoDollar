"""
Production portfolio allocation: capital, risk, and the target portfolio.

CAPITAL ALLOCATION AND RISK ALLOCATION ARE DIFFERENT DECISIONS
--------------------------------------------------------------
Conflating them is the central modelling error this module exists to avoid.

* **Capital allocation** answers "how many rupees does this sleeve get?". It is
  bounded by the cash that exists and sums to the available capital.
* **Risk allocation** answers "how much of the portfolio's volatility budget may
  this sleeve consume?". It is bounded by the volatility target and sums to
  roughly 1.0 of that budget.

They are not proportional to one another and must not be derived from one
another. A low-volatility long-term sleeve holding 50% of capital might consume
25% of the risk budget; an intraday sleeve holding 10% of capital might consume
35%. Sizing the second from its capital share would systematically under-measure
what it can lose, and sizing the first from its risk share would leave capital
idle for no reason.

Both are produced, both are reported, and the binding constraint is recorded
separately for each.

CASH IS A VALID ALLOCATION
--------------------------
There is no path in this module that deploys capital because capital exists. If
the opportunity set is thin, the regime is hostile, the data is stale, the
drawdown limit is breached or the kill switch is engaged, the target portfolio
is CASH and the reason says which. "Fully invested" is an outcome, not a goal.

THE EXISTING PORTFOLIO IS THE STARTING POINT, NOT AN AFTERTHOUGHT
------------------------------------------------------------------
Targets are computed against the CURRENT book. A monthly contribution buys the
delta needed to move toward the target; it does not liquidate and rebuild.
Turnover is measured on that delta and is itself a hard constraint, because a
target that is optimal before costs and unreachable after them is not optimal.

EVERY DECISION IS REPRODUCIBLE
------------------------------
:class:`AllocationInputs` carries everything the engine reads. It hashes to a
content fingerprint, and :class:`AllocationSnapshot` stores inputs, outputs and
that fingerprint together — so any past allocation can be re-run and checked
against what it produced.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StrategyBucket(str, Enum):
    LONG_TERM = "longterm"
    SWING = "swing"
    INTRADAY = "intraday"
    CASH = "cash"


#: The three deployable sleeves. CASH is the residual and is never "allocated
#: to" — it is what remains when the others have taken what they justified.
DEPLOYABLE = (StrategyBucket.LONG_TERM, StrategyBucket.SWING, StrategyBucket.INTRADAY)


# --------------------------------------------------------------------------- #
#  Limits                                                                       #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RiskLimits:
    """
    Hard bounds. Every one of them can only ever REDUCE a position.

    None of these are targets to be reached. `max_position_pct` is not "put 10%
    in your best idea"; it is "never more than 10% in anything".
    """

    max_position_pct: float = 0.10
    max_sector_pct: float = 0.25
    max_strategy_pct: float = 0.60
    max_portfolio_drawdown_pct: float = 0.15
    max_daily_loss_pct: float = 0.02
    target_portfolio_vol: float = 0.15
    max_portfolio_vol: float = 0.25
    #: Maximum fraction of a name's median daily traded value we may hold.
    max_liquidity_participation: float = 0.05
    #: Maximum one-way turnover per rebalance, as a fraction of portfolio value.
    max_turnover_pct: float = 0.35
    #: Cash that must remain uninvested regardless of opportunity.
    min_cash_pct: float = 0.05
    max_positions: int = 20
    #: Fractional Kelly multiplier. NEVER 1.0 — full Kelly is optimal only with
    #: a known-exact edge, and every edge here is estimated from a finite
    #: sample. A quarter-Kelly cap is the standard conservative choice, and it
    #: is a CAP, applied after every other constraint.
    kelly_fraction: float = 0.25
    max_kelly_weight: float = 0.15

    def __post_init__(self) -> None:
        if not 0 < self.kelly_fraction <= 0.5:
            raise ValueError(
                "kelly_fraction must be in (0, 0.5]. Unconstrained or full "
                "Kelly is not permitted: it assumes the edge estimate is exact."
            )
        for name in ("max_position_pct", "max_sector_pct", "max_strategy_pct"):
            v = getattr(self, name)
            if not 0 < v <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 <= self.min_cash_pct < 1.0:
            raise ValueError("min_cash_pct must be in [0, 1)")


# --------------------------------------------------------------------------- #
#  Inputs                                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class PositionInput:
    """One currently-held position, as the book records it."""

    symbol: str
    quantity: int
    average_price: float
    last_price: float
    strategy: str = "unknown"
    sector: str = "UNKNOWN"

    @property
    def market_value(self) -> float:
        return float(self.quantity) * float(self.last_price)


@dataclass
class SignalInput:
    """One proposal from a strategy, with everything sizing needs."""

    symbol: str
    strategy: str
    direction: str
    edge_score: float
    expected_return: float
    expected_return_std: float
    price: float
    sector: str = "UNKNOWN"
    #: Annualised realised volatility. Required for risk allocation — a signal
    #: without one cannot be risk-sized and is dropped with a stated reason.
    volatility: Optional[float] = None
    #: Median daily traded VALUE in rupees. Absent, the liquidity limit cannot
    #: be applied and the name is dropped rather than assumed liquid.
    median_traded_value: Optional[float] = None
    #: Age of the quote that produced `price`, in seconds.
    quote_age_sec: Optional[float] = None


@dataclass
class AllocationInputs:
    """
    Everything the engine reads. Nothing is fetched inside the engine.

    That is what makes an allocation reproducible: re-running with the same
    inputs must produce the same target, and the fingerprint below proves which
    inputs were used.
    """

    as_of: datetime
    total_capital: float
    cash: float
    positions: list[PositionInput] = field(default_factory=list)
    signals: list[SignalInput] = field(default_factory=list)
    contribution: float = 0.0

    strategy_health: dict[str, str] = field(default_factory=dict)
    regime: str = "UNKNOWN"

    #: Symbol -> annualised volatility, for names already held.
    volatilities: dict[str, float] = field(default_factory=dict)
    #: Symbol-ordered correlation matrix, if one is available.
    correlation_symbols: list[str] = field(default_factory=list)
    correlation_matrix: Optional[list[list[float]]] = None

    current_drawdown_pct: float = 0.0
    daily_pnl_pct: float = 0.0
    realised_vol: Optional[float] = None

    cost_bps_per_side: float = 25.0
    limits: RiskLimits = field(default_factory=RiskLimits)

    kill_switch_active: bool = False
    market_data_stale: bool = False
    trading_permitted: bool = True

    def fingerprint(self) -> str:
        """
        Content hash of every input.

        Deterministic across processes: floats are rounded to a fixed precision
        before hashing, because otherwise the last bits of a float make two
        identical allocations look different.
        """
        payload = json.dumps(
            _roundtrip(asdict(self)), sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _roundtrip(obj: Any, ndigits: int = 8) -> Any:
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _roundtrip(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_roundtrip(v, ndigits) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
#  Outputs                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class BindingConstraint:
    """A limit that actually bound, and by how much."""

    name: str
    scope: str
    limit: float
    requested: float
    applied: float

    @property
    def reduction(self) -> float:
        return max(0.0, self.requested - self.applied)

    def __str__(self) -> str:
        return (
            f"{self.name}[{self.scope}]: wanted {self.requested:.4f}, "
            f"limit {self.limit:.4f}, applied {self.applied:.4f}"
        )


@dataclass
class PositionTarget:
    symbol: str
    strategy: str
    sector: str
    target_weight: float
    target_value: float
    target_quantity: int
    current_quantity: int
    delta_quantity: int
    price: float
    expected_vol: Optional[float]
    #: Contribution to portfolio variance, as a fraction of total. This is the
    #: RISK number; target_weight is the CAPITAL number. They differ.
    risk_contribution_pct: Optional[float]
    reason: str
    constraints: list[str] = field(default_factory=list)

    @property
    def delta_value(self) -> float:
        return float(self.delta_quantity) * float(self.price)


@dataclass
class StrategyTarget:
    """
    Per-sleeve capital AND risk, kept separate throughout.

    ``capital_pct`` and ``risk_pct`` answer different questions and are not
    expected to match. A large divergence between them is informative, not an
    error — it says the sleeve is holding assets whose volatility differs from
    the portfolio average.
    """

    bucket: StrategyBucket
    capital_amount: float
    capital_pct: float
    risk_budget_pct: float
    risk_used_pct: float
    n_positions: int
    health: str
    reason: str
    constraints: list[str] = field(default_factory=list)


@dataclass
class TargetPortfolio:
    """The complete allocation decision, with its reasoning attached."""

    as_of: datetime
    input_fingerprint: str
    is_no_trade: bool
    strategies: list[StrategyTarget] = field(default_factory=list)
    positions: list[PositionTarget] = field(default_factory=list)
    cash_reserve: float = 0.0
    cash_reserve_pct: float = 0.0
    expected_portfolio_vol: Optional[float] = None
    expected_turnover_pct: float = 0.0
    estimated_cost: float = 0.0
    estimated_cost_pct: float = 0.0
    binding_constraints: list[BindingConstraint] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def deployed_value(self) -> float:
        return sum(p.target_value for p in self.positions)

    def summary(self) -> str:
        if self.is_no_trade:
            return f"NO TRADE / CASH — {'; '.join(self.reasons) or 'no reason recorded'}"
        return (
            f"{len(self.positions)} target position(s), "
            f"Rs {self.deployed_value:,.0f} deployed, "
            f"Rs {self.cash_reserve:,.0f} cash ({self.cash_reserve_pct:.1%}), "
            f"turnover {self.expected_turnover_pct:.1%}, "
            f"est. cost Rs {self.estimated_cost:,.0f}"
        )

    def as_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        d["binding_constraints"] = [str(c) for c in self.binding_constraints]
        for s in d["strategies"]:
            s["bucket"] = s["bucket"].value if hasattr(s["bucket"], "value") else str(s["bucket"])
        return d


@dataclass
class AllocationSnapshot:
    """
    Inputs, outputs and fingerprint stored together.

    An allocation that cannot be reproduced cannot be reviewed, and one that
    cannot be reviewed cannot be trusted with money.
    """

    fingerprint: str
    created_utc: str
    inputs: dict
    target: dict

    @classmethod
    def build(cls, inputs: AllocationInputs, target: TargetPortfolio) -> "AllocationSnapshot":
        return cls(
            fingerprint=inputs.fingerprint(),
            created_utc=datetime.now(timezone.utc).isoformat(),
            inputs=_roundtrip(asdict(inputs)),
            target=target.as_dict(),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)
