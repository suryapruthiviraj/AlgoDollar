"""Risk limits dataclass and state tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


def normalize_loss(pnl: float) -> float:
    """Convert a signed P&L figure into a non-negative LOSS magnitude.

    ``RiskState`` loss fields are POSITIVE magnitudes (₹5,000 of loss is
    ``5000.0``, not ``-5000.0``). Upstream P&L is usually signed, so use this
    at the ingest boundary:

        >>> normalize_loss(-50_000.0)   # a ₹50k loss
        50000.0
        >>> normalize_loss(12_000.0)    # a profit is not a loss
        0.0
    """
    return abs(min(0.0, float(pnl)))


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    BREACH = "breach"


@dataclass
class LimitBreached:
    limit_name: str
    current_value: float
    limit_value: float
    severity: Severity
    message: str

    @property
    def breach_pct(self) -> float:
        if self.limit_value == 0:
            return 0.0
        return (self.current_value / self.limit_value) * 100.0


@dataclass
class RiskLimits:
    """All configurable risk limits — sourced from UserSettings.

    EVERY field declared here MUST have a corresponding check in
    :func:`check_all_limits`, registered in :data:`LIMIT_CHECKS`. A limit that
    is declared but never evaluated is worse than no limit at all: it reads as
    protection that does not exist. ``test_risk_numerics.py`` asserts that the
    two stay in lockstep, so adding a field here without adding a check (and a
    registry entry) fails the test suite.
    """

    # Loss limits (₹, positive magnitudes — see RiskState docstring)
    max_daily_loss: float = 10_000.0
    max_weekly_loss: float = 25_000.0
    max_monthly_loss: float = 75_000.0

    # Drawdown limits (fraction of portfolio, e.g. 0.10 = 10%)
    max_drawdown_pct: float = 0.10
    soft_drawdown_pct: float = 0.07    # warning threshold

    # Position limits
    max_open_positions: int = 20
    max_single_stock_pct: float = 0.10   # 10% of portfolio
    max_sector_pct: float = 0.30         # 30% of portfolio

    # Trade risk
    max_risk_per_trade_pct: float = 0.01   # 1% of capital per trade
    max_daily_risk: float = 50_000.0       # ₹ absolute

    # Leverage (gross exposure / equity)
    max_leverage: float = 2.0

    # Intraday capital allocation.
    #
    # AUTHORITATIVE VALUE — this is the single source of truth for the
    # intraday capital cap. It previously read 0.50 here while
    # app/portfolio/allocator.py used `max_intraday_pct = 0.10` and
    # app/core/config.py used `max_intraday_capital_pct = 0.10`: a 5x
    # discrepancy that let the allocator's cap silently disagree with the
    # limit that actually trips. Aligned to 0.10.
    #
    # NOTE FOR allocator.py OWNER: allocator.py must consume this value
    # (`RiskLimits().max_intraday_capital_pct`) instead of re-declaring its own
    # `max_intraday_pct` default. Do not re-introduce a second constant.
    max_intraday_capital_pct: float = 0.10


@dataclass
class RiskState:
    """Mutable risk state updated as trades occur throughout the day/week/month.

    LOSS SIGN CONVENTION (important)
    --------------------------------
    ``daily_loss`` / ``weekly_loss`` / ``monthly_loss`` are **non-negative
    magnitudes of loss in ₹**. A ₹50,000 loss is ``50_000.0``. Zero means "no
    loss" (a profitable day is still ``0.0``, not a negative number).

    Passing a signed P&L (``-50_000.0`` for a ₹50k loss) used to silently
    disable every loss limit: ``-50_000 >= 10_000`` is False, so a breaching
    day produced an EMPTY breach list. That is now a hard error. Normalise
    signed P&L at the ingest boundary with :func:`normalize_loss`, or build the
    state with :meth:`RiskState.from_pnl`, which does it for you.

    Exposure fields (all optional, default empty/zero — but required for the
    corresponding limit to be evaluated):
      position_values  : symbol -> signed ₹ market value, for max_single_stock_pct
      sector_values    : sector -> signed ₹ market value, for max_sector_pct
      trade_risk_amounts : trade id -> ₹ at risk (qty * |entry - stop|),
                           for max_risk_per_trade_pct
      gross_exposure   : sum(|position value|) in ₹, for max_leverage
    """

    daily_loss: float = 0.0
    weekly_loss: float = 0.0
    monthly_loss: float = 0.0
    current_drawdown: float = 0.0      # fraction, e.g. 0.05 = 5%
    positions_count: int = 0
    peak_portfolio_value: float = 0.0
    current_portfolio_value: float = 0.0
    daily_risk_used: float = 0.0
    intraday_capital_used: float = 0.0
    total_capital: float = 0.0

    # Exposure detail required to enforce concentration / leverage limits.
    position_values: dict[str, float] = field(default_factory=dict)
    sector_values: dict[str, float] = field(default_factory=dict)
    trade_risk_amounts: dict[str, float] = field(default_factory=dict)
    gross_exposure: float = 0.0

    _LOSS_FIELDS = ("daily_loss", "weekly_loss", "monthly_loss")

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ #
    #  Sign-convention enforcement                                        #
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        """Raise if any loss field is negative (i.e. signed P&L was passed).

        Called on construction *and* at the top of :func:`check_all_limits`, so
        a post-construction assignment (``state.daily_loss = -50_000``) cannot
        sneak past either.
        """
        for name in self._LOSS_FIELDS:
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(
                    f"RiskState.{name} must be a non-negative loss magnitude, "
                    f"got {value!r}. It looks like a signed P&L was passed: a "
                    f"negative value here silently disables the limit. Use "
                    f"normalize_loss(pnl) or RiskState.from_pnl(...)."
                )

    @classmethod
    def from_pnl(
        cls,
        daily_pnl: float = 0.0,
        weekly_pnl: float = 0.0,
        monthly_pnl: float = 0.0,
        **kwargs,
    ) -> "RiskState":
        """Build a RiskState from SIGNED P&L, normalising losses at ingest.

        >>> RiskState.from_pnl(daily_pnl=-50_000.0).daily_loss
        50000.0
        """
        return cls(
            daily_loss=normalize_loss(daily_pnl),
            weekly_loss=normalize_loss(weekly_pnl),
            monthly_loss=normalize_loss(monthly_pnl),
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    #  Derived quantities                                                 #
    # ------------------------------------------------------------------ #

    def equity_base(self) -> float:
        """₹ base used to express concentration / leverage limits as fractions.

        Prefers the live portfolio value, falls back to total capital.
        Returns 0.0 when neither is known (caller must handle).
        """
        if self.current_portfolio_value > 0:
            return self.current_portfolio_value
        if self.total_capital > 0:
            return self.total_capital
        return 0.0

    def update_drawdown(self) -> None:
        """Recompute drawdown fraction from peak vs current portfolio value."""
        if self.peak_portfolio_value > 0:
            self.current_drawdown = max(
                0.0,
                (self.peak_portfolio_value - self.current_portfolio_value)
                / self.peak_portfolio_value,
            )
        if self.current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.current_portfolio_value


# --------------------------------------------------------------------------- #
#  Limit -> check registry                                                      #
#                                                                               #
#  Maps every field of RiskLimits to the `limit_name` that check_all_limits     #
#  emits when that limit is breached. Keeping this explicit means a declared    #
#  limit with no enforcement is a test failure rather than a silent no-op.      #
# --------------------------------------------------------------------------- #
LIMIT_CHECKS: dict[str, str] = {
    "max_daily_loss": "daily_loss",
    "max_weekly_loss": "weekly_loss",
    "max_monthly_loss": "monthly_loss",
    "max_drawdown_pct": "drawdown",
    "soft_drawdown_pct": "drawdown_soft",
    "max_open_positions": "positions_count",
    "max_single_stock_pct": "single_stock_concentration",
    "max_sector_pct": "sector_concentration",
    "max_risk_per_trade_pct": "trade_risk",
    "max_daily_risk": "daily_risk",
    "max_leverage": "leverage",
    "max_intraday_capital_pct": "intraday_capital",
}


def check_all_limits(
    state: RiskState,
    limits: RiskLimits,
) -> list[LimitBreached]:
    """
    Evaluate all risk limits against current state.

    Every field declared on :class:`RiskLimits` is evaluated here; see
    :data:`LIMIT_CHECKS` for the field -> ``limit_name`` mapping.

    Returns a list of LimitBreached items, sorted by severity (worst first).
    An empty list means all limits are within bounds.

    Raises
    ------
    ValueError
        If a loss field is negative (signed P&L passed instead of a loss
        magnitude), or if exposure detail is supplied without an equity base
        to express it against.
    """
    # Fail loudly rather than silently passing every loss limit.
    state.validate()

    breaches: list[LimitBreached] = []

    # 1. Daily loss
    if state.daily_loss >= limits.max_daily_loss:
        breaches.append(LimitBreached(
            limit_name="daily_loss",
            current_value=state.daily_loss,
            limit_value=limits.max_daily_loss,
            severity=Severity.BREACH,
            message=f"Daily loss ₹{state.daily_loss:,.2f} exceeds limit ₹{limits.max_daily_loss:,.2f}.",
        ))
    elif state.daily_loss >= limits.max_daily_loss * 0.80:
        breaches.append(LimitBreached(
            limit_name="daily_loss",
            current_value=state.daily_loss,
            limit_value=limits.max_daily_loss,
            severity=Severity.WARNING,
            message=f"Daily loss ₹{state.daily_loss:,.2f} approaching limit.",
        ))

    # 2. Weekly loss
    if state.weekly_loss >= limits.max_weekly_loss:
        breaches.append(LimitBreached(
            limit_name="weekly_loss",
            current_value=state.weekly_loss,
            limit_value=limits.max_weekly_loss,
            severity=Severity.BREACH,
            message=f"Weekly loss ₹{state.weekly_loss:,.2f} exceeds limit.",
        ))

    # 3. Monthly loss
    if state.monthly_loss >= limits.max_monthly_loss:
        breaches.append(LimitBreached(
            limit_name="monthly_loss",
            current_value=state.monthly_loss,
            limit_value=limits.max_monthly_loss,
            severity=Severity.BREACH,
            message=f"Monthly loss ₹{state.monthly_loss:,.2f} exceeds limit.",
        ))

    # 4. Drawdown
    if state.current_drawdown >= limits.max_drawdown_pct:
        breaches.append(LimitBreached(
            limit_name="drawdown",
            current_value=state.current_drawdown,
            limit_value=limits.max_drawdown_pct,
            severity=Severity.CRITICAL,
            message=(
                f"Drawdown {state.current_drawdown:.1%} breaches max "
                f"{limits.max_drawdown_pct:.1%}."
            ),
        ))
    elif state.current_drawdown >= limits.soft_drawdown_pct:
        breaches.append(LimitBreached(
            limit_name="drawdown_soft",
            current_value=state.current_drawdown,
            limit_value=limits.soft_drawdown_pct,
            severity=Severity.WARNING,
            message=(
                f"Drawdown {state.current_drawdown:.1%} approaching soft limit "
                f"{limits.soft_drawdown_pct:.1%}."
            ),
        ))

    # 5. Open positions
    if state.positions_count >= limits.max_open_positions:
        breaches.append(LimitBreached(
            limit_name="positions_count",
            current_value=float(state.positions_count),
            limit_value=float(limits.max_open_positions),
            severity=Severity.WARNING,
            message=(
                f"Open positions {state.positions_count}/{limits.max_open_positions} at limit."
            ),
        ))

    # 6. Daily risk used
    if state.daily_risk_used >= limits.max_daily_risk:
        breaches.append(LimitBreached(
            limit_name="daily_risk",
            current_value=state.daily_risk_used,
            limit_value=limits.max_daily_risk,
            severity=Severity.BREACH,
            message=f"Daily risk ₹{state.daily_risk_used:,.0f} ≥ limit ₹{limits.max_daily_risk:,.0f}.",
        ))

    # 7. Intraday capital
    if state.total_capital > 0:
        intraday_pct = state.intraday_capital_used / state.total_capital
        if intraday_pct >= limits.max_intraday_capital_pct:
            breaches.append(LimitBreached(
                limit_name="intraday_capital",
                current_value=intraday_pct,
                limit_value=limits.max_intraday_capital_pct,
                severity=Severity.WARNING,
                message=f"Intraday capital {intraday_pct:.1%} ≥ limit {limits.max_intraday_capital_pct:.1%}.",
            ))

    # ------------------------------------------------------------------ #
    #  Exposure-based limits.                                             #
    #  These were previously DECLARED on RiskLimits but never evaluated.  #
    # ------------------------------------------------------------------ #
    equity = state.equity_base()
    needs_equity = bool(
        state.position_values
        or state.sector_values
        or state.trade_risk_amounts
        or state.gross_exposure
    )
    if needs_equity and equity <= 0:
        raise ValueError(
            "Exposure detail was supplied (position_values / sector_values / "
            "trade_risk_amounts / gross_exposure) but neither "
            "current_portfolio_value nor total_capital is positive, so "
            "concentration and leverage limits cannot be evaluated. Refusing "
            "to skip them silently."
        )

    # 8. Single-stock concentration (max_single_stock_pct)
    for symbol, value in state.position_values.items():
        pct = abs(value) / equity
        if pct >= limits.max_single_stock_pct:
            breaches.append(LimitBreached(
                limit_name="single_stock_concentration",
                current_value=pct,
                limit_value=limits.max_single_stock_pct,
                severity=Severity.BREACH,
                message=(
                    f"{symbol} is {pct:.1%} of portfolio, at/over the "
                    f"single-stock limit {limits.max_single_stock_pct:.1%}."
                ),
            ))

    # 9. Sector concentration (max_sector_pct)
    for sector, value in state.sector_values.items():
        pct = abs(value) / equity
        if pct >= limits.max_sector_pct:
            breaches.append(LimitBreached(
                limit_name="sector_concentration",
                current_value=pct,
                limit_value=limits.max_sector_pct,
                severity=Severity.BREACH,
                message=(
                    f"Sector '{sector}' is {pct:.1%} of portfolio, at/over the "
                    f"sector limit {limits.max_sector_pct:.1%}."
                ),
            ))

    # 10. Per-trade risk (max_risk_per_trade_pct)
    max_trade_risk = limits.max_risk_per_trade_pct * equity
    for trade_id, risk_amount in state.trade_risk_amounts.items():
        if abs(risk_amount) >= max_trade_risk:
            pct = abs(risk_amount) / equity
            breaches.append(LimitBreached(
                limit_name="trade_risk",
                current_value=abs(risk_amount),
                limit_value=max_trade_risk,
                severity=Severity.BREACH,
                message=(
                    f"Trade '{trade_id}' risks ₹{abs(risk_amount):,.0f} "
                    f"({pct:.2%}), at/over the per-trade limit "
                    f"₹{max_trade_risk:,.0f} "
                    f"({limits.max_risk_per_trade_pct:.2%})."
                ),
            ))

    # 11. Leverage (max_leverage) — gross exposure / equity
    if state.gross_exposure:
        leverage = abs(state.gross_exposure) / equity
        if leverage >= limits.max_leverage:
            breaches.append(LimitBreached(
                limit_name="leverage",
                current_value=leverage,
                limit_value=limits.max_leverage,
                severity=Severity.CRITICAL,
                message=(
                    f"Leverage {leverage:.2f}x at/over the max "
                    f"{limits.max_leverage:.2f}x "
                    f"(gross ₹{abs(state.gross_exposure):,.0f} on equity "
                    f"₹{equity:,.0f})."
                ),
            ))

    # Sort: BREACH first, then CRITICAL, WARNING, INFO
    _sev_order = {
        Severity.BREACH: 0,
        Severity.CRITICAL: 1,
        Severity.WARNING: 2,
        Severity.INFO: 3,
    }
    breaches.sort(key=lambda b: _sev_order.get(b.severity, 99))
    return breaches
