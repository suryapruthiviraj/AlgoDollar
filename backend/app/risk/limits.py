"""Risk limits dataclass and state tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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
    """All configurable risk limits — sourced from UserSettings."""

    # Loss limits (₹)
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

    # Leverage
    max_leverage: float = 2.0

    # Intraday capital allocation
    max_intraday_capital_pct: float = 0.50


@dataclass
class RiskState:
    """Mutable risk state updated as trades occur throughout the day/week/month."""

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


def check_all_limits(
    state: RiskState,
    limits: RiskLimits,
) -> list[LimitBreached]:
    """
    Evaluate all risk limits against current state.

    Returns a list of LimitBreached items, sorted by severity (worst first).
    An empty list means all limits are within bounds.
    """
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
            limit_name="drawdown",
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

    # Sort: BREACH first, then CRITICAL, WARNING, INFO
    _sev_order = {
        Severity.BREACH: 0,
        Severity.CRITICAL: 1,
        Severity.WARNING: 2,
        Severity.INFO: 3,
    }
    breaches.sort(key=lambda b: _sev_order.get(b.severity, 99))
    return breaches
