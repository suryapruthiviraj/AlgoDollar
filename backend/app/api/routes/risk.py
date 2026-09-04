"""
Risk state and limits, read from what the system actually holds.

Every number here is derived from the live execution stack and the database.
Where a value cannot be computed — no positions, no broker, no correlation
history — the field is null and `unavailable` names it, rather than a zero
standing in for an unknown. A risk page showing 0% drawdown because it could
not read the account is worse than one showing nothing.
"""

from __future__ import annotations

from typing import Optional

import structlog
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import get_current_user
from app.database.models import AccountCash, Position, User
from app.database.session import get_async_session

logger = structlog.get_logger(__name__)
router = APIRouter()


class LimitStatus(BaseModel):
    """One limit, its configured value, and how close the book is to it."""

    name: str
    label: str
    limit: float
    current: Optional[float] = None
    utilisation: Optional[float] = None
    breached: bool = False
    #: Null current means "not measurable right now", which is NOT the same as
    #: zero utilisation and must not be rendered as a comfortable green bar.
    measurable: bool = True
    detail: str = ""


class RiskStateResponse(BaseModel):
    trading_mode: str
    trading_permitted: bool
    kill_switch_active: bool
    kill_switch_reason: Optional[str] = None
    reconciliation_state: Optional[str] = None

    portfolio_value: Optional[float] = None
    cash: Optional[float] = None
    invested: Optional[float] = None
    open_positions: int = 0

    current_drawdown_pct: Optional[float] = None
    realized_pnl: Optional[float] = None

    largest_position_pct: Optional[float] = None
    largest_position_symbol: Optional[str] = None
    sector_exposures: dict[str, float] = {}

    limits: list[LimitStatus] = []
    active_breaches: list[str] = []
    unavailable: list[str] = []


class LimitsResponse(BaseModel):
    limits: list[LimitStatus] = []
    source: str = "app.core.config.settings"


def _limit_rows(
    *,
    portfolio_value: Optional[float],
    largest_pct: Optional[float],
    sector_exposures: dict[str, float],
    drawdown: Optional[float],
    n_positions: int,
    cash_pct: Optional[float],
) -> list[LimitStatus]:
    """Build the limit table, marking anything unmeasurable as such."""
    rows: list[LimitStatus] = []

    def add(
        name: str, label: str, limit: float, current: Optional[float],
        detail: str = "", higher_is_worse: bool = True,
    ) -> None:
        util = None
        breached = False
        # Bound explicitly rather than relying on `measurable`: a truthiness
        # flag does not narrow Optional[float] for the checker, and the whole
        # point of this block is that an unmeasurable limit stays unmeasured.
        measurable = current is not None
        if current is not None and limit:
            util = float(current) / float(limit)
            breached = (current >= limit) if higher_is_worse else (current <= limit)
        rows.append(LimitStatus(
            name=name, label=label, limit=limit, current=current,
            utilisation=util, breached=breached, measurable=measurable,
            detail=detail or (
                "" if measurable else "not measurable from current state"
            ),
        ))

    add("max_single_stock_pct", "Max single position",
        settings.max_single_stock_pct, largest_pct,
        "Largest holding as a fraction of portfolio value.")
    add("max_sector_pct", "Max sector exposure", settings.max_sector_pct,
        max(sector_exposures.values()) if sector_exposures else None,
        "Largest sector concentration.")
    add("max_portfolio_drawdown_pct", "Max drawdown",
        settings.max_portfolio_drawdown_pct, drawdown,
        "Peak-to-trough decline of the account.")
    add("max_daily_loss_pct", "Max daily loss", settings.max_daily_loss_pct, None,
        "Requires an intraday P&L series, which is not yet persisted.")
    add("max_positions", "Max open positions", float(settings.max_positions),
        float(n_positions), "Count of open positions.")
    add("max_intraday_capital_pct", "Max intraday capital",
        settings.max_intraday_capital_pct, None,
        "Requires per-sleeve capital attribution on open positions.")
    if cash_pct is not None:
        add("min_cash_pct", "Minimum cash", 0.05, cash_pct,
            "Uninvested cash as a fraction of the portfolio.",
            higher_is_worse=False)
    return rows


@router.get("/state", response_model=RiskStateResponse)
async def get_risk_state(
    request: Request,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> RiskStateResponse:
    """
    The live risk picture: exposures, limits and what is currently blocking.

    Fields that cannot be computed are null and listed in `unavailable`. That
    distinction matters more here than anywhere else in the API — a zero
    drawdown reported because the account could not be read looks like safety.
    """
    stack = getattr(request.app.state, "execution_stack", None)
    unavailable: list[str] = []
    breaches: list[str] = []

    trading_mode = str(settings.trading_mode)
    trading_permitted = bool(getattr(stack, "trading_permitted", False)) if stack else False
    if stack is None:
        unavailable.append(
            "execution stack is not running; trading state cannot be read"
        )

    kill_active = False
    kill_reason: Optional[str] = None
    if stack is not None:
        try:
            kill_active, kill_reason = stack.service.kill_switch.is_active()
        except Exception as exc:  # noqa: BLE001
            # A switch that cannot be read counts as ENGAGED everywhere else in
            # this system; the UI must show the same thing.
            kill_active, kill_reason = True, f"kill switch unreadable: {exc}"
    else:
        kill_active = True
        kill_reason = "no execution stack; treated as engaged"

    if kill_active:
        breaches.append(kill_reason or "kill switch engaged")
    if not trading_permitted:
        breaches.append("trading not permitted (reconciliation has not succeeded)")

    recon_state = None
    if stack is not None:
        recon_state = str(getattr(stack.recovery, "state", "") or "") or None

    # ---- book, from the database -------------------------------------- #
    rows = (await session.execute(
        select(Position).where(
            Position.user_id == current_user.id, Position.is_open.is_(True)
        )
    )).scalars().all()

    invested = 0.0
    sector_value: dict[str, float] = {}
    largest_val, largest_sym = 0.0, None
    for p in rows:
        px = float(p.current_price or p.average_price or 0.0)
        val = float(p.quantity) * px
        invested += val
        sec = p.sector or "UNKNOWN"
        sector_value[sec] = sector_value.get(sec, 0.0) + val
        if val > largest_val:
            largest_val, largest_sym = val, p.symbol

    cash_row = (await session.execute(
        select(AccountCash).where(
            AccountCash.user_id == current_user.id,
            AccountCash.trading_mode == trading_mode,
        )
    )).scalar_one_or_none()

    cash: Optional[float] = float(cash_row.cash) if cash_row else None
    realized = float(cash_row.realized_pnl) if cash_row else None
    if cash is None:
        unavailable.append(
            f"no AccountCash row for mode '{trading_mode}'; cash and portfolio "
            f"value are unknown"
        )

    portfolio_value = (cash + invested) if cash is not None else None
    largest_pct = (
        largest_val / portfolio_value
        if portfolio_value and portfolio_value > 0 else None
    )
    sector_pcts = {
        k: v / portfolio_value for k, v in sector_value.items()
    } if portfolio_value and portfolio_value > 0 else {}
    cash_pct = (
        cash / portfolio_value if portfolio_value and portfolio_value > 0 and cash is not None
        else None
    )

    # Drawdown needs an equity-curve high-water mark, which is not persisted.
    # Reported as unavailable rather than as 0%.
    unavailable.append(
        "current drawdown requires a persisted equity high-water mark, which "
        "this system does not yet store"
    )

    limits = _limit_rows(
        portfolio_value=portfolio_value, largest_pct=largest_pct,
        sector_exposures=sector_pcts, drawdown=None,
        n_positions=len(rows), cash_pct=cash_pct,
    )
    breaches.extend(f"{r.label} breached" for r in limits if r.breached)

    return RiskStateResponse(
        trading_mode=trading_mode,
        trading_permitted=trading_permitted,
        kill_switch_active=kill_active,
        kill_switch_reason=kill_reason,
        reconciliation_state=recon_state,
        portfolio_value=portfolio_value,
        cash=cash,
        invested=invested,
        open_positions=len(rows),
        current_drawdown_pct=None,
        realized_pnl=realized,
        largest_position_pct=largest_pct,
        largest_position_symbol=largest_sym,
        sector_exposures={k: round(v, 6) for k, v in sector_pcts.items()},
        limits=limits,
        active_breaches=breaches,
        unavailable=unavailable,
    )


@router.get("/limits", response_model=LimitsResponse)
async def get_limits() -> LimitsResponse:
    """The configured limits, with no book attached."""
    return LimitsResponse(
        limits=_limit_rows(
            portfolio_value=None, largest_pct=None, sector_exposures={},
            drawdown=None, n_positions=0, cash_pct=None,
        )
    )
