from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user
from app.database.models import CapitalAllocation, Position, Trade, User
from app.database.session import get_async_session

router = APIRouter()
logger = structlog.get_logger(__name__)


# ── Schemas ────────────────────────────────────────────────────────────────────

class PortfolioOverview(BaseModel):
    total_capital: float
    invested: float
    cash: float
    current_value: float
    unrealized_pnl: float
    realized_pnl: float
    today_pnl: float
    monthly_pnl: float
    total_return_pct: float
    drawdown_pct: float
    volatility: float
    sharpe: float
    sortino: float


class PositionOut(BaseModel):
    id: int
    symbol: str
    exchange: str
    quantity: int
    average_price: float
    current_price: Optional[float]
    strategy: str
    entry_date: datetime
    stop_loss: Optional[float]
    target_price: Optional[float]
    signal_strength: Optional[float]
    sector: Optional[str]
    is_open: bool
    unrealized_pnl: float
    unrealized_pnl_pct: float


class AllocationBreakdown(BaseModel):
    month_year: str
    contribution_amount: float
    longterm_amount: float
    swing_amount: float
    intraday_amount: float
    cash_amount: float
    longterm_risk_pct: float
    swing_risk_pct: float
    intraday_risk_pct: float
    regime: Optional[str]


class EquityCurvePoint(BaseModel):
    date: str
    portfolio_value: float
    benchmark_value: Optional[float]
    drawdown: float


class ContributionRequest(BaseModel):
    amount: float


class ContributionResponse(BaseModel):
    message: str
    allocation_id: Optional[int]
    allocation: Optional[AllocationBreakdown]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _position_to_out(p: Position) -> PositionOut:
    current = float(p.current_price or p.average_price)
    avg = float(p.average_price)
    qty = p.quantity
    unrealized = (current - avg) * qty
    unrealized_pct = ((current - avg) / avg) * 100 if avg else 0.0
    return PositionOut(
        id=p.id,
        symbol=p.symbol,
        exchange=p.exchange,
        quantity=qty,
        average_price=avg,
        current_price=current,
        strategy=p.strategy,
        entry_date=p.entry_date,
        stop_loss=float(p.stop_loss) if p.stop_loss else None,
        target_price=float(p.target_price) if p.target_price else None,
        signal_strength=p.signal_strength,
        sector=p.sector,
        is_open=p.is_open,
        unrealized_pnl=unrealized,
        unrealized_pnl_pct=unrealized_pct,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/overview", response_model=PortfolioOverview)
async def portfolio_overview(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> PortfolioOverview:
    # Open positions
    positions_result = await session.execute(
        select(Position).where(Position.user_id == current_user.id, Position.is_open == True)
    )
    open_positions = positions_result.scalars().all()

    invested = sum(float(p.average_price) * p.quantity for p in open_positions)
    current_value = sum(
        float(p.current_price or p.average_price) * p.quantity for p in open_positions
    )
    unrealized_pnl = current_value - invested

    # Realized PnL from trades (sum of net_values for sells minus buys)
    trades_result = await session.execute(
        select(Trade).where(Trade.user_id == current_user.id)
    )
    trades = trades_result.scalars().all()
    realized_pnl = sum(
        float(t.net_value) if t.transaction_type == "SELL" else -float(t.net_value)
        for t in trades
    )

    # Capital from latest allocation
    alloc_result = await session.execute(
        select(CapitalAllocation)
        .where(CapitalAllocation.user_id == current_user.id)
        .order_by(CapitalAllocation.created_at.desc())
        .limit(1)
    )
    latest_alloc = alloc_result.scalar_one_or_none()
    total_capital = float(latest_alloc.contribution_amount) if latest_alloc else 0.0
    cash = max(total_capital - invested, 0.0)

    total_return_pct = (realized_pnl + unrealized_pnl) / total_capital * 100 if total_capital else 0.0

    return PortfolioOverview(
        total_capital=total_capital,
        invested=invested,
        cash=cash,
        current_value=current_value,
        unrealized_pnl=unrealized_pnl,
        realized_pnl=realized_pnl,
        today_pnl=unrealized_pnl,  # simplified; full impl would use EOD prices
        monthly_pnl=realized_pnl + unrealized_pnl,
        total_return_pct=total_return_pct,
        drawdown_pct=0.0,   # full impl: compute from equity curve
        volatility=0.0,
        sharpe=0.0,
        sortino=0.0,
    )


@router.get("/positions", response_model=list[PositionOut])
async def portfolio_positions(
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    is_open: bool = Query(True),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[PositionOut]:
    stmt = select(Position).where(
        Position.user_id == current_user.id,
        Position.is_open == is_open,
    )
    if strategy:
        stmt = stmt.where(Position.strategy == strategy)

    result = await session.execute(stmt.order_by(Position.entry_date.desc()))
    positions = result.scalars().all()
    return [_position_to_out(p) for p in positions]


@router.get("/allocation", response_model=Optional[AllocationBreakdown])
async def portfolio_allocation(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> Optional[AllocationBreakdown]:
    result = await session.execute(
        select(CapitalAllocation)
        .where(CapitalAllocation.user_id == current_user.id)
        .order_by(CapitalAllocation.created_at.desc())
        .limit(1)
    )
    alloc = result.scalar_one_or_none()
    if not alloc:
        return None
    return AllocationBreakdown(
        month_year=alloc.month_year,
        contribution_amount=float(alloc.contribution_amount),
        longterm_amount=float(alloc.longterm_amount),
        swing_amount=float(alloc.swing_amount),
        intraday_amount=float(alloc.intraday_amount),
        cash_amount=float(alloc.cash_amount),
        longterm_risk_pct=alloc.longterm_risk_pct,
        swing_risk_pct=alloc.swing_risk_pct,
        intraday_risk_pct=alloc.intraday_risk_pct,
        regime=alloc.regime,
    )


@router.get("/performance", response_model=list[EquityCurvePoint])
async def portfolio_performance(
    period: str = Query("1M", description="1W / 1M / 3M / 6M / 1Y / ALL"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[EquityCurvePoint]:
    # In a full implementation this would pull trade history, compute daily
    # portfolio value, and compute drawdown series.  Here we return an empty
    # series so the API is callable without crashing.
    trades_result = await session.execute(
        select(Trade)
        .where(Trade.user_id == current_user.id)
        .order_by(Trade.created_at.asc())
    )
    trades = trades_result.scalars().all()

    if not trades:
        return []

    # Build a simplified equity curve bucketed by date
    from collections import defaultdict

    daily: dict[str, float] = defaultdict(float)
    running = 0.0
    for t in trades:
        day = t.created_at.date().isoformat()
        delta = float(t.net_value) if t.transaction_type == "SELL" else -float(t.net_value)
        daily[day] = daily[day] + delta

    points: list[EquityCurvePoint] = []
    peak = 0.0
    cumulative = 0.0
    for day in sorted(daily):
        cumulative += daily[day]
        peak = max(peak, cumulative)
        drawdown = (peak - cumulative) / peak * 100 if peak else 0.0
        points.append(
            EquityCurvePoint(
                date=day,
                portfolio_value=cumulative,
                benchmark_value=None,
                drawdown=drawdown,
            )
        )
    return points


@router.post("/contribution", response_model=ContributionResponse)
async def portfolio_contribution(
    body: ContributionRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> ContributionResponse:
    if body.amount <= 0:
        raise HTTPException(status_code=422, detail="Contribution amount must be positive")

    from datetime import date

    month_year = date.today().strftime("%Y-%m")

    # Simple allocation: 60% long-term, 30% swing, 10% intraday
    longterm = round(body.amount * 0.60, 2)
    swing = round(body.amount * 0.30, 2)
    intraday = round(body.amount * 0.10, 2)
    cash = round(body.amount - longterm - swing - intraday, 2)

    alloc = CapitalAllocation(
        user_id=current_user.id,
        month_year=month_year,
        contribution_amount=body.amount,
        longterm_amount=longterm,
        swing_amount=swing,
        intraday_amount=intraday,
        cash_amount=cash,
        longterm_risk_pct=0.02,
        swing_risk_pct=0.01,
        intraday_risk_pct=0.005,
        regime="neutral",
        explanation="Default 60/30/10 split; override with AI allocation.",
    )
    session.add(alloc)
    await session.flush()

    breakdown = AllocationBreakdown(
        month_year=month_year,
        contribution_amount=body.amount,
        longterm_amount=longterm,
        swing_amount=swing,
        intraday_amount=intraday,
        cash_amount=cash,
        longterm_risk_pct=0.02,
        swing_risk_pct=0.01,
        intraday_risk_pct=0.005,
        regime="neutral",
    )
    return ContributionResponse(
        message="Allocation created successfully",
        allocation_id=alloc.id,
        allocation=breakdown,
    )
