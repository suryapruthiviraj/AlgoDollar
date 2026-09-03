from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user
from app.database.models import Signal, StrategyPerformance, User
from app.database.session import get_async_session

router = APIRouter()
logger = structlog.get_logger(__name__)

STRATEGY_NAMES = ("longterm", "swing", "intraday")


# ── Schemas ────────────────────────────────────────────────────────────────────

class StrategyHealth(BaseModel):
    name: str
    status: str  # healthy / reduced / paused / disabled
    last_updated: Optional[date]
    sharpe: Optional[float]
    sortino: Optional[float]
    win_rate: Optional[float]
    num_trades: int
    net_return_30d: Optional[float]
    max_drawdown: Optional[float]


class StrategyDetail(BaseModel):
    name: str
    status: str
    description: str
    sharpe: Optional[float]
    sortino: Optional[float]
    win_rate: Optional[float]
    num_trades: int
    gross_return: Optional[float]
    net_return: Optional[float]
    max_drawdown: Optional[float]
    last_updated: Optional[date]


class StrategyStatusUpdate(BaseModel):
    status: str


class SignalOut(BaseModel):
    id: int
    symbol: str
    direction: str
    score: float
    expected_return: Optional[float]
    probability: Optional[float]
    model_name: str
    created_at: datetime
    expires_at: Optional[datetime]
    acted_upon: bool


class PerformancePoint(BaseModel):
    date: str
    net_return: Optional[float]
    sharpe: Optional[float]
    num_trades: int


STRATEGY_DESCRIPTIONS = {
    "longterm": "Monthly rebalanced long-term momentum and value portfolio with 2% per-trade risk.",
    "swing": "Multi-day swing trades lasting 3–15 days, driven by technical and fundamental signals.",
    "intraday": "Same-day scalping and momentum strategies, capital limited to 10% of portfolio.",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _get_latest_perf(
    session: AsyncSession, strategy: str
) -> Optional[StrategyPerformance]:
    result = await session.execute(
        select(StrategyPerformance)
        .where(StrategyPerformance.strategy == strategy)
        .order_by(StrategyPerformance.date.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[StrategyHealth])
async def list_strategies(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[StrategyHealth]:
    items: list[StrategyHealth] = []
    for name in STRATEGY_NAMES:
        perf = await _get_latest_perf(session, name)
        items.append(
            StrategyHealth(
                name=name,
                status=perf.status if perf else "healthy",
                last_updated=perf.date if perf else None,
                sharpe=perf.sharpe if perf else None,
                sortino=perf.sortino if perf else None,
                win_rate=perf.win_rate if perf else None,
                num_trades=perf.num_trades if perf else 0,
                net_return_30d=perf.net_return if perf else None,
                max_drawdown=perf.max_drawdown if perf else None,
            )
        )
    return items


@router.get("/{name}", response_model=StrategyDetail)
async def get_strategy(
    name: str = Path(..., description="Strategy name: longterm | swing | intraday"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> StrategyDetail:
    if name not in STRATEGY_NAMES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown strategy '{name}'. Valid: {', '.join(STRATEGY_NAMES)}",
        )
    perf = await _get_latest_perf(session, name)
    return StrategyDetail(
        name=name,
        status=perf.status if perf else "healthy",
        description=STRATEGY_DESCRIPTIONS.get(name, ""),
        sharpe=perf.sharpe if perf else None,
        sortino=perf.sortino if perf else None,
        win_rate=perf.win_rate if perf else None,
        num_trades=perf.num_trades if perf else 0,
        gross_return=perf.gross_return if perf else None,
        net_return=perf.net_return if perf else None,
        max_drawdown=perf.max_drawdown if perf else None,
        last_updated=perf.date if perf else None,
    )


@router.put("/{name}/status", response_model=StrategyDetail)
async def update_strategy_status(
    body: StrategyStatusUpdate,
    name: str = Path(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> StrategyDetail:
    valid_statuses = {"healthy", "reduced", "paused", "disabled"}
    if body.status not in valid_statuses:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status '{body.status}'. Must be one of {sorted(valid_statuses)}",
        )
    if name not in STRATEGY_NAMES:
        raise HTTPException(status_code=404, detail=f"Unknown strategy '{name}'")

    today = date.today()
    result = await session.execute(
        select(StrategyPerformance).where(
            StrategyPerformance.strategy == name,
            StrategyPerformance.date == today,
        )
    )
    perf = result.scalar_one_or_none()
    if perf:
        perf.status = body.status
    else:
        perf = StrategyPerformance(
            strategy=name,
            date=today,
            status=body.status,
        )
        session.add(perf)

    await session.flush()
    logger.info("strategy_status_updated", name=name, status=body.status, user=current_user.id)

    return StrategyDetail(
        name=name,
        status=perf.status,
        description=STRATEGY_DESCRIPTIONS.get(name, ""),
        sharpe=perf.sharpe,
        sortino=perf.sortino,
        win_rate=perf.win_rate,
        num_trades=perf.num_trades,
        gross_return=perf.gross_return,
        net_return=perf.net_return,
        max_drawdown=perf.max_drawdown,
        last_updated=perf.date,
    )


@router.get("/{name}/signals", response_model=list[SignalOut])
async def strategy_signals(
    name: str = Path(...),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[SignalOut]:
    if name not in STRATEGY_NAMES:
        raise HTTPException(status_code=404, detail=f"Unknown strategy '{name}'")

    result = await session.execute(
        select(Signal)
        .where(Signal.user_id == current_user.id, Signal.strategy == name)
        .order_by(Signal.created_at.desc())
        .limit(limit)
    )
    signals = result.scalars().all()
    return [
        SignalOut(
            id=s.id,
            symbol=s.symbol,
            direction=s.direction,
            score=s.score,
            expected_return=s.expected_return,
            probability=s.probability,
            model_name=s.model_name,
            created_at=s.created_at,
            expires_at=s.expires_at,
            acted_upon=s.acted_upon,
        )
        for s in signals
    ]


@router.get("/{name}/performance", response_model=list[PerformancePoint])
async def strategy_performance(
    name: str = Path(...),
    days: int = Query(90, ge=7, le=365),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[PerformancePoint]:
    if name not in STRATEGY_NAMES:
        raise HTTPException(status_code=404, detail=f"Unknown strategy '{name}'")

    from datetime import timedelta

    cutoff = date.today() - timedelta(days=days)
    result = await session.execute(
        select(StrategyPerformance)
        .where(
            StrategyPerformance.strategy == name,
            StrategyPerformance.date >= cutoff,
        )
        .order_by(StrategyPerformance.date.asc())
    )
    records = result.scalars().all()
    return [
        PerformancePoint(
            date=r.date.isoformat(),
            net_return=r.net_return,
            sharpe=r.sharpe,
            num_trades=r.num_trades,
        )
        for r in records
    ]
