from __future__ import annotations

from datetime import datetime
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user
from app.database.models import Order, Trade, User
from app.database.session import get_async_session

router = APIRouter()
logger = structlog.get_logger(__name__)


# ── Schemas ────────────────────────────────────────────────────────────────────

class TradeOut(BaseModel):
    id: int
    symbol: str
    exchange: str
    transaction_type: str
    quantity: int
    price: float
    value: float
    brokerage: float
    stt: float
    exchange_charges: float
    gst: float
    stamp_duty: float
    sebi_charges: float
    total_costs: float
    net_value: float
    strategy: Optional[str]
    created_at: datetime
    order_id: Optional[int]


class TradeSummary(BaseModel):
    total_trades: int
    total_buys: int
    total_sells: int
    total_brokerage: float
    total_stt: float
    total_costs: float
    total_buy_value: float
    total_sell_value: float
    realized_pnl: float
    win_rate: Optional[float]


class PaginatedTrades(BaseModel):
    items: list[TradeOut]
    total: int
    page: int
    page_size: int
    has_next: bool


# ── Helpers ────────────────────────────────────────────────────────────────────

def _trade_to_out(t: Trade) -> TradeOut:
    return TradeOut(
        id=t.id,
        symbol=t.symbol,
        exchange=t.exchange,
        transaction_type=t.transaction_type,
        quantity=t.quantity,
        price=float(t.price),
        value=float(t.value),
        brokerage=float(t.brokerage),
        stt=float(t.stt),
        exchange_charges=float(t.exchange_charges),
        gst=float(t.gst),
        stamp_duty=float(t.stamp_duty),
        sebi_charges=float(t.sebi_charges),
        total_costs=float(t.total_costs),
        net_value=float(t.net_value),
        strategy=t.strategy,
        created_at=t.created_at,
        order_id=t.order_id,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("", response_model=PaginatedTrades)
async def list_trades(
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    symbol: Optional[str] = Query(None),
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> PaginatedTrades:
    stmt = select(Trade).where(Trade.user_id == current_user.id)

    if strategy:
        stmt = stmt.where(Trade.strategy == strategy)
    if symbol:
        stmt = stmt.where(Trade.symbol == symbol.upper())
    if start:
        stmt = stmt.where(Trade.created_at >= start)
    if end:
        stmt = stmt.where(Trade.created_at <= end)

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await session.execute(count_stmt)
    total = total_result.scalar_one()

    offset = (page - 1) * page_size
    result = await session.execute(
        stmt.order_by(Trade.created_at.desc()).offset(offset).limit(page_size)
    )
    trades = result.scalars().all()

    return PaginatedTrades(
        items=[_trade_to_out(t) for t in trades],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(offset + page_size) < total,
    )


@router.get("/summary", response_model=TradeSummary)
async def trade_summary(
    strategy: Optional[str] = Query(None),
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> TradeSummary:
    stmt = select(Trade).where(Trade.user_id == current_user.id)
    if strategy:
        stmt = stmt.where(Trade.strategy == strategy)
    if start:
        stmt = stmt.where(Trade.created_at >= start)
    if end:
        stmt = stmt.where(Trade.created_at <= end)

    result = await session.execute(stmt)
    trades = result.scalars().all()

    buys = [t for t in trades if t.transaction_type == "BUY"]
    sells = [t for t in trades if t.transaction_type == "SELL"]

    total_buy_value = sum(float(t.value) for t in buys)
    total_sell_value = sum(float(t.value) for t in sells)
    total_costs = sum(float(t.total_costs) for t in trades)
    realized_pnl = total_sell_value - total_buy_value - total_costs

    return TradeSummary(
        total_trades=len(trades),
        total_buys=len(buys),
        total_sells=len(sells),
        total_brokerage=sum(float(t.brokerage) for t in trades),
        total_stt=sum(float(t.stt) for t in trades),
        total_costs=total_costs,
        total_buy_value=total_buy_value,
        total_sell_value=total_sell_value,
        realized_pnl=realized_pnl,
        win_rate=None,  # requires matching buy/sell pairs per symbol
    )


@router.get("/{trade_id}", response_model=TradeOut)
async def get_trade(
    trade_id: int = Path(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> TradeOut:
    result = await session.execute(
        select(Trade).where(
            Trade.id == trade_id,
            Trade.user_id == current_user.id,
        )
    )
    trade = result.scalar_one_or_none()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    return _trade_to_out(trade)
