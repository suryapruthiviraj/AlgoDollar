from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.security import get_current_user
from app.database.models import User

router = APIRouter()
logger = structlog.get_logger(__name__)


# ── Schemas ────────────────────────────────────────────────────────────────────

class IndexSnapshot(BaseModel):
    symbol: str
    last_price: float
    change: float
    change_pct: float
    high: float
    low: float


class MarketOverview(BaseModel):
    timestamp: str
    nifty: IndexSnapshot
    banknifty: IndexSnapshot
    vix: float
    advances: int
    declines: int
    unchanged: int
    breadth_ratio: float
    top_gainers: list[dict]
    top_losers: list[dict]


class RegimeInfo(BaseModel):
    regime: str  # bullish / bearish / neutral / sideways / high_volatility
    confidence: float
    vix: float
    breadth_ratio: float
    description: str
    timestamp: str


class SectorPerformance(BaseModel):
    sector: str
    change_pct: float
    top_stock: str
    top_stock_change_pct: float
    num_advances: int
    num_declines: int


class Opportunity(BaseModel):
    symbol: str
    sector: str
    strategy: str
    signal_score: float
    expected_return: float
    risk_level: str
    rationale: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fetch_kite_quote(symbols: list[str]) -> dict:
    """Return quote dict from Kite or empty dict if not configured."""
    from app.core.config import settings

    if not settings.kite_api_key or not settings.kite_access_token:
        return {}
    try:
        from kiteconnect import KiteConnect

        kite = KiteConnect(api_key=settings.kite_api_key)
        kite.set_access_token(settings.kite_access_token)
        return kite.quote(symbols)
    except Exception as exc:
        logger.warning("kite_quote_failed", error=str(exc))
        return {}


def _mock_index(symbol: str, last: float, chg: float) -> IndexSnapshot:
    return IndexSnapshot(
        symbol=symbol,
        last_price=last,
        change=chg,
        change_pct=round(chg / last * 100, 2),
        high=last + abs(chg) * 1.2,
        low=last - abs(chg) * 0.8,
    )


SECTOR_LIST = [
    "IT", "Banking", "FMCG", "Pharma", "Auto", "Metals", "Energy",
    "Infra", "Realty", "Media",
]


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/overview", response_model=MarketOverview)
async def market_overview(
    current_user: User = Depends(get_current_user),
) -> MarketOverview:
    quotes = _fetch_kite_quote(["NSE:NIFTY 50", "NSE:NIFTY BANK", "NSE:INDIA VIX"])

    if quotes:
        nf = quotes.get("NSE:NIFTY 50", {})
        bn = quotes.get("NSE:NIFTY BANK", {})
        vix_data = quotes.get("NSE:INDIA VIX", {})
        nifty = IndexSnapshot(
            symbol="NIFTY 50",
            last_price=nf.get("last_price", 0),
            change=nf.get("net_change", 0),
            change_pct=nf.get("change", 0),
            high=nf.get("ohlc", {}).get("high", 0),
            low=nf.get("ohlc", {}).get("low", 0),
        )
        banknifty = IndexSnapshot(
            symbol="BANKNIFTY",
            last_price=bn.get("last_price", 0),
            change=bn.get("net_change", 0),
            change_pct=bn.get("change", 0),
            high=bn.get("ohlc", {}).get("high", 0),
            low=bn.get("ohlc", {}).get("low", 0),
        )
        vix = vix_data.get("last_price", 15.0)
    else:
        # Fallback mock data when broker not connected
        nifty = _mock_index("NIFTY 50", 24500.0, 120.5)
        banknifty = _mock_index("BANKNIFTY", 52000.0, -230.0)
        vix = 14.5

    return MarketOverview(
        timestamp=datetime.now(timezone.utc).isoformat(),
        nifty=nifty,
        banknifty=banknifty,
        vix=vix,
        advances=1200,
        declines=800,
        unchanged=100,
        breadth_ratio=1.5,
        top_gainers=[
            {"symbol": "TATAMOTORS", "change_pct": 3.5},
            {"symbol": "HCLTECH", "change_pct": 2.8},
        ],
        top_losers=[
            {"symbol": "ONGC", "change_pct": -2.1},
            {"symbol": "COALINDIA", "change_pct": -1.9},
        ],
    )


@router.get("/regime", response_model=RegimeInfo)
async def market_regime(
    current_user: User = Depends(get_current_user),
) -> RegimeInfo:
    # In production: compute regime from VIX, NIFTY trend, breadth, etc.
    vix = 14.5
    breadth_ratio = 1.5

    if vix > 25:
        regime = "high_volatility"
        confidence = 0.85
        desc = "VIX above 25 signals high fear; reduce intraday exposure."
    elif breadth_ratio > 1.3:
        regime = "bullish"
        confidence = 0.70
        desc = "Broad market breadth favors long positions."
    elif breadth_ratio < 0.8:
        regime = "bearish"
        confidence = 0.65
        desc = "Declining breadth suggests defensive positioning."
    else:
        regime = "neutral"
        confidence = 0.55
        desc = "Mixed signals; maintain balanced allocation."

    return RegimeInfo(
        regime=regime,
        confidence=confidence,
        vix=vix,
        breadth_ratio=breadth_ratio,
        description=desc,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/sectors", response_model=list[SectorPerformance])
async def market_sectors(
    current_user: User = Depends(get_current_user),
) -> list[SectorPerformance]:
    import random

    random.seed(42)
    results: list[SectorPerformance] = []
    for sector in SECTOR_LIST:
        chg = round(random.uniform(-3.0, 4.0), 2)
        results.append(
            SectorPerformance(
                sector=sector,
                change_pct=chg,
                top_stock=f"{sector[:3].upper()}STOCK",
                top_stock_change_pct=round(chg + random.uniform(0.5, 2.0), 2),
                num_advances=random.randint(3, 12),
                num_declines=random.randint(1, 8),
            )
        )
    results.sort(key=lambda x: x.change_pct, reverse=True)
    return results


@router.get("/opportunities", response_model=list[Opportunity])
async def market_opportunities(
    strategy: Optional[str] = None,
    current_user: User = Depends(get_current_user),
) -> list[Opportunity]:
    # In production: query Signal table for recent high-score signals per strategy.
    mock: list[Opportunity] = [
        Opportunity(
            symbol="INFY",
            sector="IT",
            strategy="swing",
            signal_score=0.82,
            expected_return=0.045,
            risk_level="medium",
            rationale="Breakout above 200-DMA with increasing volume.",
        ),
        Opportunity(
            symbol="RELIANCE",
            sector="Energy",
            strategy="longterm",
            signal_score=0.75,
            expected_return=0.12,
            risk_level="low",
            rationale="Strong earnings growth and improving RoCE.",
        ),
        Opportunity(
            symbol="HDFCBANK",
            sector="Banking",
            strategy="intraday",
            signal_score=0.68,
            expected_return=0.008,
            risk_level="medium",
            rationale="Pre-open gap fill potential based on futures premium.",
        ),
    ]
    if strategy:
        mock = [o for o in mock if o.strategy == strategy]
    return mock
