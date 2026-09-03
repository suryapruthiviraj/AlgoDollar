from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Path, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import KillSwitchActiveError, RiskLimitExceededError
from app.core.security import get_current_user
from app.database.models import CapitalAllocation, User, UserSettings
from app.database.session import get_async_session

router = APIRouter()
logger = structlog.get_logger(__name__)


# ── Schemas ────────────────────────────────────────────────────────────────────

class CalculateRequest(BaseModel):
    contribution: float


class AllocationRecommendation(BaseModel):
    allocation_id: Optional[int]
    month_year: str
    contribution_amount: float
    longterm_amount: float
    swing_amount: float
    intraday_amount: float
    cash_amount: float
    longterm_risk_pct: float
    swing_risk_pct: float
    intraday_risk_pct: float
    regime: str
    explanation: str
    gates_passed: bool
    gate_failures: list[str]


class ExecuteRequest(BaseModel):
    allocation_id: int


class OrderOutcome(BaseModel):
    """Per-order result. `submitted` is true only if a broker received it."""
    symbol: Optional[str] = None
    submitted: bool
    outcome: str
    broker_order_id: Optional[str] = None
    reason: Optional[str] = None


class ExecuteResponse(BaseModel):
    message: str
    executed: bool
    allocation_id: int
    warnings: list[str]
    # Reported from the execution boundary. `executed` is true only when at
    # least one order actually reached a broker — this route previously
    # returned executed=True while placing no orders whatsoever.
    trading_mode: Optional[str] = None
    trading_permitted: Optional[bool] = None
    blocked_reason: Optional[str] = None
    orders: list[OrderOutcome] = []


class AllocationHistoryItem(BaseModel):
    id: int
    month_year: str
    contribution_amount: float
    longterm_amount: float
    swing_amount: float
    intraday_amount: float
    cash_amount: float
    regime: Optional[str]
    created_at: datetime


class ExplainResponse(BaseModel):
    allocation_id: int
    explanation: str
    factors: list[dict]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _determine_regime() -> str:
    """Placeholder regime classifier.  In production, reads VIX and breadth."""
    return "neutral"


def _compute_allocation(
    contribution: float,
    regime: str,
    user_settings: Optional[UserSettings],
) -> tuple[float, float, float, float, float, float, float]:
    """Return (longterm, swing, intraday, cash, lt_risk_pct, sw_risk_pct, id_risk_pct)."""
    if regime == "bullish":
        lt_pct, sw_pct, id_pct = 0.65, 0.25, 0.10
    elif regime == "bearish":
        lt_pct, sw_pct, id_pct = 0.70, 0.20, 0.10
    else:  # neutral
        lt_pct, sw_pct, id_pct = 0.60, 0.30, 0.10

    # Disable strategies per settings
    if user_settings:
        if not user_settings.longterm_enabled:
            sw_pct += lt_pct * 0.5
            id_pct += lt_pct * 0.5
            lt_pct = 0.0
        if not user_settings.swing_enabled:
            lt_pct += sw_pct
            sw_pct = 0.0
        if not user_settings.intraday_enabled:
            lt_pct += id_pct
            id_pct = 0.0

    longterm = round(contribution * lt_pct, 2)
    swing = round(contribution * sw_pct, 2)
    intraday = round(contribution * id_pct, 2)
    cash = round(contribution - longterm - swing - intraday, 2)

    return longterm, swing, intraday, cash, 0.02, 0.01, 0.005


def _check_gates(
    contribution: float,
    user_settings: Optional[UserSettings],
) -> list[str]:
    failures: list[str] = []
    if contribution <= 0:
        failures.append("Contribution must be positive.")
    if user_settings and user_settings.kill_switch_active:
        failures.append("Kill switch is active; no new allocations permitted.")
    return failures


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/calculate", response_model=AllocationRecommendation)
async def calculate_allocation(
    body: CalculateRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> AllocationRecommendation:
    settings_result = await session.execute(
        select(UserSettings).where(UserSettings.user_id == current_user.id)
    )
    user_settings = settings_result.scalar_one_or_none()

    regime = _determine_regime()
    gate_failures = _check_gates(body.contribution, user_settings)
    gates_passed = len(gate_failures) == 0

    longterm, swing, intraday, cash, lt_r, sw_r, id_r = _compute_allocation(
        body.contribution, regime, user_settings
    )

    explanation = (
        f"Market regime detected as '{regime}'. "
        f"Allocating {longterm:.2f} to long-term, {swing:.2f} to swing, "
        f"{intraday:.2f} to intraday, and {cash:.2f} to cash reserve."
    )

    return AllocationRecommendation(
        allocation_id=None,
        month_year=date.today().strftime("%Y-%m"),
        contribution_amount=body.contribution,
        longterm_amount=longterm,
        swing_amount=swing,
        intraday_amount=intraday,
        cash_amount=cash,
        longterm_risk_pct=lt_r,
        swing_risk_pct=sw_r,
        intraday_risk_pct=id_r,
        regime=regime,
        explanation=explanation,
        gates_passed=gates_passed,
        gate_failures=gate_failures,
    )


@router.post("/execute", response_model=ExecuteResponse)
async def execute_allocation(
    body: ExecuteRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> ExecuteResponse:
    result = await session.execute(
        select(CapitalAllocation).where(
            CapitalAllocation.id == body.allocation_id,
            CapitalAllocation.user_id == current_user.id,
        )
    )
    alloc = result.scalar_one_or_none()
    if not alloc:
        raise HTTPException(status_code=404, detail="Allocation not found")

    settings_result = await session.execute(
        select(UserSettings).where(UserSettings.user_id == current_user.id)
    )
    user_settings = settings_result.scalar_one_or_none()

    warnings: list[str] = []

    if user_settings and user_settings.kill_switch_active:
        raise HTTPException(
            status_code=403,
            detail="Kill switch is active. Deactivate it before executing allocations.",
        )

    # ── Route through the single execution boundary ───────────────────────
    #
    # This endpoint previously returned executed=True after checking a kill
    # switch and logging — it placed no orders, consulted no risk engine, and
    # reached no broker, while telling the caller the allocation had been
    # executed. Reporting success for work that did not happen is worse than
    # reporting failure, so the response now reflects what actually occurred.
    service = getattr(request.app.state, "execution_service", None)
    stack = getattr(request.app.state, "execution_stack", None)

    if service is None:
        logger.error("allocation_execute_no_service", allocation_id=alloc.id)
        raise HTTPException(
            status_code=503,
            detail=(
                "Execution service is unavailable — startup did not complete. "
                "No orders can be placed."
            ),
        )

    trading_permitted = bool(getattr(stack, "trading_permitted", False))
    blocked_reason = getattr(stack, "startup_reason", None)

    if not trading_permitted:
        logger.error(
            "allocation_execute_blocked",
            allocation_id=alloc.id,
            reason=blocked_reason,
        )
        return ExecuteResponse(
            message=(
                "Allocation NOT executed: trading is blocked because startup "
                "reconciliation has not succeeded."
            ),
            executed=False,
            allocation_id=alloc.id,
            warnings=warnings,
            trading_mode=service.mode.value,
            trading_permitted=False,
            blocked_reason=blocked_reason,
            orders=[],
        )

    # Signals come from the strategy layer. No strategy is currently validated
    # or wired to produce production signals, so there is nothing to submit.
    # That is reported honestly rather than dressed up as a successful run.
    signals = getattr(alloc, "pending_signals", None) or []

    outcomes: list[OrderOutcome] = []
    for sig, size in signals:
        result = await service.submit_signal(
            sig, size, portfolio_allocation={"allocation_id": alloc.id},
        )
        outcomes.append(OrderOutcome(
            symbol=result.audit.symbol,
            submitted=result.submitted,
            outcome=result.outcome.value,
            broker_order_id=result.broker_order_id,
            reason=result.reason,
        ))

    any_submitted = any(o.submitted for o in outcomes)

    if not signals:
        warnings.append(
            "No signals were available to execute. No validated strategy is "
            "currently wired to produce production signals."
        )

    if service.mode.value == "paper":
        warnings.append(
            "Paper mode: orders route to the paper broker. No real capital moved."
        )

    logger.info(
        "allocation_execute_completed",
        user_id=current_user.id,
        allocation_id=alloc.id,
        amount=float(alloc.contribution_amount),
        orders=len(outcomes),
        submitted=any_submitted,
    )

    return ExecuteResponse(
        message=(
            f"{sum(1 for o in outcomes if o.submitted)}/{len(outcomes)} orders "
            f"submitted." if outcomes else
            "No orders were placed — nothing to execute."
        ),
        executed=any_submitted,
        allocation_id=alloc.id,
        warnings=warnings,
        trading_mode=service.mode.value,
        trading_permitted=True,
        blocked_reason=None,
        orders=outcomes,
    )


@router.get("/history", response_model=list[AllocationHistoryItem])
async def allocation_history(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[AllocationHistoryItem]:
    result = await session.execute(
        select(CapitalAllocation)
        .where(CapitalAllocation.user_id == current_user.id)
        .order_by(CapitalAllocation.created_at.desc())
        .limit(24)
    )
    allocs = result.scalars().all()
    return [
        AllocationHistoryItem(
            id=a.id,
            month_year=a.month_year,
            contribution_amount=float(a.contribution_amount),
            longterm_amount=float(a.longterm_amount),
            swing_amount=float(a.swing_amount),
            intraday_amount=float(a.intraday_amount),
            cash_amount=float(a.cash_amount),
            regime=a.regime,
            created_at=a.created_at,
        )
        for a in allocs
    ]


@router.get("/explain/{allocation_id}", response_model=ExplainResponse)
async def explain_allocation(
    allocation_id: int = Path(..., description="Allocation ID to explain"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> ExplainResponse:
    result = await session.execute(
        select(CapitalAllocation).where(
            CapitalAllocation.id == allocation_id,
            CapitalAllocation.user_id == current_user.id,
        )
    )
    alloc = result.scalar_one_or_none()
    if not alloc:
        raise HTTPException(status_code=404, detail="Allocation not found")

    explanation = alloc.explanation or "No explanation recorded for this allocation."

    factors = [
        {
            "factor": "Market Regime",
            "value": alloc.regime or "unknown",
            "weight": 0.4,
            "description": "Detected market regime drives strategy weight adjustments.",
        },
        {
            "factor": "Contribution Amount",
            "value": float(alloc.contribution_amount),
            "weight": 1.0,
            "description": "Total capital being deployed this month.",
        },
        {
            "factor": "Long-Term Allocation",
            "value": float(alloc.longterm_amount),
            "weight": alloc.longterm_risk_pct,
            "description": "Capital allocated to long-term momentum/value strategies.",
        },
        {
            "factor": "Swing Allocation",
            "value": float(alloc.swing_amount),
            "weight": alloc.swing_risk_pct,
            "description": "Capital allocated to multi-day swing strategies.",
        },
        {
            "factor": "Intraday Allocation",
            "value": float(alloc.intraday_amount),
            "weight": alloc.intraday_risk_pct,
            "description": "Capital allocated to intraday scalping/momentum.",
        },
    ]

    return ExplainResponse(
        allocation_id=allocation_id,
        explanation=explanation,
        factors=factors,
    )
