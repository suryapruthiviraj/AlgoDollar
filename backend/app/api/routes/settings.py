from __future__ import annotations

from datetime import datetime
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user
from app.database.models import AuditLog, RiskEvent, User, UserSettings
from app.database.session import get_async_session

router = APIRouter()
logger = structlog.get_logger(__name__)


# ── Schemas ────────────────────────────────────────────────────────────────────

class UserSettingsOut(BaseModel):
    user_id: int
    monthly_capital: float
    risk_tolerance: str
    max_drawdown_pct: float
    intraday_enabled: bool
    swing_enabled: bool
    longterm_enabled: bool
    max_positions: int
    max_sector_exposure_pct: float
    max_single_stock_pct: float
    cash_reserve_pct: float
    auto_execution_enabled: bool
    paper_trading_mode: bool
    kill_switch_active: bool
    updated_at: datetime


class UserSettingsUpdate(BaseModel):
    monthly_capital: Optional[float] = Field(None, ge=0)
    risk_tolerance: Optional[str] = None
    max_drawdown_pct: Optional[float] = Field(None, ge=0.01, le=0.50)
    intraday_enabled: Optional[bool] = None
    swing_enabled: Optional[bool] = None
    longterm_enabled: Optional[bool] = None
    max_positions: Optional[int] = Field(None, ge=1, le=100)
    max_sector_exposure_pct: Optional[float] = Field(None, ge=0.05, le=1.0)
    max_single_stock_pct: Optional[float] = Field(None, ge=0.01, le=0.5)
    cash_reserve_pct: Optional[float] = Field(None, ge=0.0, le=0.5)
    auto_execution_enabled: Optional[bool] = None
    paper_trading_mode: Optional[bool] = None

    @field_validator("risk_tolerance")
    @classmethod
    def validate_risk_tolerance(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("low", "medium", "high"):
            raise ValueError("risk_tolerance must be 'low', 'medium', or 'high'")
        return v


class KillSwitchRequest(BaseModel):
    activate: bool
    reason: Optional[str] = None


class KillSwitchResponse(BaseModel):
    kill_switch_active: bool
    message: str


class CostModel(BaseModel):
    brokerage_intraday_pct: float
    brokerage_delivery_pct: float
    brokerage_max_per_order: float
    stt_delivery_pct: float
    stt_intraday_sell_pct: float
    exchange_charges_pct: float
    sebi_turnover_fee_pct: float
    gst_pct: float
    stamp_duty_buy_pct: float
    notes: str


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _get_or_create_settings(
    user_id: int, session: AsyncSession
) -> UserSettings:
    result = await session.execute(
        select(UserSettings).where(UserSettings.user_id == user_id)
    )
    us = result.scalar_one_or_none()
    if us is None:
        us = UserSettings(user_id=user_id)
        session.add(us)
        await session.flush()
    return us


def _settings_to_out(us: UserSettings) -> UserSettingsOut:
    return UserSettingsOut(
        user_id=us.user_id,
        monthly_capital=float(us.monthly_capital),
        risk_tolerance=us.risk_tolerance,
        max_drawdown_pct=us.max_drawdown_pct,
        intraday_enabled=us.intraday_enabled,
        swing_enabled=us.swing_enabled,
        longterm_enabled=us.longterm_enabled,
        max_positions=us.max_positions,
        max_sector_exposure_pct=us.max_sector_exposure_pct,
        max_single_stock_pct=us.max_single_stock_pct,
        cash_reserve_pct=us.cash_reserve_pct,
        auto_execution_enabled=us.auto_execution_enabled,
        paper_trading_mode=us.paper_trading_mode,
        kill_switch_active=us.kill_switch_active,
        updated_at=us.updated_at,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("", response_model=UserSettingsOut)
async def get_settings(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> UserSettingsOut:
    us = await _get_or_create_settings(current_user.id, session)
    return _settings_to_out(us)


@router.put("", response_model=UserSettingsOut)
async def update_settings(
    body: UserSettingsUpdate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> UserSettingsOut:
    us = await _get_or_create_settings(current_user.id, session)

    before: dict = _settings_to_out(us).model_dump()

    updates = body.model_dump(exclude_none=True)
    for field, value in updates.items():
        setattr(us, field, value)

    await session.flush()

    after: dict = _settings_to_out(us).model_dump()
    audit = AuditLog(
        user_id=current_user.id,
        action="update_settings",
        entity_type="UserSettings",
        entity_id=str(us.id),
        before_state=before,
        after_state=after,
    )
    session.add(audit)

    logger.info("settings_updated", user_id=current_user.id, changes=updates)
    return _settings_to_out(us)


@router.post("/kill-switch", response_model=KillSwitchResponse)
async def toggle_kill_switch(
    body: KillSwitchRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> KillSwitchResponse:
    us = await _get_or_create_settings(current_user.id, session)
    us.kill_switch_active = body.activate
    await session.flush()

    severity = "critical" if body.activate else "info"
    action_verb = "activated" if body.activate else "deactivated"
    risk_event = RiskEvent(
        user_id=current_user.id,
        event_type="kill_switch",
        description=f"Kill switch {action_verb} by user. Reason: {body.reason or 'not provided'}",
        severity=severity,
        action_taken=f"kill_switch_{action_verb}",
    )
    session.add(risk_event)

    logger.warning(
        "kill_switch_toggled",
        user_id=current_user.id,
        activated=body.activate,
        reason=body.reason,
    )

    msg = (
        "Kill switch ACTIVATED. All automated trading is now halted."
        if body.activate
        else "Kill switch DEACTIVATED. Trading may resume."
    )
    return KillSwitchResponse(kill_switch_active=body.activate, message=msg)


@router.get("/cost-model", response_model=CostModel)
async def get_cost_model(
    current_user: User = Depends(get_current_user),
) -> CostModel:
    # Zerodha fee schedule (as of 2024)
    return CostModel(
        brokerage_intraday_pct=0.0003,        # 0.03% or Rs 20/order (whichever lower)
        brokerage_delivery_pct=0.0,           # Zero brokerage on delivery
        brokerage_max_per_order=20.0,
        stt_delivery_pct=0.001,               # 0.1% on delivery (both legs)
        stt_intraday_sell_pct=0.00025,        # 0.025% only on sell side
        exchange_charges_pct=0.0000297,       # NSE: 0.00297%
        sebi_turnover_fee_pct=0.000001,       # Rs 10 per crore
        gst_pct=0.18,                         # 18% GST on brokerage + exchange charges
        stamp_duty_buy_pct=0.00015,           # 0.015% on buy (delivery), 0.003% intraday
        notes=(
            "Zerodha fee schedule effective Jan 2024. "
            "Brokerage capped at Rs 20/order for intraday/F&O. "
            "Delivery brokerage is zero. "
            "STT, exchange charges and SEBI fee are pass-through costs."
        ),
    )
