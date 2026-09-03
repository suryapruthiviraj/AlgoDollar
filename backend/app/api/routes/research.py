from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user
from app.database.models import ModelVersion, User
from app.database.session import get_async_session

router = APIRouter()
logger = structlog.get_logger(__name__)


# ── Schemas ────────────────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    strategy: str
    start_date: date
    end_date: date
    initial_capital: float = Field(default=100000.0, ge=10000.0)
    parameters: dict[str, Any] = Field(default_factory=dict)


class BacktestResult(BaseModel):
    backtest_id: str
    strategy: str
    start_date: date
    end_date: date
    initial_capital: float
    final_value: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate: float
    total_trades: int
    avg_trade_duration_days: float
    profit_factor: float
    total_costs: float
    parameters: dict[str, Any]
    equity_curve: list[dict]
    status: str
    created_at: datetime


class BacktestListItem(BaseModel):
    backtest_id: str
    strategy: str
    start_date: date
    end_date: date
    sharpe_ratio: float
    total_return_pct: float
    status: str
    created_at: datetime


class ModelOut(BaseModel):
    id: int
    model_name: str
    version: str
    strategy: str
    training_start: Optional[date]
    training_end: Optional[date]
    validation_sharpe: Optional[float]
    oos_sharpe: Optional[float]
    oos_return: Optional[float]
    is_active: bool
    git_commit: Optional[str]
    created_at: datetime


class WalkForwardRequest(BaseModel):
    strategy: str
    start_date: date
    end_date: date
    in_sample_months: int = Field(default=12, ge=3, le=36)
    out_of_sample_months: int = Field(default=3, ge=1, le=12)
    parameters: dict[str, Any] = Field(default_factory=dict)


class WalkForwardResult(BaseModel):
    task_id: str
    strategy: str
    start_date: date
    end_date: date
    windows: int
    status: str
    message: str
    submitted_at: datetime


# ── Helpers ────────────────────────────────────────────────────────────────────

# In-memory store for demo backtests (replace with DB or Celery task store in prod)
_backtest_store: dict[str, BacktestResult] = {}


def _run_backtest_sync(req: BacktestRequest) -> BacktestResult:
    """
    NOT IMPLEMENTED — deliberately refuses rather than inventing a result.

    This function used to fabricate its output. A seeded RNG produced a Sharpe
    ratio between 0.8 and 2.2, an annualized return between 8% and 22%, a win
    rate, a profit factor and a monthly equity curve, and returned them in a
    `BacktestResult` indistinguishable from a real one. The dashboard's "Run
    Backtest" button calls this endpoint, so a user was shown invented
    performance figures presented as measured ones.

    That is disqualifying for this project. Every report in docs/ states that
    no strategy has been validated and that backtest results must never be
    fabricated; an endpoint quietly doing exactly that undermines all of it.

    A real engine already exists — `app.backtesting.engine.EventDrivenBacktester`
    with a real Zerodha cost model, and `app.research.pipeline` for purged
    walk-forward validation. Until this endpoint is wired to them, refusing is
    the honest behaviour: returning nothing is strictly better than returning
    fiction.
    """
    days = (req.end_date - req.start_date).days
    if days < 30:
        raise ValueError("Backtest window must be at least 30 days.")

    raise NotImplementedError(
        "Backtesting via this endpoint is not implemented. It previously "
        "returned randomly generated performance metrics, which have been "
        "removed. Use app.backtesting.engine.EventDrivenBacktester with "
        "app.research.pipeline; see docs/REAL_DATA_VALIDATION_REPORT.md. "
        "PRODUCTION MODEL = NONE."
    )


def _unreachable_reference_implementation(req: "BacktestRequest", rng, days: int):
    """
    Retained only to document the response schema for whoever wires the real
    engine to this endpoint. Never called.
    """
    annual_ret = rng.uniform(0.08, 0.22)
    years = days / 365.25
    total_ret = (1 + annual_ret) ** years - 1
    final_value = req.initial_capital * (1 + total_ret)
    sharpe = rng.uniform(0.8, 2.2)
    sortino = sharpe * rng.uniform(1.1, 1.5)
    max_dd = rng.uniform(0.05, 0.18)
    calmar = annual_ret / max_dd if max_dd else 0
    win_rate = rng.uniform(0.45, 0.65)
    n_trades = int(days / rng.uniform(3, 10))
    costs = final_value * 0.002

    # Simplified equity curve (monthly snapshots)
    curve: list[dict] = []
    current = req.initial_capital
    cursor = req.start_date
    while cursor <= req.end_date:
        monthly_ret = rng.uniform(-0.04, 0.06)
        current *= 1 + monthly_ret
        curve.append({"date": cursor.isoformat(), "value": round(current, 2)})
        # advance ~1 month
        m = cursor.month + 1 if cursor.month < 12 else 1
        y = cursor.year if cursor.month < 12 else cursor.year + 1
        cursor = cursor.replace(year=y, month=m)

    bt_id = str(uuid.uuid4())
    result = BacktestResult(
        backtest_id=bt_id,
        strategy=req.strategy,
        start_date=req.start_date,
        end_date=req.end_date,
        initial_capital=req.initial_capital,
        final_value=round(final_value, 2),
        total_return_pct=round(total_ret * 100, 2),
        annualized_return_pct=round(annual_ret * 100, 2),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        max_drawdown_pct=round(max_dd * 100, 2),
        calmar_ratio=round(calmar, 3),
        win_rate=round(win_rate * 100, 2),
        total_trades=n_trades,
        avg_trade_duration_days=round(days / n_trades, 1) if n_trades else 0,
        profit_factor=round(rng.uniform(1.1, 2.5), 2),
        total_costs=round(costs, 2),
        parameters=req.parameters,
        equity_curve=curve,
        status="completed",
        created_at=datetime.now(timezone.utc),
    )
    _backtest_store[bt_id] = result
    return result


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/backtest", response_model=BacktestResult)
async def run_backtest(
    body: BacktestRequest,
    current_user: User = Depends(get_current_user),
) -> BacktestResult:
    valid_strategies = {"longterm", "swing", "intraday"}
    if body.strategy not in valid_strategies:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid strategy '{body.strategy}'. Valid: {sorted(valid_strategies)}",
        )
    if body.start_date >= body.end_date:
        raise HTTPException(status_code=422, detail="start_date must be before end_date")
    if body.end_date > date.today():
        raise HTTPException(status_code=422, detail="end_date cannot be in the future")

    try:
        result = _run_backtest_sync(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NotImplementedError as exc:
        # 501, not 500: this endpoint is deliberately unimplemented rather than
        # broken. It used to return fabricated performance metrics; refusing is
        # the correct behaviour until it is wired to the real backtester.
        logger.warning(
            "backtest_endpoint_not_implemented",
            user_id=current_user.id,
            strategy=body.strategy,
        )
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    logger.info(
        "backtest_completed",
        user_id=current_user.id,
        strategy=body.strategy,
        backtest_id=result.backtest_id,
        sharpe=result.sharpe_ratio,
    )
    return result


@router.get("/backtests", response_model=list[BacktestListItem])
async def list_backtests(
    current_user: User = Depends(get_current_user),
) -> list[BacktestListItem]:
    items = [
        BacktestListItem(
            backtest_id=bt.backtest_id,
            strategy=bt.strategy,
            start_date=bt.start_date,
            end_date=bt.end_date,
            sharpe_ratio=bt.sharpe_ratio,
            total_return_pct=bt.total_return_pct,
            status=bt.status,
            created_at=bt.created_at,
        )
        for bt in _backtest_store.values()
    ]
    items.sort(key=lambda x: x.created_at, reverse=True)
    return items


@router.get("/models", response_model=list[ModelOut])
async def list_models(
    strategy: Optional[str] = None,
    active_only: bool = False,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[ModelOut]:
    stmt = select(ModelVersion).order_by(ModelVersion.created_at.desc())
    if strategy:
        stmt = stmt.where(ModelVersion.strategy == strategy)
    if active_only:
        stmt = stmt.where(ModelVersion.is_active == True)

    result = await session.execute(stmt)
    models = result.scalars().all()
    return [
        ModelOut(
            id=m.id,
            model_name=m.model_name,
            version=m.version,
            strategy=m.strategy,
            training_start=m.training_start,
            training_end=m.training_end,
            validation_sharpe=m.validation_sharpe,
            oos_sharpe=m.oos_sharpe,
            oos_return=m.oos_return,
            is_active=m.is_active,
            git_commit=m.git_commit,
            created_at=m.created_at,
        )
        for m in models
    ]


@router.post("/walkforward", response_model=WalkForwardResult)
async def run_walkforward(
    body: WalkForwardRequest,
    current_user: User = Depends(get_current_user),
) -> WalkForwardResult:
    import uuid

    valid_strategies = {"longterm", "swing", "intraday"}
    if body.strategy not in valid_strategies:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid strategy '{body.strategy}'.",
        )
    if body.start_date >= body.end_date:
        raise HTTPException(status_code=422, detail="start_date must be before end_date")

    days = (body.end_date - body.start_date).days
    window_days = (body.in_sample_months + body.out_of_sample_months) * 30
    if days < window_days:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Date range ({days} days) is shorter than one walk-forward window "
                f"({window_days} days)."
            ),
        )

    num_windows = max(1, days // (body.out_of_sample_months * 30))
    task_id = str(uuid.uuid4())

    logger.info(
        "walkforward_submitted",
        user_id=current_user.id,
        strategy=body.strategy,
        task_id=task_id,
        windows=num_windows,
    )

    # In production: submit to Celery and return task_id for polling.
    return WalkForwardResult(
        task_id=task_id,
        strategy=body.strategy,
        start_date=body.start_date,
        end_date=body.end_date,
        windows=num_windows,
        status="submitted",
        message=(
            f"Walk-forward analysis submitted with {num_windows} windows "
            f"({body.in_sample_months}m in-sample / {body.out_of_sample_months}m OOS). "
            "Poll GET /research/backtests for results."
        ),
        submitted_at=datetime.now(timezone.utc),
    )
