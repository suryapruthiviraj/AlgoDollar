from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.routes import api_router
from app.core.config import settings
from app.core.exceptions import (
    AlgoDollarError,
    BrokerConnectionError,
    KillSwitchActiveError,
    RiskLimitExceededError,
)
from app.core.logging import setup_logging
from app.database.session import create_all_tables, engine

logger = structlog.get_logger(__name__)

# ── Rate limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])


# ── WebSocket connection manager ───────────────────────────────────────────────

class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        logger.info("ws_connected", total=len(self.active))

    def disconnect(self, ws: WebSocket) -> None:
        self.active.remove(ws)
        logger.info("ws_disconnected", total=len(self.active))

    async def broadcast(self, data: dict[str, Any]) -> None:
        payload = json.dumps(data)
        dead: list[WebSocket] = []
        for connection in self.active:
            try:
                await connection.send_text(payload)
            except Exception:
                dead.append(connection)
        for d in dead:
            self.active.remove(d)


ws_manager = ConnectionManager()


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    setup_logging()
    log = structlog.get_logger("startup")

    log.info(
        "algodollar_starting",
        env=settings.app_env,
        trading_mode=settings.trading_mode,
        live_enabled=settings.is_live_trading_enabled,
    )

    try:
        await create_all_tables()
        log.info("database_ready")
    except Exception as exc:
        log.error("database_init_failed", error=str(exc))

    try:
        # `redis.asyncio`, not `aioredis`. The standalone aioredis package was
        # merged into redis-py and is unmaintained; aioredis 2.0.1 cannot even
        # be imported on Python 3.11+ (`duplicate base class TimeoutError`).
        # Because that ImportError was caught here, Redis silently reported
        # "unavailable" on every startup and the kill-switch store backed by it
        # would never have worked in production.
        import redis.asyncio as redis_asyncio

        redis = redis_asyncio.from_url(settings.redis_url, socket_connect_timeout=5)
        await redis.ping()
        await redis.aclose()
        log.info("redis_ready")
    except Exception as exc:
        log.warning("redis_unavailable", error=str(exc))

    # ── Execution stack + startup reconciliation ──────────────────────────
    #
    # The execution layer used to be unreachable: nothing outside
    # app/execution and app/broker imported either package, reconcile() was
    # never called, and the eligibility gate was enforced nowhere. It is wired
    # here so there is exactly one place an order can originate.
    #
    # Reconciliation runs BEFORE trading is permitted. If it does not reach
    # RECONCILIATION_OK the service is still constructed but its trading gate
    # stays closed, so every attempt produces an audited rejection rather than
    # an unhandled error somewhere upstream.
    app.state.execution_stack = None
    app.state.trading_pipeline = None
    try:
        # build_production_stack, NOT build_execution_stack(). The latter takes
        # every collaborator as an argument and defaults them all to None, which
        # is correct for a test but produced a stack that could never trade:
        # no data_broker meant the paper broker had no prices, no local_state
        # meant reconciliation reported UNAVAILABLE and the gate never opened,
        # and no persistence meant nothing an order did was written down.
        from app.execution.runtime import build_production_stack

        stack = await build_production_stack(
            paper_state_path=settings.paper_state_path,
            audit_path=settings.execution_audit_path,
        )
        app.state.execution_stack = stack
        app.state.execution_service = stack.service

        # The signal pipeline: market data -> strategy -> sizing -> the
        # execution service above. Published here so the API can run a cycle;
        # it is NOT started on a timer. Nothing schedules itself into placing
        # orders — a cycle happens because something asked for one.
        try:
            from app.engine.pipeline import build_default_pipeline

            app.state.trading_pipeline = build_default_pipeline(
                execution_service=stack.service,
                data_broker=getattr(stack.broker, "_data_broker", None),
            )
            log.info(
                "trading_pipeline_ready",
                strategies=[type(s).__name__ for s in app.state.trading_pipeline.strategies],
                universe=len(app.state.trading_pipeline.universe),
            )
        except Exception as exc:  # noqa: BLE001
            app.state.trading_pipeline = None
            log.error("trading_pipeline_unavailable", error=str(exc))

        if stack.trading_permitted:
            log.info("execution_stack_ready", trading_permitted=True)
        else:
            log.error(
                "execution_stack_blocked",
                trading_permitted=False,
                reason=stack.startup_reason,
                detail="Orders will be rejected until reconciliation succeeds.",
            )
    except Exception as exc:
        # Failing to build the execution stack must NOT leave a half-configured
        # object behind that might later be mistaken for a working one.
        app.state.execution_stack = None
        app.state.execution_service = None
        app.state.trading_pipeline = None
        log.error(
            "execution_stack_unavailable",
            error=str(exc),
            detail="Trading is unavailable. The API will serve read-only data.",
        )

    log.info("algodollar_started")
    yield

    # Shutdown
    log.info("algodollar_stopping")
    await engine.dispose()
    log.info("algodollar_stopped")


# ── Application factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="AlgoDollar API",
        description="Quantitative trading platform backend",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Rate limiting
    app.state.limiter = limiter
    # The ignore below is an upstream typing gap, not a real mismatch.
    # Starlette types every handler as taking `Exception`, while slowapi's
    # handler is declared to take the narrower `RateLimitExceeded`. Starlette
    # dispatches by exception CLASS, so this handler can only ever be called
    # with the type it is registered for — a contravariance Starlette's
    # annotation cannot express. Scoped to this one call and one error code.
    app.add_exception_handler(
        RateLimitExceeded, _rate_limit_exceeded_handler  # type: ignore[arg-type]
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(api_router, prefix="/api/v1")

    # ── Exception handlers ────────────────────────────────────────────────────

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        log = structlog.get_logger("exception")
        log.warning(
            "http_exception",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "status_code": exc.status_code},
        )

    @app.exception_handler(KillSwitchActiveError)
    async def kill_switch_handler(
        request: Request, exc: KillSwitchActiveError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content={
                "detail": exc.message,
                "error_type": "KillSwitchActiveError",
                "details": exc.details,
            },
        )

    @app.exception_handler(RiskLimitExceededError)
    async def risk_limit_handler(
        request: Request, exc: RiskLimitExceededError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": exc.message,
                "error_type": "RiskLimitExceededError",
                "details": exc.details,
            },
        )

    @app.exception_handler(BrokerConnectionError)
    async def broker_error_handler(
        request: Request, exc: BrokerConnectionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "detail": exc.message,
                "error_type": "BrokerConnectionError",
                "details": exc.details,
            },
        )

    @app.exception_handler(AlgoDollarError)
    async def algodollar_error_handler(
        request: Request, exc: AlgoDollarError
    ) -> JSONResponse:
        log = structlog.get_logger("exception")
        log.error(
            "algodollar_error",
            error_type=type(exc).__name__,
            message=exc.message,
            details=exc.details,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": exc.message,
                "error_type": type(exc).__name__,
                "details": exc.details,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log = structlog.get_logger("exception")
        log.exception("unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "status_code": 500},
        )

    # ── Root endpoints ─────────────────────────────────────────────────────────

    @app.get("/", tags=["root"])
    @limiter.limit("60/minute")
    async def root(request: Request) -> dict:
        return {
            "service": "AlgoDollar API",
            "version": "0.1.0",
            "trading_mode": settings.trading_mode,
            "docs": "/docs",
            "health": "/api/v1/health",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── WebSocket ──────────────────────────────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_portfolio(websocket: WebSocket) -> None:
        await ws_manager.connect(websocket)
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                    msg_type = msg.get("type", "ping")
                except json.JSONDecodeError:
                    msg_type = "ping"

                if msg_type == "ping":
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "pong",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                    )
                elif msg_type == "subscribe":
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "subscribed",
                                "channels": msg.get("channels", []),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                    )
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    return app


app = create_app()
