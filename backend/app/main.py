from __future__ import annotations

import asyncio
import random
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException, Request, WebSocket
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
from app.database.session import engine, migration_status, verify_schema
from app.realtime import ChannelHub, TickPublisher, websocket_endpoint

logger = structlog.get_logger(__name__)

# ── Rate limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])


# ── Realtime pub/sub hub ───────────────────────────────────────────────────────
# Channel-based, in-process, allowlisted (see app/realtime/channels.py).
# Publishers (lifespan events, the tick feed when armed) and the /ws endpoint
# share this one hub; the frontend subscribes to exactly the channels it
# renders. Published frames carry {"type": <channel>, ...}.
hub = ChannelHub()


async def _pub_system(**data: Any) -> None:
    """Fan-out a lifecycle event on the ``system`` channel; never fatal."""
    try:
        await hub.publish("system", data)
    except Exception:  # noqa: BLE001
        pass


async def _fetch_base_prices(
    delegate: Any, symbols: list[str], seed: int
) -> tuple[dict[str, float], bool]:
    """
    Previous-close prices to walk the mock feed around.

    Best-effort: ask the underlying data broker for a live last price per
    symbol; anything it cannot provide falls back to a DETERMINISTIC synthetic
    price (seed-derived).  ``used_fallback`` is returned so the caller can log
    loudly that part of the mock session is priced synthetically — the paper
    book's arithmetic must never pretend those prices were real.
    """
    base: dict[str, float] = {}
    if delegate is not None and hasattr(delegate, "get_quote"):
        try:
            quotes = await delegate.get_quote(symbols)
            for sym in symbols:
                q = quotes.get(sym) or quotes.get(f"NSE:{sym}") or {}
                last = float(q.get("last_price") or 0.0)
                if last > 0:
                    base[sym] = last
        except Exception as exc:  # noqa: BLE001
            logger.warning("base_price_fetch_failed", error=str(exc))

    used_fallback = False
    rng = random.Random(int(seed))
    for sym in symbols:
        if sym in base and base[sym] > 0:
            continue
        used_fallback = True
        base[sym] = round(1000.0 + rng.uniform(-400.0, 900.0), 2)
    return base, used_fallback


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
        mig = await migration_status()
        if mig["at_head"]:
            log.info("database_ready", revision=mig["head"])
        else:
            log.warning(
                "migrations_pending",
                current=mig["current"],
                head=mig["head"],
                version_table=mig["version_table_exists"],
                repair="alembic -c backend/alembic.ini upgrade head",
            )
        drift = await verify_schema()
        if drift:
            log.error("schema_drift", items=drift)
        await _pub_system(
            event="database",
            at_head=bool(mig["at_head"]),
            schema_drift_items=len(drift),
        )
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
        await _pub_system(event="redis", available=True)
    except Exception as exc:
        log.warning("redis_unavailable", error=str(exc))
        await _pub_system(event="redis", available=False)

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
            await _pub_system(event="execution_stack", trading_permitted=True)
        else:
            log.error(
                "execution_stack_blocked",
                trading_permitted=False,
                reason=stack.startup_reason,
                detail="Orders will be rejected until reconciliation succeeds.",
            )
            await _pub_system(
                event="execution_stack",
                trading_permitted=False,
                reason=stack.startup_reason,
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
        await _pub_system(event="execution_stack", trading_permitted=False, error=str(exc))

    # ── Automated intraday loop (paper, mock feed) ──────────────────────────
    #
    # Armed ONLY when a human turned it on AND we are paper AND the tick feed
    # is the deterministic mock.  Each condition is an independent switch, so
    # no single mis-set flag can arm the loop by accident:
    #
    #   * settings.auto_trade_enabled  — the explicit decision to trade
    #   * settings.trading_mode        — never arms on a live account
    #   * settings.tick_mode == "mock" — never arms on a live feed
    #   * settings.intraday_enabled    — the intraday sleeve must be opted in
    #
    # The loop still routes every order through ExecutionService, so nothing
    # here can override the kill switch, trading gate, mode or eligibility.
    app.state.auto_trader_stop = None
    app.state.auto_trader_tasks = []
    app.state.auto_trader = None
    app.state.tick_feed = None
    app.state.tick_aggregator = None

    autotrade_armed = (
        settings.auto_trade_enabled
        and settings.intraday_enabled
        and settings.tick_mode == "mock"
        and not settings.is_live_trading_enabled
        and getattr(app.state, "execution_stack", None) is not None
        and getattr(app.state, "execution_service", None) is not None
    )
    if autotrade_armed:
        try:
            from app.broker.tickdata import (
                MockTickFeed,
                TickBarAggregator,
                TickQuoteSource,
            )
            from app.engine.trader import IntradayAutoTrader
            from app.strategies.intraday import IntradayStrategy

            stack = app.state.execution_stack
            service = app.state.execution_service
            broker = getattr(stack, "broker", None)
            if broker is None or not hasattr(broker, "_data_broker"):
                raise RuntimeError(
                    "broker has no _data_broker to swap for the mock tick source"
                )

            pipeline = getattr(app.state, "trading_pipeline", None)
            universe = list(getattr(pipeline, "universe", None) or [])[: settings.trader_universe_size]
            if not universe:
                universe = [
                    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN",
                    "BHARTIARTL", "ITC", "LT", "HINDUNILVR", "AXISBANK", "KOTAKBANK",
                    "M&M", "SUNPHARMA", "BAJFINANCE", "MARUTI", "TITAN", "ADANIENT",
                ][: settings.trader_universe_size]

            underlying = getattr(broker, "_data_broker", None)
            base_prices, used_fallback = await _fetch_base_prices(
                underlying, universe, settings.tick_seed
            )
            if used_fallback:
                log.warning(
                    "mock_base_prices_synthetic",
                    count=sum(1 for s in universe if base_prices.get(s, 0) <= 0),
                    detail=(
                        "Part of the mock feed is priced from seeded synthetic "
                        "base prices (not real closes). Paper-only; labelled in "
                        "every quote payload."
                    ),
                )

            aggregator = TickBarAggregator()
            quote_source = TickQuoteSource(aggregator, delegate=underlying)
            # The PaperBroker fills against whatever data source it holds;
            # pointing it at the aggregator is what makes fills and signals
            # resolve the SAME minute-bar price.
            broker._data_broker = quote_source

            stop_event = asyncio.Event()
            app.state.auto_trader_stop = stop_event
            app.state.tick_aggregator = aggregator
            app.state.tick_feed = MockTickFeed(
                universe, base_prices,
                seed=settings.tick_seed,
                interval_ms=settings.tick_interval_ms,
            )

            strategy = IntradayStrategy(paper_mode=True)
            trader = IntradayAutoTrader(
                execution_service=service,
                strategy=strategy,
                bar_source=aggregator,
                broker=broker,
                universe=universe,
                cycle_seconds=settings.trader_cycle_seconds,
                use_broker_stop=settings.intraday_use_broker_stop,
            )
            app.state.auto_trader = trader

            # Fan ticks out to the "ticks" channel as they flow into the
            # aggregator. No-op until a dashboard subscribes; scheduled, never
            # awaited, so the feed's synchronous path is untouched.
            tick_publisher = TickPublisher(hub, channel="ticks")

            def _on_tick(symbol: Any, ts: Any, price: float, volume: float) -> None:
                aggregator.on_tick(symbol, ts, price, volume)
                tick_publisher(symbol, ts, price, volume)

            app.state.auto_trader_tasks = [
                asyncio.create_task(app.state.tick_feed.run(stop_event, _on_tick), name="mock-tick-feed"),
                asyncio.create_task(trader.run(stop_event), name="intraday-auto-trader"),
            ]
            log.info(
                "auto_trader_armed",
                universe=len(universe),
                cycle_seconds=settings.trader_cycle_seconds,
                broker_stop=settings.intraday_use_broker_stop,
                synthetic_base=used_fallback,
            )
            await _pub_system(event="auto_trader", armed=True, universe=len(universe))
        except Exception as exc:  # noqa: BLE001
            app.state.auto_trader_stop = None
            app.state.auto_trader_tasks = []
            app.state.auto_trader = None
            app.state.tick_feed = None
            app.state.tick_aggregator = None
            log.error(
                "auto_trader_unavailable",
                error=str(exc),
                detail="Intraday loop not armed; the API will serve read-only data.",
            )

    if not autotrade_armed:
        await _pub_system(event="auto_trader", armed=False)

    log.info("algodollar_started")
    await _pub_system(event="started")

    yield

    # Shutdown
    #
    # Ordered deliberately: release the execution stack (and the Redis clients
    # it opened) BEFORE disposing the database engine, because closing the
    # stack can still want to write. Each step is guarded so one failure does
    # not abort the rest — a shutdown path that raises turns a clean stop into
    # a crash and leaves connections open.
    log.info("algodollar_stopping")
    await _pub_system(event="stopping")

    # Stop the auto-trader and tick feed BEFORE releasing the execution stack:
    # the loop submits orders through stack.service, and a half-disposed stack
    # is not a safe place to still be trading.  Signalled cooperatively, then
    # forcibly cancelled if it does not stop promptly — the SL-M stops at the
    # broker confine the risk while we finish shutting down.
    stop_event = getattr(app.state, "auto_trader_stop", None)
    if stop_event is not None:
        stop_event.set()
        tasks = getattr(app.state, "auto_trader_tasks", []) or []
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as exc:  # noqa: BLE001
                log.error("auto_trader_shutdown_error", error=str(exc))
        for task in tasks:
            if not task.done():
                task.cancel()
        log.info("auto_trader_stopped", tasks=len(tasks))

    # A distinct name: `stack` is already bound to the ExecutionStack built
    # during startup, and rebinding it to Optional[Any] here is what mypy
    # objected to.
    running_stack = getattr(app.state, "execution_stack", None)
    if running_stack is not None and hasattr(running_stack, "aclose"):
        try:
            await running_stack.aclose()
            log.info("execution_stack_closed")
        except Exception as exc:  # noqa: BLE001
            log.error("execution_stack_close_failed", error=str(exc))

    try:
        await engine.dispose()
        log.info("database_engine_disposed")
    except Exception as exc:  # noqa: BLE001
        log.error("database_dispose_failed", error=str(exc))

    log.info("algodollar_stopped")
    await _pub_system(event="stopped")


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
    app.state.realtime_hub = hub
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
        # Channel-based pub/sub: subscribe to "ticks", "orders", "portfolio",
        # "risk", "strategy", "audit" or "system", and published frames arrive
        # tagged {"type": <channel>, ...}. See app/realtime/ws.py for the
        # client<->server message contract.
        await websocket_endpoint(websocket, hub)

    return app


app = create_app()
