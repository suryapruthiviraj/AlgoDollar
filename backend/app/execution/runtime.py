"""
Production assembly of the execution stack from application settings.

``build_execution_stack`` takes every collaborator as an argument, which is what
makes it testable. This module is what actually SUPPLIES those arguments in a
running deployment, and it exists because supplying none of them — which is what
``main.py`` used to do — produced a stack that was wired but inert:

    data_broker=None   -> the paper broker had no prices and could never fill
    local_state=None   -> reconciliation reported UNAVAILABLE, gate stayed shut
    persistence=None   -> nothing an order did was ever written down

Each of those defaults is individually correct (fail closed, invent nothing).
Together they meant the application could start, report healthy, and never be
able to trade.

WHAT GETS DEGRADED AND WHAT GETS REFUSED
----------------------------------------
Some dependencies have honest fallbacks and some do not:

* **Database** — REQUIRED. Without it there is no durable record, so
  reconciliation has nothing to compare and the gate stays shut. Not optional.
* **Redis** — optional for paper. Absent, the kill switch is process-local and
  the order store is in-memory; both are logged at WARNING and both are treated
  as non-durable, which reconciliation already accounts for.
* **Market data** — REQUIRED to fill anything. A stack built without it is
  constructed but can only produce refusals.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.core.config import settings
from app.execution.bootstrap import (
    ExecutionStack,
    InMemoryKillSwitchStore,
    build_execution_stack,
    settings_kill_switch_probe,
)
from app.execution.persistence import (
    SqlAlchemyExecutionPersistence,
    ensure_opening_cash,
    get_or_create_system_user,
)

logger = logging.getLogger(__name__)

#: Opening paper balance, used only the FIRST time a paper account is created.
#: Never re-applied: see ensure_opening_cash.
DEFAULT_PAPER_OPENING_CASH = 1_000_000.0


class SessionScopedLocalState:
    """
    ``LocalStateStore`` backed by a session FACTORY rather than one session.

    ``SqlAlchemyLocalStateStore`` holds the session it is given. Holding one
    open for the lifetime of the process means reconciliation reads through a
    stale identity map — it would compare the broker against whatever this
    session saw the first time it asked, not against what is committed now.
    Since reconciliation exists precisely to detect divergence, reading stale
    local state defeats it.

    Every call therefore opens its own short-lived session. Failures propagate
    as ``LocalStateUnavailable``, never as an empty list.
    """

    def __init__(
        self, session_factory: Any, user_id: int, *, trading_mode: Optional[str] = None
    ) -> None:
        self._sf = session_factory
        self._user_id = int(user_id)
        self._mode = trading_mode or str(settings.trading_mode)

    async def _with_store(self, method: str) -> Any:
        from app.execution.reconciliation import SqlAlchemyLocalStateStore

        async with self._sf() as session:
            store = SqlAlchemyLocalStateStore(
                session, self._user_id, trading_mode=self._mode
            )
            return await getattr(store, method)()

    async def get_positions(self) -> list[dict]:
        return await self._with_store("get_positions")

    async def get_orders(self) -> list[dict]:
        return await self._with_store("get_orders")

    async def get_trades(self) -> list[dict]:
        return await self._with_store("get_trades")

    async def get_cash(self) -> dict:
        return await self._with_store("get_cash")


async def _redis_clients() -> tuple[Optional[Any], Optional[Any]]:
    """
    Return ``(async_client, sync_client)``, or ``(None, None)`` with the reason.

    Two clients, not one, because they serve two different call shapes: the
    order store is awaited from async code, while the kill-switch probe is
    synchronous and must stay that way (see RedisKillSwitchStore).
    """
    try:
        import redis as redis_sync
        import redis.asyncio as redis_asyncio

        aclient = redis_asyncio.from_url(settings.redis_url, socket_connect_timeout=5)
        await aclient.ping()
        sclient = redis_sync.from_url(settings.redis_url, socket_connect_timeout=5)
        sclient.ping()
        return aclient, sclient
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Redis unavailable (%s). The kill switch will be process-local and "
            "the order store non-durable; neither survives a restart. This is "
            "acceptable for paper, and reconciliation already refuses to report "
            "OK against a non-durable store.", exc,
        )
        return None, None


def _build_data_broker() -> Optional[Any]:
    """The price source for paper fills. None means nothing can fill."""
    try:
        from app.broker.marketdata import MarketDataBroker
        from app.data.providers import YahooDataProvider

        return MarketDataBroker(YahooDataProvider())
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "No market data source could be constructed (%s). The execution "
            "stack will be built, but with no prices every order is refused.",
            exc,
        )
        return None


async def build_production_stack(
    *,
    session_factory: Any = None,
    data_broker: Any = None,
    live_broker: Any = None,
    paper_state_path: Optional[str] = None,
    audit_path: Optional[str] = None,
    opening_cash: float = DEFAULT_PAPER_OPENING_CASH,
    paper_clock: Any = None,
) -> ExecutionStack:
    """
    Assemble the execution stack a running deployment actually uses.

    Every argument exists so a test can substitute an EXTERNAL boundary — the
    database, the market data feed, the broker. None of them bypass an internal
    gate: the same ExecutionService, the same safety layer, the same
    reconciliation runs in both cases.
    """
    if session_factory is None:
        from app.database.session import async_session_maker

        session_factory = async_session_maker

    mode = str(settings.trading_mode)

    # -- identity and opening balance ------------------------------------- #
    user_id = await get_or_create_system_user(session_factory)
    current_cash = await ensure_opening_cash(
        session_factory, user_id=user_id, mode=mode, opening_cash=opening_cash
    )
    logger.info(
        "execution identity: user_id=%s mode=%s cash=Rs %.2f", user_id, mode, current_cash
    )

    persistence = SqlAlchemyExecutionPersistence(session_factory, user_id=user_id)
    local_state = SessionScopedLocalState(session_factory, user_id, trading_mode=mode)

    # -- optional Redis --------------------------------------------------- #
    aredis, sredis = await _redis_clients()
    kill_switch_store: Any
    order_store: Any = None
    if aredis is not None and sredis is not None:
        from app.execution.lifecycle import RedisOrderStore

        kill_switch_store = RedisKillSwitchStore(sredis)
        order_store = RedisOrderStore(aredis)
    else:
        kill_switch_store = InMemoryKillSwitchStore()

    # -- market data ------------------------------------------------------ #
    if data_broker is None:
        data_broker = _build_data_broker()

    # build_execution_stack already installs store_kill_switch_probe(store) for
    # the store passed in, so it is not added again here.
    stack = await build_execution_stack(
        data_broker=data_broker,
        live_broker=live_broker,
        kill_switch_store=kill_switch_store,
        local_state=local_state,
        persistence=persistence,
        order_store=order_store,
        audit_path=audit_path,
        initial_cash=current_cash,
        paper_state_path=paper_state_path,
        paper_clock=paper_clock,
    )

    # Hang the Redis clients off the stack so shutdown can close them. Without
    # a handle they were unreachable, so every restart leaked its connections
    # and a container that cycled repeatedly would exhaust the server's client
    # limit.
    stack.redis_clients = tuple(c for c in (aredis, sredis) if c is not None)

    # Persist the reconciliation verdict so the reason a process refused to
    # trade is still answerable after it exits.
    try:
        await persistence.record_reconciliation(
            mode=mode,
            state=str(getattr(stack.recovery, "state", "") or "UNKNOWN"),
            trading_permitted=bool(stack.trading_permitted),
            detail=stack.startup_reason or "",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("could not persist startup reconciliation: %s", exc)

    return stack


class RedisKillSwitchStore:
    """
    Kill-switch store backed by Redis, read SYNCHRONOUSLY at probe time.

    A synchronous client is used deliberately. ``KillSwitch`` probes are
    synchronous, so an async client would force the value to be cached and
    refreshed on a timer — and a cached kill switch is not a kill switch: an
    operator engaging it would not be seen until the next refresh tick, which
    is the entire window the switch exists to close.

    A Redis failure RAISES rather than returning None, because ``KillSwitch``
    treats a raising probe as ACTIVE. "I cannot tell whether trading was
    halted" must mean stop, never carry on.
    """

    def __init__(self, client: Any, prefix: str = "algodollar:") -> None:
        if client is None:
            raise ValueError("RedisKillSwitchStore requires a redis client")
        self._c = client
        self._p = prefix

    def get(self, key: str) -> Any:
        raw = self._c.get(f"{self._p}{key}")
        return raw.decode() if isinstance(raw, bytes) else raw

    def set(self, key: str, value: Any) -> None:
        self._c.set(f"{self._p}{key}", value)

    def delete(self, key: str) -> None:
        self._c.delete(f"{self._p}{key}")

    def engage(self, reason: str = "manual") -> None:
        self.set("kill_switch", "1")
        self.set("kill_switch_reason", reason)

    def release(self) -> None:
        self.delete("kill_switch")


__all__ = [
    "DEFAULT_PAPER_OPENING_CASH",
    "RedisKillSwitchStore",
    "SessionScopedLocalState",
    "build_production_stack",
    "settings_kill_switch_probe",
]
