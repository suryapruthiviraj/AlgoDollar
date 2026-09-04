"""
Assembly of the execution stack at application startup.

WHAT THIS DOES
--------------
Builds the one `ExecutionService` the application will use, and runs startup
reconciliation before trading is permitted. Wiring lives here rather than in
`main.py` so it can be exercised by tests without starting a web server — the
previous arrangement had no wiring at all, and untestable wiring is how it
would get there again.

THE STARTUP CONTRACT
--------------------
    establish dependencies
    -> select broker for the configured mode
    -> verify broker connectivity
    -> reconcile broker state against local state
    -> only on RECONCILIATION_OK does the trading gate open

Failure at any step leaves the gate CLOSED and the service still constructed.
That distinction matters: a service that exists but refuses to trade produces
an audited rejection for every attempt, which is far more diagnosable than a
missing object producing an AttributeError somewhere upstream.

PAPER IS THE DEFAULT
--------------------
Broker selection is driven by `settings.trading_mode`, which defaults to
`paper`. Live requires BOTH `trading_mode=live` AND an explicit authorization
flag. There is no fallback in either direction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from app.core.config import settings
from app.execution.audit import (
    AuditJournal,
    AuditSink,
    InMemoryAuditSink,
    JsonlAuditSink,
)
from app.execution.lifecycle import InMemoryOrderStore
from app.execution.order_manager import OrderManager
from app.execution.reconciliation import ReconciliationEngine
from app.execution.recovery import RecoveryManager
from app.execution.safety import ExecutionSafety
from app.execution.service import (
    ExecutionService,
    KillSwitch,
    TradingGate,
    TradingMode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kill-switch sources
# ---------------------------------------------------------------------------

class InMemoryKillSwitchStore:
    """
    Minimal kill-switch store for paper runs and tests.

    Implements the `get`/`set` surface `ExecutionSafety` and
    `ReconciliationEngine` expect. Not durable — a real deployment must back
    this with Redis or the database so an engaged switch survives a restart.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    # convenience for operators and tests
    def engage(self, reason: str = "manual") -> None:
        self._data["kill_switch"] = "1"
        self._data["kill_switch_reason"] = reason

    def release(self) -> None:
        self._data.pop("kill_switch", None)


def store_kill_switch_probe(store, key: str = "kill_switch") -> Callable[[], bool]:
    """Probe reading the store that ExecutionSafety and reconciliation share."""

    def probe() -> bool:
        # Deliberately not wrapped: KillSwitch treats a raising probe as
        # ACTIVE, which is the behaviour we want if the store is unreachable.
        return bool(store.get(key))

    probe.__name__ = f"store[{key}]"
    return probe


def settings_kill_switch_probe(
    fetch_user_settings: Callable[[], Any],
) -> Callable[[], bool]:
    """
    Probe reading `UserSettings.kill_switch_active` from the database.

    This exists because the application had two unconnected kill switches: the
    UI wrote this database flag while the execution layer read a Redis key.
    Both are now consulted, and either one stops trading.
    """

    def probe() -> bool:
        us = fetch_user_settings()
        return bool(getattr(us, "kill_switch_active", False))

    probe.__name__ = "user_settings.kill_switch_active"
    return probe


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ExecutionStack:
    """Everything assembled, plus whether trading was permitted."""
    service: ExecutionService
    order_manager: OrderManager
    safety: ExecutionSafety
    broker: Any
    recovery: RecoveryManager
    kill_switch_store: Any
    audit: AuditJournal
    startup_ok: bool
    startup_reason: Optional[str] = None

    @property
    def trading_permitted(self) -> bool:
        return self.recovery.trading_permitted


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

async def build_execution_stack(
    *,
    data_broker=None,
    live_broker=None,
    kill_switch_store=None,
    local_state=None,
    audit_path: Optional[str] = None,
    initial_cash: float = 1_000_000.0,
    extra_kill_switch_probes: tuple[Callable[[], bool], ...] = (),
    run_recovery: bool = True,
) -> ExecutionStack:
    """
    Build the execution stack and run startup reconciliation.

    Parameters
    ----------
    data_broker : BrokerInterface, optional
        Source of live prices for the paper broker. Paper trading uses REAL
        market data and simulated fills; without a data source the paper
        broker has no prices and cannot fill anything.
    live_broker : BrokerInterface, optional
        Only used when trading_mode is live AND live trading is authorized.
    kill_switch_store : store with get/set, optional
    local_state : LocalStateStore, optional
        Real local state for reconciliation. Absent, reconciliation reports
        UNAVAILABLE and the trading gate stays closed — which is correct: a
        process that cannot read its own records has nothing to compare.
    audit_path : str, optional
        JSONL destination. In-memory only if omitted.
    run_recovery : bool
        Leave True. False is for constructing the stack in tests that drive
        recovery themselves.
    """
    mode = TradingMode(settings.trading_mode)
    live_authorized = bool(settings.is_live_trading_enabled)

    store = kill_switch_store if kill_switch_store is not None else InMemoryKillSwitchStore()

    # Annotated to the protocol: inference from the first element alone makes
    # this list[InMemoryAuditSink], so appending the durable JSONL sink — the
    # one that matters for an audit trail — looks like a type error.
    sinks: list[AuditSink] = [InMemoryAuditSink()]
    if audit_path:
        sinks.append(JsonlAuditSink(audit_path))
    audit = AuditJournal(*sinks)

    # -- broker selection: paper unless live is BOTH configured and authorized
    if mode is TradingMode.LIVE:
        if not live_authorized:
            raise RuntimeError(
                "trading_mode=live but live trading is not authorized. Refusing "
                "to build a live execution stack."
            )
        if live_broker is None:
            raise RuntimeError(
                "trading_mode=live but no live broker supplied. There is no "
                "fallback to paper — that would silently change which account "
                "is at risk."
            )
        broker = live_broker
    else:
        from app.broker.paper import PaperBroker

        broker = PaperBroker(data_broker=data_broker, initial_cash=initial_cash)

    safety = ExecutionSafety(store)
    order_manager = OrderManager(safety, store=InMemoryOrderStore())

    engine = ReconciliationEngine(store, local_state=local_state)
    recovery = RecoveryManager(engine, local_state=local_state)

    probes = [store_kill_switch_probe(store), *extra_kill_switch_probes]
    kill_switch = KillSwitch(*probes)

    def _eligibility():
        from app.governance.eligibility import (
            assess_live_trading_eligibility,
            gather_repo_evidence,
        )
        return assess_live_trading_eligibility(gather_repo_evidence())

    service = ExecutionService(
        broker=broker,
        order_manager=order_manager,
        trading_mode=mode,
        kill_switch=kill_switch,
        trading_gate=TradingGate(recovery),
        audit=audit,
        eligibility_provider=_eligibility,
        live_authorized=live_authorized,
    )

    startup_ok, reason = True, None
    if run_recovery:
        startup_ok, reason = await _run_startup_recovery(recovery, broker)

    return ExecutionStack(
        service=service,
        order_manager=order_manager,
        safety=safety,
        broker=broker,
        recovery=recovery,
        kill_switch_store=store,
        audit=audit,
        startup_ok=startup_ok,
        startup_reason=reason,
    )


async def _run_startup_recovery(
    recovery: RecoveryManager, broker,
) -> tuple[bool, Optional[str]]:
    """
    Run reconciliation. Any failure leaves the gate closed.

    Connectivity is established first: reconciling against a broker that was
    never connected produces empty broker state, and empty-versus-empty used
    to compare equal and report OK. That specific fail-open is the reason this
    step is explicit.
    """
    try:
        if hasattr(broker, "connect"):
            await broker.connect()
    except Exception as exc:
        msg = f"broker connect failed: {exc!r}"
        logger.error("startup_recovery_blocked: %s", msg)
        try:
            recovery.block(msg)
        except Exception:
            logger.exception("could not mark recovery blocked")
        return False, msg

    try:
        await recovery.recover(broker)
    except Exception as exc:
        msg = f"startup reconciliation raised: {exc!r}"
        logger.error("startup_recovery_blocked: %s", msg)
        return False, msg

    permitted = recovery.trading_permitted
    if not permitted:
        msg = recovery.blocked_reason or "reconciliation did not reach OK"
        logger.error(
            "TRADING BLOCKED AFTER STARTUP RECONCILIATION: %s. No orders will "
            "be accepted until this is resolved.", msg,
        )
        return False, msg

    logger.info("startup reconciliation OK — trading gate open")
    return True, None
