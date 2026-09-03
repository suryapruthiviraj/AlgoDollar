"""
Restart recovery — the gate between "process started" and "process may trade".

A trading process that restarts mid-flight can be in any of a dozen states:
an order was about to be sent, was sent but the response was lost, was
acknowledged but not written down, partially filled, fully filled, or was
being cancelled.  Until the local record and the broker agree, the process
does not know its own position and MUST NOT trade.

Startup state machine
---------------------
    (process start) --> BLOCKED
                          |  recover()
                          v
                      RECOVERING
                       /        \\
        reconciliation OK    anything else
                     /            \\
                    v              v
                 READY           BLOCKED

* The initial state is BLOCKED, not "unknown" — a process that has never run
  recovery may not trade.
* READY is reachable ONLY through :meth:`RecoveryManager.recover` ending in
  ``RECONCILIATION_OK``.  There is no other transition into READY, and
  ``_enter_ready`` re-asserts that invariant before flipping the flag.
* Once BLOCKED after a failed recovery, only another successful recovery can
  clear it.

Recovery sequence (in order)
----------------------------
    1. load persistent local state
    2. query broker
    3. reconcile orders
    4. reconcile positions
    5. reconcile cash
    6. identify unknown / ambiguous orders
    7. identify mismatches (classify)
    8. resolve safely
    9. permit trading — only if everything reconciled
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Optional, Sequence

from ..broker.base import BrokerInterface
from .reconciliation import (
    Discrepancy,
    DiscrepancyKind,
    KillSwitchActivationError,
    LocalStateStore,
    ReconciliationEngine,
    ReconciliationResult,
    ReconciliationSnapshot,
    ReconciliationStatus,
    classify_order_status,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  States                                                                      #
# --------------------------------------------------------------------------- #

class StartupState(str, Enum):
    RECOVERING = "RECOVERING"
    READY = "READY"
    BLOCKED = "BLOCKED"

    def __bool__(self) -> bool:
        """Only READY may ever read as truthy."""
        return self is StartupState.READY

    @property
    def permits_trading(self) -> bool:
        return self is StartupState.READY


class RecoveryPhase(str, Enum):
    LOAD_LOCAL_STATE = "load_local_state"
    QUERY_BROKER = "query_broker"
    RECONCILE_ORDERS = "reconcile_orders"
    RECONCILE_POSITIONS = "reconcile_positions"
    RECONCILE_CASH = "reconcile_cash"
    IDENTIFY_UNKNOWN_ORDERS = "identify_unknown_orders"
    IDENTIFY_MISMATCHES = "identify_mismatches"
    RESOLVE = "resolve_safely"
    VERIFY = "verify_reconciliation"
    PERMIT_TRADING = "permit_trading"


class TradingBlockedError(RuntimeError):
    """Raised when trading is attempted before recovery reached READY."""


# --------------------------------------------------------------------------- #
#  Report types                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class PhaseOutcome:
    phase: RecoveryPhase
    ok: bool
    detail: str = ""
    discrepancies: list[Discrepancy] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.phase.value}: {'OK' if self.ok else 'FAILED'} — {self.detail}"


@dataclass
class RecoveryReport:
    state: StartupState
    status: ReconciliationStatus
    phases: list[PhaseOutcome] = field(default_factory=list)
    result: Optional[ReconciliationResult] = None
    unresolved: list[Discrepancy] = field(default_factory=list)
    resolutions: list[str] = field(default_factory=list)
    blocked_reason: str = ""
    kill_switch_activated: bool = False
    kill_switch_error: Optional[str] = None

    @property
    def trading_permitted(self) -> bool:
        """True only when recovery ended in READY via a successful reconciliation."""
        return (
            self.state is StartupState.READY
            and self.status is ReconciliationStatus.RECONCILIATION_OK
        )

    def __bool__(self) -> bool:
        return self.trading_permitted

    def phase(self, phase: RecoveryPhase) -> Optional[PhaseOutcome]:
        for p in self.phases:
            if p.phase is phase:
                return p
        return None

    def summary(self) -> str:
        return (
            f"state={self.state.value} status={self.status.value} "
            f"trading_permitted={self.trading_permitted} "
            f"unresolved={len(self.unresolved)}"
            + (f" reason={self.blocked_reason}" if self.blocked_reason else "")
        )


#: A resolver may attempt safe repairs.  It receives the broker and the failed
#: reconciliation result and returns human-readable descriptions of what it
#: actually did.  Returning an empty list means "nothing was resolved".
Resolver = Callable[
    [BrokerInterface, ReconciliationResult], Awaitable[Sequence[str]]
]


# --------------------------------------------------------------------------- #
#  RecoveryManager                                                             #
# --------------------------------------------------------------------------- #

class RecoveryManager:
    """
    Drives startup recovery and owns the trading permission flag.

    Nothing in the execution path should consult reconciliation directly;
    it should ask this object (``trading_permitted`` / ``require_ready()``).
    """

    def __init__(
        self,
        engine: ReconciliationEngine,
        *,
        local_state: Optional[LocalStateStore] = None,
        resolver: Optional[Resolver] = None,
        disambiguate_unknown_orders: bool = True,
    ) -> None:
        self._engine = engine
        self._local_state = local_state
        self._resolver = resolver
        self._disambiguate = disambiguate_unknown_orders
        # Fail closed: a process that has not recovered may not trade.
        self._state: StartupState = StartupState.BLOCKED
        self._last_report: Optional[RecoveryReport] = None
        self._blocked_reason = "recovery has not run yet"

    # ------------------------------------------------------------------ #
    #  Permission surface                                                  #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> StartupState:
        return self._state

    @property
    def trading_permitted(self) -> bool:
        return self._state is StartupState.READY

    @property
    def blocked_reason(self) -> str:
        return "" if self.trading_permitted else self._blocked_reason

    @property
    def last_report(self) -> Optional[RecoveryReport]:
        return self._last_report

    def require_ready(self) -> None:
        """Raise unless recovery has completed with RECONCILIATION_OK."""
        if not self.trading_permitted:
            raise TradingBlockedError(
                f"Trading is blocked (state={self._state.value}): {self._blocked_reason}"
            )

    # ------------------------------------------------------------------ #
    #  The recovery sequence                                              #
    # ------------------------------------------------------------------ #

    async def recover(
        self,
        broker: BrokerInterface,
        db_session=None,
        *,
        local_state: Optional[LocalStateStore] = None,
        raise_on_kill_switch_failure: bool = True,
    ) -> RecoveryReport:
        """
        Run the full startup sequence.

        Returns a :class:`RecoveryReport`.  Reconciliation failures do NOT
        raise — they end in ``StartupState.BLOCKED`` — because a restart must
        always reach a defined safe state.  A failure to ACTIVATE the kill
        switch does propagate (after the state is already BLOCKED), because
        silently continuing would mean reporting a safety action that never
        happened.
        """
        logger.info("Startup recovery: begin.")
        self._state = StartupState.RECOVERING
        self._blocked_reason = "recovery in progress"
        phases: list[PhaseOutcome] = []

        try:
            snap = ReconciliationSnapshot()

            # 1. load persistent local state -----------------------------
            store = self._engine.resolve_local_state(
                db_session, local_state or self._local_state, snap
            )
            await self._engine.fetch_local(store, snap)
            local_missing = [n for n in snap.unavailable if n.startswith("local_")]
            phases.append(PhaseOutcome(
                RecoveryPhase.LOAD_LOCAL_STATE,
                ok=not local_missing,
                detail=(
                    "local state loaded"
                    if not local_missing
                    else f"local state UNAVAILABLE: {sorted(local_missing)}"
                ),
            ))

            # 2. query broker --------------------------------------------
            await self._engine.fetch_broker(broker, snap)
            broker_missing = [n for n in snap.unavailable if n.startswith("broker_")]
            phases.append(PhaseOutcome(
                RecoveryPhase.QUERY_BROKER,
                ok=not broker_missing,
                detail=(
                    "broker queried"
                    if not broker_missing
                    else f"broker UNREACHABLE: {sorted(broker_missing)}"
                ),
            ))

            # 3-6. the four comparisons ----------------------------------
            discrepancies: list[Discrepancy] = []
            for phase, fn in (
                (RecoveryPhase.RECONCILE_ORDERS, self._engine.reconcile_orders),
                (RecoveryPhase.RECONCILE_POSITIONS, self._engine.reconcile_positions),
                (RecoveryPhase.RECONCILE_CASH, self._engine.reconcile_cash),
                (RecoveryPhase.IDENTIFY_UNKNOWN_ORDERS,
                 self._engine.identify_unknown_orders),
            ):
                found = list(fn(snap))
                discrepancies.extend(found)
                phases.append(PhaseOutcome(
                    phase,
                    ok=not found,
                    detail=(
                        "clean" if not found
                        else f"{len(found)} finding(s): "
                             + "; ".join(str(d) for d in found[:5])
                    ),
                    discrepancies=found,
                ))

            # 7. identify mismatches (classify) --------------------------
            result = self._engine.classify(snap, discrepancies)
            phases.append(PhaseOutcome(
                RecoveryPhase.IDENTIFY_MISMATCHES,
                ok=result.ok,
                detail=result.summary(),
                discrepancies=list(result.discrepancies),
            ))

            # 8. resolve safely ------------------------------------------
            resolutions: list[str] = []
            if not result.ok:
                resolutions = await self._resolve_safely(broker, result)
            phases.append(PhaseOutcome(
                RecoveryPhase.RESOLVE,
                ok=result.ok or bool(resolutions),
                detail=(
                    "nothing to resolve" if result.ok
                    else (
                        "; ".join(resolutions) if resolutions
                        else "NOTHING could be resolved automatically"
                    )
                ),
            ))

            # 8b. re-verify: resolution alone never unlocks trading -------
            if resolutions:
                verified = await self._engine.evaluate(
                    broker,
                    db_session,
                    local_state=local_state or self._local_state,
                )
                phases.append(PhaseOutcome(
                    RecoveryPhase.VERIFY,
                    ok=verified.ok,
                    detail=f"re-reconciled after resolution: {verified.summary()}",
                    discrepancies=list(verified.discrepancies),
                ))
                result = verified

        except KillSwitchActivationError as exc:
            # Defensive: block first, then let the (critical) failure propagate.
            self._state = StartupState.BLOCKED
            self._blocked_reason = str(exc)
            raise
        except Exception as exc:  # noqa: BLE001 -- any failure => BLOCKED
            logger.exception("Startup recovery raised; blocking trading.")
            result = ReconciliationResult(
                status=ReconciliationStatus.RECONCILIATION_ERROR,
                recommendation="Recovery itself failed.",
                error=f"{type(exc).__name__}: {exc}",
            )
            phases.append(PhaseOutcome(
                RecoveryPhase.IDENTIFY_MISMATCHES,
                ok=False,
                detail=f"recovery aborted: {type(exc).__name__}: {exc}",
            ))
            resolutions = []

        # 9. permit trading — only if everything reconciled ---------------
        return self._finish(result, phases, resolutions, raise_on_kill_switch_failure)

    # ------------------------------------------------------------------ #
    #  Step 8 — safe resolution                                            #
    # ------------------------------------------------------------------ #

    async def _resolve_safely(
        self,
        broker: BrokerInterface,
        result: ReconciliationResult,
    ) -> list[str]:
        """
        Attempt only resolutions that cannot themselves cause a trade.

        Built in: re-query the broker for each ambiguous order id
        (``get_order_status``) to convert "unknown" into a definite state.
        This is read-only.  Anything that would place, cancel or amend an
        order, or rewrite local records, must be supplied explicitly as a
        ``resolver`` — and even then the result is re-reconciled afterwards,
        so a resolver cannot by itself unlock trading.
        """
        actions: list[str] = []

        if self._disambiguate:
            unknown = result.of_kind(DiscrepancyKind.UNKNOWN_ORDER_STATE)
            for d in unknown:
                if not d.order_id:
                    logger.error(
                        "Ambiguous order for %s has no broker id — cannot be "
                        "disambiguated automatically.", d.symbol,
                    )
                    continue
                try:
                    status = await broker.get_order_status(d.order_id)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Could not disambiguate order %s: %s", d.order_id, exc
                    )
                    continue
                resolved_status = (status or {}).get("status")
                if classify_order_status(resolved_status) == "unknown":
                    logger.error(
                        "Broker also reports order %s as %r — still ambiguous.",
                        d.order_id, resolved_status,
                    )
                    continue
                actions.append(
                    f"order {d.order_id} disambiguated at broker as {resolved_status}"
                )

        if self._resolver is not None:
            try:
                extra = await self._resolver(broker, result)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Custom resolver failed.")
                actions.append(f"resolver FAILED: {type(exc).__name__}: {exc}")
            else:
                actions.extend(str(a) for a in (extra or []))

        return actions

    # ------------------------------------------------------------------ #
    #  Step 9 — transition                                                 #
    # ------------------------------------------------------------------ #

    def _finish(
        self,
        result: ReconciliationResult,
        phases: list[PhaseOutcome],
        resolutions: list[str],
        raise_on_kill_switch_failure: bool,
    ) -> RecoveryReport:
        report = RecoveryReport(
            state=StartupState.BLOCKED,
            status=result.status,
            phases=phases,
            result=result,
            unresolved=list(result.discrepancies),
            resolutions=list(resolutions),
        )

        if result.ok:
            self._enter_ready(result)
            report.state = StartupState.READY
            report.unresolved = []
            phases.append(PhaseOutcome(
                RecoveryPhase.PERMIT_TRADING,
                ok=True,
                detail="reconciliation OK — trading permitted",
            ))
            self._last_report = report
            logger.info("Startup recovery: %s", report.summary())
            return report

        # Not OK: stay blocked, then try to persist the halt.
        self._state = StartupState.BLOCKED
        self._blocked_reason = result.recommendation or result.summary()
        report.blocked_reason = self._blocked_reason
        phases.append(PhaseOutcome(
            RecoveryPhase.PERMIT_TRADING,
            ok=False,
            detail=f"TRADING BLOCKED — {report.blocked_reason}",
        ))
        self._last_report = report
        logger.critical("Startup recovery: %s", report.summary())

        try:
            self._engine.enforce(result, raise_on_failure=False)
            report.kill_switch_activated = result.kill_switch_activated
        except KillSwitchActivationError as exc:
            # The halt could not be persisted.  Never pretend it was.
            report.kill_switch_activated = False
            report.kill_switch_error = str(exc)
            logger.critical("Startup recovery: %s", exc)
            if raise_on_kill_switch_failure:
                raise
        return report

    def _enter_ready(self, result: ReconciliationResult) -> None:
        """The ONLY transition into READY."""
        if result.status is not ReconciliationStatus.RECONCILIATION_OK:
            raise AssertionError(
                "READY is reachable only via RECONCILIATION_OK; "
                f"refusing transition from {result.status.value}"
            )
        self._state = StartupState.READY
        self._blocked_reason = ""
        logger.info("Startup recovery: state READY — trading permitted.")

    # ------------------------------------------------------------------ #
    #  Re-blocking                                                         #
    # ------------------------------------------------------------------ #

    def block(self, reason: str) -> None:
        """Force the process back to BLOCKED (e.g. a periodic pass failed)."""
        self._state = StartupState.BLOCKED
        self._blocked_reason = reason
        logger.critical("Trading BLOCKED: %s", reason)


__all__ = [
    "PhaseOutcome",
    "RecoveryManager",
    "RecoveryPhase",
    "RecoveryReport",
    "Resolver",
    "StartupState",
    "TradingBlockedError",
]
