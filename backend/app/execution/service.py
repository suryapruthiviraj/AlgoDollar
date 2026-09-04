"""
The single authoritative order-submission path.

WHY THIS MODULE EXISTS
----------------------
Every safety component in this codebase worked and none of them was connected.
`ExecutionSafety`, `OrderManager`, `ReconciliationEngine`, `RecoveryManager`
and the eligibility gate all had passing tests, while a repository-wide search
for imports of `app.execution` or `app.broker` from anywhere else returned
nothing. The execution layer was unreachable dead code, which meant the safety
guarantees were real in isolation and vacuous in practice.

`ExecutionService` is the one place an order can be created. Everything else —
API routes, workers, strategy code — goes through it or does not trade.

THE ORDER OF CHECKS IS THE DESIGN
---------------------------------
Checks run cheapest-and-most-absolute first, so that the conditions which must
*never* be overridden are evaluated before anything that could itself fail:

    1. Kill switch          operator override; beats everything
    2. Trading gate         startup reconciliation must have succeeded
    3. Mode + authorization paper is default; live needs explicit eligibility
    4. Eligibility          evaluated always, enforced for live
    5. Risk + safety gates  delegated to OrderManager -> ExecutionSafety
    6. Broker submission    the only call site that reaches a broker

Every step can only ever *reduce* the set of permitted orders. There is no
branch anywhere in this file that turns a rejection into an approval.

FAIL CLOSED, INCLUDING ON OUR OWN BUGS
--------------------------------------
Any exception raised while evaluating a gate blocks the order. That covers
missing data, stale data, timeouts, unavailable dependencies and unknown
states — and also covers defects in this file. A safety check that cannot
complete has not passed.

PAPER IS THE DEFAULT AND THERE IS NO FALLBACK
---------------------------------------------
There is deliberately no path from a failed live submission to a paper one, or
from a failed paper submission to a live one. Such a fallback would silently
change which account is at risk at the worst possible moment.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from app.broker.base import BrokerInterface
from app.core.exceptions import (
    AlgoDollarError,
    AmbiguousOrderStateError,
    KillSwitchActiveError,
)
from app.execution.audit import (
    AuditJournal,
    ExecutionAuditRecord,
    ExecutionOutcome,
    InMemoryAuditSink,
)
from app.strategies.base import Signal

logger = logging.getLogger(__name__)


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class ExecutionBlocked(AlgoDollarError):
    """Raised when the execution boundary refuses to submit an order."""

    def __init__(self, outcome: ExecutionOutcome, reason: str,
                 record: Optional[ExecutionAuditRecord] = None) -> None:
        super().__init__(reason)
        self.outcome = outcome
        self.reason = reason
        self.record = record


@dataclass
class ExecutionResult:
    """Outcome of one submission attempt through the boundary."""
    outcome: ExecutionOutcome
    audit: ExecutionAuditRecord
    broker_order_id: Optional[str] = None
    reason: Optional[str] = None

    @property
    def submitted(self) -> bool:
        return self.outcome.reached_broker

    @property
    def blocked(self) -> bool:
        return not self.submitted


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

KillSwitchProbe = Callable[[], bool]


class KillSwitch:
    """
    Aggregates every kill-switch source. Any source saying stop means stop.

    The application previously had TWO unconnected switches: the API wrote
    `UserSettings.kill_switch_active` in the database, while `ExecutionSafety`
    read a `kill_switch` key in Redis. A user pressing the button in the UI did
    not stop the execution layer.

    Rather than pick a winner and silently ignore the other, this consults all
    of them. A source that raises counts as ACTIVE: a kill switch whose state
    cannot be read must be assumed engaged, because the alternative is trading
    through an outage in the one control designed to stop trading.
    """

    def __init__(self, *probes: KillSwitchProbe) -> None:
        self._probes = list(probes)

    def add_probe(self, probe: KillSwitchProbe) -> None:
        self._probes.append(probe)

    def is_active(self) -> tuple[bool, Optional[str]]:
        """Returns (active, reason). No probes configured is itself active."""
        if not self._probes:
            return True, (
                "no kill-switch source configured — refusing to trade without "
                "a working stop control"
            )
        for probe in self._probes:
            name = getattr(probe, "__name__", repr(probe))
            try:
                if probe():
                    return True, f"kill switch ACTIVE (source: {name})"
            except Exception as exc:
                return True, (
                    f"kill-switch source {name} could not be read ({exc!r}); "
                    f"treating as ACTIVE"
                )
        return False, None


# ---------------------------------------------------------------------------
# Trading gate
# ---------------------------------------------------------------------------

class TradingGate:
    """
    Whether startup reconciliation has succeeded.

    Defaults to closed. A process that has not reconciled against the broker
    does not know what it owns, and must not act on a portfolio it cannot
    describe.
    """

    def __init__(self, recovery_manager=None) -> None:
        self._recovery = recovery_manager

    def check(self) -> tuple[bool, Optional[str]]:
        if self._recovery is None:
            return False, (
                "no recovery manager wired — startup reconciliation has not "
                "run, so local state is unverified"
            )
        try:
            self._recovery.require_ready()
            return True, None
        except Exception as exc:
            return False, f"trading gate closed: {exc}"

    @property
    def state(self) -> str:
        if self._recovery is None:
            return "NO_RECOVERY_MANAGER"
        return str(getattr(self._recovery, "state", "UNKNOWN"))


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------

class ExecutionService:
    """
    The only supported way to place an order.

    Parameters
    ----------
    broker : BrokerInterface
        The broker orders are routed to. In paper mode this MUST be a
        PaperBroker; the constructor verifies it rather than trusting the
        caller, because a misconfiguration here trades real money.
    order_manager : OrderManager
        Owns idempotency, risk validation (via ExecutionSafety) and the
        broker call itself.
    trading_mode : TradingMode
    kill_switch : KillSwitch
    trading_gate : TradingGate
    audit : AuditJournal
    eligibility_provider : callable() -> EligibilityReport, optional
        Evaluated on every attempt. Enforced for live; recorded for paper.
    live_authorized : bool
        A second, explicit switch that must ALSO be true for live submission.
        Being eligible is not the same as being authorized.
    """

    def __init__(
        self,
        *,
        broker: BrokerInterface,
        order_manager,
        trading_mode: TradingMode = TradingMode.PAPER,
        kill_switch: Optional[KillSwitch] = None,
        trading_gate: Optional[TradingGate] = None,
        audit: Optional[AuditJournal] = None,
        eligibility_provider: Optional[Callable[[], Any]] = None,
        live_authorized: bool = False,
        persistence: Optional[Any] = None,
    ) -> None:
        self.mode = TradingMode(trading_mode)
        self.broker = broker
        self.order_manager = order_manager
        self.kill_switch = kill_switch or KillSwitch()
        self.trading_gate = trading_gate or TradingGate()
        self.audit = audit or AuditJournal(InMemoryAuditSink())
        self.eligibility_provider = eligibility_provider
        self.live_authorized = bool(live_authorized)
        # Durable record of every order, fill, position and cash movement.
        # Attached HERE, on the single authoritative path, so an order cannot
        # be placed without being recorded — a caller cannot forget, because a
        # caller cannot reach the broker any other way.
        #
        # None means "no durable record configured". That is legitimate for a
        # unit test, and it is why the durable-state requirement is enforced by
        # the RECONCILIATION gate rather than here: with no local state to
        # compare, reconciliation reports UNAVAILABLE and the trading gate stays
        # shut, so a production process cannot quietly trade unrecorded.
        self.persistence = persistence

        self._assert_broker_matches_mode()

    # -- construction-time safety ----------------------------------------

    def _assert_broker_matches_mode(self) -> None:
        """
        A paper-mode service must not hold a live broker, and vice versa.

        Checked once at construction rather than per order, so a
        misconfiguration fails at startup instead of at the first trade.
        """
        from app.broker.paper import PaperBroker

        is_paper_broker = isinstance(self.broker, PaperBroker)

        if self.mode is TradingMode.PAPER and not is_paper_broker:
            raise ExecutionBlocked(
                ExecutionOutcome.BLOCKED_MODE,
                f"trading_mode=paper but broker is {type(self.broker).__name__}, "
                f"not PaperBroker. Refusing to construct a service that would "
                f"route paper orders to a live venue.",
            )
        if self.mode is TradingMode.LIVE and is_paper_broker:
            raise ExecutionBlocked(
                ExecutionOutcome.BLOCKED_MODE,
                "trading_mode=live but broker is PaperBroker. Refusing to "
                "construct a service that reports paper fills as live.",
            )
        if self.mode is TradingMode.LIVE and not self.live_authorized:
            raise ExecutionBlocked(
                ExecutionOutcome.BLOCKED_MODE,
                "trading_mode=live requires live_authorized=True, set only by "
                "an explicit human decision.",
            )

    # -- the boundary ----------------------------------------------------

    async def submit_signal(
        self,
        signal: Signal,
        position_size: int,
        *,
        exchange: str = "NSE",
        product: Optional[str] = None,
        reference_price: float = 0.0,
        idempotency_key: Optional[str] = None,
        portfolio_allocation: Optional[dict[str, Any]] = None,
        **risk_context: Any,
    ) -> ExecutionResult:
        """
        Submit one signal. This is the ONLY path to a broker.

        Returns an ExecutionResult in every case, including blocks. Callers
        must inspect `.submitted`; a returned result is not an executed order.

        `risk_context` is forwarded to OrderManager.submit_order (available
        cash, portfolio value, current positions, limits, and so on). Risk
        limits are enforced there and in ExecutionSafety; this method does not
        re-implement them, it just guarantees they cannot be skipped.
        """
        rec = self.audit.new_record(self.mode.value)
        rec.symbol = getattr(signal, "symbol", None)
        rec.exchange = exchange
        rec.side = _side_of(signal)
        rec.quantity = int(position_size) if position_size is not None else None
        rec.product = product
        rec.strategy = getattr(signal, "strategy_name", None)
        rec.signal_edge_score = _f(getattr(signal, "edge_score", None))
        rec.expected_return = _f(getattr(signal, "expected_return", None))
        rec.expected_return_std = _f(getattr(signal, "expected_return_std", None))
        rec.stop_loss_pct = _f(getattr(signal, "stop_loss_pct", None))
        rec.target_pct = _f(getattr(signal, "target_pct", None))
        rec.signal_timestamp = _s(getattr(signal, "timestamp", None))
        rec.portfolio_allocation = portfolio_allocation
        meta = getattr(signal, "metadata", None) or {}
        rec.model_name = meta.get("model_name")
        rec.model_version = meta.get("model_version")

        try:
            self._check_kill_switch(rec)
            self._check_trading_gate(rec)
            self._check_mode_authorization(rec)
            self._check_eligibility(rec)
            return await self._submit_to_broker(
                signal, position_size, rec,
                exchange=exchange, product=product,
                reference_price=reference_price,
                idempotency_key=idempotency_key,
                **risk_context,
            )
        except ExecutionBlocked as blocked:
            rec.outcome = blocked.outcome.value
            rec.rejection_reason = blocked.reason
            self.audit.record(rec)
            return ExecutionResult(blocked.outcome, rec, reason=blocked.reason)
        except Exception as exc:
            # An unexpected failure anywhere above is a BLOCK, never a pass.
            rec.outcome = ExecutionOutcome.ERROR.value
            rec.error = repr(exc)
            rec.rejection_reason = f"unexpected error at execution boundary: {exc!r}"
            self.audit.record(rec)
            logger.exception("execution boundary raised; order blocked")
            return ExecutionResult(
                ExecutionOutcome.ERROR, rec, reason=rec.rejection_reason
            )

    # -- individual gates ------------------------------------------------

    def _check_kill_switch(self, rec: ExecutionAuditRecord) -> None:
        active, reason = self.kill_switch.is_active()
        rec.kill_switch_active = active
        if active:
            raise ExecutionBlocked(
                ExecutionOutcome.BLOCKED_KILL_SWITCH, reason or "kill switch active"
            )

    def _check_trading_gate(self, rec: ExecutionAuditRecord) -> None:
        ok, reason = self.trading_gate.check()
        rec.reconciliation_state = self.trading_gate.state
        if not ok:
            raise ExecutionBlocked(
                ExecutionOutcome.BLOCKED_NOT_RECONCILED,
                reason or "startup reconciliation has not succeeded",
            )

    def _check_mode_authorization(self, rec: ExecutionAuditRecord) -> None:
        """Paper needs nothing extra. Live needs an explicit human decision."""
        if self.mode is TradingMode.PAPER:
            return
        if not self.live_authorized:
            raise ExecutionBlocked(
                ExecutionOutcome.BLOCKED_MODE,
                "live trading is not authorized",
            )

    def _check_eligibility(self, rec: ExecutionAuditRecord) -> None:
        """
        Evaluate eligibility always; enforce it for live.

        Recording it in paper mode is the point: it makes the gap between "what
        we would need" and "what we have" visible on every single paper order,
        rather than only at the moment someone tries to go live.

        Enforcing LIVE_ELIGIBLE in paper mode would be wrong in the other
        direction — it would block all paper trading, which is the very
        activity that produces the evidence the gates require.
        """
        if self.eligibility_provider is None:
            rec.eligibility_state = "NOT_EVALUATED"
            rec.eligibility_permits_live = None
            if self.mode is TradingMode.LIVE:
                raise ExecutionBlocked(
                    ExecutionOutcome.BLOCKED_ELIGIBILITY,
                    "live mode requires an eligibility provider; none configured",
                )
            return

        try:
            report = self.eligibility_provider()
        except Exception as exc:
            rec.eligibility_state = "EVALUATION_FAILED"
            rec.eligibility_permits_live = False
            if self.mode is TradingMode.LIVE:
                raise ExecutionBlocked(
                    ExecutionOutcome.BLOCKED_ELIGIBILITY,
                    f"eligibility could not be evaluated ({exc!r}); failing closed",
                ) from exc
            return

        rec.eligibility_state = _report_state(report)
        permits = bool(getattr(report, "permits_live_trading", False))
        rec.eligibility_permits_live = permits
        try:
            rec.failed_gates = [
                r.name for r in getattr(report, "results", []) if not r.passed
            ] or None
        except Exception:
            rec.failed_gates = None

        if self.mode is TradingMode.LIVE:
            from app.governance.eligibility import require_live_eligible

            # Delegated rather than re-checked here: a second implementation of
            # the same rule is a second place for it to drift.
            require_live_eligible(report, action="place a live order")

    # -- broker submission -----------------------------------------------

    async def _lookup_decline_reason(
        self, client_order_id: Optional[str],
    ) -> tuple[Optional[str], Optional[list[str]]]:
        """
        Retrieve why OrderManager declined, by client order id.

        `OrderManager` records the failing gates on the order record before
        returning None (state RISK_REJECTED, reason = the joined failed
        checks). Looking the record up by its id recovers the ACTUAL cause.

        Returns (reason, failed_checks). Both are None when the cause cannot be
        established — the caller then says so rather than guessing. Reporting a
        plausible-but-wrong cause is worse than reporting an unknown one,
        because it sends whoever is investigating in the wrong direction. An
        earlier version reported "duplicate suppression" for every decline,
        including stale-data rejections.
        """
        if not client_order_id:
            return None, None

        store = getattr(self.order_manager, "_store", None) or getattr(
            self.order_manager, "store", None
        )
        if store is None or not hasattr(store, "get"):
            return None, None

        try:
            # OrderStore.get is async. An earlier version called it without
            # awaiting, so `record` was a coroutine object — truthy, with no
            # `reason` attribute — and this function could never return a
            # reason. Every decline was therefore reported as "no reason
            # recorded", which is exactly the misleading-audit failure this
            # lookup exists to prevent.
            record = store.get(client_order_id)
            if inspect.isawaitable(record):
                record = await record
        except Exception:
            return None, None
        if record is None:
            return None, None

        reason = getattr(record, "reason", None) or getattr(
            record, "reject_reason", None
        )
        if not reason:
            return None, None

        # OrderManager joins failed gate names with "; ".
        checks = [c.strip() for c in str(reason).split(";") if c.strip()]
        return str(reason), (checks or None)

    def _to_order_intent(
        self,
        signal: Signal,
        rec: ExecutionAuditRecord,
        *,
        exchange: str,
        product: Optional[str],
        quantity: int,
        reference_price: float = 0.0,
        idempotency_key: Optional[str] = None,
    ):
        """
        Translate an ALPHA signal into an ORDER intent.

        These are two genuinely different objects and the system has one of
        each: `app.strategies.base.Signal` says "RELIANCE should outperform
        over the next five days", while `app.execution.order_manager.Signal`
        says "BUY 10 RELIANCE on NSE, MIS, stop 2842". Nothing connected them,
        which is why the wired path failed on a missing `client_order_id` the
        first time it was run end to end.

        The translation is deliberately explicit rather than a field copy,
        because two decisions here have real consequences:

        * **Product.** An intraday strategy must trade MIS and a longer-horizon
          one CNC. Getting this wrong changes both the margin treatment and the
          cost schedule (MIS and CNC differ by roughly 2.5x per round trip).
        * **Stop price.** `stop_loss_pct` is a fraction; the broker needs an
          absolute trigger. This is the risk anchor the position sizer used, so
          it must be carried through rather than recomputed downstream — and
          without it, `OrderManager` treats the whole notional as at risk.
        """
        from app.broker.base import OrderType, Product, TransactionType
        from app.execution.lifecycle import deterministic_client_order_id
        from app.execution.order_manager import Signal as OrderSignal

        direction = str(getattr(getattr(signal, "direction", None), "value", "")).upper()
        txn = TransactionType.SELL if direction in {"SHORT", "EXIT"} else TransactionType.BUY

        strategy = (getattr(signal, "strategy_name", "") or "").lower()
        if product:
            prod = Product[product.upper()]
        else:
            prod = Product.MIS if strategy == "intraday" else Product.CNC

        # Absolute stop from the fractional stop the sizer used. Zero means
        # "no broker-side stop", which OrderManager treats as full-notional
        # risk — conservative, and the correct default when unknown.
        stop_pct = _f(getattr(signal, "stop_loss_pct", None)) or 0.0
        stop_price = 0.0
        if reference_price > 0 and stop_pct > 0:
            stop_price = (
                reference_price * (1 - stop_pct) if txn is TransactionType.BUY
                else reference_price * (1 + stop_pct)
            )

        rec.product = prod.value
        rec.side = txn.value

        # ── Idempotency key ────────────────────────────────────────────────
        #
        # This MUST be deterministic for a given logical intent. An earlier
        # version let OrderSignal mint a fresh random id on construction, so
        # every call — including a retry of the same submission — produced a
        # new key. Idempotency was therefore defeated at the boundary: N
        # submissions of one logical order created N broker orders, which is
        # the exact failure the client order id exists to prevent.
        #
        # The key is derived from what makes an intent distinct: instrument,
        # venue, side, size, product, strategy and the signal's own timestamp.
        # Two retries of one decision collapse to one key; two genuinely
        # different decisions do not collide, because the signal timestamp
        # differs.
        client_order_id = idempotency_key or deterministic_client_order_id(
            signal.symbol,
            exchange,
            txn.value,
            int(quantity),
            prod.value,
            getattr(signal, "strategy_name", "") or "",
            _s(getattr(signal, "signal_date", None))
            or _s(getattr(signal, "timestamp", None))
            or "",
            prefix="SIG",
        )
        rec.idempotency_key = client_order_id

        return OrderSignal(
            symbol=signal.symbol,
            exchange=exchange,
            txn_type=txn,
            order_type=OrderType.MARKET,
            product=prod,
            price=float(reference_price or 0.0),
            stop_price=float(stop_price),
            strategy=getattr(signal, "strategy_name", "") or "",
            client_order_id=client_order_id,
            intent_kind="exit" if direction == "EXIT" else "entry",
        )

    async def _submit_to_broker(
        self,
        signal: Signal,
        position_size: int,
        rec: ExecutionAuditRecord,
        *,
        exchange: str = "NSE",
        product: Optional[str] = None,
        reference_price: float = 0.0,
        idempotency_key: Optional[str] = None,
        **risk_context: Any,
    ) -> ExecutionResult:
        """
        The only call site in the application that reaches a broker.

        Risk validation lives inside OrderManager.submit_order, which consults
        ExecutionSafety. It is not duplicated here.
        """
        order_row_id: Optional[int] = None
        try:
            order_intent = self._to_order_intent(
                signal, rec, exchange=exchange, product=product,
                quantity=int(position_size),
                reference_price=reference_price,
                idempotency_key=idempotency_key,
            )

            # Claim the idempotency key BEFORE the broker is called. If this
            # order has been submitted before — a retry, a redelivered task, a
            # second worker racing on the same signal — we stop here rather
            # than sending a duplicate to the exchange. The claim is a UNIQUE
            # constraint, so two concurrent claimants cannot both win.
            #
            # A failure to claim BLOCKS. Not being able to prove this is not a
            # duplicate is not permission to assume it is not one.
            if self.persistence is not None:
                claim = await self.persistence.claim_idempotency_key(
                    getattr(order_intent, "client_order_id", None),
                    intent=order_intent,
                    quantity=int(position_size),
                    mode=self.mode.value,
                )
                order_row_id = claim.order_row_id
                if claim.duplicate:
                    rec.outcome = ExecutionOutcome.BLOCKED_DUPLICATE.value
                    rec.rejection_reason = (
                        f"duplicate submission: client_order_id "
                        f"{claim.client_order_id} was already submitted. No "
                        f"second order sent."
                    )
                    rec.idempotency_key = claim.client_order_id
                    self.audit.record(rec)
                    return ExecutionResult(
                        ExecutionOutcome.BLOCKED_DUPLICATE, rec,
                        reason=rec.rejection_reason,
                    )

            order_id = await self.order_manager.submit_order(
                order_intent, int(position_size), self.broker, **risk_context
            )
        except KillSwitchActiveError as exc:
            rec.outcome = ExecutionOutcome.BLOCKED_KILL_SWITCH.value
            rec.kill_switch_active = True
            rec.rejection_reason = str(exc)
            await self._persist_block(order_row_id, "BLOCKED_KILL_SWITCH", str(exc))
            self.audit.record(rec)
            return ExecutionResult(
                ExecutionOutcome.BLOCKED_KILL_SWITCH, rec, reason=str(exc)
            )
        except AmbiguousOrderStateError as exc:
            # The broker may or may not have the order. Never retried here.
            rec.outcome = ExecutionOutcome.AMBIGUOUS.value
            rec.rejection_reason = str(exc)
            rec.final_state = "UNKNOWN"
            # UNKNOWN is written DURABLY, not just to the audit log. This is the
            # one state that must survive the process: on the next start,
            # reconciliation has to know there is an order whose fate was never
            # established, so it can ask the broker by client_order_id instead
            # of assuming either outcome.
            await self._persist_block(order_row_id, "UNKNOWN", str(exc))
            self.audit.record(rec)
            logger.error(
                "AMBIGUOUS order state for %s — reconciliation required before "
                "any further action on this order.", rec.symbol,
            )
            return ExecutionResult(
                ExecutionOutcome.AMBIGUOUS, rec, reason=str(exc)
            )
        except Exception as exc:
            rec.outcome = ExecutionOutcome.BLOCKED_RISK.value
            rec.risk_checks_passed = False
            rec.rejection_reason = f"{type(exc).__name__}: {exc}"
            await self._persist_block(
                order_row_id, "BLOCKED_RISK", rec.rejection_reason
            )
            self.audit.record(rec)
            return ExecutionResult(
                ExecutionOutcome.BLOCKED_RISK, rec, reason=rec.rejection_reason
            )

        if order_id is None:
            # OrderManager declined without raising. The specific cause — a
            # duplicate, a stale tick, a failed gate — was recorded on the
            # order record before it returned, so it can be recovered rather
            # than guessed at.
            reason, checks = await self._lookup_decline_reason(
                getattr(order_intent, "client_order_id", None)
            )
            rec.outcome = ExecutionOutcome.BLOCKED_RISK.value
            rec.risk_checks_passed = False
            rec.failed_risk_checks = checks
            rec.rejection_reason = reason or (
                "order manager declined to submit and recorded no reason; see "
                "order_manager logs for the specific gate"
            )
            await self._persist_block(
                order_row_id, "BLOCKED_RISK", rec.rejection_reason
            )
            self.audit.record(rec)
            return ExecutionResult(
                ExecutionOutcome.BLOCKED_RISK, rec, reason=rec.rejection_reason
            )

        rec.risk_checks_passed = True
        rec.broker_order_id = str(order_id)
        rec.idempotency_key = rec.idempotency_key or str(order_id)
        rec.outcome = ExecutionOutcome.SUBMITTED.value
        rec.final_state = "SUBMITTED"

        # Ask the broker what it actually did, then persist THAT. An accepted
        # order is not a filled order, so nothing here is inferred from the
        # fact that submit_order returned without raising.
        if self.persistence is not None and order_row_id is not None:
            try:
                await self.persistence.attach_broker_order_id(order_row_id, str(order_id))
                sync = await self.persistence.sync_from_broker(
                    order_row_id, self.broker, str(order_id), mode=self.mode.value,
                )
                rec.final_state = sync.status or rec.final_state
                rec.fill_quantity = sync.filled_quantity
                rec.average_fill_price = sync.average_fill_price
                if sync.errors:
                    # The money may already have moved. Surfacing this is what
                    # lets reconciliation find the divergence; swallowing it is
                    # what would hide it.
                    rec.error = "; ".join(sync.errors)[:2000]
                    logger.error(
                        "order %s submitted but its outcome was not fully "
                        "recorded: %s", order_id, rec.error,
                    )
            except Exception as exc:  # noqa: BLE001
                rec.error = f"post-submission persistence failed: {exc!r}"
                logger.exception(
                    "order %s reached the broker but could not be recorded", order_id
                )

        self.audit.record(rec)
        return ExecutionResult(
            ExecutionOutcome.SUBMITTED, rec, broker_order_id=str(order_id)
        )

    async def _persist_block(
        self, order_row_id: Optional[int], state: str, reason: str
    ) -> None:
        """Record a blocked order durably. Never raises into the order path."""
        if self.persistence is None or order_row_id is None:
            return
        try:
            await self.persistence.record_block(order_row_id, state, reason)
        except Exception as exc:  # noqa: BLE001
            logger.error("could not persist block for order %s: %s", order_row_id, exc)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _f(v) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _s(v) -> Optional[str]:
    return None if v is None else str(v)


def _side_of(signal: Signal) -> Optional[str]:
    d = getattr(signal, "direction", None)
    if d is None:
        return None
    val = getattr(d, "value", d)
    return {"LONG": "BUY", "SHORT": "SELL", "EXIT": "SELL"}.get(str(val), str(val))


def _report_state(report) -> Optional[str]:
    state = getattr(report, "state", None)
    if state is None:
        return None
    return str(getattr(state, "value", state)).upper()
