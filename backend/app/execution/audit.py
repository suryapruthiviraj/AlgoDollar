"""
Execution audit journal.

WHY THIS EXISTS
---------------
Every order attempt must leave a record, including — especially — the ones
that never reached a broker. A journal that only records successful trades
cannot answer the question that actually matters after an incident: *why did
the system do that?* Rejections, risk vetoes and eligibility blocks are the
most informative entries in it.

TWO RULES
---------
1. **Never fabricate a field.** A value that was not available is recorded as
   `None`, not as zero, not as an empty string, and not as a plausible guess.
   `None` means "not known"; `0.0` means "measured as zero". Conflating them
   makes the journal actively misleading.

2. **Writing the journal must never block a safety decision.** If persistence
   fails, the failure is logged and the order still proceeds through its
   normal path — but the failure is surfaced, never swallowed silently. An
   audit sink that can veto trading is a new outage mode; an audit sink that
   fails quietly is a compliance hole. Neither is acceptable, so the journal
   records the sink failure itself.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class ExecutionOutcome(str, Enum):
    """Terminal classification of an execution attempt."""
    SUBMITTED = "SUBMITTED"                    # reached the broker
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED_BY_BROKER = "REJECTED_BY_BROKER"
    BLOCKED_KILL_SWITCH = "BLOCKED_KILL_SWITCH"
    BLOCKED_ELIGIBILITY = "BLOCKED_ELIGIBILITY"
    BLOCKED_RISK = "BLOCKED_RISK"
    BLOCKED_NOT_RECONCILED = "BLOCKED_NOT_RECONCILED"
    BLOCKED_MODE = "BLOCKED_MODE"
    BLOCKED_DUPLICATE = "BLOCKED_DUPLICATE"
    AMBIGUOUS = "AMBIGUOUS"                    # broker outcome unknown
    ERROR = "ERROR"

    @property
    def reached_broker(self) -> bool:
        return self in {
            ExecutionOutcome.SUBMITTED, ExecutionOutcome.FILLED,
            ExecutionOutcome.PARTIALLY_FILLED,
            ExecutionOutcome.REJECTED_BY_BROKER, ExecutionOutcome.AMBIGUOUS,
        }


@dataclass
class ExecutionAuditRecord:
    """
    One execution attempt, from intent to terminal outcome.

    Every field that was not observed is None. Downstream consumers must treat
    None as "unknown", never as a default value.
    """
    # --- identity -------------------------------------------------------
    audit_id: str
    idempotency_key: Optional[str]
    timestamp: str                          # ISO-8601 UTC
    trading_mode: str                       # "paper" | "live"

    # --- intent ---------------------------------------------------------
    symbol: Optional[str] = None
    exchange: Optional[str] = None
    side: Optional[str] = None
    quantity: Optional[int] = None
    order_type: Optional[str] = None
    product: Optional[str] = None

    # --- provenance -----------------------------------------------------
    strategy: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    signal_edge_score: Optional[float] = None
    expected_return: Optional[float] = None
    expected_return_std: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    target_pct: Optional[float] = None
    signal_timestamp: Optional[str] = None

    # --- decisioning ----------------------------------------------------
    intended_notional: Optional[float] = None
    trade_risk: Optional[float] = None
    portfolio_allocation: Optional[dict[str, Any]] = None
    eligibility_state: Optional[str] = None
    eligibility_permits_live: Optional[bool] = None
    failed_gates: Optional[list[str]] = None
    risk_checks_passed: Optional[bool] = None
    failed_risk_checks: Optional[list[str]] = None
    kill_switch_active: Optional[bool] = None
    reconciliation_state: Optional[str] = None

    # --- execution ------------------------------------------------------
    broker_order_id: Optional[str] = None
    broker_response: Optional[dict[str, Any]] = None
    fill_quantity: Optional[int] = None
    average_fill_price: Optional[float] = None
    costs: Optional[float] = None
    slippage: Optional[float] = None

    # --- outcome --------------------------------------------------------
    outcome: str = ExecutionOutcome.ERROR.value
    rejection_reason: Optional[str] = None
    final_state: Optional[str] = None
    error: Optional[str] = None
    sink_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, sort_keys=True)

    @property
    def reached_broker(self) -> bool:
        try:
            return ExecutionOutcome(self.outcome).reached_broker
        except ValueError:
            return False

    def summary(self) -> str:
        qty = "?" if self.quantity is None else self.quantity
        return (
            f"[{self.outcome}] {self.side or '?'} {qty} {self.symbol or '?'} "
            f"({self.trading_mode}) "
            f"{'-> ' + self.broker_order_id if self.broker_order_id else ''}"
            f"{' | ' + self.rejection_reason if self.rejection_reason else ''}"
        )


class AuditSink(Protocol):
    """Anywhere an audit record can be durably written."""

    def write(self, record: ExecutionAuditRecord) -> None: ...


class InMemoryAuditSink:
    """
    Non-durable sink for tests and for local paper runs.

    Deliberately NOT the default in any live configuration: an audit trail that
    vanishes on restart is not an audit trail.
    """

    def __init__(self) -> None:
        self.records: list[ExecutionAuditRecord] = []

    def write(self, record: ExecutionAuditRecord) -> None:
        self.records.append(record)

    # -- query helpers used by tests and the API -------------------------

    def by_outcome(self, outcome: ExecutionOutcome) -> list[ExecutionAuditRecord]:
        return [r for r in self.records if r.outcome == outcome.value]

    def reached_broker(self) -> list[ExecutionAuditRecord]:
        return [r for r in self.records if r.reached_broker]

    def blocked(self) -> list[ExecutionAuditRecord]:
        return [r for r in self.records if not r.reached_broker]


class JsonlAuditSink:
    """Append-only JSON Lines sink. One record per line, never rewritten."""

    def __init__(self, path) -> None:
        from pathlib import Path
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: ExecutionAuditRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")


class AuditJournal:
    """
    Records execution attempts to one or more sinks.

    Multiple sinks are supported so a durable store and an in-memory view can
    coexist. A sink that raises does not prevent the others from being written
    and does not propagate — but the failure is attached to the record and
    logged at ERROR, so a silently broken audit pipeline is still visible.
    """

    def __init__(self, *sinks: AuditSink) -> None:
        self._sinks: list[AuditSink] = list(sinks) or [InMemoryAuditSink()]

    @property
    def sinks(self) -> list[AuditSink]:
        return list(self._sinks)

    def add_sink(self, sink: AuditSink) -> None:
        self._sinks.append(sink)

    @staticmethod
    def new_record(
        trading_mode: str, idempotency_key: Optional[str] = None,
    ) -> ExecutionAuditRecord:
        return ExecutionAuditRecord(
            audit_id=str(uuid.uuid4()),
            idempotency_key=idempotency_key,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            trading_mode=trading_mode,
        )

    def record(self, rec: ExecutionAuditRecord) -> ExecutionAuditRecord:
        errors: list[str] = []
        for sink in self._sinks:
            try:
                sink.write(rec)
            except Exception as exc:
                errors.append(f"{type(sink).__name__}: {exc!r}")
                logger.error(
                    "AUDIT SINK FAILURE — execution record may not be durably "
                    "stored. sink=%s error=%r record=%s",
                    type(sink).__name__, exc, rec.summary(),
                )
        if errors:
            rec.sink_error = "; ".join(errors)

        log = logger.warning if not rec.reached_broker else logger.info
        log("execution_audit %s", rec.summary())
        return rec
