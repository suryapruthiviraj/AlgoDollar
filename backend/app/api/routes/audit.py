"""
Execution audit: what the system decided, and WHY.

This exists because "No trade" is not an answer. Every execution attempt —
including every refusal — produces an audit record carrying the specific gate
that stopped it, and until now none of that was reachable from the API. The UI
could only say that nothing happened, which is indistinguishable from a quiet
market, a broken feed, an engaged kill switch and a breached sector limit.

So the contract here is that a rejection is reported as:

    RELIANCE BUY x12 rejected — sector exposure limit (ENERGY at 25% cap)

and never as an absence.

WHERE THE RECORDS COME FROM
---------------------------
The in-memory sink attached to the running execution stack, which holds this
process's decisions. The JSONL sink on disk is the durable record and survives
restarts; it is read as a fallback when the in-memory sink is empty, so a
freshly restarted process can still explain what it did yesterday.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app.core.config import settings

logger = structlog.get_logger(__name__)
router = APIRouter()


#: The VERB only. The specific reason is appended separately, so the headline
#: reads "RELIANCE BUY x10 rejected — sector exposure limit" rather than
#: repeating the category and the reason back to back.
_OUTCOME_VERB = {
    "SUBMITTED": "submitted",
    "BLOCKED_KILL_SWITCH": "blocked",
    "BLOCKED_ELIGIBILITY": "blocked",
    "BLOCKED_RISK": "rejected",
    "BLOCKED_NOT_RECONCILED": "blocked",
    "BLOCKED_MODE": "blocked",
    "BLOCKED_DUPLICATE": "suppressed",
    "AMBIGUOUS": "UNKNOWN",
    "ERROR": "errored",
}

#: Used only when no more specific reason could be extracted.
_OUTCOME_FALLBACK = {
    "BLOCKED_KILL_SWITCH": "kill switch engaged",
    "BLOCKED_ELIGIBILITY": "live-trading eligibility not met",
    "BLOCKED_RISK": "a risk check failed",
    "BLOCKED_NOT_RECONCILED": "startup reconciliation has not succeeded",
    "BLOCKED_MODE": "trading mode not authorised",
    "BLOCKED_DUPLICATE": "an identical order was already submitted",
    "AMBIGUOUS": "the broker outcome could not be determined",
    "ERROR": "an error at the execution boundary",
}

#: Risk-check identifiers mapped to plain language. A UI showing
#: `SECTOR_EXPOSURE_EXCEEDED` to a person has not explained anything.
_RISK_CHECK_LABEL = {
    "SECTOR_EXPOSURE": "sector exposure limit",
    "SECTOR_EXPOSURE_EXCEEDED": "sector exposure limit",
    "MAX_SECTOR_PCT": "sector exposure limit",
    "POSITION_SIZE": "position size limit",
    "MAX_POSITION_PCT": "single-position size limit",
    "MAX_POSITIONS": "maximum open positions reached",
    "INSUFFICIENT_CASH": "insufficient cash",
    "INSUFFICIENT_HOLDINGS": "insufficient holdings to sell",
    "DAILY_LOSS_LIMIT": "daily loss limit reached",
    "DAILY_RISK_BUDGET": "daily risk budget exhausted",
    "MAX_DRAWDOWN": "portfolio drawdown limit",
    "STALE_DATA": "market data too old to price the order",
    "STALE_TICK": "market data too old to price the order",
    "LIQUIDITY": "insufficient liquidity for this size",
    "MAX_PARTICIPATION": "order too large for the traded volume",
    "MARKET_CLOSED": "market closed",
    "SQUAREOFF_WINDOW": "inside the intraday square-off window",
    "DUPLICATE_ORDER": "duplicate of an order already working",
    "INSTRUMENT_INVALID": "instrument not tradeable",
    "RISK_LIMIT": "daily risk limit",
    "SHORT_SELL_NOT_SUPPORTED": "short selling is not supported",
    "INSUFFICIENT_FUNDS": "insufficient funds at the broker",
    "MARKET_CLOSED_BROKER": "market closed",
    "CASH": "insufficient cash",
    "HOLDINGS": "insufficient holdings to sell",
    "POSITION_LIMIT": "maximum open positions reached",
    "MARKET_HOURS": "market closed",
    "DATA_FRESHNESS": "market data too old to price the order",
    "KILL_SWITCH": "kill switch engaged",
}


class AuditEntry(BaseModel):
    """One execution decision, phrased so a person can act on it."""

    audit_id: str
    timestamp: str
    trading_mode: str
    symbol: Optional[str] = None
    side: Optional[str] = None
    quantity: Optional[int] = None
    strategy: Optional[str] = None

    outcome: str
    submitted: bool
    #: "RELIANCE BUY x12 rejected — sector exposure limit"
    headline: str
    #: The specific gate, in plain language.
    reason: Optional[str] = None
    #: The raw reason exactly as the engine recorded it, kept so nothing is lost
    #: in translation and an operator can search for it.
    raw_reason: Optional[str] = None
    #: The numeric specifics behind the reason, casing preserved — e.g.
    #: "Daily risk limit would be breached: used Rs 1 + this Rs 500 > max Rs 1."
    detail: Optional[str] = None
    failed_checks: list[str] = []
    failed_gates: list[str] = []

    kill_switch_active: Optional[bool] = None
    reconciliation_state: Optional[str] = None
    eligibility_state: Optional[str] = None

    broker_order_id: Optional[str] = None
    fill_quantity: Optional[int] = None
    average_fill_price: Optional[float] = None
    intended_notional: Optional[float] = None
    edge_score: Optional[float] = None
    expected_return: Optional[float] = None
    portfolio_allocation: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class AuditResponse(BaseModel):
    entries: list[AuditEntry] = []
    total: int = 0
    submitted: int = 0
    rejected: int = 0
    source: str = "none"
    #: Set when the audit trail itself is unavailable. Distinguishing "nothing
    #: was decided" from "we cannot see what was decided" is the entire point.
    unavailable_reason: Optional[str] = None


def _check_identifier(check: str) -> str:
    """
    The gate name out of a composite check string.

    The safety layer records checks as ``"risk_limit: Daily risk limit would be
    breached: used Rs 1 ..."`` — an identifier, a colon, then detail. Matching
    the whole string against the label table always missed, so every rejection
    fell through to the raw text.
    """
    return str(check).split(":", 1)[0].strip().upper().replace(" ", "_")


def _check_detail(check: str) -> Optional[str]:
    """The human detail after the identifier, with its original casing kept."""
    parts = str(check).split(":", 1)
    return parts[1].strip() if len(parts) == 2 and parts[1].strip() else None


def _broker_reason(record: dict) -> Optional[str]:
    """
    The broker's own refusal reason, translated where we recognise it.

    Read from `broker_response`, which is what the venue actually said —
    distinct from our pre-trade checks, which never ran for an order the broker
    itself turned down.
    """
    resp = record.get("broker_response") or {}
    if not isinstance(resp, dict):
        return None
    raw = (
        resp.get("reject_reason")
        or resp.get("status_message")
        or resp.get("message")
    )
    if not raw:
        return None
    key = str(raw).strip().upper().replace(" ", "_")
    if key in _RISK_CHECK_LABEL:
        return _RISK_CHECK_LABEL[key]
    return str(raw)


def _plain_reason(record: dict) -> Optional[str]:
    """
    The most specific human-readable reason available.

    Preference order: a named risk check, then the kill switch, then
    reconciliation, then a failed eligibility gate, then the engine's own text.

    Casing is preserved. An earlier version lowercased the whole string, which
    turned "used Rs 1" into "used rs 1" — mangling currency in a message an
    operator is meant to act on.
    """
    checks = record.get("failed_risk_checks") or []
    for c in checks:
        key = _check_identifier(c)
        if key in _RISK_CHECK_LABEL:
            return _RISK_CHECK_LABEL[key]
    if checks:
        ident = _check_identifier(checks[0])
        return ident.replace("_", " ").lower()

    if record.get("kill_switch_active"):
        return "kill switch engaged"

    state = record.get("reconciliation_state")
    if state and str(state).upper() not in ("RECONCILIATION_OK", "OK"):
        return f"reconciliation state is {state}"

    gates = record.get("failed_gates") or []
    if gates:
        return f"eligibility gate: {gates[0]}"

    raw = record.get("rejection_reason")
    return str(raw) if raw else _OUTCOME_FALLBACK.get(str(record.get("outcome") or ""))


def _headline(record: dict, reason: Optional[str]) -> str:
    sym = record.get("symbol") or "?"
    side = record.get("side") or ""
    qty = record.get("quantity")
    outcome = str(record.get("outcome") or "")
    verb = _OUTCOME_VERB.get(outcome, outcome.lower().replace("_", " "))

    lead = f"{sym} {side}".strip()
    if qty:
        lead = f"{lead} x{qty}"

    if outcome == "SUBMITTED":
        # An order can reach the broker and be refused THERE. The execution
        # outcome is still SUBMITTED — it did reach the venue — but reporting
        # that alone would tell an operator the order went through when the
        # broker turned it down. The broker's own final state wins the headline.
        final = str(record.get("final_state") or "").upper()
        if final in ("REJECTED", "CANCELLED", "EXPIRED"):
            broker_reason = _broker_reason(record)
            word = "rejected by broker" if final == "REJECTED" else final.lower()
            return f"{lead} {word}" + (f" — {broker_reason}" if broker_reason else "")
        oid = record.get("broker_order_id")
        filled = record.get("fill_quantity")
        if filled is not None and qty and 0 < int(filled) < int(qty):
            return f"{lead} partially filled {filled}/{qty}" + (
                f" (order {oid})" if oid else ""
            )
        return f"{lead} submitted" + (f" (order {oid})" if oid else "")
    # Exactly one reason clause: the verb says WHAT happened, the reason says
    # WHY. Appending both a category and a reason repeated the same words twice.
    return f"{lead} {verb}" + (f" — {reason}" if reason else "")


def _to_entry(record: dict) -> AuditEntry:
    outcome = str(record.get("outcome") or "ERROR")
    reason = _plain_reason(record)
    return AuditEntry(
        audit_id=str(record.get("audit_id") or ""),
        timestamp=str(record.get("timestamp") or ""),
        trading_mode=str(record.get("trading_mode") or "unknown"),
        symbol=record.get("symbol"),
        side=record.get("side"),
        quantity=record.get("quantity"),
        strategy=record.get("strategy"),
        outcome=outcome,
        submitted=outcome == "SUBMITTED",
        headline=_headline(record, reason),
        reason=reason,
        raw_reason=record.get("rejection_reason"),
        detail=next(
            (d for d in (_check_detail(c) for c in (record.get("failed_risk_checks") or [])) if d),
            None,
        ),
        failed_checks=[str(c) for c in (record.get("failed_risk_checks") or [])],
        failed_gates=[str(g) for g in (record.get("failed_gates") or [])],
        kill_switch_active=record.get("kill_switch_active"),
        reconciliation_state=record.get("reconciliation_state"),
        eligibility_state=record.get("eligibility_state"),
        broker_order_id=record.get("broker_order_id"),
        fill_quantity=record.get("fill_quantity"),
        average_fill_price=record.get("average_fill_price"),
        intended_notional=record.get("intended_notional"),
        edge_score=record.get("signal_edge_score"),
        expected_return=record.get("expected_return"),
        portfolio_allocation=record.get("portfolio_allocation"),
        error=record.get("error"),
    )


def _records_from_memory(request: Request) -> Optional[list[dict]]:
    stack = getattr(request.app.state, "execution_stack", None)
    if stack is None or getattr(stack, "audit", None) is None:
        return None
    out: list[dict] = []
    for sink in stack.audit.sinks:
        records = getattr(sink, "records", None)
        if records is None:
            continue
        for r in records:
            out.append(r.to_dict() if hasattr(r, "to_dict") else dict(vars(r)))
    return out or None


def _records_from_disk(limit: int) -> Optional[list[dict]]:
    """
    Tail the durable JSONL sink.

    Read from the end so a long-running deployment does not load its whole
    history to show the last twenty decisions.
    """
    path = settings.execution_audit_path
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        lines = p.read_text().splitlines()
    except Exception as exc:  # noqa: BLE001
        logger.warning("audit_jsonl_unreadable", error=str(exc))
        return None
    out: list[dict] = []
    for line in lines[-(limit * 3):]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # A truncated final line is normal if the process died mid-write.
            # Skip it rather than failing the whole request.
            continue
    return out or None


@router.get("", response_model=AuditResponse)
async def get_audit(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    rejected_only: bool = Query(False),
    symbol: Optional[str] = Query(None),
    strategy: Optional[str] = Query(None),
) -> AuditResponse:
    """
    Recent execution decisions, newest first, each with the reason it happened.

    `rejected_only=true` returns just the refusals — the view that answers "why
    isn't it trading?", which is the question an empty trade list cannot.
    """
    records = _records_from_memory(request)
    source = "memory"
    if not records:
        records = _records_from_disk(limit)
        source = "jsonl" if records else "none"

    if not records:
        stack = getattr(request.app.state, "execution_stack", None)
        return AuditResponse(
            entries=[], total=0, submitted=0, rejected=0, source="none",
            unavailable_reason=(
                "No execution stack is running, so no decisions can be read."
                if stack is None else
                "No execution attempt has been recorded yet. This means nothing "
                "has been attempted — NOT that attempts were made and refused."
            ),
        )

    entries = [_to_entry(r) for r in records]
    if symbol:
        entries = [e for e in entries if (e.symbol or "").upper() == symbol.upper()]
    if strategy:
        entries = [e for e in entries if (e.strategy or "") == strategy]

    submitted = sum(1 for e in entries if e.submitted)
    rejected = len(entries) - submitted

    if rejected_only:
        entries = [e for e in entries if not e.submitted]

    entries.sort(key=lambda e: e.timestamp, reverse=True)
    return AuditResponse(
        entries=entries[:limit],
        total=len(records),
        submitted=submitted,
        rejected=rejected,
        source=source,
    )


@router.get("/{audit_id}", response_model=AuditEntry)
async def get_audit_entry(audit_id: str, request: Request) -> AuditEntry:
    """One decision in full, for drilling into a specific refusal."""
    records = _records_from_memory(request) or _records_from_disk(500) or []
    for r in records:
        if str(r.get("audit_id")) == audit_id:
            return _to_entry(r)
    raise HTTPException(status_code=404, detail=f"No audit record {audit_id}")
