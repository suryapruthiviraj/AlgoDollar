"""
app/engine/equity.py — intraday equity (P&L) tracking and high-water mark.

O2 slice: give the daily-loss and drawdown gates numbers instead of "no
evidence recorded".  Each session minute the paper book is marked to market,
the running peak (high-water mark) is kept, the drawdown off it and the day's
loss are derived, and :func:`check_all_limits` is evaluated against every
snapshot so the nearest daily-loss / drawdown breach is RECORDED even when
nobody is watching live.

The tracker only RECORDS.  It never halts or trades — deciding to act on a
breach stays the kill switch's job, so recording cannot be mistaken for
enforcement.  See the O2 note in docs/ROADMAP.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence

from app.risk.limits import (
    LimitBreached,
    RiskLimits,
    RiskState,
    check_all_limits,
)


@dataclass(frozen=True)
class EquitySnapshot:
    """One minute's marked-to-market equity during a session."""

    minute: str             # "0931" .. "1530" (IST)
    ts: str                 # ISO timestamp (IST, minutes)
    cash: float             # free cash held by the broker
    position_value: float   # sum(qty * live price) for open positions
    equity: float           # cash + position_value
    peak_equity: float      # high-water mark up to and including this minute
    drawdown_pct: float     # (peak - equity) / peak, 0.0 when flat at peak
    daily_loss_rupees: float  # non-negative loss magnitude vs session base

    def to_dict(self) -> dict:
        return {
            "minute": self.minute,
            "ts": self.ts,
            "cash": round(self.cash, 2),
            "position_value": round(self.position_value, 2),
            "equity": round(self.equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "drawdown_pct": round(self.drawdown_pct, 6),
            "daily_loss_rupees": round(self.daily_loss_rupees, 2),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EquitySnapshot":
        return cls(
            minute=str(d["minute"]),
            ts=str(d["ts"]),
            cash=float(d["cash"]),
            position_value=float(d["position_value"]),
            equity=float(d["equity"]),
            peak_equity=float(d["peak_equity"]),
            drawdown_pct=float(d["drawdown_pct"]),
            daily_loss_rupees=float(d["daily_loss_rupees"]),
        )


@dataclass
class EquityTracker:
    """
    Marks the paper book to market minute by minute.

    ``base_equity`` is the session's reference point — the equity of the FIRST
    snapshot, so a resumed campaign whose books open with carried positions is
    not penalised before it trades.  ``daily_loss`` is the positive magnitude
    of ``base_equity - equity`` and is 0.0 on a profitable session (mirrors
    ``normalize_loss``).
    """

    cash_start: float = 0.0
    snapshots: List[EquitySnapshot] = field(default_factory=list)
    base_equity: Optional[float] = None

    def record(
        self,
        ts: datetime,
        cash: float,
        positions: Sequence[dict],
        prices: dict,
    ) -> EquitySnapshot:
        """
        Record one marked-to-market snapshot.

        Positions with ``quantity != 0`` MUST have a live price in ``prices``;
        silently valuing a live position at zero or average cost is exactly the
        D5 defect this subsystem exists to prevent, so it raises instead.
        """
        position_value = 0.0
        for p in positions or []:
            qty = int(p.get("quantity", 0) or 0)
            if qty == 0:
                continue
            sym = str(p.get("symbol", ""))
            px = prices.get(sym)
            if px is None or not (px > 0):
                raise ValueError(
                    f"No live price for open position {sym} (qty {qty}); "
                    "refusing to value an open book at zero or average cost."
                )
            position_value += qty * float(px)

        equity = float(cash) + position_value
        running_peak = max(peak(self.snapshots) or 0.0, equity)
        drawdown = (running_peak - equity) / running_peak if running_peak > 0 else 0.0

        if self.base_equity is None:
            self.base_equity = equity
        base = self.base_equity
        daily_loss = max(0.0, base - equity)

        snap = EquitySnapshot(
            minute=ts.strftime("%H%M"),
            ts=ts.isoformat(),
            cash=cash,
            position_value=position_value,
            equity=equity,
            peak_equity=running_peak,
            drawdown_pct=max(0.0, drawdown),
            daily_loss_rupees=daily_loss,
        )
        self.snapshots.append(snap)
        return snap

    # ------------------------------------------------------------------ #
    #  Derived quantities                                                 #
    # ------------------------------------------------------------------ #

    @property
    def start_equity(self) -> Optional[float]:
        if not self.snapshots:
            return None
        return self.snapshots[0].equity

    @property
    def end_equity(self) -> Optional[float]:
        if not self.snapshots:
            return None
        return self.snapshots[-1].equity

    @property
    def peak_equity(self) -> Optional[float]:
        return peak(self.snapshots)

    @property
    def min_equity(self) -> Optional[float]:
        if not self.snapshots:
            return None
        return min(s.equity for s in self.snapshots)

    @property
    def max_drawdown_pct(self) -> Optional[float]:
        if not self.snapshots:
            return None
        return max(s.drawdown_pct for s in self.snapshots)

    @property
    def max_daily_loss_rupees(self) -> float:
        if not self.snapshots:
            return 0.0
        return max(s.daily_loss_rupees for s in self.snapshots)


def peak(snapshots: Sequence[EquitySnapshot]) -> Optional[float]:
    """Running peak of a snapshot series."""
    if not snapshots:
        return None
    return max(s.peak_equity for s in snapshots)


def risk_breaches_for(
    snapshot: EquitySnapshot,
    limits: Optional[RiskLimits] = None,
) -> List[LimitBreached]:
    """Evaluate daily-loss / drawdown RiskLimits against one snapshot."""
    state = RiskState(
        daily_loss=snapshot.daily_loss_rupees,
        current_portfolio_value=snapshot.equity,
        peak_portfolio_value=snapshot.peak_equity,
        current_drawdown=snapshot.drawdown_pct,
    )
    return check_all_limits(state, limits if limits is not None else RiskLimits())


def worst_risk(
    snapshots: Sequence[EquitySnapshot],
    limits: Optional[RiskLimits] = None,
) -> dict:
    """
    The worst daily-loss / drawdown posture seen across a session's snapshots.

    :func:`check_all_limits` is evaluated on every snapshot and the summary
    keeps the maximum loss, maximum drawdown and the set of limit names that
    ever breached.  Purely advisory: it is written to the campaign ledger, so
    a breach is EVIDENCE, and acting on it remains enforcement.
    """
    limits = limits if limits is not None else RiskLimits()
    breached_names: set[str] = set()
    max_loss = 0.0
    max_dd = 0.0
    for snap in snapshots:
        max_loss = max(max_loss, snap.daily_loss_rupees)
        max_dd = max(max_dd, snap.drawdown_pct)
        for b in risk_breaches_for(snap, limits):
            breached_names.add(b.limit_name)
    return {
        "daily_loss_max_rupees": round(max_loss, 2),
        "drawdown_max_pct": round(max_dd, 6),
        "breached": bool(breached_names),
        "breaches": sorted(breached_names),
        "limit_daily_loss_rupees": float(limits.max_daily_loss),
        "limit_drawdown_pct": float(limits.max_drawdown_pct),
    }
