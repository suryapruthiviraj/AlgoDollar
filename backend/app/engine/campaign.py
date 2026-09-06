"""
app/engine/campaign.py — offline multi-day PAPER campaign.

This is the O1 artefact: it turns "zero days run" into a persisted,
restartable multi-day paper run through the EXACT production pipeline the
live trader uses (PaperBroker -> ExecutionService -> recovery/reconciliation
-> strategy -> IntradayAutoTrader).

What makes it a *campaign* rather than a replay:

* **Continuous book** — the paper broker is built over a ``state_path``, so
  cash and positions carry across days exactly as a live restart would.
* **Crash-safe ledger** — every completed day is appended to a JSONL ledger
  the moment it finishes; a killed process loses nothing.
* **Restart / resume** — a later ``run_campaign`` call with the same ledger
  and paper state picks up at the next day, reports how many previous days
  were already completed, and *reconciles the persisted paper book against
  the ledger before trading a single bar*.  A disagreement or an unreadable
  book stops the campaign cold (fail-closed) and is recorded, never papered
  over.
* **Honest evidence** — the summary is operational only: days completed,
  restarts resumed, reconciliation verdicts, trade counts and realised
  rupee P&L against cost-aware paper fills.  Synthetic bars are flagged on
  every public surface and DSR/PBO are deliberately NOT computed on a
  synthetic campaign (that analysis lives in quant/ on real data).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from app.broker.paper import PaperBroker
from app.broker.tickdata import TickQuoteSource, to_ist
from app.engine.equity import EquityTracker, worst_risk
from app.engine.replay import (
    ReplayClock,
    SessionResult,
    _build_stack,
    _prepare_aggregator,
    _recent_sessions,
    build_strategy_config,
    drive_session,
    load_bars_from_csv,
    sessions_in,
    synthetic_bar_source,
)
from app.execution.file_stores import FileKillSwitchStore, FileOrderStore
from app.risk.limits import RiskLimits
from app.strategies.intraday import IntradayStrategy

logger = logging.getLogger(__name__)

CAMPAIGN_NOTE = (
    "Operational paper-run evidence ONLY — days run, restart recovery, "
    "reconciliation verdicts, cost-aware fills. For synthetic bars this is "
    "NOT a performance claim; DSR/PBO are computed only on real data in "
    "quant/experiments."
)


def _paper_metrics(records: Sequence[dict]) -> dict:
    """Operational paper metrics from the persisted intraday equity curves.

    Curve peak, drawdown and Sharpe come from the marked-to-market equity
    tracker embedded in each day record.  This makes the daily-loss / drawdown
    gates MEASURABLE on a running campaign (the O2 slice); it is not a
    performance claim, and the ``synthetic`` flag on the summary keeps that
    distinction explicit.
    """
    days = len(records)
    day_loss_max = 0.0
    drawdown_max = 0.0
    risk_breached = False
    risk_halts = 0
    points: list[float] = []
    returns: list[float] = []
    for r in records:
        eq = r.get("equity") or {}
        if eq.get("start") is None or eq.get("end") is None:
            continue
        start = float(eq["start"])
        end = float(eq["end"])
        if start <= 0:
            continue
        points.append(end)
        returns.append((end - start) / start)
        day_loss_max = max(day_loss_max, float(eq.get("daily_loss_rupees", 0.0) or 0.0))
        drawdown_max = max(drawdown_max, float(eq.get("max_drawdown_pct", 0.0) or 0.0))
        risk = r.get("risk") or {}
        if bool(risk.get("breached", False)):
            risk_breached = True
        if bool(risk.get("enforced_halted", False)):
            risk_halts += 1

    base = {  # nothing measurable yet
        "paper_trading_days": days,
        "paper_peak_equity_rupees": None,
        "paper_max_drawdown_pct": None,
        "paper_daily_loss_max_rupees": None,
        "paper_drawdown_limit_breached": None,
        "paper_sharpe": None,
        "paper_risk_halts": 0,
    }
    if not points:
        return base

    curve_peak = points[0]
    curve_dd = 0.0
    for e in points:
        curve_peak = max(curve_peak, e)
        if curve_peak > 0:
            curve_dd = max(curve_dd, (curve_peak - e) / curve_peak)

    sharpe: Optional[float] = None
    if len(returns) >= 2:
        sd = statistics.stdev(returns) if len(returns) > 1 else 0.0
        if sd > 0:
            sharpe = round(statistics.mean(returns) / sd * math.sqrt(252.0), 4)

    return {
        "paper_trading_days": days,
        "paper_peak_equity_rupees": round(float(max(points)), 2),
        "paper_max_drawdown_pct": round(max(curve_dd, drawdown_max), 6),
        "paper_daily_loss_max_rupees": round(day_loss_max, 2),
        "paper_drawdown_limit_breached": risk_breached,
        "paper_sharpe": sharpe,
        "paper_risk_halts": risk_halts,
    }


@dataclass
class PaperCampaignConfig:
    symbols: List[str]
    opening_cash: float = 1_000_000.0
    seed: int = 0
    source: str = "synthetic"  # "synthetic" | "csv" | "custom"
    data_dir: Optional[str] = None
    frames: Optional[Mapping[str, pd.DataFrame]] = None  # "custom" (tests)
    sessions: Optional[Sequence[date]] = None
    strategy_config: Optional[Mapping[str, Any]] = None
    risk_limits: Optional[Mapping[str, Any]] = None


def _campaign_limits(limits: Optional[Mapping[str, Any]]) -> Optional[RiskLimits]:
    """Build :class:`RiskLimits` from the config mapping, if enforcement is on.

    ``None`` means recording-only: the tracker records daily-loss / drawdown
    breaches but nothing is acted on.  Providing a mapping opts the campaign
    into enforcement — a HARD breach engages the durable kill switch.
    """
    if limits is None:
        return None
    known = {f for f in RiskLimits.__dataclass_fields__}
    unknown = sorted(set(limits) - known)
    if unknown:
        raise ValueError(f"unknown risk limit(s): {', '.join(unknown)}")
    return RiskLimits(**{k: v for k, v in limits.items() if k in known})


class _CampaignLocalState:
    """Local-state double backed by the campaign ledger.

    The ledger IS the campaign's own bookkeeping record: it mirrors the
    broker's full book (cash, positions, orders, fills) as of the last
    completed day — exactly what the ORM-backed local store would hold in
    production.  Reconciling the freshly-loaded paper book against it at
    every restart is precisely the invariant that proves the book survived
    the gap.  A forged or drifted ledger therefore blocks the campaign cold.
    """

    def __init__(self, opening_cash: float, records: Sequence[dict]) -> None:
        last = records[-1] if records else None
        self.cash = (
            float(last["cash_end"])
            if last and last.get("cash_end") is not None
            else float(opening_cash)
        )
        self._positions = list((last or {}).get("positions", []))
        self._orders = list((last or {}).get("orders", []))
        self._trades = list((last or {}).get("fills", []))

    async def get_positions(self) -> list[dict]:
        return list(self._positions)

    async def get_orders(self) -> list[dict]:
        return list(self._orders)

    async def get_trades(self) -> list[dict]:
        return list(self._trades)

    async def get_cash(self) -> dict:
        return {"cash": self.cash}


# --------------------------------------------------------------------------- #
#  Ledger persistence (JSONL — one completed day per line)                     #
# --------------------------------------------------------------------------- #

def load_ledger(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def day_records(raw: Sequence[dict]) -> List[dict]:
    """Only completed-days out of a raw ledger (runs vs. days are both stored)."""
    return [r for r in raw if r.get("kind", "day") == "day"]


def run_start_count(raw: Sequence[dict]) -> int:
    return sum(1 for r in raw if r.get("kind") == "run_start")


def _append_record(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


# --------------------------------------------------------------------------- #
#  Day planning / bar slicing                                                  #
# --------------------------------------------------------------------------- #

def plan_days(config: PaperCampaignConfig, total: int) -> List[date]:
    """The campaign's FULL schedule.

    Returns the whole configured session list (or ``total`` recent business
    days) rather than slicing it: bars must be generated over the complete
    schedule so a resumed run produces bit-identical data to a contiguous
    one — the synthetic generator is length-dependent for symbols after the
    first.  ``run_campaign`` iterates the schedule from ``prior:total``.
    """
    if config.sessions is not None and len(config.sessions):
        plan = sorted({d for d in config.sessions})
    else:
        plan = _recent_sessions(total)
    if len(plan) < total:
        raise ValueError(
            f"campaign of {total} days needs {total} sessions; "
            f"only {len(plan)} available (source={config.source})"
        )
    return plan


def build_bar_source(
    config: PaperCampaignConfig, days: Sequence[date],
) -> Mapping[str, pd.DataFrame]:
    if config.source == "synthetic":
        return synthetic_bar_source(config.symbols, list(days), seed=config.seed)
    if config.source == "csv":
        if not config.data_dir:
            raise ValueError("source='csv' requires data_dir")
        frames = load_bars_from_csv(config.data_dir, config.symbols)
        if not frames:
            raise ValueError(f"no bar files found under {config.data_dir}")
        return frames
    if config.source == "custom":
        if config.frames is None:
            raise ValueError("source='custom' requires frames")
        return dict(config.frames)
    raise ValueError(f"unknown campaign source: {config.source}")


def _bars_for_day(
    frames: Mapping[str, pd.DataFrame], day: date,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sym, df in frames.items():
        if df is None or df.empty or "time" not in df.columns:
            continue
        parsed = pd.to_datetime(df["time"])
        keep = [to_ist(ts.to_pydatetime()).date() == day for ts in parsed]
        sub = df.loc[keep]
        if not sub.empty:
            out[sym] = sub
    return out


# --------------------------------------------------------------------------- #
#  The campaign                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class CampaignResult:
    config_snapshot: Dict[str, Any]
    ledger_path: str
    paper_state_path: str
    planned_days: int
    records: List[dict] = field(default_factory=list)
    run_starts: int = 0
    skipped_days: int = 0
    stop_reason: Optional[str] = None
    stopped_after_days: int = 0
    kill_switch_engaged: bool = False
    kill_switch_reason: Optional[str] = None

    @property
    def synthetic(self) -> bool:
        return bool(self.config_snapshot.get("source") == "synthetic")

    @property
    def completed(self) -> bool:
        return self.stop_reason is None and len(self.records) >= self.planned_days

    def summary(self) -> Dict[str, Any]:
        recs = self.records
        sources = sorted({r.get("source") for r in recs}) or ["none"]
        net = sum(float(r.get("net_pnl_rupees", 0.0) or 0.0) for r in recs)
        gross = sum(float(r.get("gross_pnl_rupees", 0.0) or 0.0) for r in recs)
        costs = sum(float(r.get("costs_rupees", 0.0) or 0.0) for r in recs)
        trades = sum(int(r.get("trades", 0)) for r in recs)
        recon_ok = sum(1 for r in recs if bool(r.get("recovery_ok", False)))
        day_errors = sum(1 for r in recs if r.get("errors"))
        paper = _paper_metrics(recs)
        return {
            "campaign": "offline_paper",
            "source": sources[0] if len(sources) == 1 else sources,
            "synthetic": bool(self.config_snapshot.get("source") == "synthetic"),
            "planned_days": self.planned_days,
            "completed_days": len(recs),
            "skipped_days": self.skipped_days,
            "restarts_resumed": max(self.run_starts - 1, 0),
            "runs_total": self.run_starts,
            "reconciliation_ok_days": recon_ok,
            "reconciliation_failed_days": len(recs) - recon_ok,
            "days_with_errors": day_errors,
            "trades": trades,
            "gross_pnl_rupees": round(gross, 2),
            "total_costs_rupees": round(costs, 2),
            "net_pnl_rupees": round(net, 2),
            **paper,
            "stopped": self.stop_reason is not None,
            "stop_reason": self.stop_reason,
            "kill_switch_engaged": self.kill_switch_engaged,
            "kill_switch_reason": self.kill_switch_reason,
            "ledger_path": self.ledger_path,
            "paper_state_path": self.paper_state_path,
            "note": CAMPAIGN_NOTE,
        }

    def write_summary(self, prefix: str) -> str:
        path = f"{prefix}_summary.json"
        with open(path, "w") as fh:
            json.dump(self.summary(), fh, indent=2, default=str)
        return path


def _ledger_positions(positions: Sequence[Mapping[str, Any]]) -> List[dict]:
    """Normalise paper position dicts into ledger-serialisable maps."""
    out = []
    for p in positions:
        if p is None:
            continue
        if int(p.get("quantity", 0) or 0) == 0:
            continue
        out.append({
            "symbol": str(p.get("symbol")),
            "exchange": str(p.get("exchange") or "NSE"),
            "quantity": int(p.get("quantity", 0)),
            "average_price": float(p.get("average_price", 0.0) or 0.0),
            "product": str(p.get("product") or "MIS"),
            "strategy": str(p.get("strategy") or "intraday"),
        })
    return out


def _equity_from_session(result: SessionResult) -> dict:
    snaps = [s.to_dict() for s in result.equity_snapshots]
    return {
        "start": round(result.start_equity, 2) if result.start_equity is not None else None,
        "end": round(result.end_equity, 2) if result.end_equity is not None else None,
        "peak": round(result.peak_equity, 2) if result.peak_equity is not None else None,
        "min": round(result.min_equity, 2) if result.min_equity is not None else None,
        "max_drawdown_pct": round(result.max_drawdown_pct, 6)
        if result.max_drawdown_pct is not None else None,
        "daily_loss_rupees": round(result.daily_loss_rupees, 2)
        if result.daily_loss_rupees is not None else None,
        "snapshots": snaps,
    }


def _record_from_session(
    result: SessionResult, day: date, *, source: str, resume_after: int,
    end_book: Mapping[str, Sequence[Mapping[str, Any]]],
    enforcement: Optional[Mapping[str, Any]] = None,
) -> dict:
    net = sum(t.net_pnl_rupees for t in result.trades)
    gross = sum(t.pnl_rupees for t in result.trades)
    costs = sum(t.costs_rupees for t in result.trades)
    risk = worst_risk(result.equity_snapshots)
    risk["enforced_halted"] = bool((enforcement or {}).get("halted", False))
    risk["enforced_halt_reason"] = (enforcement or {}).get("reason")
    return {
        "kind": "day",
        "day": day.isoformat(),
        "source": source,
        "resume_after": resume_after,
        "ts": datetime.now(timezone.utc).isoformat(),
        "cash_start": round(result.cash_start, 2),
        "cash_end": round(result.cash_end, 2),
        "equity_return_pct": round(result.equity_return_pct, 4),
        "entries": result.entries,
        "exits": result.exits,
        "stops_placed": result.stops_placed,
        "stops_cancelled": result.stops_cancelled,
        "trades": len(result.trades),
        "net_pnl_rupees": round(net, 2),
        "gross_pnl_rupees": round(gross, 2),
        "costs_rupees": round(costs, 2),
        "open_at_end": result.open_at_end,
        "positions": _ledger_positions(end_book.get("positions", [])),
        "orders": list(end_book.get("orders", [])),
        "fills": list(end_book.get("trades", [])),
        "recovery_ok": result.recovery_ok,
        "reconciliation": result.recovery_detail,
        "errors": result.errors,
        "rejected": result.rejected,
        "equity": _equity_from_session(result),
        "risk": risk,
    }


async def _run_campaign_day(
    bar_source: Mapping[str, pd.DataFrame],
    day: date,
    symbols: Sequence[str],
    *,
    config: PaperCampaignConfig,
    ledger_records: Sequence[dict],
    paper_state_path: str,
    kill_switch_store: Any,
    order_store: Any,
    risk_limits: Optional[RiskLimits],
    on_breach: Optional[Callable[[Sequence[Any]], None]],
) -> tuple[SessionResult, Dict[str, List[dict]]]:
    aggregator = _prepare_aggregator(bar_source, day, symbols)
    local = _CampaignLocalState(config.opening_cash, ledger_records)
    clock = ReplayClock(day)
    broker = PaperBroker(
        data_broker=TickQuoteSource(aggregator),
        initial_cash=config.opening_cash,
        clock=clock,
        state_path=paper_state_path,
    )
    stack = _build_stack(
        broker, cash=local.cash, local_state=local,
        kill_switch_store=kill_switch_store,
        order_store=order_store,
    )
    strategy = IntradayStrategy(paper_mode=True,
                                **(build_strategy_config(config.strategy_config)))
    tracker = EquityTracker(cash_start=float(local.cash))
    completed = False
    try:
        result = await drive_session(
            bar_source,
            day=day,
            symbols=symbols,
            aggregator=aggregator,
            broker=broker,
            stack=stack,
            strategy=strategy,
            clock=clock,
            collect_reports=False,
            equity_tracker=tracker,
            risk_limits=risk_limits,
            on_breach=on_breach,
        )
        completed = True
        end_book = {
            "positions": _ledger_positions(await broker.get_positions()),
            "orders": await broker.get_orders(),
            "trades": await broker.get_trades(),
        }
        return result, end_book
    finally:
        # Only persist the book when the session actually ran.  A failed
        # connect (corrupt/unreadable state) must never trigger _save_state,
        # which would overwrite the evidence with a fresh empty book.
        if completed:
            await broker.disconnect()


async def run_campaign(
    total_days: int,
    config: PaperCampaignConfig,
    *,
    ledger_path: str,
    paper_state_path: str,
) -> CampaignResult:
    build_strategy_config(config.strategy_config)
    days = plan_days(config, total_days)
    bar_source = build_bar_source(config, days)

    records = day_records(load_ledger(ledger_path))
    run_starts = run_start_count(load_ledger(ledger_path))
    prior = len(records)
    if prior >= total_days:
        logger.warning("campaign already complete (%d/%d days)", prior, total_days)
        return CampaignResult(
            config_snapshot=asdict(config),
            ledger_path=ledger_path,
            paper_state_path=paper_state_path,
            planned_days=total_days,
            records=records,
            run_starts=run_starts,
        )

    # Durable stores are keyed to THIS book (not just its directory): a
    # restarted campaign overlaps the same book, day for day, with the same
    # deterministic entry ids — exactly the case the order store must remember
    # — while a *different* book sharing the directory is a different campaign
    # and must get its own stores.  Keying on the book filename keeps restart
    # protection and campaign isolation at the same time.
    book_name = os.path.basename(os.path.abspath(paper_state_path)) or "book.json"
    durable_dir = os.path.dirname(os.path.abspath(paper_state_path))
    kill_store = FileKillSwitchStore(
        os.path.join(durable_dir, f"{book_name}.kill_switch.json")
    )
    order_store = FileOrderStore(
        os.path.join(durable_dir, f"{book_name}.orders.json")
    )

    # A persisted halt must stop the campaign before it serves a single bar —
    # the durable switch is the memory a restart otherwise loses.
    if kill_store.active:
        logger.error(
            "durable kill switch already engaged (%s); refusing to run",
            kill_store.reason(),
        )
        return CampaignResult(
            config_snapshot=asdict(config),
            ledger_path=ledger_path,
            paper_state_path=paper_state_path,
            planned_days=total_days,
            records=records,
            run_starts=run_starts,
            stop_reason=(
                "durable kill switch is engaged ("
                f"{kill_store.reason() or 'no reason recorded'}); "
                "release it to resume the campaign"
            ),
            kill_switch_engaged=True,
            kill_switch_reason=kill_store.reason(),
        )

    # Mark this process run before any day executes, so a crash between
    # process start and the first completed day is still visible in the ledger.
    _append_record(ledger_path, {
        "kind": "run_start",
        "ts": datetime.now(timezone.utc).isoformat(),
        "resume_after": prior,
        "planned_days": total_days,
    })
    run_starts += 1

    limits = _campaign_limits(config.risk_limits)
    enforcement: Dict[str, Any] = {"halted": False, "reason": None}

    def on_breach(hard: Sequence[Any]) -> None:
        """Engage the durable kill switch on the FIRST hard breach."""
        if enforcement["halted"]:
            return
        enforcement["halted"] = True
        names = "; ".join(sorted({str(b.limit_name) for b in hard}))
        enforcement["reason"] = names
        kill_store.engage(f"campaign risk enforcement: {names}")
        logger.critical(
            "CAMPAIGN HALT — engaging durable kill switch on %s", names
        )

    result = CampaignResult(
        config_snapshot=asdict(config),
        ledger_path=ledger_path,
        paper_state_path=paper_state_path,
        planned_days=total_days,
        records=list(records),
        run_starts=run_starts,
    )

    for day in days[prior:total_days]:
        bars = _bars_for_day(bar_source, day)
        if not bars:
            result.stop_reason = (
                f"day {day.isoformat()}: planned session has no bars — "
                "campaign failed closed instead of silently continuing"
            )
            result.stopped_after_days = len(result.records)
            logger.error("campaign stopped: %s", result.stop_reason)
            return result

        try:
            session, end_book = await _run_campaign_day(
                bars, day, list(bars),
                config=config,
                ledger_records=result.records,
                paper_state_path=paper_state_path,
                kill_switch_store=kill_store,
                order_store=order_store,
                risk_limits=limits,
                on_breach=on_breach,
            )
        except Exception as exc:  # noqa: BLE001 — fail closed on any restart wound
            result.stop_reason = (
                f"day {day.isoformat()}: {type(exc).__name__}: {exc}"
            )
            result.stopped_after_days = len(result.records)
            logger.error("campaign stopped: %s", result.stop_reason)
            return result

        if not session.recovery_ok:
            result.stop_reason = (
                f"day {day.isoformat()}: recovery/reconciliation blocked: "
                f"{session.recovery_detail}"
            )
            result.stopped_after_days = len(result.records)
            logger.error("campaign stopped: %s", result.stop_reason)
            return result

        record = _record_from_session(
            session, day, source=config.source, resume_after=prior,
            end_book=end_book, enforcement=enforcement,
        )
        _append_record(ledger_path, record)
        result.records.append(record)
        prior += 1
        logger.info(
            "campaign day %s complete | trades=%d net=Rs %.2f recon=%s",
            day.isoformat(), record["trades"], record["net_pnl_rupees"],
            record["reconciliation"],
        )
        if enforcement["halted"]:
            result.stop_reason = (
                f"day {day.isoformat()}: durable kill switch engaged by "
                f"daily-loss/drawdown breach: {enforcement['reason']} — "
                "release the switch to resume"
            )
            result.stopped_after_days = len(result.records)
            result.kill_switch_engaged = True
            result.kill_switch_reason = kill_store.reason()
            logger.error("campaign halted: %s", result.stop_reason)
            return result
    return result


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

async def _main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Offline multi-day paper campaign through the production stack"
    )
    parser.add_argument("--synthetic", type=int, metavar="DAYS",
                        help="run N business days of synthetic bars")
    parser.add_argument("--dir", help="per-symbol 1-min CSV directory (real bars)")
    parser.add_argument("--symbols",
                        help="comma-separated symbol filter (default RELIANCE,TCS,INFY)")
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ledger", default="data/campaign/paper/ledger.jsonl")
    parser.add_argument("--paper-state", default="data/campaign/paper/book.json")
    parser.add_argument("--out", default="data/campaign/paper/run",
                        help="output prefix for the summary JSON")
    parser.add_argument(
        "--limit-daily-loss", type=float, metavar="RUPEES",
        help="halt the campaign when a day's paper loss reaches this",
    )
    parser.add_argument(
        "--limit-drawdown", type=float, metavar="PCT",
        help="halt the campaign when drawdown off the session high reaches this",
    )
    args = parser.parse_args(list(argv))

    if not args.synthetic and not args.dir:
        parser.print_usage(sys.stderr)
        print("provide --synthetic DAYS or --dir DATA_DIR", file=sys.stderr)
        return 2

    symbols = ([s.strip().upper() for s in args.symbols.split(",")]
               if args.symbols else ["RELIANCE", "TCS", "INFY"])
    source = "synthetic" if args.synthetic else "csv"
    risk_limits = None
    if args.limit_daily_loss is not None or args.limit_drawdown is not None:
        risk_limits = {}
        if args.limit_daily_loss is not None:
            risk_limits["max_daily_loss"] = args.limit_daily_loss
        if args.limit_drawdown is not None:
            risk_limits["max_drawdown_pct"] = args.limit_drawdown
    config = PaperCampaignConfig(
        symbols=symbols,
        opening_cash=args.cash,
        seed=args.seed,
        source=source,
        data_dir=args.dir,
        risk_limits=risk_limits,
    )
    if source == "csv":
        # CSV runs default to the full available calendar.
        frames = load_bars_from_csv(args.dir, symbols)
        if not frames:
            print("no bar files found under", args.dir, file=sys.stderr)
            return 2
        days = args.synthetic or len(sessions_in(frames))
    else:
        days = args.synthetic

    result = await run_campaign(
        days,
        config,
        ledger_path=args.ledger,
        paper_state_path=args.paper_state,
    )
    summary = result.summary()
    print(json.dumps(summary, indent=2))
    print("wrote", result.write_summary(args.out))
    if not result.completed:
        print("campaign INCOMPLETE:", result.stop_reason or "unknown",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(asyncio.run(_main(sys.argv[1:])))
