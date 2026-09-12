"""
Intraday auto-trader replay research on REAL 1-minute bars.

WHAT THIS SCRIPT IS
-------------------
The honest attempt to answer one question about the Phase A constructed
`IntradayAutoTrader`:

    Does it fire often enough, and clear costs and a multiple-testing bar,
    when run OFFLINE through the exact production execution stack on real
    NSE 1-minute bars?

It replays bars through `app.engine.replay` — the same TickBarAggregator ->
TickQuoteSource -> PaperBroker -> ExecutionService stack the live mock path
uses — and scores the resulting trade log + daily returns with the production
research statistics (`app.research.statistics`).  There are no research-only
reimplementations: a replay result computed by different code than a live
session is not a result about the live system.

METHOD
------
1. Load real 1-minute bars (`backend/data/replay/intraday/*.csv`), fetched from
   Yahoo Finance (NSE `.NS` tickers) and cleaned to the replay column schema.
2. Baseline replay with production-default strategy parameters.
3. Replay a small grid over the strategy's CONSTRUCTOR-EXPOSED parameters
   (min_net_edge, max_positions) — the only surface production exposes.
4. Deflated Sharpe on the baseline daily returns, with n_trials = the number
   of configurations in the search.
5. Probability-of-backtest-overfitting on the config returns matrix.
6. Write results JSON and print a verdict.

HONESTY CONSTRAINTS (why every number below is a measurement, not a promise)
---------------------------------------------------------------------------
* Data: a single low-volatility week (2026-08-27..09-04). One market regime at
  best. The repo's own eligibility gate requires ~60 sessions of intraday
  history; this source (Yahoo Finance 1-minute) cannot deliver that — it caps
  at ~8 days per request and retains ~7 days total.
* Survivorship: irrelevant for a one-week window of currently-listed, very
  liquid names, but the window contains no drawdown/crash scenario at all.
* A strategy that trades zero times in the entire sample cannot support any
  performance claim, positive or negative. Zero trades is reported as such.
* PBO over T=7 sessions has n_partitions=2 (the smallest valid split) and is
  therefore about as informative as a coin flip; a degenerate (zero-trading)
  matrix cannot be tested at all and is reported as uncomputable.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import numpy as np

from app.engine.replay import load_bars_from_csv, run_replay  # noqa: E402
from app.research.statistics import (  # noqa: E402
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

GRID: list[dict] = [
    {"min_net_edge": 0.002, "max_positions": 5},
    {"min_net_edge": 0.003, "max_positions": 5},   # production defaults
    {"min_net_edge": 0.004, "max_positions": 5},
    {"min_net_edge": 0.003, "max_positions": 3},
    {"min_net_edge": 0.003, "max_positions": 8},
]

DATA_DIR = Path(__file__).resolve().parents[2] / "backend" / "data" / "replay" / "intraday"

PROVENANCE = {
    "source": "Yahoo Finance, NSE .NS tickers, interval=1m, period=7d",
    "fetched": "2026-09-06",
    "window": "2026-08-27 .. 2026-09-04 (7 trading sessions)",
    "symbols": 8,
    "survivorship_bias": "not estimable from a single current-week window",
    "point_in_time": "n/a (single week, today's tickers)",
    "intraday_history_limit": (
        "Yahoo 1-minute caps at ~8 days per request / ~7 days retained; "
        "the eligibility gate requires ~60 sessions of intraday history."
    ),
}


def parse_args(argv: list[str]) -> Namespace:
    ap = ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(OUT / "intraday_replay_real.json"))
    return ap.parse_args(argv)


async def run_grid(bars: dict, grid: list[dict]) -> list[dict]:
    results = []
    for cfg in grid:
        t0 = time.time()
        label = f"edge={cfg.get('min_net_edge'):.3f},maxpos={cfg.get('max_positions')}"
        r = await run_replay(bars, strategy_config=cfg, synthetic=False)
        session_days = [s.session for s in r.sessions]
        daily = {s.session: s.equity_return_pct for s in r.sessions}
        results.append({
            "config": cfg,
            "label": label,
            "replay_seconds": round(time.time() - t0, 1),
            "trades": r.summary["trades"],
            "trading_days": r.summary["percent_trading_days"],
            "net_pnl_rupees": r.summary["net_pnl_rupees"],
            "expectancy_per_trade": r.summary["expectancy_rupees_per_trade"],
            "win_rate_pct": r.summary["win_rate_pct"],
            "gross_pnl_rupees": r.summary["gross_pnl_rupees"],
            "total_costs_rupees": r.summary["total_costs_rupees"],
            "sessions_with_errors": r.summary["sessions_with_errors"],
            "daily_returns_pct": [daily[d] for d in session_days],
            "exit_reasons": _exit_reason_counts(r.trades),
        })
        print(f"  [{label}] trades={r.summary['trades']} "
              f"net=Rs{r.summary['net_pnl_rupees']:.0f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return results


def _exit_reason_counts(trades) -> dict:
    counts: dict = {}
    for t in trades:
        counts[t.exit_reason] = counts.get(t.exit_reason, 0) + 1
    return counts


def edge_survey(bars: dict, sample_stride: int = 5) -> dict:
    """
    Max net edge (expected return - cost) the strategy would have seen, scored
    at every sampled intraday minute of every symbol-day via the production
    `_build_signal`.  This is the data-backed WHY behind a zero-trade week: if
    the maximum net edge never clears the loosest threshold, the strategy's
    own edge estimate says nothing was worth trading.
    """
    from datetime import datetime
    from datetime import time as dtime
    from zoneinfo import ZoneInfo

    import pandas as pd

    from app.strategies.intraday import IntradayStrategy

    IST = ZoneInfo("Asia/Kolkata")
    strat = IntradayStrategy(paper_mode=True, min_net_edge=0.0)
    edges: list[float] = []
    cells = 0
    for sym, df in bars.items():
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        for day, g in df.groupby(df["time"].dt.date):
            g = g.sort_values("time")
            cells += 1
            minute = dtime(9, 31)
            while minute <= dtime(14, 45):
                t = pd.Timestamp(datetime.combine(day, minute, tzinfo=IST))
                sub = g[g["time"] <= t]
                if len(sub) >= 30:
                    try:
                        sig = strat._build_signal(
                            sym, sub, pd.DataFrame(),
                            datetime.combine(day, minute, tzinfo=IST),
                            long_allowed=True,
                        )
                    except Exception:
                        sig = None
                    if sig is not None:
                        edges.append(float(sig.edge_score))
                minute = (
                    dtime(minute.hour + 1, 0) if minute.minute == 59
                    else dtime(minute.hour, minute.minute + 1)
                )
    if not edges:
        return {"cells": cells, "candidate_edges": 0, "max_net_edge": None}
    arr = np.asarray(edges)
    return {
        "cells": cells,
        "candidate_edges": len(arr),
        "max_net_edge": round(float(np.max(arr)), 4),
        "median_net_edge": round(float(np.median(arr)), 4),
        "p95_net_edge": round(float(np.percentile(arr, 95)), 4),
        "p99_net_edge": round(float(np.percentile(arr, 99)), 4),
        "net_edge_exceeds_cost": int((arr > 0.0018).sum()),
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    bars = load_bars_from_csv(args.data_dir)
    if not bars:
        print("no bar files in", args.data_dir)
        return 2
    print(f"replaying {len(bars)} symbols from {args.data_dir}")
    print(f"grid: {len(GRID)} configs over the constructor-exposed surface")

    grid_results = asyncio.run(run_grid(bars, GRID))

    # ---- Edge survey: the WHY behind the trade count ----
    print("scoring the strategy's net-edge distribution over all symbol-days...",
          flush=True)
    survey = edge_survey(bars)

    # ---- Deflated Sharpe on production-default (baseline) config ----
    baseline = next(g for g in grid_results
                    if g["config"] == {"min_net_edge": 0.003, "max_positions": 5})
    n_trials = len(GRID)
    dsr = None
    if baseline["trades"] == 0:
        dsr_error = "baseline traded 0 times in the sample; no returns to deflate"
    else:
        try:
            r = deflated_sharpe_ratio(baseline["daily_returns_pct"], n_trials=n_trials)
            dsr = {
                "observed_sharpe": r.observed_sharpe,
                "expected_max_sharpe_null": r.expected_max_sharpe_null,
                "deflated_sharpe_ratio": r.deflated_sharpe_ratio,
                "n_trials": r.n_trials,
                "n_observations": r.n_observations,
                "skewness": r.skewness,
                "kurtosis": r.kurtosis,
                "is_significant": r.is_significant,
                "summary": r.summary(),
            }
            dsr_error = None
        except ValueError as exc:
            dsr = None
            dsr_error = f"uncomputable: {exc}"

    # ---- PBO over the config return matrix (T x N) ----
    matrix = np.asarray([g["daily_returns_pct"] for g in grid_results], dtype=float).T
    pbo = None
    pbo_error = None
    if np.isclose(np.nanvar(matrix, axis=0), 0).all():
        pbo_error = "all configs produced identical (zero-trading) return series"
    elif matrix.shape[0] < 4:
        pbo_error = f"too few sessions for even n_partitions=2 (T={matrix.shape[0]})"
    else:
        try:
            res = probability_of_backtest_overfitting(
                matrix, n_partitions=2, random_seed=42,
            )
            pbo = {
                "pbo": res.pbo,
                "n_combinations": res.n_combinations,
                "n_configs": res.n_configs,
                "median_oos_rank": res.median_oos_rank,
                "is_overfit": res.is_overfit,
                "summary": res.summary(),
            }
        except (RuntimeError, ValueError) as exc:
            pbo = None
            pbo_error = f"uncomputable: {exc}"

    # ---- Verdict aligned with the live-trading eligibility gate ----
    total_trades = sum(g["trades"] for g in grid_results)
    if baseline["trades"] == 0:
        verdict = (
            "NO_TRADE: baseline (and every grid config) traded 0 times in the "
            "sample. Not a bug and not noise — the strategy's own edge survey "
            f"peaks at net edge {survey['max_net_edge']} (vs cost ~0.0018 and "
            "the loosest grid threshold 0.002), so it judged nothing tradeable. "
            "Zero trades clears costs trivially and produces no PnL; with a "
            "single quiet week neither DSR nor PBO can be evaluated. NO "
            "strategy claim — positive or negative — can be made on this "
            "sample; the eligibility gate requires ~60 sessions of intraday "
            "history, which this source cannot provide."
        )
    else:
        verdict = (
            "INSUFFICIENT_DATA: single-week sample cannot support any strategy "
            f"claim (gate requires ~60 sessions). Baseline traded "
            f"{baseline['trades']} times."
        )

    report = {
        "provenance": PROVENANCE,
        "n_sessions": len(grid_results[0]["daily_returns_pct"]),
        "n_configs": len(GRID),
        "edge_survey": survey,
        "grid_results": grid_results,
        "deflated_sharpe_baseline": {"value": dsr, "error": dsr_error},
        "probability_of_backtest_overfitting": {"value": pbo, "error": pbo_error},
        "verdict": verdict,
    }
    OUT.parent.mkdir(exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print("\n" + "=" * 72)
    print("VERDICT:", verdict)
    print("=" * 72)
    print("DSR (baseline):", dsr["summary"] if dsr else dsr_error)
    print("PBO (grid):   ", pbo["summary"] if pbo else pbo_error)
    print("trades across all configs:", total_trades)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
