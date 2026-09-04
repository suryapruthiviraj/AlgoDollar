#!/usr/bin/env python
"""
Run the full research study and write the validation report.

Everything this prints is computed from the acquired dataset. Nothing is
hard-coded, and the verdict comes from criteria fixed in app/research/study.py
before the study was ever run.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from app.research.datasets import DEFAULT_ROOT, build_dataset_bundle  # noqa: E402
from app.research.study import run_study  # noqa: E402

OUT_JSON = DEFAULT_ROOT / "study_result.json"


def main() -> int:
    bundle = build_dataset_bundle()
    manifest = bundle.manifest or {}
    stress_syms = set((manifest.get("stress_universe") or {}).get("symbols") or [])

    all_syms = bundle.daily.symbols()
    # The default research universe EXCLUDES the hand-picked failure set. That
    # set is a selected sample and belongs only in the sensitivity test; folding
    # it into the main universe would trade survivorship bias for selection bias.
    universe = [s for s in all_syms if s not in stress_syms]

    prices = bundle.daily.panel(universe)
    volume = bundle.daily.panel(universe, field="volume")
    bench = bundle.benchmark.returns()
    stress_prices = (
        bundle.daily.panel(sorted(stress_syms)) if stress_syms else None
    )

    print(f"universe        : {prices.shape[1]} symbols (excl. {len(stress_syms)} stress names)")
    print(f"dates           : {prices.shape[0]} sessions "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"stale bars NaN'd: {sum(bundle.daily.stale_bars_masked.values())}")
    print()

    result = run_study(
        prices, benchmark=bench, stress_prices=stress_prices,
        limitations=bundle.limitations(), volume=volume,
    )

    print("=== IN-SAMPLE (reference only, NOT evidence) ===")
    print(f"{'signal':22} {'Sharpe':>7} {'CAGR':>8} {'exCAGR':>8} {'MDD':>8}")
    for name, m in result.in_sample.items():
        ex = f"{m['excess_cagr']:8.2%}" if m["excess_cagr"] is not None else "     n/a"
        print(f"{name:22} {m['sharpe']:7.2f} {m['cagr']:8.2%} {ex} {m['max_drawdown']:8.2%}")

    print("\n=== WALK-FORWARD FOLDS (selection on train, measured on test) ===")
    print(f"{'#':>2} {'test window':25} {'selected':22} {'trainSR':>8} {'testSR':>8}")
    for f in result.folds:
        print(f"{f.fold:>2} {f.test_start}..{f.test_end:12} {f.selected_signal:22} "
              f"{f.train_sharpe:8.2f} {f.test_sharpe:8.2f}")

    o = result.oos_metrics
    print("\n=== OUT-OF-SAMPLE (stitched test windows) ===")
    for k in ("n_observations", "sharpe", "cagr", "benchmark_cagr", "excess_cagr",
              "deflated_sharpe_ratio", "pbo", "n_trials"):
        print(f"  {k:24}: {o.get(k)}")

    b = result.robustness.get("bootstrap", {})
    if "p05" in b:
        print(f"\n  bootstrap Sharpe 90% CI: [{b['p05']}, {b['p95']}]  "
              f"P(>0)={b['fraction_above_zero']}")

    us = result.robustness.get("universe_sensitivity")
    if us:
        print(f"\n=== UNIVERSE SENSITIVITY (+{us['added_symbols']} known failures) ===")
        print(f"  survivor-only : Sharpe {us['survivor_only']['sharpe']:.2f}  "
              f"CAGR {us['survivor_only']['cagr']:.2%}")
        print(f"  with failures : Sharpe {us['with_failures']['sharpe']:.2f}  "
              f"CAGR {us['with_failures']['cagr']:.2%}")
        print(f"  delta         : Sharpe {us['sharpe_delta']:+.2f}  "
              f"CAGR {us['cagr_delta']:+.2%}")

    cs = result.robustness.get("cost_sensitivity", {})
    if cs:
        print("\n=== COST SENSITIVITY ===")
        for lvl, v in cs.items():
            print(f"  {lvl:>8}: Sharpe {v['sharpe']:6.2f}  CAGR {v['cagr']:7.2%}")

    print("\n=== CRITERIA ===")
    for c in result.criteria:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name:26} "
              f"observed={c.observed}  threshold={c.threshold}")

    print(f"\n=== VERDICT: {result.verdict} ===")
    print("\nLIMITATIONS:")
    for lim in result.limitations:
        print(f"  - {lim[:160]}")

    doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": result.verdict,
        "universe_size": int(prices.shape[1]),
        "sessions": int(prices.shape[0]),
        "date_range": [str(prices.index[0].date()), str(prices.index[-1].date())],
        "criteria": [asdict(c) for c in result.criteria],
        "in_sample": result.in_sample,
        "folds": [asdict(f) for f in result.folds],
        "out_of_sample": result.oos_metrics,
        "robustness": result.robustness,
        "limitations": result.limitations,
    }
    OUT_JSON.write_text(json.dumps(doc, indent=2, default=str))
    print(f"\nwritten -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
