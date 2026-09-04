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
from app.research.universe import (  # noqa: E402
    PointInTimeUniverse,
    load_listing_dates,
)

OUT_JSON = DEFAULT_ROOT / "study_result.json"


def main() -> int:
    bundle = build_dataset_bundle()
    manifest = bundle.manifest or {}
    # The stress universe (known corporate failures) is no longer held back:
    # under point-in-time membership those names are eligible only for the
    # period they actually traded, which is the correct treatment. Recorded here
    # for provenance so the report can say they are IN the pool.
    stress_syms = set((manifest.get("stress_universe") or {}).get("symbols") or [])

    # The POOL is every symbol with price history. Membership on each date is
    # decided by the point-in-time provider, NOT by pre-filtering the pool —
    # pre-filtering is what a survivor snapshot does.
    #
    # Known-delisted names are part of the pool now rather than held back for a
    # sensitivity test: with point-in-time membership they are eligible only for
    # the period they actually traded, which is the correct treatment. Excluding
    # them would reintroduce the bias this whole exercise removes.
    pool = bundle.daily.symbols()

    prices = bundle.daily.panel(pool)
    volume = bundle.daily.panel(pool, field="volume")
    bench = bundle.benchmark.returns()

    pit = PointInTimeUniverse(
        prices, volume, listing_dates=load_listing_dates(DEFAULT_ROOT)
    )
    umanifest = pit.manifest()
    cov = pit.coverage()

    print(f"pool            : {prices.shape[1]} symbols "
          f"(incl. {len(stress_syms & set(pool))} known-delisted)")
    print(f"dates           : {prices.shape[0]} sessions "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"stale bars NaN'd: {sum(bundle.daily.stale_bars_masked.values())}")
    print(f"stopped trading : {cov.n_stopped_trading} "
          f"({umanifest['pool_completeness']['stopped_trading_pct']}% of pool)")
    print(f"universe fp     : {umanifest['fingerprint']}")
    print(f"unavailable <   : {umanifest['unavailable_before']}")
    for when in (prices.index[100], prices.index[len(prices)//2], prices.index[-1]):
        try:
            n = len(pit.get_members(when))
            print(f"  members on {when.date()}: {n}")
        except Exception as exc:  # noqa: BLE001
            print(f"  members on {when.date()}: UNAVAILABLE ({exc})")
    print()

    dataset_fp = str((manifest.get("universe") or {}).get("source", "")) + "|" + \
        str(len(pool)) + "|" + str(prices.index[0].date()) + "|" + \
        str(prices.index[-1].date())
    import hashlib as _h
    dataset_fp = _h.sha256(dataset_fp.encode()).hexdigest()[:16]

    # The bundle's limitations describe the SNAPSHOT universe it would have
    # served. A point-in-time provider is now in force, so that sentence is no
    # longer true and must not be carried into the report — a stale limitation
    # is as misleading as a missing one.
    limitations = [
        lim for lim in bundle.limitations()
        if "SURVIVORSHIP BIAS" not in lim.upper()
    ]
    limitations.insert(0, (
        "POINT-IN-TIME UNIVERSE APPLIED: membership on each date requires the "
        "symbol to have been listed, still trading, and liquid over the "
        "trailing window on that date. "
        f"{umanifest['pool_completeness']['stopped_trading']} of "
        f"{umanifest['pool_completeness']['pool_size']} pool symbols "
        f"({umanifest['pool_completeness']['stopped_trading_pct']}%) stopped "
        "trading during the period and are excluded from dates after they "
        "stopped, so companies that failed are held for exactly the period they "
        "were tradeable."
    ))
    limitations.insert(1, (
        "RESIDUAL POOL BIAS: NSE's equity list contains only companies listed "
        "TODAY, so companies that delisted before it was published are absent "
        "unless added by name. The universe definition is point-in-time; the "
        "POOL it draws from is still incomplete by an unmeasured amount."
    ))

    result = run_study(
        prices, benchmark=bench, stress_prices=None,
        limitations=limitations, volume=volume,
        universe=pit, dataset_fingerprint=dataset_fp,
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

    if result.provenance:
        p = result.provenance
        print("\n=== PROVENANCE ===")
        print(f"  experiment_id      : {p.experiment_id}")
        print(f"  dataset            : {p.dataset_fingerprint}")
        print(f"  universe           : {p.universe_fingerprint}")
        print(f"  strategy           : {p.strategy_version}")
        print(f"  validation period  : {p.validation_start} -> {p.validation_end}")

    print(f"\n=== VERDICT: {result.verdict} ===")
    print("\nLIMITATIONS:")
    for lim in result.limitations:
        print(f"  - {lim[:160]}")

    doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": result.verdict,
        "provenance": result.provenance.to_dict() if result.provenance else None,
        "universe_manifest": umanifest,
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
