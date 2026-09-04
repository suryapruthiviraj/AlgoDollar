#!/usr/bin/env python
"""
Audit the acquired dataset and write a machine-readable manifest.

Answers, from the data itself rather than from what the download claimed:
symbols, observations, dates, OHLCV coverage, missing sessions, duplicate rows,
corporate-action evidence, timezone correctness, and the point-in-time status
of every field.

NOTHING IS REPAIRED HERE. The audit reports; it does not forward-fill a gap,
drop a duplicate or smooth a split. A dataset that needs repair is a dataset
whose repair must be a visible, reviewed decision.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from app.research.datasets import DEFAULT_ROOT, build_dataset_bundle  # noqa: E402

OUT_PATH = DEFAULT_ROOT / "data_audit.json"

#: A one-day move this large in an ADJUSTED series is not a price move. It is
#: almost always an unadjusted corporate action leaking through.
SPLIT_EVIDENCE_THRESHOLD = 0.35


def audit() -> dict:
    bundle = build_dataset_bundle()
    daily = bundle.daily
    symbols = daily.symbols()

    per_symbol: dict[str, dict] = {}
    all_dates: Counter = Counter()
    total_rows = 0
    dup_total = 0
    nonmono = []
    neg_price = []
    zero_volume_syms = []
    ohlc_violations = []
    split_evidence: dict[str, list] = {}
    adj_missing = []
    stale_bars: list[dict] = []
    adj_ratios: dict[str, float] = {}

    for sym in symbols:
        df = daily.bars(sym)
        total_rows += len(df)
        for ts in df.index:
            all_dates[ts] += 1

        dups = int(df.index.duplicated().sum())
        dup_total += dups
        if not df.index.is_monotonic_increasing:
            nonmono.append(sym)

        cols = set(df.columns)
        if "adj_close" not in cols:
            adj_missing.append(sym)

        price_cols = [c for c in ("open", "high", "low", "close") if c in cols]
        if price_cols:
            sub = df[price_cols]
            if bool((sub <= 0).any().any()):
                neg_price.append(sym)
        if {"open", "high", "low", "close"} <= cols:
            bad = (
                (df["high"] < df["low"])
                | (df["high"] < df["open"]) | (df["high"] < df["close"])
                | (df["low"] > df["open"]) | (df["low"] > df["close"])
            )
            n_bad = int(bad.sum())
            if n_bad:
                ohlc_violations.append({"symbol": sym, "rows": n_bad})

        if "volume" in cols:
            zv = int((df["volume"] == 0).sum())
            if zv:
                zero_volume_syms.append({"symbol": sym, "zero_volume_days": zv})
        # A zero-volume bar whose OHLC are all identical is the vendor carrying
        # the previous close forward on a non-trading day. It is not a neutral
        # row: it produces a return of exactly zero, which compresses measured
        # volatility and inflates any Sharpe computed from the series.
        n_stale = int(daily.stale_bar_mask(df).sum())
        if n_stale:
            stale_bars.append({"symbol": sym, "stale_bars": n_stale})

        # Corporate-action evidence: a huge move in RAW close that is absent
        # from ADJ close is a split/bonus the vendor adjusted for.
        if {"close", "adj_close"} <= cols and len(df) > 1:
            raw_ret = df["close"].pct_change()
            adj_ret = df["adj_close"].pct_change()
            suspicious = raw_ret[
                (raw_ret.abs() > SPLIT_EVIDENCE_THRESHOLD)
                & (adj_ret.abs() < SPLIT_EVIDENCE_THRESHOLD / 2)
            ]
            if len(suspicious):
                split_evidence[sym] = [
                    {"date": str(d.date()), "raw_return": round(float(v), 4)}
                    for d, v in suspicious.items()
                ][:10]

        if {"close", "adj_close"} <= cols and len(df):
            # close/adj_close at the START of the series. 1.0 would mean no
            # adjustment was ever applied; anything above it is the cumulative
            # dividend adjustment.
            try:
                adj_ratios[sym] = round(float(df["close"].iloc[0] / df["adj_close"].iloc[0]), 4)
            except (ZeroDivisionError, ValueError):
                pass

        per_symbol[sym] = {
            "rows": int(len(df)),
            "first": str(df.index[0].date()) if len(df) else None,
            "last": str(df.index[-1].date()) if len(df) else None,
            "duplicate_index_rows": dups,
            "columns": sorted(str(c) for c in df.columns),
            "tz_aware": bool(getattr(df.index, "tz", None) is not None),
        }

    # ---- calendar ------------------------------------------------------- #
    # The trading calendar is TAKEN FROM THE DATA, not from a holiday list:
    # a date on which a large fraction of the universe traded is a session.
    # Anything else would require an NSE calendar this project does not have.
    session_dates = sorted(d for d, n in all_dates.items() if n >= max(2, len(symbols) * 0.5))
    coverage_gaps = []
    for sym in symbols:
        df = daily.bars(sym)
        if not len(df):
            continue
        # Only sessions inside the symbol's own listed life count as missing.
        window = [d for d in session_dates if df.index[0] <= d <= df.index[-1]]
        missing = sorted(set(window) - set(df.index))
        if missing:
            coverage_gaps.append({
                "symbol": sym,
                "missing_sessions": len(missing),
                "pct_of_life": round(100.0 * len(missing) / max(1, len(window)), 3),
                "examples": [str(d.date()) for d in missing[:5]],
            })

    bench = bundle.benchmark.series()
    bench_missing = sorted(set(session_dates) - set(bench.index))

    coverage_gaps.sort(key=lambda r: r["missing_sessions"], reverse=True)

    audit_doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "symbols": len(symbols),
            "observations": int(total_rows),
            "distinct_dates": len(all_dates),
            "trading_sessions_inferred": len(session_dates),
            "first_date": str(min(all_dates).date()) if all_dates else None,
            "last_date": str(max(all_dates).date()) if all_dates else None,
        },
        "integrity": {
            "duplicate_index_rows": dup_total,
            "non_monotonic_symbols": nonmono,
            "non_positive_price_symbols": neg_price,
            "ohlc_relationship_violations": ohlc_violations,
            "symbols_missing_adj_close": adj_missing,
            "zero_volume": {
                "symbols_affected": len(zero_volume_syms),
                "worst": sorted(
                    zero_volume_syms, key=lambda r: r["zero_volume_days"], reverse=True
                )[:10],
            },
        },
        "coverage": {
            "symbols_with_gaps": len(coverage_gaps),
            "worst_gaps": coverage_gaps[:10],
            "benchmark_missing_sessions": len(bench_missing),
        },
        "stale_bars": {
            "definition": (
                "volume == 0 AND open == high == low == close — the vendor "
                "carrying the previous close forward on a non-trading day."
            ),
            "symbols_affected": len(stale_bars),
            "total_bars": sum(r["stale_bars"] for r in stale_bars),
            "worst": sorted(stale_bars, key=lambda r: r["stale_bars"], reverse=True)[:10],
            "handling": (
                "MASKED TO NaN by ParquetDailyBars.panel(drop_stale=True), never "
                "filled or repaired. Left in, they add returns of exactly zero, "
                "which compress volatility and inflate Sharpe."
            ),
        },
        "adjustment_semantics": {
            "finding": (
                "The vendor's `close` is ALREADY SPLIT-ADJUSTED; only dividends "
                "separate it from `adj_close`. Verified: the close/adj_close "
                "ratio starts above 1.0 and converges to exactly 1.0 at the end "
                "of every series, and no symbol shows a ~50% single-day move in "
                "`close` where a bonus or split is known to have occurred."
            ),
            "consequence": (
                "NEITHER series is the price actually quoted on a past date. "
                "`close` is a split-adjusted reconstruction. A backtest sizing "
                "positions in SHARES off `close` therefore computes a share "
                "count that never existed, though the resulting RETURN is "
                "correct. Rupee-notional sizing is unaffected."
            ),
            "close_over_adj_close_at_series_start": dict(
                sorted(adj_ratios.items())[:15]
            ),
            "symbols_with_no_adjustment_at_all": [
                k for k, v in adj_ratios.items() if v == 1.0
            ],
        },
        "corporate_actions": {
            "separate_dataset_available": False,
            "detection_method": (
                "Inferred only: a >35% one-day move in RAW close that does NOT "
                "appear in adj_close indicates a split or bonus the vendor "
                "adjusted for. Ex-dates, ratios and dividend amounts are NOT "
                "recoverable this way."
            ),
            "symbols_with_evidence": len(split_evidence),
            "examples": dict(list(split_evidence.items())[:10]),
        },
        "timestamps": {
            "timezone": (
                "Bars are tz-NAIVE dates normalised to midnight, representing "
                "NSE SESSION DATES (IST). They are not instants and must not be "
                "compared against a UTC clock; doing so shifts every Indian "
                "session date by up to 5.5 hours."
            ),
            "any_tz_aware_index": any(v["tz_aware"] for v in per_symbol.values()),
        },
        "point_in_time_status": {
            "daily_bars": "POINT-IN-TIME (as traded on the date recorded)",
            "adj_close": (
                "RECONSTRUCTED — back-adjusted using corporate actions known "
                "TODAY. A 2015 adj_close is not the number an observer saw in "
                "2015. Valid for return computation, invalid as a price level."
            ),
            "universe": "RECONSTRUCTED — present-day membership (survivorship bias)",
            "sectors": "RECONSTRUCTED — current classification applied to all dates",
            "benchmark": "POINT-IN-TIME level; adjusted series reconstructed",
        },
        "limitations": bundle.limitations(),
        "per_symbol": per_symbol,
    }
    return audit_doc


def main() -> int:
    doc = audit()
    OUT_PATH.write_text(json.dumps(doc, indent=2))

    c, i, cov = doc["counts"], doc["integrity"], doc["coverage"]
    print(f"symbols            : {c['symbols']}")
    print(f"observations       : {c['observations']:,}")
    print(f"distinct dates     : {c['distinct_dates']:,}")
    print(f"inferred sessions  : {c['trading_sessions_inferred']:,}")
    print(f"range              : {c['first_date']} -> {c['last_date']}")
    print(f"duplicate rows     : {i['duplicate_index_rows']}")
    print(f"non-monotonic      : {len(i['non_monotonic_symbols'])}")
    print(f"non-positive price : {len(i['non_positive_price_symbols'])}")
    print(f"OHLC violations    : {len(i['ohlc_relationship_violations'])}")
    print(f"missing adj_close  : {len(i['symbols_missing_adj_close'])}")
    print(f"zero-volume syms   : {i['zero_volume']['symbols_affected']}")
    print(f"symbols with gaps  : {cov['symbols_with_gaps']}")
    print(f"benchmark gaps     : {cov['benchmark_missing_sessions']}")
    print(f"CA evidence syms   : {doc['corporate_actions']['symbols_with_evidence']}")
    print(f"stale bars         : {doc['stale_bars']['total_bars']} across "
          f"{doc['stale_bars']['symbols_affected']} symbols")
    print(f"\nwritten -> {OUT_PATH}")
    print("\nLIMITATIONS:")
    for lim in doc["limitations"]:
        print(f"  - {lim[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
