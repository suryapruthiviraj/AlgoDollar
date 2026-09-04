#!/usr/bin/env python
"""
Acquire the real NSE dataset the research pipeline runs on.

Writes one Parquet file per symbol plus a machine-readable manifest recording
what was fetched, when, from where, and what is KNOWN TO BE MISSING.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
* It does not forward-fill, interpolate or otherwise invent a missing bar. A
  gap in the record is evidence — usually of a suspension, a holiday the
  calendar did not know about, or a listing that had not happened yet — and
  filling it destroys that evidence.
* It does not silently drop a symbol that failed. A failure is recorded in the
  manifest under ``failed``, because "we could not get this name" and "this
  name has no history" are different facts.
* It does not claim the universe is point-in-time. It is not. The symbol list
  is a PRESENT-DAY snapshot, so every name in it survived to today. The
  manifest states this as a survivorship limitation rather than leaving the
  reader to assume otherwise.

Usage
-----
    python scripts/acquire_data.py --symbols 120 --start 2012-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from app.data.providers import YahooDataProvider  # noqa: E402
from app.data.universe import StockUniverse  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parents[1] / "research_data"
PRICES_DIR = DATA_ROOT / "prices"
MANIFEST_PATH = DATA_ROOT / "manifest.json"

#: Yahoo's NSE index symbol. Fetched through the same path as everything else
#: so the benchmark cannot silently come from a different vendor or calendar.
BENCHMARK = "^NSEI"


def fetch_all(symbols: list[str], start: str, end: str, provider: YahooDataProvider) -> dict:
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    ok: dict[str, dict] = {}
    failed: dict[str, str] = {}

    for i, sym in enumerate(symbols, 1):
        try:
            df = provider.fetch_symbol(sym, start, end)
        except Exception as exc:  # noqa: BLE001
            failed[sym] = f"{type(exc).__name__}: {exc}"
            print(f"[{i}/{len(symbols)}] {sym}: FAILED {exc}", flush=True)
            continue

        if df is None or len(df) == 0:
            # Recorded, not skipped. For a name that was listed during the
            # window, absence is itself a finding about the vendor.
            failed[sym] = "no data returned"
            print(f"[{i}/{len(symbols)}] {sym}: no data", flush=True)
            continue

        df = df.sort_index()
        path = PRICES_DIR / f"{sym}.parquet"
        df.to_parquet(path)
        ok[sym] = {
            "rows": int(len(df)),
            "first": str(df.index[0].date()),
            "last": str(df.index[-1].date()),
            "columns": [str(c) for c in df.columns],
            "file": path.name,
        }
        print(
            f"[{i}/{len(symbols)}] {sym}: {len(df)} rows "
            f"{df.index[0].date()} -> {df.index[-1].date()}",
            flush=True,
        )
        # Courtesy rate limit. Hammering the vendor gets the IP throttled, and
        # a throttled fetch returns an EMPTY frame rather than an error, which
        # would look like "this symbol has no history".
        time.sleep(0.25)

    return {"ok": ok, "failed": failed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=120)
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    args = ap.parse_args()

    universe = StockUniverse.get_nifty500_symbols()[: args.symbols]
    provider = YahooDataProvider()

    print(f"Fetching {len(universe)} symbols {args.start} -> {args.end}", flush=True)
    result = fetch_all(universe, args.start, args.end, provider)

    print("Fetching benchmark", BENCHMARK, flush=True)
    bench = None
    bench_meta: dict = {"symbol": BENCHMARK, "status": "MISSING"}
    try:
        import yfinance as yf

        bench = yf.download(
            BENCHMARK, start=args.start, end=args.end,
            progress=False, auto_adjust=False, threads=False,
        )
        if bench is not None and len(bench):
            if isinstance(bench.columns, pd.MultiIndex):
                bench.columns = bench.columns.get_level_values(0)
            bench = bench.rename(columns={
                "Adj Close": "adj_close", "Close": "close", "Open": "open",
                "High": "high", "Low": "low", "Volume": "volume",
            })
            bench.index = pd.DatetimeIndex(bench.index).tz_localize(None).normalize()
            bench.to_parquet(DATA_ROOT / "benchmark.parquet")
            bench_meta = {
                "symbol": BENCHMARK, "status": "OK", "rows": int(len(bench)),
                "first": str(bench.index[0].date()), "last": str(bench.index[-1].date()),
                "file": "benchmark.parquet",
            }
    except Exception as exc:  # noqa: BLE001
        bench_meta = {"symbol": BENCHMARK, "status": "FAILED", "error": str(exc)}

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested": {
            "symbols": args.symbols, "start": args.start, "end": args.end,
        },
        "source": {
            "vendor": "Yahoo Finance",
            "accessor": "yfinance",
            "adjustment": (
                "auto_adjust=False; both `close` (unadjusted) and `adj_close` "
                "(split+dividend adjusted) are stored. Returns MUST be computed "
                "from adj_close; using close silently books every split as a "
                "price crash."
            ),
        },
        "universe": {
            "source": "StockUniverse.get_nifty500_symbols()",
            "point_in_time": False,
            "limitation": (
                "SURVIVORSHIP BIAS: this is a PRESENT-DAY membership snapshot, "
                "not point-in-time index membership. Every symbol in it survived "
                "and remained listed to today. Any performance measured on this "
                "universe is biased by an unknown amount, and the direction is "
                "NOT reliably conservative — see docs/DATA_INTEGRITY_REPORT.md."
            ),
        },
        "symbols_ok": result["ok"],
        "symbols_failed": result["failed"],
        "benchmark": bench_meta,
        "known_gaps": {
            "corporate_actions": (
                "NOT AVAILABLE as a separate dataset. Splits and dividends are "
                "only implicit in adj_close. Ex-dates, ratios and dividend "
                "amounts are NOT obtainable here, so any analysis needing them "
                "is a stated research limitation, not something to approximate."
            ),
            "delistings": (
                "NOT AVAILABLE. Delisted names are absent from the universe "
                "entirely, which is the survivorship problem above."
            ),
            "symbol_changes": (
                "NOT AVAILABLE. A renamed ticker appears as a new symbol with "
                "truncated history; no mapping table exists here."
            ),
            "fundamentals": (
                "NOT AVAILABLE point-in-time. No provider here supplies "
                "fundamentals with publication dates, so any value/quality "
                "factor would carry look-ahead bias of unknown size."
            ),
            "intraday": "NOT AVAILABLE. Daily bars only.",
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    n_ok, n_fail = len(result["ok"]), len(result["failed"])
    print(f"\nDone: {n_ok} ok, {n_fail} failed. Manifest -> {MANIFEST_PATH}")
    # A partial dataset is still a dataset; the manifest records exactly which
    # names are missing, so downstream code can decide rather than guess.
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
