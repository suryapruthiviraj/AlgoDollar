#!/usr/bin/env python
"""
Fetch price history for the FULL pre-study listing pool.

No selection: every EQ-series symbol NSE reports as listed before the study
start is fetched, plus every known-delisted name from the survivorship probe.
Choosing a subset — by size, by liquidity, by anything — would put a selection
decision inside the universe the study is supposed to measure.

Symbols already on disk are skipped, so this is resumable.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data.providers import YahooDataProvider  # noqa: E402
from app.research.datasets import DEFAULT_ROOT  # noqa: E402

PRICES = DEFAULT_ROOT / "prices"
START, END = "2012-01-01", date.today().isoformat()
POOL_CUTOFF = "2012-01-01"


def main() -> int:
    ref = json.loads((DEFAULT_ROOT / "universe_reference.json").read_text())
    listings = ref["listings"]

    pool = sorted(s for s, v in listings.items() if v["listing_date"] < POOL_CUTOFF)

    probe_path = DEFAULT_ROOT / "survivorship_probe.json"
    failures: list[str] = []
    if probe_path.exists():
        probe = json.loads(probe_path.read_text())
        failures = sorted(
            s for s, v in (probe.get("reachable") or {}).items() if v.get("rows", 0) > 500
        )

    targets = sorted(set(pool) | set(failures))
    have = {p.stem for p in PRICES.glob("*.parquet")}
    todo = [s for s in targets if s not in have]

    print(f"pool (listed < {POOL_CUTOFF}): {len(pool)}")
    print(f"known delisted added        : {len(failures)}")
    print(f"already on disk             : {len(have)}")
    print(f"to fetch                    : {len(todo)}", flush=True)

    provider = YahooDataProvider()
    ok, failed = 0, {}
    for i, sym in enumerate(todo, 1):
        try:
            df = provider.fetch_symbol(sym, START, END)
        except Exception as exc:  # noqa: BLE001
            failed[sym] = f"{type(exc).__name__}: {exc}"
            continue
        if df is None or len(df) == 0:
            failed[sym] = "no data returned"
        else:
            df.sort_index().to_parquet(PRICES / f"{sym}.parquet")
            ok += 1
        if i % 50 == 0:
            print(f"  [{i}/{len(todo)}] ok={ok} failed={len(failed)}", flush=True)
        time.sleep(0.15)

    (DEFAULT_ROOT / "pool_manifest.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pool_cutoff": POOL_CUTOFF,
        "selection_rule": (
            "EVERY EQ-series symbol NSE reports as listed before the cutoff, "
            "plus known-delisted names from the survivorship probe. No "
            "performance, size or liquidity criterion was applied — those are "
            "decided point-in-time by the universe provider, not here."
        ),
        "pool_size": len(targets),
        "fetched_ok": ok,
        "already_present": len(have),
        "failed": failed,
        "residual_bias": (
            "EQUITY_L.csv lists companies listed TODAY, so companies that "
            "delisted before today are absent unless added explicitly. The pool "
            "is therefore still incomplete; the count of names whose series ENDS "
            "before the dataset end is reported by the universe provider and is "
            "the measurable part of what remains."
        ),
    }, indent=2))
    print(f"\ndone: {ok} fetched, {len(failed)} failed. Total on disk: "
          f"{len(list(PRICES.glob('*.parquet')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
