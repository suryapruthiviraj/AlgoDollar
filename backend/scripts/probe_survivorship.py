#!/usr/bin/env python
"""
Measure the survivorship gap instead of only declaring it.

The research universe is a PRESENT-DAY snapshot of NIFTY 500 members, so every
name in it survived to today. The usual response is to state that as a caveat
and move on. This script goes one step further: it tries to fetch a hand-listed
set of Indian companies that FAILED, were suspended, or collapsed, and reports
how many of them the vendor can still serve.

WHAT THE ANSWER MEANS
---------------------
* If the vendor CANNOT serve them, the gap is confirmed as unfixable from this
  source, and the bias in every result is real and unquantified.
* If the vendor CAN serve some, their returns give a first-order estimate of
  what the survivor universe is leaving out.

Either way the output is evidence rather than an assertion.

THE LIST IS NOT A UNIVERSE
--------------------------
These names were chosen because they are well-known Indian corporate failures,
not by any systematic rule. That makes this a PROBE, not a reconstructed
point-in-time universe: it cannot be used to re-run a backtest without bias,
because a hand-picked set of failures is itself a selected sample. It can only
answer "is the missing tail reachable at all, and roughly how bad was it".
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data.providers import YahooDataProvider  # noqa: E402
from app.research.datasets import DEFAULT_ROOT  # noqa: E402

OUT_PATH = DEFAULT_ROOT / "survivorship_probe.json"

#: Indian listed companies that failed, were suspended, delisted, or lost the
#: overwhelming majority of their value. Every one of them was investable at
#: some point in the 2012-2026 window the research dataset covers, and NONE of
#: them appear in a present-day NIFTY 500 snapshot.
KNOWN_FAILURES = {
    "DHFL": "Dewan Housing — collapsed 2019, insolvency, delisted",
    "RCOM": "Reliance Communications — insolvency 2019",
    "JETAIRWAYS": "Jet Airways — ceased operations 2019",
    "YESBANK": "Yes Bank — near-failure, RBI reconstruction 2020 (still listed)",
    "SUZLON": "Suzlon Energy — debt restructuring, ~99% drawdown",
    "VIDEOIND": "Videocon Industries — insolvency",
    "KTKBANK": "Karnataka Bank — control name, expected to survive",
    "PNB": "Punjab National Bank — Nirav Modi fraud 2018, control name",
    "IDEA": "Vodafone Idea — near-failure, ~95% drawdown",
    "RPOWER": "Reliance Power — ~98% drawdown",
    "RELINFRA": "Reliance Infrastructure — ~95% drawdown",
    "GTLINFRA": "GTL Infrastructure — penny stock after collapse",
    "UNITECH": "Unitech — fraud, suspended",
    "IL&FSENGG": "IL&FS Engineering — IL&FS collapse 2018",
    "ALOKTEXT": "Alok Industries — insolvency",
    "JPASSOCIAT": "Jaiprakash Associates — insolvency",
}

#: Fetched as a control. These ARE in the research universe, so a failure to
#: fetch them would mean the probe itself is broken rather than the names being
#: unavailable.
CONTROLS = ["RELIANCE", "TCS", "INFY"]


def main() -> int:
    provider = YahooDataProvider()
    start, end = "2012-01-01", datetime.now(timezone.utc).date().isoformat()

    reachable: dict[str, dict] = {}
    unreachable: dict[str, str] = {}

    for sym, note in KNOWN_FAILURES.items():
        try:
            df = provider.fetch_symbol(sym, start, end)
        except Exception as exc:  # noqa: BLE001
            unreachable[sym] = f"{note} | fetch raised: {exc}"
            print(f"{sym:12} UNREACHABLE ({exc})", flush=True)
            continue
        if df is None or len(df) == 0:
            unreachable[sym] = f"{note} | no data returned"
            print(f"{sym:12} UNREACHABLE (no data)", flush=True)
            continue

        col = "adj_close" if "adj_close" in df.columns else "close"
        s = df[col].dropna()
        total = float(s.iloc[-1] / s.iloc[0] - 1.0) if len(s) > 1 else float("nan")
        peak = s.cummax()
        mdd = float((s / peak - 1.0).min()) if len(s) > 1 else float("nan")
        reachable[sym] = {
            "note": note,
            "rows": int(len(df)),
            "first": str(df.index[0].date()),
            "last": str(df.index[-1].date()),
            "total_return": round(total, 4),
            "max_drawdown": round(mdd, 4),
        }
        print(
            f"{sym:12} {len(df):5d} rows  {df.index[0].date()}->{df.index[-1].date()}  "
            f"total {total:+.1%}  MDD {mdd:.1%}",
            flush=True,
        )

    controls = {}
    for sym in CONTROLS:
        df = provider.fetch_symbol(sym, start, end)
        controls[sym] = int(len(df)) if df is not None else 0

    n_reach = len(reachable)
    n_total = len(KNOWN_FAILURES)
    losers = [v for v in reachable.values() if v["total_return"] < 0]
    mean_ret = (
        round(sum(v["total_return"] for v in reachable.values()) / n_reach, 4)
        if n_reach else None
    )
    mean_mdd = (
        round(sum(v["max_drawdown"] for v in reachable.values()) / n_reach, 4)
        if n_reach else None
    )

    doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Hand-listed Indian corporate failures fetched from the same vendor "
            "as the research dataset. A PROBE, not a reconstructed point-in-time "
            "universe: the names were selected because they failed, so their "
            "statistics are a biased sample of the missing tail and must NOT be "
            "merged into a backtest."
        ),
        "controls": controls,
        "counts": {
            "probed": n_total,
            "reachable": n_reach,
            "unreachable": len(unreachable),
        },
        "summary": {
            "reachable_with_negative_total_return": len(losers),
            "mean_total_return": mean_ret,
            "mean_max_drawdown": mean_mdd,
        },
        "reachable": reachable,
        "unreachable": unreachable,
        "interpretation": (
            "Names the vendor cannot serve are permanently absent from any "
            "backtest run here. Names it CAN serve are absent only because the "
            "universe list is a present-day snapshot — which means the "
            "survivorship gap is partly a UNIVERSE CONSTRUCTION problem, not "
            "purely a data availability one."
        ),
    }
    OUT_PATH.write_text(json.dumps(doc, indent=2))
    print(f"\nreachable {n_reach}/{n_total}; controls={controls}")
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
