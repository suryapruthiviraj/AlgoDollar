"""
Fetch NSE 1-minute bars from Yahoo Finance into the replay CSV schema.

WHY THIS SCRIPT EXIST
---------------------
The offline intraday replay harness (`app.engine.replay`) and its research
experiment (`quant/experiments/intraday_replay_research.py`) consume one-minute
OHLCV bars stored as per-symbol CSV files with the exact schema:

    time, open, high, low, close, volume     (time: aware IST)

`git` ignores fetched data (backend/data/replay/), so this script is how the
artifact is REGENERATED reproducibly.  Provenance is recorded in the produced
directory and mirrored into the research results JSON.

HONEST SOURCE LIMITATIONS (must print with the data)
----------------------------------------------------
* Yahoo Finance caps 1-minute requests at ~8 calendar days and retains ~7
  days of 1-minute granularity.  That is one market regime at best, nowhere
  near the ~60 sessions the live-eligibility gate requires.
* Volume is Yahoo's convention for NSE shares (not rupee turnover).
* `auto_adjust=True` is used (price history adjusted for the current payout
  regime); intraday uses are typically unaffected in a one-week window.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

DEFAULT_SYMBOLS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK",
    "SBIN", "ICICIBANK", "HINDUNILVR", "BHARTIARTL",
]
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "replay" / "intraday"


def parse_args(argv: list[str]) -> Namespace:
    ap = ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--period", default="7d", help="yfinance period for interval=1m")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    import yfinance as yf

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wrote: list[str] = []
    for sym in symbols:
        raw = f"{sym}.NS"
        df = yf.download(raw, period=args.period, interval=args.interval,
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            print(f"EMPTY {raw}; skipped")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns={"Datetime": "time"})[
            ["time", "Open", "High", "Low", "Close", "Volume"]
        ].rename(columns={"Open": "open", "High": "high", "Low": "low",
                          "Close": "close", "Volume": "volume"})
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Asia/Kolkata")
        df = df.sort_values("time").reset_index(drop=True)
        dest = out / f"{sym}.csv"
        df.to_csv(dest, index=False)
        wrote.append(f"{sym}:{len(df)}rows:{df['time'].dt.date.nunique()}d")
        print(f"wrote {dest} ({len(df)} rows across {df['time'].dt.date.nunique()} days)")

    meta = out / "PROVENANCE.txt"
    meta.write_text(
        "\n".join([
            "source=yfinance (Yahoo Finance), NSE .NS tickers",
            f"period={args.period} interval={args.interval} auto_adjust=True",
            "schema=time(IST-aware),open,high,low,close,volume",
            "wrote=" + ",".join(wrote),
            "LIMIT=1m data caps at ~8 days/request, ~7 days retained;",
            "      NOT sufficient (~60 sessions) for an eligibility claim.",
        ]) + "\n"
    )
    print("provenance ->", meta)
    return 0 if wrote else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
