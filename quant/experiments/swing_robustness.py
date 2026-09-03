"""
Robustness analysis: attempt to DISPROVE the leading swing candidate.

PURPOSE
-------
The leaderboard run found that no candidate is statistically significant after
multiple-testing correction, and that the best ECONOMICS belonged to the
simplest rule (12-1 momentum, 12% turnover) rather than to the model with the
best information coefficient. This script tries to break that finding.

The tests are chosen because each one has, historically, killed a strategy
that looked good in a first-pass backtest:

  COST STRESS         Slippage estimates are always optimistic in research.
  PARAMETER STRESS    A result that exists only at one lookback is curve fit.
  TIME STRESS         A strategy that works in two years out of fifteen is a
                      regime bet, not an edge.
  REGIME STRESS       Performance that appears only in bull markets is beta.
  CONCENTRATION       An edge carried by one sector, one name, or five
                      rebalances is not an edge.
  UNIVERSE STRESS     Sensitivity to which names are included is fragility.

Only after all of that is the FINAL HOLDOUT touched, exactly once.

INTERPRETING ANYTHING POSITIVE HERE
-----------------------------------
The universe is survivorship-filtered. Every number below is biased upward.
A negative result is therefore trustworthy; a positive one is not.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.data.features import FeatureEngine  # noqa: E402
from app.research.pipeline import (  # noqa: E402
    build_forward_return_labels, cross_sectional_ic, stack_to_panel,
)
from app.research.statistics import deflated_sharpe_ratio  # noqa: E402

CACHE = Path(__file__).resolve().parents[2] / "data_cache"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

HORIZON = 5
MIN_NAMES = 20
COST_PER_SIDE = 0.0011
HOLDOUT_START = pd.Timestamp("2022-01-01")
N_TRIALS = 10          # honest count from the leaderboard run

# Coarse sector map for concentration analysis. Approximate but sufficient to
# answer "is the entire result one sector?".
SECTORS = {
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "MPHASIS": "IT", "PERSISTENT": "IT", "LTIM": "IT", "OFSS": "IT",
    "HDFCBANK": "FIN", "ICICIBANK": "FIN", "SBIN": "FIN", "KOTAKBANK": "FIN",
    "AXISBANK": "FIN", "INDUSINDBK": "FIN", "BANKBARODA": "FIN", "PNB": "FIN",
    "CANBK": "FIN", "BAJFINANCE": "FIN", "BAJAJFINSV": "FIN", "SBILIFE": "FIN",
    "HDFCLIFE": "FIN", "MUTHOOTFIN": "FIN", "CHOLAFIN": "FIN",
    "LICHSGFIN": "FIN", "RECLTD": "FIN", "PFC": "FIN", "ICICIPRULI": "FIN",
    "ICICIGI": "FIN",
    "SUNPHARMA": "PHARMA", "CIPLA": "PHARMA", "DRREDDY": "PHARMA",
    "DIVISLAB": "PHARMA", "LUPIN": "PHARMA", "AUROPHARMA": "PHARMA",
    "TORNTPHARM": "PHARMA", "BIOCON": "PHARMA", "GLENMARK": "PHARMA",
    "ALKEM": "PHARMA", "ZYDUSLIFE": "PHARMA", "ABBOTINDIA": "PHARMA",
    "APOLLOHOSP": "PHARMA",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TATACONSUM": "FMCG", "DABUR": "FMCG",
    "GODREJCP": "FMCG", "MARICO": "FMCG", "COLPAL": "FMCG",
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "EICHERMOT": "AUTO",
    "HEROMOTOCO": "AUTO", "BAJAJ-AUTO": "AUTO", "M&M": "AUTO",
    "MOTHERSON": "AUTO", "BOSCHLTD": "AUTO", "ASHOKLEY": "AUTO",
    "TVSMOTOR": "AUTO", "BALKRISIND": "AUTO",
    "TATASTEEL": "METAL", "JSWSTEEL": "METAL", "HINDALCO": "METAL",
    "VEDL": "METAL", "JINDALSTEL": "METAL", "SAIL": "METAL", "NMDC": "METAL",
    "COALINDIA": "METAL",
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "BPCL": "ENERGY",
    "IOC": "ENERGY", "GAIL": "ENERGY", "NTPC": "ENERGY",
    "POWERGRID": "ENERGY", "TATAPOWER": "ENERGY",
    "ULTRACEMCO": "CEMENT", "SHREECEM": "CEMENT", "AMBUJACEM": "CEMENT",
    "ACC": "CEMENT", "GRASIM": "CEMENT",
}


# ---------------------------------------------------------------------------

def load():
    close = pd.read_parquet(CACHE / "PANEL_close_clean.parquet")
    volume = pd.read_parquet(CACHE / "PANEL_volume.parquet").reindex_like(close)
    high = pd.read_parquet(CACHE / "PANEL_high.parquet").reindex_like(close)
    low = pd.read_parquet(CACHE / "PANEL_low.parquet").reindex_like(close)
    bench = pd.read_parquet(CACHE / "PANEL_bench.parquet")["close"]
    first = bench.dropna().index.min()
    close = close[close.index >= first]
    volume, high, low = (d[d.index >= first] for d in (volume, high, low))
    return close, volume, high, low, bench.reindex(close.index).ffill()


def momentum_scores(close: pd.DataFrame, skip: int = 21, lookback: int = 252):
    """Generalized 12-1 momentum so the lookback can be perturbed."""
    return np.log(close.shift(skip) / close.shift(lookback))


def long_only_excess(
    scores_wide: pd.DataFrame,
    labels_wide: pd.DataFrame,
    rebal: pd.DatetimeIndex,
    cost_mult: float = 1.0,
    quantile: float = 0.2,
    symbols: list[str] | None = None,
) -> tuple[pd.Series, float, dict]:
    """
    Long-only top-quintile excess return over the equal-weight universe.

    Returns (per-rebalance excess series, turnover, per-rebalance holdings).
    """
    if symbols is not None:
        scores_wide = scores_wide[symbols]
        labels_wide = labels_wide[symbols]

    out, holdings, prev, turns = {}, {}, None, []
    for d in rebal:
        if d not in scores_wide.index:
            continue
        s = scores_wide.loc[d].dropna()
        y = labels_wide.loc[d].dropna()
        common = s.index.intersection(y.index)
        if len(common) < MIN_NAMES:
            continue
        s, y = s[common], y[common]
        k = max(1, int(len(s) * quantile))
        top = s.sort_values(ascending=False).head(k).index
        if prev is not None and len(prev):
            turns.append(1.0 - len(set(top) & set(prev)) / len(prev))
        prev = top
        holdings[d] = list(top)
        out[d] = float(y[top].mean() - y.mean())

    ser = pd.Series(out).sort_index()
    turnover = float(np.mean(turns)) if turns else 1.0
    ser = ser - turnover * 2 * COST_PER_SIDE * cost_mult
    return ser, turnover, holdings


def stats(x: pd.Series) -> dict:
    x = x.dropna()
    ppy = 252.0 / HORIZON
    if len(x) < 10 or x.std(ddof=1) == 0:
        return {"n": len(x), "sharpe": float("nan"), "ann_return": float("nan")}
    return {
        "n": int(len(x)),
        "sharpe": round(float(x.mean() / x.std(ddof=1) * np.sqrt(ppy)), 3),
        "ann_return": round(float(x.mean() * ppy), 5),
    }


# ---------------------------------------------------------------------------

def main() -> None:
    close, volume, high, low, bench = load()
    labels = build_forward_return_labels(close, HORIZON)
    labels = labels.sub(labels.mean(axis=1), axis=0)   # market-neutral

    dev = close.index[close.index < HOLDOUT_START]
    rebal_dev = dev[::HORIZON]

    base = momentum_scores(close)
    results: dict = {}

    print("=" * 78)
    print("ROBUSTNESS — attempting to DISPROVE 12-1 momentum (long-only)")
    print("=" * 78)
    print("Baseline from the leaderboard: +4.25%/yr excess vs equal weight,")
    print("Sharpe 0.41, DSR 0.461 (NOT significant at 0.95).")
    print()

    # ---- 1. COST STRESS -------------------------------------------------
    print("-" * 78)
    print("1. COST / SLIPPAGE STRESS")
    print("-" * 78)
    cost_rows = {}
    for m in (0.5, 1.0, 1.5, 2.0, 3.0):
        s, turn, _ = long_only_excess(base, labels, rebal_dev, cost_mult=m)
        st = stats(s)
        cost_rows[f"{m}x"] = st
        print(f"  cost x{m:<4}  excess={st['ann_return']:+7.2%}/yr  "
              f"Sharpe={st['sharpe']:+.3f}")
    results["cost_stress"] = cost_rows

    # ---- 2. PARAMETER STRESS -------------------------------------------
    print()
    print("-" * 78)
    print("2. PARAMETER STRESS (momentum lookback / skip)")
    print("-" * 78)
    par_rows = {}
    for lb in (126, 189, 252, 315, 378):
        for sk in (10, 21, 42):
            s, _, _ = long_only_excess(momentum_scores(close, sk, lb),
                                       labels, rebal_dev)
            st = stats(s)
            par_rows[f"lb{lb}_skip{sk}"] = st
    shp = [v["sharpe"] for v in par_rows.values() if np.isfinite(v["sharpe"])]
    for k, v in par_rows.items():
        print(f"  {k:16s} excess={v['ann_return']:+7.2%}/yr  Sharpe={v['sharpe']:+.3f}")
    print(f"  --> across {len(shp)} settings: mean Sharpe={np.mean(shp):+.3f}, "
          f"min={np.min(shp):+.3f}, max={np.max(shp):+.3f}, "
          f"{sum(1 for x in shp if x > 0)}/{len(shp)} positive")
    results["parameter_stress"] = par_rows

    # ---- 3. TIME STRESS -------------------------------------------------
    print()
    print("-" * 78)
    print("3. TIME STRESS (per calendar year)")
    print("-" * 78)
    s_base, turn, holdings = long_only_excess(base, labels, rebal_dev)
    by_year = s_base.groupby(s_base.index.year).agg(["mean", "count"])
    ppy = 252.0 / HORIZON
    yr_rows = {}
    for yr, row in by_year.iterrows():
        ann = row["mean"] * ppy
        yr_rows[int(yr)] = round(float(ann), 5)
        bar = "+" if ann > 0 else "-"
        print(f"  {yr}  excess={ann:+7.2%}/yr  ({int(row['count'])} rebalances) {bar}")
    pos_years = sum(1 for v in yr_rows.values() if v > 0)
    print(f"  --> {pos_years}/{len(yr_rows)} years positive")
    results["time_stress"] = yr_rows
    results["positive_years"] = f"{pos_years}/{len(yr_rows)}"

    # ---- 4. LEAVE-ONE-YEAR-OUT -----------------------------------------
    print()
    print("-" * 78)
    print("4. CONCENTRATION — leave one year out")
    print("-" * 78)
    loo = {}
    for yr in sorted(yr_rows):
        sub = s_base[s_base.index.year != yr]
        loo[int(yr)] = stats(sub)
    worst = min(loo.items(), key=lambda kv: kv[1]["sharpe"])
    for yr, st in loo.items():
        print(f"  excluding {yr}: Sharpe={st['sharpe']:+.3f} "
              f"excess={st['ann_return']:+7.2%}/yr")
    print(f"  --> removing {worst[0]} drops Sharpe to {worst[1]['sharpe']:+.3f} "
          f"(full-sample {stats(s_base)['sharpe']:+.3f})")
    results["leave_one_year_out"] = loo

    # ---- 5. LEAVE-ONE-SECTOR-OUT ---------------------------------------
    print()
    print("-" * 78)
    print("5. CONCENTRATION — leave one sector out")
    print("-" * 78)
    sec_rows = {}
    all_syms = [c for c in close.columns]
    for sec in sorted(set(SECTORS.values())):
        keep = [s for s in all_syms if SECTORS.get(s) != sec]
        if len(keep) < 40:
            continue
        ss, _, _ = long_only_excess(base, labels, rebal_dev, symbols=keep)
        sec_rows[sec] = stats(ss)
        print(f"  excluding {sec:8s} ({len(all_syms)-len(keep):2d} names): "
              f"Sharpe={sec_rows[sec]['sharpe']:+.3f}  "
              f"excess={sec_rows[sec]['ann_return']:+7.2%}/yr")
    results["leave_one_sector_out"] = sec_rows

    # ---- 6. TOP-REBALANCE CONCENTRATION --------------------------------
    print()
    print("-" * 78)
    print("6. CONCENTRATION — is the result a handful of rebalances?")
    print("-" * 78)
    ranked = s_base.sort_values(ascending=False)
    full = stats(s_base)
    conc = {"full": full}
    for n in (1, 5, 10, 25):
        trimmed = s_base.drop(ranked.head(n).index)
        conc[f"drop_top_{n}"] = stats(trimmed)
        print(f"  dropping best {n:2d} of {len(s_base)} rebalances: "
              f"Sharpe={conc[f'drop_top_{n}']['sharpe']:+.3f}  "
              f"excess={conc[f'drop_top_{n}']['ann_return']:+7.2%}/yr")
    results["rebalance_concentration"] = conc

    # ---- 7. UNIVERSE STRESS --------------------------------------------
    print()
    print("-" * 78)
    print("7. UNIVERSE STRESS (random half-universes)")
    print("-" * 78)
    rng = np.random.default_rng(42)
    uni = []
    for i in range(20):
        pick = list(rng.choice(all_syms, size=len(all_syms) // 2, replace=False))
        ss, _, _ = long_only_excess(base, labels, rebal_dev, symbols=pick)
        st = stats(ss)
        if np.isfinite(st["sharpe"]):
            uni.append(st["sharpe"])
    print(f"  20 random half-universes: mean Sharpe={np.mean(uni):+.3f}  "
          f"sd={np.std(uni):.3f}  min={np.min(uni):+.3f}  max={np.max(uni):+.3f}  "
          f"{sum(1 for x in uni if x>0)}/{len(uni)} positive")
    results["universe_stress"] = {
        "mean_sharpe": round(float(np.mean(uni)), 3),
        "sd": round(float(np.std(uni)), 3),
        "positive": f"{sum(1 for x in uni if x>0)}/{len(uni)}",
    }

    # ---- 8. REGIME STRESS ----------------------------------------------
    print()
    print("-" * 78)
    print("8. REGIME STRESS")
    print("-" * 78)
    bret = np.log(bench / bench.shift(1))
    sma200 = bench.rolling(200).mean()
    vol60 = bret.rolling(60).std() * np.sqrt(252)
    vol_med = vol60.median()
    reg_rows = {}
    regimes = {
        "bull (above 200DMA)": bench > sma200,
        "bear (below 200DMA)": bench <= sma200,
        "high vol": vol60 > vol_med,
        "low vol": vol60 <= vol_med,
    }
    for label, mask in regimes.items():
        d = s_base.index[s_base.index.isin(bench.index[mask.fillna(False)])]
        st = stats(s_base.reindex(d))
        reg_rows[label] = st
        print(f"  {label:22s} n={st['n']:4d}  Sharpe={st['sharpe']:+.3f}  "
              f"excess={st['ann_return']:+7.2%}/yr")
    results["regime_stress"] = reg_rows

    # ---- 9. FINAL HOLDOUT — touched once -------------------------------
    print()
    print("=" * 78)
    print("9. FINAL HOLDOUT (2022-01-01 onward) — evaluated ONCE")
    print("=" * 78)
    hold = close.index[close.index >= HOLDOUT_START]
    rebal_hold = hold[::HORIZON]
    s_hold, turn_h, _ = long_only_excess(base, labels, rebal_hold)
    st_h = stats(s_hold)
    dsr_h = deflated_sharpe_ratio(s_hold.dropna().values, n_trials=N_TRIALS)
    print(f"  periods={st_h['n']}  turnover={turn_h:.0%}")
    print(f"  excess vs equal-weight = {st_h['ann_return']:+.2%}/yr  "
          f"Sharpe={st_h['sharpe']:+.3f}")
    print(f"  DSR (n_trials={N_TRIALS}) = {dsr_h.deflated_sharpe_ratio:.4f}  "
          f"{'SIGNIFICANT' if dsr_h.is_significant else 'NOT SIGNIFICANT'}")
    print()
    print(f"  development-period Sharpe was {stats(s_base)['sharpe']:+.3f}; "
          f"holdout is {st_h['sharpe']:+.3f}")
    results["final_holdout"] = {
        **st_h,
        "turnover": round(turn_h, 4),
        "dsr": round(dsr_h.deflated_sharpe_ratio, 4),
        "dsr_significant": bool(dsr_h.is_significant),
        "development_sharpe": stats(s_base)["sharpe"],
    }

    (OUT / "swing_robustness.json").write_text(json.dumps(results, indent=2))
    print()
    print(f"written to {OUT / 'swing_robustness.json'}")


if __name__ == "__main__":
    main()
