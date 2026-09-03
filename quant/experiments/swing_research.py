"""
Swing-horizon cross-sectional research on REAL NSE data.

WHAT THIS SCRIPT IS
-------------------
The honest attempt to answer one question:

    Does any signal in this feature set predict 5-day forward relative returns
    across NSE large/mid-caps, well enough to survive costs and the fact that
    we searched for it?

It runs entirely through the production modules — `app.data.features`,
`app.research.pipeline`, `app.research.statistics`. There are no research-only
reimplementations, because a research result computed by different code than
production runs is not a result about production.

METHOD
------
1. Load the real panel, cleaned of unadjusted corporate actions.
2. Compute features with the production FeatureEngine (all verified causal).
3. Reserve a FINAL HOLDOUT that nothing in the selection process touches.
4. Evaluate rule-based baselines BEFORE any model. If a plain momentum rule
   works and a gradient booster does not beat it, the booster is not the
   answer.
5. Evaluate fitted models under purged, embargoed walk-forward.
6. Score everything on the same out-of-sample dates so the leaderboard is
   comparable.
7. Apply the multiple-testing correction using the HONEST trial count —
   every rule and every model tried, not just the winner.

WHAT WOULD MAKE THIS RESULT MEANINGLESS
---------------------------------------
The universe is survivorship-filtered: delisted names are absent from the
data source (verified — Satyam, DHFL and Videocon all return zero rows over
periods when they were listed). That biases every result UPWARD.

The consequence is asymmetric and worth stating before any number is read:
a POSITIVE result here is uninterpretable, because it cannot be separated
from the bias. A NEGATIVE result is robust, because a strategy that fails
even after the losers have been deleted from the sample has genuinely failed.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.data.features import FeatureEngine  # noqa: E402
from app.research.pipeline import (  # noqa: E402
    build_forward_return_labels,
    cross_sectional_ic,
    long_short_returns,
    stack_to_panel,
)
from app.research.statistics import (  # noqa: E402
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from app.research.validation import PurgedWalkForward, deflated_tstat  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("swing_research")
log.setLevel(logging.INFO)

CACHE = Path(__file__).resolve().parents[2] / "data_cache"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

HORIZON = 5                    # 5 trading-day forward return
FINAL_HOLDOUT_START = "2022-01-01"   # never used for any selection decision
MIN_NAMES = 20
COST_PER_SIDE = 0.0011         # ~11 bps, delivery round trip ≈ 22bps (see costs.py)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_panel():
    close = pd.read_parquet(CACHE / "PANEL_close_clean.parquet")
    volume = pd.read_parquet(CACHE / "PANEL_volume.parquet").reindex_like(close)
    high = pd.read_parquet(CACHE / "PANEL_high.parquet").reindex_like(close)
    low = pd.read_parquet(CACHE / "PANEL_low.parquet").reindex_like(close)
    bench = pd.read_parquet(CACHE / "PANEL_bench.parquet")["close"]

    # The benchmark series starts later than the price panel. Silently
    # forward-filling from nothing leaves NaN at the head, which propagates
    # into every market-relative feature and into the CAGR calculation. Trim
    # the study to the period where the benchmark actually exists.
    first = bench.dropna().index.min()
    close = close[close.index >= first]
    volume, high, low = (d[d.index >= first] for d in (volume, high, low))
    bench = bench.reindex(close.index).ffill()
    return close, volume, high, low, bench


# Features excluded from MODEL training (they remain available as baselines).
#
# pvt and obv_slope are cumulative sums from the start of the sample, so their
# LEVEL depends on an arbitrary start date and drifts without bound. A model
# trained on 2006-2015 levels sees entirely different values in 2016-2021.
# They are non-stationary and will not generalize.
NON_STATIONARY = {"pvt", "obv_slope_10d"}


def build_features(close, volume, high, low, bench) -> pd.DataFrame:
    fe = FeatureEngine()
    log.info("computing features for %d symbols x %d dates...",
             close.shape[1], close.shape[0])
    feats = fe.compute_all_features(
        prices_df=close, volume_df=volume, nifty_df=bench.to_frame("close"),
        high_df=high, low_df=low,
    )
    log.info("features: %s", feats.shape)
    return feats


# ---------------------------------------------------------------------------
# Scoring rules (baselines) — no fitting, so no training window needed
# ---------------------------------------------------------------------------

BASELINES: dict[str, tuple[str, float]] = {
    # name -> (feature column, sign).  sign=+1 means higher feature => buy.
    "momentum_12_1":      ("momentum_12_1", +1.0),
    "reversal_5d":        ("log_return_5d", -1.0),
    "reversal_21d":       ("log_return_21d", -1.0),
    "low_volatility":     ("realized_vol_63d", -1.0),
    "trend_sma200":       ("price_to_sma200", +1.0),
    "rsi_contrarian":     ("rsi_14", -1.0),
    "excess_vs_nifty_21": ("excess_return_vs_nifty_21d", +1.0),
    "high_volume":        ("relative_volume_zscore", +1.0),
}


def _quantile_leg_returns(
    scores: pd.Series, labels: pd.Series, quantile: float = 0.2,
) -> pd.DataFrame:
    """Per-date top-quintile and bottom-quintile mean forward returns."""
    df = pd.DataFrame({"s": scores, "y": labels}).dropna()
    rows = {}
    for date, g in df.groupby(level="date"):
        if len(g) < MIN_NAMES:
            continue
        k = max(1, int(len(g) * quantile))
        r = g.sort_values("s", ascending=False)
        rows[date] = {
            "long": float(r.head(k)["y"].mean()),
            "short": float(r.tail(k)["y"].mean()),
            "names": tuple(r.head(k).index.get_level_values("symbol")),
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def evaluate_scores(
    scores: pd.Series, labels: pd.Series, name: str, n_trials: int,
    rebalance_dates: pd.DatetimeIndex | None = None,
) -> dict:
    """
    Score a prediction series.

    NON-OVERLAPPING SAMPLING
    ------------------------
    A 5-day forward return sampled every day produces overlapping periods.
    Overlapping returns are strongly autocorrelated, so their standard
    deviation understates the true dispersion and the resulting Sharpe is
    inflated — and annualizing by sqrt(252/5) on a daily-sampled series
    compounds the error. Positions are therefore sampled every HORIZON-th
    date, giving genuinely independent, tradeable rebalance periods.

    LONG-ONLY IS REPORTED SEPARATELY AND MATTERS MORE
    -------------------------------------------------
    A quintile long-short book is not implementable by a retail investor in
    the Indian cash segment: naked short selling is prohibited, and SLB
    borrow is thin and expensive. The long-short figure is a research
    diagnostic. The long-only figure, measured against the same universe's
    equal-weight return, is what this platform could actually trade.
    """
    ic = cross_sectional_ic(scores, labels, min_names_per_date=MIN_NAMES)
    legs = _quantile_leg_returns(scores, labels)

    if len(ic) < 30 or len(legs) < 30:
        return {"strategy": name, "usable": False,
                "reason": f"only {len(ic)} scored dates"}

    # Non-overlapping rebalance dates. A COMMON grid is used across every
    # candidate so the leaderboard compares like with like and the CSCV
    # overfitting test has aligned columns to work with.
    if rebalance_dates is not None:
        legs_no = legs.reindex(legs.index.intersection(rebalance_dates))
    else:
        legs_no = legs.iloc[::HORIZON]
    if len(legs_no) < 30:
        return {"strategy": name, "usable": False,
                "reason": f"only {len(legs_no)} non-overlapping periods"}

    # Universe equal-weight return over the same periods = the passive
    # alternative within this universe.
    ew = labels.groupby(level="date").mean().reindex(legs_no.index)

    # Turnover: fraction of the long book replaced each rebalance.
    prev, turns = None, []
    for names in legs_no["names"]:
        if prev is not None and len(prev):
            turns.append(1.0 - len(set(names) & set(prev)) / len(prev))
        prev = names
    turnover = float(np.mean(turns)) if turns else 1.0

    # Cost is charged in proportion to what actually trades.
    long_cost = turnover * 2 * COST_PER_SIDE
    ls_cost = 2 * turnover * 2 * COST_PER_SIDE

    ls_gross = legs_no["long"] - legs_no["short"]
    ls_net = ls_gross - ls_cost
    lo_gross = legs_no["long"]
    lo_net = lo_gross - long_cost
    lo_excess = lo_net - ew            # long-only minus equal-weight universe

    periods_per_year = 252.0 / HORIZON
    ann = np.sqrt(periods_per_year)

    def sharpe(x: pd.Series) -> float:
        x = x.dropna()
        sd = x.std(ddof=1)
        return float(x.mean() / sd * ann) if sd > 0 and len(x) > 2 else float("nan")

    def maxdd(x: pd.Series) -> float:
        c = (1 + x.dropna()).cumprod()
        return float((c / c.cummax() - 1).min()) if len(c) else float("nan")

    ic_mean, ic_sd = float(ic.mean()), float(ic.std(ddof=1))
    dsr = deflated_sharpe_ratio(ls_net.dropna().values, n_trials=n_trials)
    dsr_lo = deflated_sharpe_ratio(lo_excess.dropna().values, n_trials=n_trials)

    return {
        "strategy": name,
        "usable": True,
        "n_ic_dates": int(len(ic)),
        "n_rebalances": int(len(legs_no)),
        "turnover_per_rebalance": round(turnover, 4),
        "ic_mean": round(ic_mean, 5),
        "ic_std": round(ic_sd, 5),
        "icir_ann": round(ic_mean / ic_sd * ann, 4) if ic_sd > 0 else None,
        "ic_t_deflated": round(deflated_tstat(ic_mean, ic_sd, len(ic), HORIZON), 3),
        # long-short (research diagnostic, NOT retail-implementable)
        "ls_sharpe_gross": round(sharpe(ls_gross), 4),
        "ls_sharpe_net": round(sharpe(ls_net), 4),
        "ls_return_ann_net": round(float(ls_net.mean() * periods_per_year), 5),
        "ls_max_drawdown": round(maxdd(ls_net), 4),
        "ls_dsr": round(dsr.deflated_sharpe_ratio, 4),
        "ls_dsr_significant": bool(dsr.is_significant),
        # long-only (what could actually be traded)
        "lo_sharpe_net": round(sharpe(lo_net), 4),
        "lo_return_ann_net": round(float(lo_net.mean() * periods_per_year), 5),
        "lo_max_drawdown": round(maxdd(lo_net), 4),
        "lo_excess_vs_ew_ann": round(float(lo_excess.mean() * periods_per_year), 5),
        "lo_excess_sharpe": round(sharpe(lo_excess), 4),
        "lo_dsr_vs_ew": round(dsr_lo.deflated_sharpe_ratio, 4),
        "lo_beats_ew_significantly": bool(dsr_lo.is_significant),
        "_ls_net": ls_net,
    }


# ---------------------------------------------------------------------------
# Fitted models
# ---------------------------------------------------------------------------

def make_models() -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    }
    try:
        import lightgbm as lgb

        def gbm():
            return lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                num_leaves=15, min_child_samples=100, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
            )

        models["lightgbm"] = gbm
    except Exception as exc:  # pragma: no cover
        log.warning("LightGBM unavailable: %s", exc)
    return models


def walk_forward_predictions(
    X: pd.DataFrame, y: pd.Series, model_factory, n_splits: int = 6,
) -> pd.Series:
    """
    Purged, embargoed walk-forward. Returns OOS predictions only.

    A fresh model is fitted per fold. Reusing a fitted model would carry
    later-fold information backwards.
    """
    dates = pd.DatetimeIndex(X.index.get_level_values("date").unique()).sort_values()
    splitter = PurgedWalkForward(
        n_splits=n_splits, label_horizon=HORIZON, embargo_frac=0.01, expanding=True,
    )
    preds = []
    for split in splitter.split(dates):
        tr_dates, te_dates = dates[split.train_idx], dates[split.test_idx]
        dv = X.index.get_level_values("date")
        Xtr, ytr = X[dv.isin(tr_dates)], y[dv.isin(tr_dates)]
        Xte = X[dv.isin(te_dates)]

        ok = Xtr.notna().all(axis=1) & ytr.notna()
        Xtr, ytr = Xtr[ok], ytr[ok]
        Xte = Xte[Xte.notna().all(axis=1)]
        if len(Xtr) < 500 or len(Xte) < MIN_NAMES:
            continue

        m = model_factory()
        m.fit(Xtr.values, ytr.values)
        preds.append(pd.Series(m.predict(Xte.values), index=Xte.index))
    if not preds:
        return pd.Series(dtype=float)
    return pd.concat(preds).sort_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    close, volume, high, low, bench = load_panel()
    print("=" * 78)
    print("SWING RESEARCH — REAL NSE DATA")
    print("=" * 78)
    print(f"universe : {close.shape[1]} symbols")
    print(f"period   : {close.index.min().date()} .. {close.index.max().date()}")
    print(f"horizon  : {HORIZON} trading days")
    print(f"holdout  : {FINAL_HOLDOUT_START} onward is NEVER used for selection")
    print()

    feats = build_features(close, volume, high, low, bench)
    labels = build_forward_return_labels(close, HORIZON)
    X, y, _ = stack_to_panel(feats, labels, list(close.columns))

    # Drop features that are entirely missing. Requiring every feature to be
    # present would otherwise leave zero usable training rows — which is
    # exactly what happened before ATR was given real high/low input.
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        print(f"dropping {len(all_nan)} all-NaN features: {all_nan}")
        X = X.drop(columns=all_nan)

    # Neutralize the market factor: a 5-day forward return is dominated by
    # the market move that day, which no stock-selection signal controls.
    # Ranking is about relative performance, so labels are demeaned within
    # each date. Without this, IC largely measures market beta.
    y = y - y.groupby(level="date").transform("mean")

    dv = X.index.get_level_values("date")
    holdout_mask = dv >= pd.Timestamp(FINAL_HOLDOUT_START)
    X_dev, y_dev = X[~holdout_mask], y[~holdout_mask]
    X_hold, y_hold = X[holdout_mask], y[holdout_mask]

    print(f"development rows : {len(X_dev):,}  ({X_dev.index.get_level_values('date').min().date()} .. {X_dev.index.get_level_values('date').max().date()})")
    print(f"holdout rows     : {len(X_hold):,}  ({X_hold.index.get_level_values('date').min().date()} .. {X_hold.index.get_level_values('date').max().date()})")
    print(f"features         : {X.shape[1]}")
    print()

    models = make_models()
    n_trials = len(BASELINES) + len(models)
    print(f"HONEST TRIAL COUNT: {n_trials} "
          f"({len(BASELINES)} rules + {len(models)} models)")
    print()

    results: list[dict] = []
    ls_matrix: dict[str, pd.Series] = {}

    # One shared rebalance grid for every candidate.
    dev_dates = pd.DatetimeIndex(
        X_dev.index.get_level_values('date').unique()
    ).sort_values()
    REBAL = dev_dates[::HORIZON]
    print(f'common rebalance grid: {len(REBAL)} non-overlapping periods')


    # ---- Baselines first -------------------------------------------------
    print("-" * 78)
    print("BASELINES (rule-based, no fitting)")
    print("-" * 78)
    for name, (col, sign) in BASELINES.items():
        if col not in X_dev.columns:
            print(f"  {name:22s} SKIPPED — feature '{col}' not present")
            continue
        scores = sign * X_dev[col]
        r = evaluate_scores(scores.dropna(), y_dev, name, n_trials, REBAL)
        results.append(r)
        if r.get("usable"):
            ls_matrix[name] = r.pop("_ls_net")
            print(f"  {name:22s} IC={r['ic_mean']:+.4f} t={r['ic_t_deflated']:+6.2f} | "
                  f"LS_SR={r['ls_sharpe_net']:+.2f} DSR={r['ls_dsr']:.3f}"
                  f"{'*' if r['ls_dsr_significant'] else ' '} | "
                  f"LO_excess={r['lo_excess_vs_ew_ann']:+.2%}/yr "
                  f"SR={r['lo_excess_sharpe']:+.2f} DSR={r['lo_dsr_vs_ew']:.3f}"
                  f"{'*' if r['lo_beats_ew_significantly'] else ' '} | "
                  f"turn={r['turnover_per_rebalance']:.0%}")
        else:
            print(f"  {name:22s} UNUSABLE: {r['reason']}")

    # ---- Fitted models ---------------------------------------------------
    print()
    print("-" * 78)
    print("FITTED MODELS (purged walk-forward, retrained per fold)")
    print("-" * 78)
    model_cols = [c for c in X_dev.columns if c not in NON_STATIONARY]
    dropped = sorted(set(X_dev.columns) - set(model_cols))
    if dropped:
        print(f"  (excluded from model input as non-stationary: {dropped})")
    X_model = X_dev[model_cols]

    for name, factory in models.items():
        preds = walk_forward_predictions(X_model, y_dev, factory)
        if preds.empty:
            print(f"  {name:22s} produced no OOS predictions")
            continue
        r = evaluate_scores(preds, y_dev, name, n_trials, REBAL)
        results.append(r)
        if r.get("usable"):
            ls_matrix[name] = r.pop("_ls_net")
            print(f"  {name:22s} IC={r['ic_mean']:+.4f} t={r['ic_t_deflated']:+6.2f} | "
                  f"LS_SR={r['ls_sharpe_net']:+.2f} DSR={r['ls_dsr']:.3f}"
                  f"{'*' if r['ls_dsr_significant'] else ' '} | "
                  f"LO_excess={r['lo_excess_vs_ew_ann']:+.2%}/yr "
                  f"SR={r['lo_excess_sharpe']:+.2f} DSR={r['lo_dsr_vs_ew']:.3f}"
                  f"{'*' if r['lo_beats_ew_significantly'] else ' '} | "
                  f"turn={r['turnover_per_rebalance']:.0%}")
        else:
            print(f"  {name:22s} UNUSABLE: {r['reason']}")

    # ---- Passive benchmark ----------------------------------------------
    print()
    print("-" * 78)
    print("PASSIVE BENCHMARK")
    print("-" * 78)
    dev_bench = bench[bench.index < pd.Timestamp(FINAL_HOLDOUT_START)].dropna()
    br = np.log(dev_bench / dev_bench.shift(1)).dropna()
    years = (dev_bench.index[-1] - dev_bench.index[0]).days / 365.25
    cagr = (dev_bench.iloc[-1] / dev_bench.iloc[0]) ** (1 / years) - 1
    bsr = float(br.mean() / br.std(ddof=1) * np.sqrt(252))
    bdd = float((dev_bench / dev_bench.cummax() - 1).min())
    print(f"  NIFTY 50 buy-and-hold  CAGR={cagr:+.2%}  Sharpe={bsr:+.3f}  MaxDD={bdd:.1%}")

    # Equal-weight universe: the passive alternative a stock picker must beat.
    ew_daily = np.log(close / close.shift(1)).mean(axis=1)
    ew_dev = ew_daily[ew_daily.index < pd.Timestamp(FINAL_HOLDOUT_START)].dropna()
    ew_curve = ew_dev.cumsum().apply(np.exp)
    ew_years = (ew_dev.index[-1] - ew_dev.index[0]).days / 365.25
    ew_cagr = ew_curve.iloc[-1] ** (1 / ew_years) - 1
    ew_sr = float(ew_dev.mean() / ew_dev.std(ddof=1) * np.sqrt(252))
    ew_dd = float((ew_curve / ew_curve.cummax() - 1).min())
    print(f"  Equal-weight universe  CAGR={ew_cagr:+.2%}  Sharpe={ew_sr:+.3f}  MaxDD={ew_dd:.1%}")
    print(f"  (the equal-weight figure is itself survivorship-inflated — these")
    print(f"   99 names are the ones that survived to 2024)")

    # ---- PBO across all candidates --------------------------------------
    print()
    print("-" * 78)
    print("OVERFITTING RISK (CSCV across all candidates)")
    print("-" * 78)
    if len(ls_matrix) >= 2:
        common = None
        for s in ls_matrix.values():
            common = s.index if common is None else common.intersection(s.index)
        M = np.column_stack([ls_matrix[k].reindex(common).values for k in ls_matrix])
        M = M[~np.isnan(M).any(axis=1)]
        if len(M) >= 40:
            pbo = probability_of_backtest_overfitting(M, n_partitions=10)
            print(f"  {pbo.summary()}")
        else:
            print(f"  insufficient common dates ({len(M)}) for CSCV")

    # ---- Persist ---------------------------------------------------------
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    (OUT / "swing_leaderboard.json").write_text(json.dumps({
        "horizon_days": HORIZON,
        "n_trials": n_trials,
        "universe_size": int(close.shape[1]),
        "development_period": [str(X_dev.index.get_level_values('date').min().date()),
                               str(X_dev.index.get_level_values('date').max().date())],
        "holdout_period": [str(X_hold.index.get_level_values('date').min().date()),
                           str(X_hold.index.get_level_values('date').max().date())],
        "benchmark": {"cagr": round(float(cagr), 5), "sharpe": round(bsr, 4),
                      "max_drawdown": round(bdd, 4)},
        "cost_per_side": COST_PER_SIDE,
        "results": clean,
    }, indent=2))
    print()
    print(f"leaderboard written to {OUT / 'swing_leaderboard.json'}")


if __name__ == "__main__":
    main()
