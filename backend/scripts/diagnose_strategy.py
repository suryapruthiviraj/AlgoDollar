#!/usr/bin/env python
"""
Diagnose the corrected point-in-time result. CHANGES NOTHING.

Reproduces the study's exact walk-forward configuration and decomposes the
resulting out-of-sample series. No threshold, parameter, universe or cost
assumption is altered anywhere in this file — every number below describes the
strategy as it already stands.

Factor proxies are built FROM THE SAME UNIVERSE using the same point-in-time
membership, so a factor's own returns are subject to the same eligibility rules
as the strategy's. Value is absent because no point-in-time fundamentals exist;
that is reported rather than approximated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.research.backtest import (  # noqa: E402
    TRADING_DAYS,
    cross_sectional_weights,
    run_backtest,
)
from app.research.datasets import DEFAULT_ROOT, build_dataset_bundle  # noqa: E402
from app.research.signals import BASELINE_SIGNALS  # noqa: E402
from app.research.statistics import stationary_bootstrap  # noqa: E402
from app.research.study import walk_forward_selection  # noqa: E402
from app.research.universe import (  # noqa: E402
    PointInTimeUniverse,
    load_listing_dates,
)

OUT = DEFAULT_ROOT / "diagnosis.json"


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return float("nan")
    total = float((1 + r).prod() - 1)
    yrs = len(r) / TRADING_DAYS
    return float((1 + total) ** (1 / yrs) - 1) if yrs > 0 and total > -1 else float("nan")


def ci90(r: pd.Series, n: int = 1000) -> tuple[float, float]:
    a = r.dropna().to_numpy()
    if len(a) < 60:
        return (float("nan"), float("nan"))
    sims = stationary_bootstrap(a, n_simulations=n, random_seed=7)
    s = [
        float(np.mean(p) / np.std(p, ddof=1) * np.sqrt(TRADING_DAYS))
        if np.std(p, ddof=1) > 0 else 0.0
        for p in sims
    ]
    return (round(float(np.percentile(s, 5)), 4), round(float(np.percentile(s, 95)), 4))


def main() -> int:
    bundle = build_dataset_bundle()
    pool = bundle.daily.symbols()
    px = bundle.daily.panel(pool)
    vol = bundle.daily.panel(pool, field="volume")
    bench = bundle.benchmark.returns()
    pit = PointInTimeUniverse(px, vol, listing_dates=load_listing_dates(DEFAULT_ROOT))

    signals = {n: f(px) for n, f in BASELINE_SIGNALS.items()}
    folds, oos = walk_forward_selection(
        signals, px, benchmark=bench, n_folds=6, cost_bps=25.0, universe=pit
    )
    oos = oos.dropna()
    b = bench.reindex(oos.index).dropna()
    common = oos.index.intersection(b.index)
    s_r, b_r = oos.loc[common], b.loc[common]

    doc: dict = {"n_obs": int(len(oos)),
                 "period": [str(oos.index[0].date()), str(oos.index[-1].date())]}

    # ---- 1. factor decomposition ------------------------------------- #
    # Each proxy is a long-short quintile portfolio built on the SAME
    # point-in-time universe, so its returns face the same eligibility rules.
    factors: dict[str, pd.Series] = {"MKT": b_r}
    proxy_defs = {
        "MOM": "momentum_12_1",
        "LOWVOL": "low_volatility",
        "REV": "short_term_reversal",
    }
    for label, sig_name in proxy_defs.items():
        res = run_backtest(
            signals[sig_name], px, benchmark=bench, cost_bps=0.0,
            long_only=False, universe=pit,
        )
        factors[label] = res.returns_net.reindex(common).fillna(0.0)

    # Liquidity/size proxy: long the least-traded eligible names, short the most.
    traded = (px * vol).rolling(63, min_periods=30).median()
    size_sig = -np.log(traded.replace(0, np.nan))
    factors["ILLIQ"] = run_backtest(
        size_sig, px, benchmark=bench, cost_bps=0.0, long_only=False, universe=pit
    ).returns_net.reindex(common).fillna(0.0)

    X = pd.DataFrame(factors).reindex(common).fillna(0.0)
    y = s_r.reindex(common).fillna(0.0)
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    coef, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
    resid = y.to_numpy() - A @ coef
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())

    mkt_only = np.column_stack([np.ones(len(X)), X["MKT"].to_numpy()])
    mcoef, *_ = np.linalg.lstsq(mkt_only, y.to_numpy(), rcond=None)
    mresid = y.to_numpy() - mkt_only @ mcoef

    doc["factor"] = {
        "raw_cagr": round(cagr(y), 4),
        "benchmark_cagr": round(cagr(b_r), 4),
        "excess_cagr": round(cagr(y) - cagr(b_r), 4),
        "market_beta": round(float(mcoef[1]), 4),
        "market_alpha_annual": round(float(mcoef[0]) * TRADING_DAYS, 4),
        "correlation_to_benchmark": round(float(np.corrcoef(y, X["MKT"])[0, 1]), 4),
        "beta_adjusted_cagr": round(cagr(pd.Series(mresid, index=common)), 4),
        "r2_market_only": round(1 - float((mresid ** 2).sum()) / ss_tot, 4),
        "r2_all_factors": round(1 - ss_res / ss_tot, 4),
        "loadings": {
            n: round(float(c), 4) for n, c in zip(X.columns, coef[1:])
        },
        "residual_alpha_annual": round(float(coef[0]) * TRADING_DAYS, 4),
        "residual_sharpe": round(sharpe(pd.Series(resid, index=common)), 4),
        "value_factor": "NOT TESTED — no point-in-time fundamentals exist",
    }

    # ---- 2. turnover ------------------------------------------------- #
    picked = max({f.selected_signal for f in folds},
                 key=lambda s: sum(1 for f in folds if f.selected_signal == s))
    w = cross_sectional_weights(
        signals[picked].where(signals[picked].notna()), rebalance_days=5
    )
    dw = (w - w.shift(1)).fillna(0.0)
    entries = dw.clip(lower=0).sum(axis=1)
    exits = (-dw.clip(upper=0)).sum(axis=1)
    held_prev = (w.shift(1).abs() > 0)
    held_now = (w.abs() > 0)
    new_names = (held_now & ~held_prev).sum(axis=1)
    dropped = (held_prev & ~held_now).sum(axis=1)
    resized = (held_now & held_prev & (dw.abs() > 1e-9)).sum(axis=1)

    doc["turnover"] = {
        "annual_total": round(float((entries + exits).mean() * TRADING_DAYS), 2),
        "annual_entries": round(float(entries.mean() * TRADING_DAYS), 2),
        "annual_exits": round(float(exits.mean() * TRADING_DAYS), 2),
        "mean_names_added_per_rebalance": round(float(new_names[new_names > 0].mean()), 2),
        "mean_names_dropped_per_rebalance": round(float(dropped[dropped > 0].mean()), 2),
        "mean_names_resized_per_rebalance": round(float(resized[resized > 0].mean()), 2),
        "mean_positions": round(float(held_now.sum(axis=1).mean()), 2),
        "note": (
            "Turnover is measured on the DELTA of weights, both sides. A name "
            "kept but re-weighted still trades, which is why turnover far "
            "exceeds the number of names entering and leaving."
        ),
    }

    cost_curve = {}
    for mult, bps in (("1x", 25.0), ("2x", 50.0), ("5x", 125.0),
                      ("10x", 250.0), ("actual", 25.0)):
        r = run_backtest(signals[picked], px, benchmark=bench,
                         cost_bps=bps, universe=pit)
        cost_curve[f"{mult} ({bps:g}bps)"] = {
            "sharpe": round(r.metrics.sharpe, 4),
            "cagr": round(r.metrics.cagr, 4),
            "excess_cagr": round(r.metrics.excess_cagr, 4)
            if r.metrics.excess_cagr is not None else None,
        }
    doc["cost_sensitivity"] = cost_curve

    # ---- 3. drawdown -------------------------------------------------- #
    curve = (1 + y).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1
    trough = dd.idxmin()
    start = curve.loc[:trough].idxmax()
    after = dd.loc[trough:]
    recovered = after[after >= -1e-9]
    rec_date = recovered.index[0] if len(recovered) else None

    dd_slice = y.loc[start:trough]
    b_slice = b_r.loc[start:trough]
    dd_beta = (
        float(np.cov(dd_slice, b_slice, ddof=1)[0, 1] / b_slice.var(ddof=1))
        if len(dd_slice) > 5 and b_slice.var(ddof=1) > 0 else float("nan")
    )
    doc["drawdown"] = {
        "max": round(float(dd.min()), 4),
        "peak_date": str(start.date()),
        "trough_date": str(trough.date()),
        "duration_sessions": int(len(y.loc[start:trough])),
        "recovery_date": str(rec_date.date()) if rec_date is not None else "NOT RECOVERED",
        "recovery_sessions": int(len(y.loc[trough:rec_date])) if rec_date is not None else None,
        "benchmark_return_over_same_window": round(float((1 + b_slice).prod() - 1), 4),
        "strategy_return_over_same_window": round(float((1 + dd_slice).prod() - 1), 4),
        "beta_during_drawdown": round(dd_beta, 4),
        "full_period_beta": doc["factor"]["market_beta"],
    }

    # ---- 4. year by year ---------------------------------------------- #
    years = []
    for yr, grp in y.groupby(y.index.year):
        bg = b_r[b_r.index.year == yr]
        c = (1 + grp).cumprod()
        years.append({
            "year": int(yr),
            "strategy": round(float((1 + grp).prod() - 1), 4),
            "benchmark": round(float((1 + bg).prod() - 1), 4) if len(bg) else None,
            "excess": round(float((1 + grp).prod() - (1 + bg).prod()), 4) if len(bg) else None,
            "sharpe": round(sharpe(grp), 3),
            "drawdown": round(float((c / c.cummax() - 1).min()), 4),
            "sessions": int(len(grp)),
        })
    doc["yearly"] = years
    pos = [r for r in years if (r["excess"] or 0) > 0]
    doc["positive_excess_years"] = f"{len(pos)}/{len(years)}"

    # ---- 5. regime (trailing-only labels) ------------------------------ #
    trail = (1 + b_r).rolling(126, min_periods=60).apply(np.prod, raw=True) - 1
    rvol = b_r.rolling(63, min_periods=30).std() * np.sqrt(TRADING_DAYS)
    vol_med = rvol.expanding(min_periods=120).median()

    regimes = {}
    for name, mask in (
        ("bull", trail > 0.10),
        ("bear", trail < -0.10),
        ("sideways", (trail >= -0.10) & (trail <= 0.10)),
        ("high_vol", rvol > vol_med),
        ("low_vol", rvol <= vol_med),
    ):
        m = mask.reindex(common).fillna(False)
        sub = y[m]
        if len(sub) < 30:
            regimes[name] = {"days": int(len(sub)), "note": "too few days"}
            continue
        lo, hi = ci90(sub, n=600)
        regimes[name] = {
            "days": int(len(sub)),
            "sharpe": round(sharpe(sub), 3),
            "mean_daily": round(float(sub.mean()), 6),
            "benchmark_mean_daily": round(float(b_r[m].mean()), 6),
            "sharpe_ci90": [lo, hi],
        }
    doc["regimes"] = regimes

    # ---- 6. recency ---------------------------------------------------- #
    halves = np.array_split(np.arange(len(y)), 2)
    thirds = np.array_split(np.arange(len(y)), 3)
    doc["recency"] = {
        "first_half": {"period": [str(y.index[halves[0][0]].date()), str(y.index[halves[0][-1]].date())],
                       "sharpe": round(sharpe(y.iloc[halves[0]]), 3),
                       "cagr": round(cagr(y.iloc[halves[0]]), 4)},
        "second_half": {"period": [str(y.index[halves[1][0]].date()), str(y.index[halves[1][-1]].date())],
                        "sharpe": round(sharpe(y.iloc[halves[1]]), 3),
                        "cagr": round(cagr(y.iloc[halves[1]]), 4)},
        "thirds": [
            {"period": [str(y.index[t[0]].date()), str(y.index[t[-1]].date())],
             "sharpe": round(sharpe(y.iloc[t]), 3), "cagr": round(cagr(y.iloc[t]), 4)}
            for t in thirds
        ],
        "rolling_252d_sharpe": {
            "min": round(float((y.rolling(252).mean() / y.rolling(252).std(ddof=1)
                                * np.sqrt(TRADING_DAYS)).min()), 3),
            "median": round(float((y.rolling(252).mean() / y.rolling(252).std(ddof=1)
                                   * np.sqrt(TRADING_DAYS)).median()), 3),
            "max": round(float((y.rolling(252).mean() / y.rolling(252).std(ddof=1)
                                * np.sqrt(TRADING_DAYS)).max()), 3),
            "pct_windows_below_zero": round(float(
                ((y.rolling(252).mean() / y.rolling(252).std(ddof=1)) < 0).mean()
            ), 4),
        },
        "last_252_sessions": {"sharpe": round(sharpe(y.tail(252)), 3),
                              "cagr": round(cagr(y.tail(252)), 4)},
        "last_504_sessions": {"sharpe": round(sharpe(y.tail(504)), 3),
                              "cagr": round(cagr(y.tail(504)), 4)},
        "fold_test_sharpes": [
            {"window": f"{f.test_start}..{f.test_end}",
             "selected": f.selected_signal, "test_sharpe": f.test_sharpe}
            for f in folds
        ],
    }

    # ---- 7. concentration (leave-out by DAY) ---------------------------- #
    # By day, not by name: the OOS series is a portfolio series, so a name's
    # contribution is not separable from it without re-running the backtest per
    # name — which would be a different experiment.
    leave_out = {}
    for k in (1, 5, 10, 25):
        drop = y.nlargest(k).index
        kept = y.drop(drop)
        leave_out[f"drop_top_{k}_days"] = {
            "sharpe": round(sharpe(kept), 4),
            "cagr": round(cagr(kept), 4),
            "excess_cagr": round(cagr(kept) - cagr(b_r.drop(drop, errors="ignore")), 4),
        }
    doc["concentration_days"] = {
        "baseline": {"sharpe": round(sharpe(y), 4), "cagr": round(cagr(y), 4)},
        **leave_out,
    }

    # Name-level: re-run holding out each of the largest holdings.
    res_full = run_backtest(signals[picked], px, benchmark=bench,
                            cost_bps=25.0, universe=pit)
    contrib = (res_full.weights.shift(1).fillna(0.0) *
               px.pct_change().reindex(res_full.weights.index)).sum(axis=0)
    top_names = contrib.nlargest(25).index.tolist()
    name_out = {}
    for k in (1, 5, 10, 25):
        drop = set(top_names[:k])
        keep_cols = [c for c in px.columns if c not in drop]
        r = run_backtest(
            signals[picked][keep_cols], px[keep_cols], benchmark=bench,
            cost_bps=25.0, universe=pit,
        )
        name_out[f"drop_top_{k}_names"] = {
            "sharpe": round(r.metrics.sharpe, 4),
            "cagr": round(r.metrics.cagr, 4),
            "excess_cagr": round(r.metrics.excess_cagr, 4)
            if r.metrics.excess_cagr is not None else None,
        }
    doc["concentration_names"] = {
        "baseline": {"sharpe": round(res_full.metrics.sharpe, 4),
                     "cagr": round(res_full.metrics.cagr, 4),
                     "excess_cagr": round(res_full.metrics.excess_cagr, 4)
                     if res_full.metrics.excess_cagr is not None else None},
        "top_contributors": top_names[:10],
        **name_out,
    }

    doc["selected_signal"] = picked
    OUT.write_text(json.dumps(doc, indent=2, default=str))
    print(json.dumps(doc, indent=2, default=str))
    print(f"\nwritten -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
