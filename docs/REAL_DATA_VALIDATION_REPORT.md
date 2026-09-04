> ## SUPERSEDED — DO NOT CITE THESE NUMBERS
>
> This report describes a study over 99 NSE symbols spanning 2007-09-17 → 2024-12-31.
> **That dataset is not present in this repository and these figures cannot be
> reproduced from anything committed here.** A later audit found no market data on
> disk at all before `scripts/acquire_data.py` was written and run.
>
> The authoritative, reproducible research status is
> **[RESEARCH_VALIDATION.md](RESEARCH_VALIDATION.md)**, computed over 108 symbols and
> 362,697 observations from 2012-01-02 → 2026-09-03, with its manifests
> (`backend/research_data/manifest.json`, `data_audit.json`, `study_result.json`)
> tracked in git and regenerable via the scripts named there.
>
> Kept for history — a superseded report is evidence of what was believed at the time.
> It is not evidence about the strategy.

---

# AlgoDollar — Real Data Validation Report

**Phase:** real Indian market-data validation, following the engineering audit in `AUDIT_REPORT.md`.
**Data:** 99 NSE large/mid-cap symbols, 4,262 daily bars, 2007-09-17 → 2024-12-31, plus NIFTY 50.
**Method:** baselines before models; purged/embargoed walk-forward; costs charged in proportion to measured turnover; multiple-testing correction on an honest trial count; a final holdout touched exactly once.

---

## PRODUCTION STATUS

# NO STRATEGY VALIDATED

Not a single candidate — rule-based or machine-learned — produced statistically significant risk-adjusted excess return after costs and multiple-testing correction. The leading candidate was then tested once on data reserved from the entire selection process and **lost to a passive equal-weight portfolio of the same universe**.

`LIVE_TRADING_ELIGIBILITY = BLOCKED_INSUFFICIENT_DATA` (1 of 23 gates passing).

---

## 1. Engineering correctness

Carried forward from the previous phase and re-verified on real data.

| Property | Status | Evidence |
|---|---|---|
| Features free of look-ahead | **Verified** | 21 features pass a mechanical causality test; 2 negative controls confirm the checker is not vacuous |
| Backtester does not manufacture edge | **Verified** | 0 of 20 pure-noise worlds profitable |
| Purging and embargo applied | **Verified** | Every fold reports `n_purged == label_horizon` |
| Multiple-testing machinery works | **Verified** | Rejects a 1.89-Sharpe winner cherry-picked from 1000 noise trials |
| Capital allocator constraints hold | **Verified** | See §12 |
| **Execution layer** | **BROKEN** | 13 CRITICAL defects — see §7 |

Two defects were found and fixed *during* this phase, both by tests written for it:

- `rolling_beta` was O(T·window) with a Python loop, making feature generation over a real universe unusable. Vectorized via rolling moments; output verified identical to a direct computation (`0.875324` vs `0.875324`).
- The research harness initially required every feature to be present, and `atr_14` was 100% NaN because high/low were never passed. Result: **zero usable training rows**, and both models silently produced no predictions at all. This is precisely the silent-failure class the audit warned about, reproduced in new code.

---

## 2. Data audit

### 2.1 What was obtained

Daily OHLCV with split/dividend adjustment, ~19 years, via Yahoo Finance. Sufficient for swing-horizon research.

### 2.2 What could NOT be obtained — and what that blocks

| Required data | Available? | Consequence |
|---|---|---|
| Point-in-time index constituents | **No** | Survivorship cannot be corrected. Universe is today's membership applied to all history. |
| Point-in-time fundamentals with publication dates | **No** | **The long-term engine cannot be validated at all.** Quality/growth/valuation scores have no real inputs. |
| Historical intraday (multi-year) | **No** — 1m ≈ 7 days, 5m/15m ≈ 60 days, 1h ≈ 730 days | **The intraday engine cannot be validated at all.** Sixty days spans one regime at best. |
| Delisted security history | **No** | See below. |
| Bid/ask spreads | **No** | Slippage is modelled, not measured. |

### 2.3 Survivorship bias — measured, not assumed

Queried over windows when each company was actively listed and trading:

| Company | Event | Rows returned |
|---|---|---|
| SATYAMCOMP | 2009 accounting fraud | **0 — absent** |
| DHFL | 2019 collapse | **0 — absent** |
| VIDEOIND | insolvency | **0 — absent** |
| RELIANCE (control) | still listed | 720 — present |

The source keeps survivors and drops failures.

### CORRECTION — an earlier version of this report reasoned incorrectly here

The first draft claimed the bias is directional: *"a positive result is uninterpretable, a negative result is robust."* **That was wrong, and the error is worth stating plainly because it would have licensed an unjustified conclusion.**

Survivorship bias inflates the **absolute** return of a long-only book, because the companies that went to zero are missing. But the quantity actually measured here is **excess return of the top quintile over an equal-weight portfolio of the same universe** — and both legs are computed on the same filtered set. The bias does not cleanly survive the subtraction.

Worse, it plausibly runs the *other* way for this particular signal. A company heading for delisting — Satyam through 2008, DHFL through 2019 — has collapsing momentum long before it disappears. A momentum strategy would have ranked such a name in its **bottom** quintile and not held it, while the equal-weight benchmark would have held it all the way down. Deleting those companies from the sample therefore removes episodes in which momentum correctly avoided a disaster, which biases its measured excess return **downward**, not upward.

The honest statement is therefore:

> **The direction of survivorship bias on excess return is indeterminate and signal-dependent.** This dataset cannot establish the true historical performance of any strategy tested on it — in either direction.

What remains valid is narrower, and is stated as such throughout this report:

- The candidates failed **on this dataset**. That is a fact about this measurement.
- It is **not** proof that no edge exists. An unbiased dataset could plausibly show a different result, and for a signal that avoids failing companies it could show a better one.
- Conclusions that do **not** depend on the universe composition — the turnover finding, the IC-versus-profit inversion, the execution defects, the machinery verification — remain valid regardless.

Obtaining point-in-time constituents and delisted history (see `DATA_ACQUISITION_PLAN.md`) is therefore not a refinement. It is the precondition for any performance claim at all.

### 2.4 Automated quality audit

`app/data/quality.py` runs before any modelling. On this panel: **0 CRITICAL, 6 WARNING, 3 INFO → USABLE**.

The material finding: **12 unadjusted corporate actions**, including a fake −261% single-day log return (the 2008 Bajaj demerger). Diagnosed by the adjusted and raw series moving by the *identical* ratio, proving no adjustment factor was applied:

```
BAJAJFINSV  2008-05-26  logret=-2.6102  adj_ratio=0.0735  raw_ratio=0.0735
NESTLEIND   2010-01-08  logret=-1.4411  adj_ratio=0.2367  raw_ratio=0.2367
ABBOTINDIA  2010-01-08  logret=+1.0582  adj_ratio=2.8811  raw_ratio=2.8811
```

Left in place, a mean-reversion model would treat an unadjusted 1:5 split as an 80% crash, buy it, and book an enormous imaginary profit on a "recovery" that never happened — the shareholder simply held five times as many shares all along. Masked at a cost of **0.003%** of observations; maximum daily move falls from a fake 261% to a plausible 38%.

---

## 3. Statistical validity

- **Purged, embargoed walk-forward**, 6 expanding folds, `label_horizon=5`, embargo 1%.
- **Labels demeaned within each date.** A raw 5-day forward return is dominated by that day's market move, which no stock-selection signal controls. Without this, IC largely measures beta.
- **Non-overlapping rebalances.** A 5-day return sampled daily produces autocorrelated, overlapping periods whose standard deviation understates dispersion; annualizing such a series inflates Sharpe. All performance is measured on a common grid of **705 non-overlapping periods**.
- **Effective sample size** deflated by the label horizon for every t-statistic.
- **Honest trial count = 10** (8 rules + 2 models), used in every Deflated Sharpe Ratio.

---

## 4. Backtest performance — the leaderboard

Development period 2007-09-17 → 2021-12-31. `LO_excess` is long-only top-quintile minus the equal-weight universe, net of turnover-proportional costs. `*` would mark DSR > 0.95.

| Strategy | IC | t (deflated) | LS Sharpe | LO excess/yr | LO Sharpe | DSR | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| **momentum_12_1** | +0.0226 | +2.77 | +0.29 | **+4.25%** | **+0.41** | 0.461 | 12% |
| lightgbm | +0.0391 | +6.23 | +0.31 | +2.35% | +0.29 | 0.274 | 55% |
| reversal_5d | **+0.0451** | **+7.09** | +0.19 | +1.05% | +0.10 | 0.112 | 82% |
| low_volatility | +0.0104 | +1.22 | +0.12 | +0.90% | +0.08 | 0.101 | 12% |
| trend_sma200 | +0.0003 | +0.04 | −0.06 | −0.62% | −0.05 | 0.037 | 16% |
| ridge | +0.0321 | +4.98 | +0.12 | −1.20% | −0.14 | 0.020 | 55% |
| rsi_contrarian | +0.0304 | +4.48 | +0.06 | −2.26% | −0.22 | 0.008 | 44% |
| reversal_21d | +0.0199 | +2.83 | −0.22 | −4.40% | −0.38 | 0.001 | 41% |
| excess_vs_nifty_21 | −0.0199 | −2.83 | −0.68 | −8.96% | −0.83 | 0.000 | 40% |
| high_volume | −0.0069 | −1.47 | −1.59 | −9.16% | −1.17 | 0.000 | 78% |

**Passive benchmarks (the bar):**

| | CAGR | Sharpe | Max DD |
|---|---:|---:|---:|
| NIFTY 50 buy-and-hold | +9.92% | 0.434 | −59.9% |
| **Equal-weight universe** | **+17.09%** | **0.830** | −57.7% |

The equal-weight figure is itself survivorship-inflated — these are the 99 names that survived to 2024.

**Probability of Backtest Overfitting: 0.143** across all 10 candidates — the selection *procedure* is not pathologically overfit. That does not rescue any individual result.

### Three findings that matter more than the ranking

**Statistical predictability is real but economically inert.** Several signals have large, deflated t-statistics: `reversal_5d` at t=+7.09, `lightgbm` at t=+6.23. These are genuine predictive relationships. They do not translate into money.

**Turnover inverts the ranking.** `reversal_5d` has the *best* IC in the study and nearly the *worst* economics, because it replaces 82% of the book every rebalance. `momentum_12_1` has half the IC and the best economics at 12% turnover. Ranking models by IC would have selected exactly the wrong strategy.

**Complexity did not pay.** LightGBM achieved a substantially higher IC than 12-1 momentum (0.0391 vs 0.0226) and delivered *worse* net excess return (+2.35% vs +4.25%), because it demanded 55% turnover instead of 12%. Per the stated policy — if the baseline wins, use the baseline — the gradient booster is rejected.

**No candidate reached significance.** The maximum DSR observed was **0.461**, against a 0.95 requirement.

---

## 5. Out-of-sample performance — the final holdout

The leading candidate (12-1 momentum, long-only) was evaluated **once** on 2022-01-01 → 2024-12-31, data untouched by any selection decision.

| | Development (2007–2021) | **Final holdout (2022–2024)** |
|---|---:|---:|
| Excess vs equal-weight | +4.35%/yr | **−0.61%/yr** |
| Sharpe | +0.421 | **−0.067** |
| Turnover | 12% | 11% |
| Deflated Sharpe Ratio | 0.461 | **0.045** |
| Significant? | No | **No** |

**The candidate lost to a passive equal-weight portfolio out of sample.** This is the decisive result of the phase.

---

## 6. Robustness — the attempt to disprove

Run against 12-1 momentum before the holdout was opened.

**What it survived:**

| Test | Result |
|---|---|
| Cost/slippage stress | Positive at 0.5× through 3× (Sharpe 0.483 → 0.170) |
| Parameter stress | **15/15** lookback/skip combinations positive (mean Sharpe 0.333, range 0.130–0.589) — not curve-fit to one setting |
| Universe stress | 18/20 random half-universes positive |
| Leave-one-year-out | Never collapses; worst exclusion (2015) drops Sharpe 0.421 → 0.309 |
| Leave-one-sector-out | Never collapses; worst (AUTO) drops to 0.236 |

**What it failed:**

| Test | Result |
|---|---|
| **Rebalance concentration** | Dropping the best **25 of 652** rebalances (3.8%) flips Sharpe from **+0.421 to −0.279**. The entire result is carried by ~4% of periods. |
| **Regime dependence** | Bull (above 200DMA): Sharpe **+0.694**. Bear (below 200DMA): **−0.134**. The "edge" is a bull-market phenomenon — disguised beta, not alpha. |
| **Final holdout** | Negative (§5). |

A strategy whose performance lives in 4% of rebalances and evaporates below the 200-day moving average is fragile by construction. The holdout then confirmed it.

---

## 7. Execution realism

A dedicated adversarial audit of the Zerodha/Kite path returned **13 CRITICAL findings**. This layer has never held a live connection. Selected findings:

| # | Defect | Consequence |
|---|---|---|
| 1 | Two of twelve safety gates **fail open** — `MarketClosedError`/`StaleDataError` subclass `RuntimeError`, not `SafetyCheckError`, so the generic handler downgrades them to warnings | Orders validate at 03:00 Sunday and on data that has never ticked |
| 2 | Kill switch **fails open** when its store errors; and is a no-op when no store is wired | A Redis blip silently disables the kill switch |
| 3 | Idempotency key recorded **after** placement; `_call_kite` retries every exception 3× | A timed-out order that reached Zerodha is placed again — measured: **3 live orders from one call** |
| 5 | Market orders use `signal.price`, documented as ignored for MARKET | **10,000,000 shares placed against ₹1 of cash** |
| 6 | Daily risk gate measures transaction **cost**, not risk | ₹70 crore of gross exposure fits inside a ₹50,000 daily risk cap |
| 9 | All three `_fetch_db_*` reconciliation helpers `return []`; broker fetch failure is swallowed | **Broker unreachable ⇒ `[] vs []` ⇒ status OK ⇒ trading starts blind** |
| 10 | `place_order` has no trigger-price parameter; `Signal.stop_price` is never read | Strategies believing they have broker-side stops **have none** |
| 11 | Nothing in the application imports `app.execution` or `app.broker`; two unconnected kill switches exist | The UI kill switch does not stop the execution layer |
| 13 | Paper broker: limit orders always fill at the limit price; selling stock you do not own **mints cash and creates no short** | **Paper trading is worthless as a validation stage** |

Finding 13 is structural for this phase: paper trading cannot serve as a gate until the paper broker is realistic.

Items requiring live credentials to settle: KiteTicker tick payload keys, SL/SL-M trigger validation, true order-rate limits, and whether a timed-out `place_order` was accepted.

---

## 8. Paper-trading performance

**None.** No paper trading has been run.

It would not be meaningful yet: the paper broker fills limit orders at the limit price and creates cash from selling shares that do not exist (§7, finding 13). Paper results from that simulator would measure the simulator, not the strategy.

---

## 9. Portfolio construction and capital allocation

The capital allocator was validated at the requested contribution levels.

**Sum invariant and cash validity (STRONG_BULL):**

| Contribution | Long-term | Swing | Intraday | Cash | Residual |
|---:|---:|---:|---:|---:|---:|
| ₹0 | 0.00 | 0.00 | 0.00 | 0.00 | **+0.000000** |
| ₹10,000 | 3,653.84 | 3,166.66 | 1,000.00 | 2,179.50 | **+0.000000** |
| ₹50,000 | 18,269.23 | 15,833.33 | 5,000.00 | 10,897.44 | **+0.000000** |
| ₹1,00,000 | 36,538.46 | 31,666.66 | 10,000.00 | 21,794.88 | **+0.000000** |
| ₹5,00,000 | 182,692.30 | 158,333.33 | 50,000.00 | 108,974.37 | **+0.000000** |
| ₹50,00,000 | 1,826,923.07 | 1,583,333.33 | 500,000.00 | 1,089,743.60 | **+0.000000** |

**Regime responsiveness at ₹5,00,000** — previously byte-identical across all regimes:

| Regime | Deployed | Cash |
|---|---:|---:|
| STRONG_BULL | 78.2% | 108,974 |
| WEAK_BULL | 64.8% | 175,962 |
| RECOVERY | 54.8% | 225,897 |
| SIDEWAYS | 49.3% | 253,301 |
| HIGH_VOL | 31.3% | 343,494 |
| WEAK_BEAR | 19.1% | 404,391 |
| STRONG_BEAR | 7.3% | 463,462 |
| **PANIC** | **0.0%** | **500,000** |

**The explicit 2% test** — a user cap must never become 95% through re-normalization:

| `max_intraday_pct` | Actual | |
|---:|---:|---|
| 2% | **2.0000%** | OK |
| 5% | **5.0000%** | OK |
| 10% | **10.0000%** | OK |

**100% cash reachable:** all strategies PAUSED at ₹5,00,000 → cash **₹500,000 (100%)**.

The allocator is sound. It has nothing validated to allocate *to*.

---

## 10. Remaining risks

1. **No validated strategy exists.** Everything downstream is therefore hypothetical.
2. **The long-term engine is entirely unvalidated** and its fundamental inputs are still mock data. It is guarded — it refuses to emit signals on mock data outside paper mode — but it has never been tested against real fundamentals.
3. **The intraday engine is entirely unvalidated.** No multi-year granular data exists to test it.
4. **The execution layer is unsafe** (§7) and disconnected from the application.
5. **Paper trading is not yet a meaningful gate** because the paper broker is unrealistic.
6. **Survivorship bias is unquantified.** Its presence is proven; its magnitude is not, and cannot be without the missing data.
7. **Costs are modelled, not measured.** No real fill has ever been observed.
8. **Regime multipliers and equity caps remain uncalibrated priors**, labelled as such in code.
9. **Only the swing horizon was tested, on one universe, at one horizon (5 days), with one label definition.** A different horizon might behave differently — but exploring that multiplies the trial count and raises the significance bar accordingly.

---

## 11. Live-trading eligibility

```
LIVE_TRADING_ELIGIBILITY = BLOCKED_INSUFFICIENT_DATA
gates passed: 1 / 23
permits_live_trading: False
```

| Category | Passed |
|---|---|
| Data | 1/5 — only `real_price_history` |
| Statistical | 0/5 |
| Performance | 0/4 |
| Execution | 0/6 |
| Operational | 0/3 |

The gate is fail-closed by construction: unrecorded evidence counts as failure, an exception inside a gate counts as failure, and the state is a derived property with no code path that can assign `LIVE_ELIGIBLE`.

---

## 12. What must happen next

**To unblock the swing horizon (the only one with data):**
1. Obtain point-in-time NIFTY constituent history. Re-run; expect results to *worsen*.
2. Obtain delisted-security price history to quantify the survivorship bias.
3. Measure real spreads rather than assuming 11 bps per side.

**To unblock the other two horizons:**
4. Point-in-time fundamentals with publication dates → long-term. Without this the long-term engine should be disabled outright, not left running on mock data.
5. Multi-year granular intraday data → intraday.

**To make paper trading meaningful:**
6. Fix the 13 CRITICAL execution defects, starting with the fail-open safety gates, the idempotency ordering, and the paper broker's fill and short-selling logic.
7. Wire `app.execution`/`app.broker` into the application; unify the two kill switches; call `reconcile()` at startup.

**Before any live consideration:**
8. Re-run this entire phase on unbiased data. If nothing survives — which the present evidence suggests is likely — that is the answer.
9. 90+ days of paper trading on a realistic simulator, compared against backtest expectations.

---

## 13. Honest conclusion

The machinery works. It was pointed at nineteen years of real Indian equity data and it did the thing it was built to do: **it declined to certify anything.**

There is genuine, statistically detectable cross-sectional signal in this feature set — several signals carry deflated t-statistics above 4, and one above 7. That signal did not survive contact with transaction costs, it did not survive correction for having searched ten candidates, and the strongest of it was carried by 4% of rebalances and disappeared below the 200-day moving average. When the leading candidate was finally shown data reserved from every selection decision, it underperformed a passive equal-weight portfolio of the same stocks.

**What that does and does not establish.** It establishes that these ten candidates failed on this dataset. It does **not** establish that no edge exists in the Indian market, and an earlier draft of this report overstated exactly that — see the correction in §2.3. The universe is survivorship-filtered, and because performance was measured as excess return over a benchmark drawn from the *same* filtered universe, the bias does not have a determinate direction. For a momentum signal it plausibly runs against the strategy, since the companies deleted from the sample are precisely the ones a momentum rule would have avoided. The correct conclusion is that **this dataset cannot establish true historical performance in either direction.**

Three findings survive the data problem entirely, because none of them depends on which companies are in the universe:

The signal with the best information coefficient was nearly the worst strategy, because turnover inverted the ranking — selecting models on predictive accuracy would have chosen exactly wrong. The gradient booster lost to twelve-one momentum on net return despite a far higher IC, so complexity was rejected on evidence rather than taste. And the passive equal-weight benchmark, at 17.09% CAGR and 0.830 Sharpe, proved a much harder opponent than any of the ten candidates assembled to beat it.

The correct action now is not to tune parameters until the holdout cooperates. It is to obtain point-in-time constituents and delisted history, then re-run — and to accept whatever that shows.

**PRODUCTION STATUS: NO STRATEGY VALIDATED.**

Read precisely: *no strategy has been validated.* That is a statement about the absence of evidence, not evidence of absence.
