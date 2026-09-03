# AlgoDollar — Adversarial Self-Audit

**Scope:** the prediction, statistics, ML, portfolio and trading logic.
**Method:** read every module, execute it, and test claims against synthetic data with known ground truth.
**Verdict up front:** the system as originally generated was **not** production-grade, and was not close. This document records what was wrong, what was fixed, what was verified, and — most importantly — what still cannot be claimed.

---

## 0. The single most important statement in this document

**No component of this system has ever been validated on real market data.**

There is no market data in the repository. No model has been trained on real prices. No backtest has been run on a real instrument. Every number in this report comes from **synthetic data with known ground truth**, which is the correct way to verify that machinery is *correct*, and says nothing about whether any strategy is *profitable*.

Therefore:

- The production model currently selected is: **none**.
- The expected return of this system is: **unknown**.
- The out-of-sample performance on Indian equities is: **unmeasured**.

Anyone reading a Sharpe ratio out of this codebase today is reading a property of a random number generator.

---

## 1. Weaknesses found

Defects are grouped by whether they would have **lost money**, **produced false confidence**, or **broken outright**. Every one was reproduced by executing the code.

### 1.1 Would have lost money in production

| # | Defect | Evidence |
|---|---|---|
| C1 | **Regime detection had zero effect on allocation.** The multiplier table was keyed on `BULL_*`/`BEAR_*` strings while the enum emitted `STRONG_BULL`/`WEAK_BEAR`/…; `.get(label, default)` swallowed every miss. | Allocations byte-identical across all 8 regimes: `STRONG_BULL → LT 488,851 / SW 353,059 / ID 108,090` and `PANIC → LT 488,851 / SW 353,059 / ID 108,090`. The system deployed 95% of capital into a market panic. |
| C2 | **User risk caps were voided by re-normalization.** Caps were applied, then weights were divided by their sum, undoing them. | `max_intraday_pct=0.02` produced an intraday allocation of **95.00%** — a 47× cap violation. |
| C3 | **The "no trade" capability did not exist.** Scores were normalized to a simplex, so deployment was a binary cliff at exactly 95% regardless of edge. | An 80% collapse in expected Sharpe (`1.0 → 0.2`) changed allocations by **₹0** — identical to the rupee. |
| C4 | **Intraday used naive local time and never squared off.** On a UTC host (the cloud norm), 15:20 IST reads as 09:50. | Verified: at 09:50 UTC (=15:20 IST, past the square-off) the strategy was *permitted to open* new positions and `should_exit` returned `False`. An intraday book carried overnight at intraday leverage. |
| C5 | **Long-term decisions were driven by random numbers.** `_MockFundamentalProvider` used a class-level RNG never reseeded per symbol, so it advanced on every call. | The same stock scored **39.3 / 50.6 / 37.7** on three consecutive calls, straddling the sell threshold of 40. Buy/sell decisions were coin flips. Nothing consumed the `data_source: "MOCK"` flag. |
| C6 | **Swing reported a 12-month return as a 5-day expected return.** Annual momentum was passed through as `edge_score` on a signal with `holding_period_days=5`. | On a **zero-drift random walk**, top `edge_score = 0.4156` — a claimed 41.6% five-day return where no edge exists. Every downstream consumer (sizing, ranking, risk budgeting) was fed a number inflated ~50×. |
| C7 | **Backtest and live execution used different cost models.** `order_manager.py` duplicated the cost logic with divergent constants and no DP charges. | CNC BUY 1000@2000: backtest **₹2,378.35** vs live **₹2,138.35** (₹240 gap). Live accounting was optimistic and fed under-sized risk into order validation. |
| C8 | **Health auto-DISABLE fired inside the noise band.** Thresholds acted on a ~21-observation Sharpe with no sample-size gate on degradation. | Monte Carlo, 20k trials, true Sharpe = 1.0: **P(auto-DISABLED) = 28.5% per month**. A genuinely profitable strategy was killed roughly every 3.5 months by sampling noise. |
| C9 | **Silent 2–3× leverage.** Each signal was sized against the full capital with no cross-signal budget, and the risk-engine check was wrapped in `except Exception: pass` — failing **open**. | 4 swing signals → 112% of capital allocated; a full book at the per-position cap → **300% gross exposure**. |
| C10 | **Drawdown halt liquidated at a future price.** On halt the loop broke but fell through to end-of-dataset liquidation. | Halt fired 2015-04-28; positions were marked out at the 2016-11-30 price — **19 months of look-ahead** in exactly the scenario the halt exists to measure. *(Found by a test written during this audit.)* |

### 1.2 Produced false confidence

| # | Defect | Evidence |
|---|---|---|
| F1 | **The entire test suite tested nothing.** All five original test files defined their own stub classes and imported **zero** production modules. | `test_risk.py`, `test_allocator.py`, `test_backtester.py`, `test_safety.py`, `test_costs.py` — real production imports: **0** each; locally-defined stub classes: 11–23 each. `test_backtester.py`'s own comment read *"Minimal backtester stub."* Four of its assertions were already failing against its own stub. |
| F2 | **Information Coefficient was pooled, not cross-sectional.** One `spearmanr` over all dates and stocks at once. | A model with **zero stock-picking skill** whose prediction level merely tracked the market scored pooled **IC = 0.685**, clearing the 0.02 gate by 34×. Correct per-date IC: **0.004**. |
| F3 | **ICIR was fabricated.** `test_icir = test_ic` — literally the same value. The real helper had zero call sites while the docstring advertised "selection criterion: highest test ICIR". |
| F4 | **Model selection read the test set.** Both the eligibility gate and the ranking key used test metrics, contradicting the function's own docstring. No multiple-comparison correction — a plain `max` over N candidates. |
| F5 | **No purging, no embargo, no multiple-testing control anywhere.** With 5-day labels sampled daily, ~80% of adjacent labels overlap; effective sample size is ~n/5, so every t-statistic was overstated by ~√5. |
| F6 | **Monte Carlo used IID resampling**, destroying the autocorrelation that drives drawdown. |
| F7 | **The ML stack was decorative.** `LinearAlphaModel`, `GBMAlphaModel`, `ModelCompetition` had **zero call sites** outside their own module. No strategy was ever instantiated anywhere. No forward-return label was constructed anywhere in the repo. The "ML alpha" was unreachable; the only live path was a hardcoded heuristic. |
| F8 | **Long-term BUY threshold was statistically unreachable.** Averaging five z-capped components shrinks dispersion. Over 1500 draws: **P(composite ≥ 65) = 0.133%** (~1 stock in 750) while **P(< 40) = 4.1%** — a structurally one-way liquidating book. |

### 1.3 Numerically wrong

| # | Defect | Before → After |
|---|---|---|
| N1 | **CVaR applied the Expected-Shortfall formula to VaR instead of σ**, double-counting z. | 5,762.29 vs Monte-Carlo ES 3,506.01 — **+64.4%**. Now 3,504.33 (**0.048%** error). |
| N2 | **Risk parity did not equalize risk contributions.** The CCD constant term ignored σ(w). | N=8 dispersion **62.78%** → **6.3e-11**. Weight error vs true ERC: 8.8pp → 0.00pp. |
| N3 | **Long/short books produced garbage risk.** `total_value == 0` was an exact float compare; a hedged book nets to ~1e-9. | Hedged book vol **1,264,911.00** → **0.0632** (analytic: 0.0632). |
| N4 | **Missing prices were silently invisible to risk** (`prices.get(s, 0.0)`), so an all-missing book reported vol = 0 on live positions. Now raises. |
| N5 | **A third of declared risk limits were never enforced**: `max_single_stock_pct`, `max_sector_pct`, `max_risk_per_trade_pct`, `max_leverage`. Sector concentrations were computed and never compared to anything. |
| N6 | **Signed losses raised no breach.** `RiskState(daily_loss=-50000)` against a ₹10,000 limit → breaches `[]`. |
| N7 | **Backtester priced every trade as delivery** (`product="CNC"` hardcoded), overstating intraday round-trip cost by **2.47×** and charging a DP fee on every sell. |
| N8 | **Slippage tiers were dead.** `_MAX_SLIPPAGE` (10bps) was *below* `_SLIPPAGE_SMALL_CAP` (15bps), and no call site passed the large-cap list — every symbol got exactly 10bps. |
| N9 | **Fractional share quantities.** The backtester bought 13.47 shares of a stock. |
| N10 | **Position sizing used free cash, not portfolio value**, systematically shrinking positions as the book filled. |
| N11 | **Two modules disagreed on the regime enum** (`MarketRegime.BULL` did not exist) — an `AttributeError` at import. |
| N12 | **11 broken imports** (`backend.app.*` vs `app.*`) — modules could not be imported at all. |

### 1.4 Structural

- **O(T²) hot loop.** The engine rebuilt a fresh 5-column DataFrame *per symbol per bar*, preceded by a dict comprehension whose every entry was immediately overwritten. Realistic backtests were unusably slow.
- **OHLC was fabricated** (`open = high = low = close`), silently. ATR is therefore identically zero and intrabar stop detection is impossible — while the docstring claimed stops were checked against bar high/low.
- **Silent failure swallowing.** An exception in the feature pipeline produced an empty DataFrame, a flat equity curve, and a "successful" backtest indistinguishable from *"the strategy chose not to trade."*
- **The code had never been executed.** No dependency was installed; nothing imported.

---

## 2. Changes made

### 2.1 New capability that did not exist before

| Module | What it provides |
|---|---|
| `app/research/validation.py` | Purged, embargoed walk-forward splitting; effective-sample-size deflation; a leakage assertion that converts a silent statistical error into a loud failure. |
| `app/research/statistics.py` | Deflated Sharpe Ratio; Probability of Backtest Overfitting via CSCV; stationary block bootstrap; Benjamini-Hochberg and Holm-Bonferroni corrections. |
| `app/research/pipeline.py` | The missing spine: data → labels → purged split → per-fold training → OOS prediction → **cross-sectional** IC → multiple-testing-adjusted verdict. |

### 2.2 Repairs

All ten money-losing defects (C1–C10), all eight false-confidence defects (F1–F8) and all twelve numerical defects (N1–N12) were addressed. Selected before/after evidence is in §5–§7.

### 2.3 Repair evidence — before → after

Every row was measured by executing the code before and after the change.

**Allocator and regime**

| Defect | Before | After |
|---|---|---|
| C1 regime effect | `STRONG_BULL` and `PANIC` byte-identical; 1 distinct allocation across 8 regimes | 8 distinct, monotone: **78.2 / 55.6 / 45.2 / 37.8 / 31.3 / 19.1 / 7.3 / 0.0%** deployed. `PANIC` → **100% cash** |
| C2 cap enforcement | 2% intraday cap → **95.00%** allocated (47×) | **2.00%** exactly; remainder to cash |
| C3 no-trade capability | Sharpe 1.0 → 0.2 changed allocation by **₹0** | 78.2% → **19.0%**; 100 distinct deployment levels across 100 score steps |
| C1b PANIC reachability | 0 / 2000 simulated crash paths reached PANIC without VIX | **441 / 500** crash paths; the −22.9% slide now classifies correctly |

**Strategies**

| Defect | Before | After |
|---|---|---|
| C6 swing edge units | zero-drift random walk → 4 signals, top `edge_score` **0.7107** claimed at 5-day horizon; 35/50 cleared the gate | **0 signals**; max abs raw score **0.0024**; 0/50 clear the gate |
| C8 false auto-disable | true Sharpe 1.0 → P(DISABLE) **28.04%/month** (~98%/yr) | **0.00%** below minimum sample; **1.61%/yr** at n=60. Genuine evidence (−4.0 σ) still disables |
| F8 unreachable buy bar | P(buy) **0.200%**, P(sell) 6.533% — a one-way liquidating book | P(buy) **10.0%**, P(sell) **20.0%**, thresholds now move with the cross-sectional distribution |
| C? over-limit buys | 30 held / 20 max → **18 BUY signals** | **0 BUY signals** |
| C9 silent leverage | swing **253%**, longterm **200%** gross intent | **100.0%** both, per-signal and per-batch |
| C4 timezone | 15:20 IST on a UTC host → new positions allowed, no square-off | IST throughout; naive datetimes raise |
| C5 mock fundamentals | same stock scored 39.3 / 50.6 / 37.7 | deterministic per symbol; raises outside paper mode, re-checked on every call |

**Risk numerics**

| Defect | Before | After |
|---|---|---|
| N1 CVaR | 5,762.29 vs MC ES 3,506.01 (**+64.4%**) | **3,504.33** (0.048% error) |
| N2 risk parity (N=8) | dispersion **62.78%** | **6.3e-11** |
| N3 hedged book vol | **1,264,911.00** | **0.0632** (analytic 0.0632) |
| N6 signed losses | breaches `[]` on a ₹50,000 loss vs ₹10,000 limit | raises; `from_pnl` → 3 breaches |

**ML evaluation**

| Defect | Before | After |
|---|---|---|
| F2 zero-skill model IC | pooled **0.685** (34× the gate) | cross-sectional **0.0040**, t=0.45, p=0.65 |
| F2b injected-skill control | — | IC **0.3246**, ICIR 2.59 (metric is not merely always ~0) |
| F4 noise winner, N=30 | val IC 0.0356 passed the 0.02 gate | E[max\|null]=0.254, **DSR 0.485 → rejected** |

**Engine**

| Defect | Before | After |
|---|---|---|
| C10 drawdown-halt look-ahead | halted 2015-04-28, liquidated at 2016-11-30 prices | liquidates at the halt bar |
| Hot loop | 126s test runtime | **14s** (9×) |
| N9 fractional shares | 13.47 shares | whole shares enforced |

### 2.4 Tests

The five stub-only test files were the most dangerous artifact in the repository: they made the system look verified while testing only themselves. `test_backtester.py` was replaced outright with tests that import the production engine. Six new test files were added, all exercising real code.

```
350 tests passing
```

---

## 3. Models tested

**None on real data.** This section cannot be filled in honestly yet.

What *was* tested is the machinery that will one day select a model:

| Component | Candidates available | Selected | Basis |
|---|---|---|---|
| Alpha model | Ridge, LightGBM GBM | **none** | No real data exists to select on. |
| Regime detection | rule-based | rule-based (default) | HMM/clustering not implemented; no evidence they would help, so none was added. |
| Covariance | sample, Ledoit-Wolf shrinkage | **Ledoit-Wolf** | Verified in use and correctly annualized. |
| Portfolio construction | mean-variance, min-vol, risk parity, vol targeting | **not selected** | All four now numerically correct; no data to choose between them. |
| Monte Carlo | IID bootstrap, stationary block bootstrap | **block** | IID understated risk of ruin by 2× (§7). |
| Overfitting control | none, DSR, PBO | **DSR + PBO** | Demonstrated to reject noise winners (§5). |

Deliberately **not** added: deep learning, transformers, reinforcement learning, PCA, HMM, GARCH. None of them can be justified without evidence, and there is no evidence yet. Adding them would have been decoration.

---

## 4. Validation methodology

1. **Causality testing.** Every feature is checked mechanically: `f(prices[:T])[-1]` must equal `f(prices[:T+k])[T-1]`. If recomputing a feature on truncated history changes the value it had at that timestamp, it is reading the future.
2. **Null-hypothesis testing.** Feed the engine independent geometric random walks. Any reported edge is a bug.
3. **Positive controls.** Inject a known signal. Failure to recover it means the pipeline is broken in the opposite direction.
4. **Negative controls on the tests themselves.** A test that never fails proves nothing, so the causality checker is fed a deliberately leaky feature and a full-sample-normalized feature and must catch both.
5. **Purged, embargoed walk-forward** for any train/test split.
6. **Multiple-testing adjustment** on any result produced by a search.

---

## 5. Results — machinery validation

**These are properties of the code, not of any trading strategy.**

### 5.1 Look-ahead: clean

21 features across price, volatility, volume, cross-sectional and market families — **all causal**. Both negative controls correctly caught. This was the most pleasant surprise of the audit: the feature layer was already right.

```
23 passed  (21 causality + 2 negative controls)
```

### 5.2 Null hypothesis: the engine does not manufacture edge

20 independent random worlds, zero true drift:

```
mean net Sharpe    : -2.00
fraction Sharpe > 0:  0.00   (0 of 20)
mean gross return  : -5.7%
```

### 5.3 Multiple testing: a noise winner is correctly rejected

Best of N pure-noise strategies, **ground truth: zero edge in every row**:

| Trials | Best raw annual Sharpe | Null bar E[max] | DSR | Verdict |
|---:|---:|---:|---:|---|
| 1 | −1.71 | 0.00 | 0.0016 | REJECTED |
| 10 | 0.94 | 0.96 | 0.4893 | REJECTED |
| 50 | 1.38 | 1.38 | 0.5004 | REJECTED |
| 200 | 1.53 | 1.72 | 0.3728 | REJECTED |
| 1000 | **1.89** | 1.94 | 0.4711 | REJECTED |

A Sharpe of **1.89** reads as an excellent strategy in any backtest report. From 1000 trials on noise, it is *below* what luck alone produces.

### 5.4 PBO: selection quality is measurable

```
40 configs, all pure noise     -> PBO = 0.571  [OVERFIT]
20 configs, one truly superior -> PBO = 0.000  [acceptable]
```

### 5.5 The pipeline finds signal only when signal exists

```
NULL world   (random walks, noise features) -> OOS IC = -0.0012   (correctly ~0)
SIGNAL world (known injected relationship)  -> OOS IC = +0.24     (recovered)
```

> A note on method: the first version of the signal-world generator was itself broken — it built prices from a separately drawn forward-return array, so the label was a rolling average of those draws rather than the draw itself, and the "signal" feature had no real predictive power. **The pipeline correctly reported IC ≈ 0.** The generator was wrong, not the pipeline. This is exactly what the null/positive control pair is for.

---

## 6. Post-cost results

The cost model's arithmetic was independently verified against the standard Indian schedule and **passes**: delivery brokerage ₹0, the `min(0.03%, ₹20)` per-order cap, STT sides (sell-only intraday, both sides delivery), GST base excluding STT and stamp duty, buy-side-only stamp duty, SEBI as a turnover fraction, DP charges on delivery sells only.

| Round trip | Total | % of turnover | Expected band |
|---|---:|---:|---|
| ₹50,000 intraday (MIS) | ₹53.64 | 0.1073% | 0.05–0.12% ✓ |
| ₹50,000 delivery (CNC) | ₹132.54 | 0.2651% | 0.2–0.3% ✓ |

Magnitudes are sane. The defects were around the model, not in it (N7, N8, C7) — all now fixed.

**Post-cost strategy performance: unmeasured.** There is no strategy result to report.

---

## 7. Robustness results

**Block vs IID bootstrap** on an autocorrelated equity curve (ρ = 0.5):

| Method | Risk of ruin | 5th percentile outcome |
|---|---:|---:|
| IID (previous default) | 16.20% | ₹433,954 |
| Block (new default) | **32.20%** | **₹302,700** |

IID resampling **understated the risk of ruin by a factor of two**. Any risk figure previously produced by this system was optimistic.

**Slippage stress** is now supported (`slippage_multiplier` at 0.5× / 1× / 1.5× / 2× / 3×) and enforced by test. It was impossible before: the slippage tiers were dead code.

**Performance:** the engine hot loop went from 126s → 14s on the test suite (**9×**), by building per-symbol frames once instead of per-bar.

---

## 8. Rejected approaches and why

| Approach | Why rejected |
|---|---|
| Deep learning / transformers / RL | No evidence of incremental value. There is not yet evidence that a *linear* model works. Complexity without a baseline is decoration. |
| PCA | Not shown to improve anything. Would add a fitting step and an interpretability cost for no measured benefit. |
| HMM regime detection | The rule-based detector is transparent and now demonstrably responsive. An HMM adds fitting instability and a vocabulary that is harder to audit, for unmeasured gain. |
| GARCH | EWMA volatility is already implemented and causal. No evidence GARCH improves forecasts here. |
| IID bootstrap | Empirically understates tail risk by 2× (§7). |
| Pooled IC | Empirically scores a zero-skill model at 0.685 (§1.2 F2). |
| Roncalli `-σ(w)/n` risk-parity form | Correct only without mid-sweep normalization; this implementation renormalizes each sweep. The scale-invariant variance form converges in ≤93 iterations vs ≤686. Documented in-code so it is not "fixed" back. |

---

## 9. Remaining limitations

1. **No real market data.** Everything above is synthetic validation.
2. **No trained model.** No production model exists.
3. **Fundamental data is mock.** The long-term engine now refuses to emit signals on mock data outside paper mode, but a real point-in-time fundamentals provider must be integrated before it can do anything useful. Point-in-time fundamentals are hard to obtain and easy to get wrong.
4. **Survivorship bias is unaddressed in substance.** The universe is a static list. A point-in-time index-constituent history is required; without it, every long-horizon backtest is upward-biased.
5. **OHLC is synthesized from close** unless real high/low data is supplied. ATR and intrabar stops do not work on close-only data. The engine now warns loudly instead of failing silently.
6. **Overlapping labels remain non-IID** even after purging. The deflated t-statistic accounts for this, but it is a deflation, not a cure.
7. **Zerodha rates require verification** against the current official schedule. The arithmetic is right; the constants are as of the codebase's authoring and are configurable, not verified live.
8. **Regime multipliers and equity caps are uncalibrated priors** — hand-chosen monotone values, explicitly labelled as such in code. They are not estimated from data.
9. **Walk-forward does not yet retrain the alpha model per window.** `app/research/pipeline.py` does this correctly; `backtesting/walkforward.py` still only searches strategy parameters. These two should be unified.
10. **No live broker testing.** The Zerodha adapter has never held a connection.
11. **`RiskEngine` has no `approve_trade` method.** Strategy sizing now fails *closed* when handed a risk engine lacking it, so nothing silently bypasses risk — but the method must be implemented before the risk engine can actually be wired into position sizing. Today the backtester passes `None`, meaning risk approval is simply not connected.

---

## 10. Production model currently selected

**None.**

No model is approved, deployed, or deployable. `TRADING_MODE` remains `paper`. Live trading must stay disabled.

---

## 11. Confidence level

| Claim | Confidence | Basis |
|---|---|---|
| The feature layer is free of look-ahead | **High** | 21 features mechanically verified; checker validated by negative controls. |
| The backtest engine does not manufacture edge | **High** | 0 of 20 noise worlds profitable. |
| The cost arithmetic is correct | **Medium-High** | Independently verified against standard rules; magnitudes in band. Rates need live verification. |
| The risk numerics are correct | **High** | CVaR matches Monte-Carlo ES to 0.048%; risk parity dispersion 6e-11. |
| The multiple-testing machinery works | **High** | Rejects a 1.89-Sharpe noise winner; PBO separates noise from signal. |
| The allocator respects its constraints | **High** | 62 invariant tests; caps hold exactly; PANIC → 100% cash. |
| **Any strategy in this repo is profitable** | **None** | No evidence whatsoever exists. |
| **This system is ready for real money** | **None** | See §12. |

---

## 12. What must happen before live trading

Ordered. Each gate must pass before the next is attempted.

**Data**
1. Acquire real Indian equity history (OHLCV, adjusted for corporate actions).
2. Acquire a **point-in-time** index-constituent history to address survivorship bias.
3. Acquire point-in-time fundamentals, or disable the long-term engine entirely.
4. Verify current Zerodha/NSE/SEBI charges against official sources and update the cost config.

**Research**
5. Run `app/research/pipeline.py` on real data. Establish whether *any* cross-sectional IC exists after purging.
6. Compare against honest baselines: buy-and-hold NIFTY, equal-weight, simple momentum, simple volatility targeting. If the model cannot beat these after costs, stop.
7. Report DSR and PBO for every candidate, with an honest trial count including discarded variants.
8. Run parameter-neighbourhood sensitivity. Reject anything that works only at one setting.
9. Run slippage stress at 0.5×–3×. Reject anything that dies above 1.5×.
10. Reserve a final holdout that is never touched during selection.

**Engineering**
11. Unify `backtesting/walkforward.py` with the research pipeline so the model is retrained per window.
12. Supply real OHLC so ATR and intrabar stops function.
13. Test the Zerodha adapter against a live connection: auth, token refresh, WebSocket reconnect, rate limits, order lifecycle, reconciliation.
14. Verify the kill switch, duplicate-order protection and startup reconciliation against a real broker session.

**Paper**
15. Run paper trading on live data for a minimum of 3–6 months.
16. Compare paper results against backtest expectations. A material divergence means the backtest is wrong; investigate rather than proceed.

**Live**
17. Only then, and only at the smallest capital increment, with the kill switch tested and a hard daily loss limit active.

---

## 13. Honest summary

The scaffolding was good: clean module boundaries, a genuine no-trade mechanism, correct scaler discipline, real early stopping, causally-clean features, and an exact capital-sum invariant. Those parts were kept.

The decision logic was not. Regime detection did nothing. Risk caps did not bind. The system could not choose cash. An intraday book would have been carried overnight. Long-term buy and sell decisions were driven by an unseeded random number generator. And the test suite that should have caught all of this tested nothing but itself.

Most of that is now fixed and covered by tests that fail when the behaviour regresses. What has **not** changed is the thing that matters most: there is still no evidence that this system can make money, because it has never seen a real price. The infrastructure to find that out honestly now exists — including the parts specifically designed to tell you *no*.

The most valuable property of this system today is that when it is eventually run on real data, it is now considerably more likely to correctly report that it has found nothing.
