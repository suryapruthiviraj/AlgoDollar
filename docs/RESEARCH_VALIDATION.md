# AlgoDollar — Research Validation Report

# STATUS: **NOT VALIDATED**

**Survivorship bias has been removed.** The point-in-time universe criterion now PASSES.
The strategy fails on a different criterion — consistency across years — and that failure is
reported rather than argued away.

Regenerate:

```bash
cd backend
python scripts/acquire_universe_reference.py   # NSE listing dates + symbol changes
python scripts/acquire_pool.py                 # full pre-2012 listing pool
python scripts/audit_data.py
python scripts/run_research.py
```

---

## 1. Universe definition (chosen, not assumed)

    Universe(D) = { s : listed_on_or_before(s, D)
                      AND still_trading_on(s, D)
                      AND trailing-60-session avg volume >= 500,000 shares
                      AND trailing-60-session avg turnover >= Rs 5,000,000
                      AND price(s, D) >= Rs 5 }

A **liquidity-screened NSE equity universe** — deliberately NOT index membership.

The original design was `StockUniverse.get_nifty500_symbols()` (a hardcoded snapshot of
today's NIFTY 500) plus `filter_liquid(min_avg_volume=500_000, min_avg_turnover=5_000_000,
lookback_days=60)`. The liquidity filter was already point-in-time; the membership list was
not. **The thresholds are carried over unchanged** — re-deriving them against the new
universe would be fitting the universe to the result.

Index membership was not used because it **cannot be established**: NSE serves no dated
historical constituent files (verified, 404), and no reachable source provides entry/exit
dates. Inferring it would fabricate the dates that decide every result.

| Element | Source |
|---|---|
| Membership date | Every trading session, computed on demand |
| Entry date | max(NSE published `DATE OF LISTING`, first observed bar) |
| Exit date | Last observed bar, when the series stops ≥20 sessions before the data ends |
| Symbol / exchange | NSE `EQUITY_L.csv`, EQ series only |
| Corporate actions | `adj_close` for returns; raw `close` for fills; ex-dates unavailable |
| Listing/delisting | Entry from listing date; exit from series end (reason not recoverable) |

## 2. Historical membership coverage

| | |
|---|---|
| Pool | **946 symbols** (827 newly fetched this phase) |
| Observations | **3,368,792** |
| Sessions | 3,626 · 2012-01-02 → 2026-09-03 |
| Membership UNAVAILABLE before | **2012-02-28** (60-session liquidity warm-up — raises, never guesses) |
| Universe fingerprint | `ae8711d55c7e23f2` |

Membership size by year — **not a snapshot**:

| 2012 | 2014 | 2016 | 2018 | 2020 | 2022 | 2024 | 2026 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 169 | 222 | 209 | 258 | 243 | 338 | 365 | 373 |

## 3. Data source / provenance

| Source | Content | Status |
|---|---|---|
| `nsearchives.nseindia.com/.../EQUITY_L.csv` | 2,288 EQ symbols + `DATE OF LISTING` | **used** |
| `.../symbolchange.csv` | 1,057 ticker changes | acquired |
| `.../ind_nifty500list.csv` | Today's NIFTY 500 | provenance only — **never applied historically** |
| Yahoo Finance (`yfinance`) | Daily OHLCV + `adj_close` | **used** |
| `DelistedCompanies.csv`, `SUSPENSION.csv` | delisting register | **404 — unavailable** |
| dated index constituents | historical membership | **404 — unavailable** |

### Phase 3 — corporate-failure validation

| Symbol | In pool | Member 2013–2017 | Member 2019+ | Correct? |
|---|---|---|---|---|
| RCOM | yes | **yes** | no | collapsed 2019 |
| UNITECH | yes | **yes** | no | suspended |
| JETAIRWAYS | yes | **yes** (to 2019) | no | ceased ops 2019 |
| RELINFRA | yes | **yes** | yes → out by 2026 | still traded |
| GTLINFRA | yes | **never** | never | **correctly excluded** — penny stock, never met the floor |

GTLINFRA is the important row: it was **not forced in** to improve the survivorship story. It
was never eligible, so it never appears.

## 4. Old (survivorship-biased) vs 5. New (point-in-time)

| Metric | OLD — 108-symbol snapshot | NEW — 946-symbol point-in-time |
|---|---:|---:|
| OOS Sharpe | 1.2559 | **1.0111** |
| CAGR | 16.84% | 19.17% |
| Benchmark CAGR | 12.03% | 11.88% |
| Excess CAGR | +4.81% | **+7.30%** |
| **DSR** (6 trials) | 0.9982 | **0.9831** |
| **PBO** | 0.0143 | **0.0000** |
| **Bootstrap 90% CI** | [0.68, 1.89] | **[0.4656, 1.5782]**, P(>0)=0.999 |
| Positive excess years | 76.9% | **53.85%** |
| Most-selected signal | low_volatility | vol_scaled_momentum |
| Sortino / MDD / Calmar | — | see below |
| Max drawdown | −44.0% | **−39.7%** |
| Beta | — | **0.947** |
| Annual turnover | — | **15.96x** |
| Cost drag | — | **3.98%/yr** |
| Distinct names held | — | 559 (mean 53.5 positions/day) |

**Sharpe fell and the confidence interval widened.** The bootstrap lower bound dropped from
0.68 to 0.47 — the result is materially less certain once the universe is honest.

### Regime — the finding the biased universe hid

| Regime | Days | Sharpe |
|---|---:|---:|
| Bull | 1,065 | +2.09 |
| Sideways | 1,844 | +1.14 |
| **Bear** | **170** | **−2.15** |

The strategy **loses badly in downturns**. A survivor-only universe cannot show this,
because the names that fell hardest were removed from it.

### Cost sensitivity

| 0bps | 10bps | 25bps | 50bps | 100bps |
|---:|---:|---:|---:|---:|
| 1.33 | 1.26 | **1.17** | 1.01 | 0.70 |

### Concentration

Top 10 days contribute 66.6% of total return; excluding them, Sharpe falls 1.17 → **0.83**.

### Parameter perturbation (stability, not search)

Rebalance 1/5/10/21d → Sharpe 0.94 / 1.17 / 1.16 / 1.14. Quantile decile/quintile/third →
1.20 / 1.17 / 1.07. Stable except at daily rebalancing, where cost dominates.

## 6–8. Criteria and verdict

| Criterion | Observed | Threshold | |
|---|---:|---:|---|
| OOS Sharpe | 1.0111 | ≥ 0.50 | PASS |
| Bootstrap 5th pct | 0.4656 | > 0 | PASS |
| Deflated Sharpe | 0.9831 | ≥ 0.95 | PASS |
| PBO | 0.0000 | ≤ 0.50 | PASS |
| Excess CAGR | +7.30% | > 0 | PASS |
| Survives 50bps | 1.0115 | ≥ 0.50 | PASS |
| **Positive excess years** | **53.85%** | **≥ 55%** | **FAIL** |
| **Point-in-time universe** | **applied** | point-in-time | **PASS** |

Excess by year: 2014 −8.4%, 2015 +7.3%, 2016 −3.3%, 2017 +14.1%, 2018 −10.9%, 2019 −9.6%,
2020 +14.4%, 2021 +59.7%, 2023 +50.7%, 2025 −17.4%. **Six of thirteen years negative**, and
the total leans heavily on 2021 and 2023.

# VERDICT: NOT VALIDATED

7 of 8 pass; the consistency criterion fails at 53.85% against a 55% threshold **fixed before
the study ran**. It was not moved. Under the previous biased universe the same criterion read
76.9% — the difference is what survivorship bias was concealing.

## Provenance (Phase 7)

```
experiment_id : 83262ea24beff98f
dataset       : 8bd6046a2735cfae
universe      : ae8711d55c7e23f2
strategy      : baselines:breakout_126d,low_volatility,momentum_12_1,
                short_term_reversal,trend_50_200,vol_scaled_momentum
period        : 2014-01-29 -> 2026-09-03
```

`gather_repo_evidence()` **rejects** any study lacking a provenance block or run without a
point-in-time universe: statistical evidence stays `None`, so every statistical gate fails.
The old biased result can no longer satisfy any gate.

## 9–12. Status

| | |
|---|---|
| **Strategy** | **NOT VALIDATED** |
| **Paper trading** | Already enabled and working; no change |
| **Live trading** | **BLOCKED** — unchanged |
| Promotion | Stays at *Research NOT VALIDATED → strategy disabled → no forced trades* |

### Remaining blockers

1. **Consistency** — 6 of 13 years negative; returns concentrated in 2021 and 2023.
2. **Bear-regime Sharpe −2.15** — no downside protection.
3. **Residual pool bias** — NSE's equity list contains only companies listed *today*, so
   companies delisted before it was published are absent. Only 1 of 946 pool symbols shows a
   series ending, which is implausibly low and confirms the pool still under-represents
   failures. The universe *definition* is point-in-time; the *pool* is not yet complete.
4. **No point-in-time fundamentals** — value and quality remain untestable.
5. **No intraday feed** — the intraday sleeve remains unvalidatable.

---

## 3. Backtest construction

### The lag convention, enforced not documented

```
signal[t]   computed from prices up to and including the close of t
weights[t]  actionable only after t
return      earned from the close of t+1 to the close of t+2
```

A two-bar offset. `assert_no_lookahead()` raises on any frame shifted by less
than the declared lag, on misaligned indexes, and on an identically-zero
forward-return frame.

### What is modelled

Costs at **25 bps per side** charged on `|w[t] − w[t−1]|` (both sides of every
rotation, and the initial build from cash); weekly rebalance with positions held
in between; long-only top-quintile; 10% single-name cap; a liquidity cap at 5%
of median traded value; gross exposure never above 1.0.

Long-only is deliberate: retail single-stock shorting is not available in the
Indian cash segment, so a long-short result would not be implementable.

### Two real bugs found while building this

Both were caught because a **number was implausible**, and both now have tests.

1. **Forward returns were identically zero.** The entry and exit shifts resolved
   to the same bar (`shift(-2)/shift(-2)`). Every baseline returned a Sharpe near
   **−6.5** — a portfolio paying costs and earning nothing. A Sharpe of −6.5 on
   a long-only equity book is not a finding; it is a bug.
2. **Returns were indexed by signal date, not realisation date.** The series sat
   two bars ahead of every other date-indexed series, giving a long-only equity
   portfolio a **beta of 0.02**. Correctly aligned: **beta 0.89, correlation
   0.77**.

A third bug was found by the test suite itself: the single-name weight cap was
**silently a no-op**, because weights were clipped and then renormalised, which
put them straight back. Fixed to scale down only; a cross-section too thin to
satisfy the cap now leaves the book partially invested rather than concentrated.

---

## 4. Walk-forward validation

Six sequential folds. Selection among the six baselines happens on **train data
only**; the winner is then measured on the following test window, separated by a
**10-session embargo**. Without the embargo, a 12-month momentum signal computed
on the first test day is built almost entirely from training-period prices.

| Fold | Test window | Selected | Train SR | **Test SR** |
|---|---|---|---|---|
| 0 | 2014-01-30 → 2016-03-14 | low_volatility | 2.05 | **2.19** |
| 1 | 2016-03-15 → 2018-04-20 | low_volatility | 2.02 | **1.98** |
| 2 | 2018-04-23 → 2020-06-05 | low_volatility | 2.01 | **0.14** |
| 3 | 2020-06-08 → 2022-07-04 | low_volatility | 1.25 | **1.83** |
| 4 | 2022-07-05 → 2024-08-13 | low_volatility | 1.41 | **2.23** |
| 5 | 2024-08-14 → 2026-09-03 | low_volatility | 1.58 | **0.05** |

The two weakest folds (0.14 and 0.05) span the COVID crash and the most recent
period. The selection procedure is not stable across regimes.

### Stitched out-of-sample result

| | |
|---|---|
| Observations | 3,085 |
| **Sharpe (net)** | **1.256** |
| CAGR | 16.84% |
| Benchmark CAGR | 12.03% |
| **Excess CAGR** | **+4.81%** |
| Deflated Sharpe Ratio (6 trials) | **0.9982** |
| PBO (CSCV) | **0.014** |
| Bootstrap Sharpe 90% CI | **[0.68, 1.89]**, P(>0) = 1.00 |

---

## 5. Baselines

All six are reported, including the losers. Selecting the winner from this table
and reporting only that is the multiple-testing error the DSR corrects for.
Parameters are textbook and were fixed before any result was seen.

*In-sample, full period, liquidity-capped — reference only, NOT evidence:*

| Signal | Sharpe | CAGR | Excess CAGR | Max DD |
|---|---|---|---|---|
| low_volatility | 0.92 | 12.74% | +0.71% | −44.0% |
| momentum_12_1 | 0.58 | 9.44% | −2.59% | −48.5% |
| vol_scaled_momentum | 0.53 | 8.59% | −3.65% | −44.0% |
| trend_50_200 | 0.51 | 8.29% | −3.74% | −40.9% |
| short_term_reversal | 0.24 | 3.30% | −8.73% | −53.9% |
| breakout_126d | 0.09 | 0.05% | −12.20% | −51.2% |

Value and quality are **untested** — they need point-in-time fundamentals.

---

## 6. Machine learning — **REJECTED**

LightGBM cross-sectional ranker, six trailing price features, target = forward
5-day return demeaned within each date. Purged by the label horizon and
embargoed, so training labels cannot overlap prediction dates.

| | Baseline (walk-forward) | LightGBM |
|---|---|---|
| OOS Sharpe | **1.256** | **0.809** |
| Improvement | — | **−0.447** |
| Mean cross-sectional IC | — | **0.0084** |

Per-fold IC: 0.0117, 0.0072, 0.0144, **0.0004**. The most recent fold has
essentially none.

**Verdict: ML REJECTED — no meaningful improvement over the simple baseline.**
A model that underperforms a volatility ranking, while adding model risk,
retraining risk and opacity, is not worth deploying. No hyperparameter search
was run; searching here would be the overfitting the study exists to detect, at
the one point the study cannot see it.

---

## 7. Robustness

**Cost sensitivity** — the result degrades smoothly and survives a doubling:

| Cost | Sharpe | CAGR |
|---|---|---|
| 0 bps | 1.61 | 21.59% |
| 10 bps | 1.51 | 20.07% |
| 25 bps *(assumed)* | 1.36 | 17.84% |
| 50 bps | 1.11 | 14.20% |
| 100 bps | 0.61 | 7.26% |

**Universe sensitivity** — adding the 12 known failures back moved Sharpe
**+0.54** and CAGR **+6.64%**. This looks backwards and must not be read as
"survivorship bias was helping us". The mechanism is that the low-volatility
signal **essentially never holds the failures** (max 0.6% of days, most 0.0%) —
they are high-volatility and rank last. The change comes from the quantile cut
moving as the cross-section widens, not from holding failed names.

**So this test does not measure the survivorship bias.** It confounds two
effects, and a hand-picked failure set is itself a selected sample. It bounds
one narrow aspect; it does not substitute for point-in-time membership.

**Subperiod** — 76.9% of out-of-sample years have positive excess return.

**Concentration, regimes, parameter perturbation** — see
`research_data/study_result.json`.

---

## 8. Acceptance criteria

Fixed as module constants in `app/research/study.py` **before** the study ran.
Changing one is a reviewable diff sitting next to the result it produced.

| Criterion | Observed | Threshold | |
|---|---|---|---|
| OOS Sharpe | 1.2559 | ≥ 0.50 | **PASS** |
| Bootstrap 5th percentile | 0.6836 | > 0.0 | **PASS** |
| Deflated Sharpe Ratio | 0.9982 | ≥ 0.95 | **PASS** |
| PBO | 0.0143 | ≤ 0.50 | **PASS** |
| Excess CAGR vs benchmark | +4.81% | > 0 | **PASS** |
| Survives 50bps costs | 1.1119 | ≥ 0.50 | **PASS** |
| Positive excess years | 76.9% | ≥ 55% | **PASS** |
| **Point-in-time universe** | **present-day snapshot** | point-in-time | **FAIL** |

### Why the single failure is decisive

The statistical criteria are strong. They were computed on a universe that
**contains no company that failed**. §2 shows what that excludes: names that
lost 80–99% of their value, all of which were investable throughout the study
window and none of which the strategy could ever have avoided *ex ante* — it
would have had to hold some of them.

A low-volatility strategy is specifically exposed to this. Its selection rule is
"hold the calmest names", and a company on its way to insolvency is frequently
calm right up until it is not. The survivor-only universe removes exactly the
population where that rule would have been tested hardest.

Reporting VALIDATED here would mean certifying a result whose most important
input is known to be wrong. **A statistically convincing result on a biased
sample is a statistically convincing description of the bias.**

---

## 9. What would change the verdict

In priority order:

1. **Reconstruct point-in-time index membership.** §2 established the data is
   largely reachable — this is universe construction, not acquisition. Removes
   the single blocking criterion.
2. **Re-run the full study on that universe.** The current numbers are then
   superseded, not adjusted. If the result survives, it means something.
3. **Acquire point-in-time fundamentals with publication dates.** Unblocks value
   and quality, currently untestable.
4. **Extend beyond 108 symbols.** Confirms the result is not a large-cap artifact.

Not on this list: tuning any parameter, adding features, or trying more models.
The blocking problem is the data, and no amount of modelling fixes it.

---

## 10. Enforcement

`gather_repo_evidence()` reads `research_data/study_result.json` and populates
`oos_sharpe`, `deflated_sharpe_ratio` and `oos_sharpe_at_stressed_costs` from
the study's own output, so the eligibility gates evaluate the real numbers
rather than restated ones. A verdict other than `VALIDATED`, or a missing study
file, records a blocking note.

Current eligibility state: **`blocked_insufficient_data` — live trading not
permitted.**

```
Research study verdict: NOT VALIDATED (OOS Sharpe 1.2559, DSR 0.9982, 6 trials).
NO VALIDATED STRATEGY: the research study did not reach VALIDATED, so nothing
may be promoted to live trading regardless of how any individual gate reads.
```

---

## 11. What was deliberately not done

No parameter search. No threshold moved after seeing a result. No symbol or
period excluded post hoc. No losing period removed. No claim about future
returns. The six baselines use textbook parameters; the LightGBM model uses
fixed, heavily-regularised settings with no tuning.

**PRODUCTION MODEL = NONE. LIVE TRADING = BLOCKED. PAPER IS THE DEFAULT.**
