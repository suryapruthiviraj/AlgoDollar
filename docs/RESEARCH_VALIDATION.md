# AlgoDollar — Research Validation Report

# STATUS: **NOT VALIDATED**

**No strategy in this repository is approved for live trading.** The execution
system refuses to promote one: `gather_repo_evidence()` reads the study's own
output file, and a verdict other than `VALIDATED` is recorded as a blocking
note in the eligibility report.

Seven of eight acceptance criteria pass. The eighth — a point-in-time universe —
fails, and it is not a technicality. It is the criterion that decides whether
the other seven measured anything real.

Everything below is computed by `backend/scripts/run_research.py` from
`backend/research_data/`. Regenerate with:

```bash
python scripts/acquire_data.py --symbols 120 --start 2012-01-01
python scripts/audit_data.py
python scripts/probe_survivorship.py
python scripts/run_research.py
```

---

## 1. The dataset

Real NSE daily OHLCV from Yahoo Finance, acquired 2026-09-04.

| | |
|---|---|
| Symbols (research universe) | **108** |
| Observations | **362,697** |
| Trading sessions | **3,623** |
| Date range | **2012-01-02 → 2026-09-03** (14.7 years) |
| Benchmark | `^NSEI` (NIFTY 50), 3,603 sessions |
| Stress universe (sensitivity only) | 12 known corporate failures |

### Integrity — what the audit found

| Check | Result |
|---|---|
| Duplicate index rows | **0** |
| Non-monotonic date indexes | **0** |
| Non-positive prices | **0** |
| OHLC relationship violations (high < low, etc.) | **0** |
| Symbols missing `adj_close` | **0** |
| Symbols with intra-life coverage gaps | 7 |
| Benchmark sessions missing vs. universe | 20 |
| **Stale carried-forward bars** | **757 across all 108 symbols** |

Nothing was repaired. The audit reports; it does not forward-fill, de-duplicate
or smooth.

### Two findings that change how the data must be used

**Stale bars.** 757 rows have `volume == 0` and `open == high == low == close` —
the vendor carrying the previous close forward on a non-trading day. They are
not neutral: each produces a return of **exactly zero**, which compresses
measured volatility and inflates any Sharpe computed from the series. Masking
them removed 136 fake zero-returns from a 20-symbol sample (314 → 178). They are
NaN'd by `ParquetDailyBars.panel(drop_stale=True)`, never filled.

**`close` is already split-adjusted.** The `close/adj_close` ratio starts above
1.0 and converges to exactly 1.0 at the end of every series, and no symbol shows
a ~50% single-day move in `close` where a split is known to have occurred. Only
dividends separate the two series. Consequence: **neither series is the price
actually quoted on a past date.** Returns are correct; a backtest sizing
positions in *shares* off `close` computes a share count that never existed.
Rupee-notional sizing is unaffected, which is what this backtester uses.

### Point-in-time vs reconstructed

| Field | Status |
|---|---|
| Daily OHLCV bars | **POINT-IN-TIME** — as traded on the date recorded |
| `adj_close` | **RECONSTRUCTED** — back-adjusted with actions known *today* |
| Universe membership | **RECONSTRUCTED** — present-day snapshot |
| Sector labels | **RECONSTRUCTED** — current classification applied to all dates |
| Benchmark level | **POINT-IN-TIME** |

### Datasets that do not exist here

Wired as raising `UnavailableProvider`s rather than left as `None`, so a study
needing one fails loudly instead of running on a substitute:

- **Intraday bars** — daily only. No intraday strategy can be researched.
- **Point-in-time index membership** — see §2.
- **Corporate actions** — no ex-dates, ratios or dividend amounts. Only implicit
  in `adj_close`.
- **Point-in-time fundamentals** — no publication dates. **Value and quality
  factors are therefore untestable here**, and are reported as untested rather
  than approximated.
- **Delistings / symbol changes** — no mapping table. 12 of 120 requested
  symbols returned nothing (`HDFC`, `LTIM`, `MCDOWELL-N`, `ADANITRANS`, `PVR`,
  `ZOMATO`…), most of them renamed or merged tickers. That is direct evidence of
  the gap, not a transient fetch error.

---

## 2. Survivorship bias — measured, not just declared

The universe is a **present-day** NIFTY 500 snapshot, so every name in it
survived to today. Rather than state that and move on,
`scripts/probe_survivorship.py` tried to fetch 16 well-known Indian corporate
failures.

**13 of 16 were reachable, with full 3,600-row history.**

| Symbol | Total return 2012→2026 | Max drawdown |
|---|---|---|
| RCOM | **−98.9%** | −99.7% |
| GTLINFRA | −87.2% | −98.6% |
| RELINFRA | −81.1% | −98.8% |
| UNITECH | −80.7% | −99.1% |
| JETAIRWAYS | −80.2% | −98.4% |
| IDEA | −70.6% | −97.6% |
| RPOWER | −67.9% | −99.2% |
| YESBANK | −47.0% | −97.2% |

Controls (RELIANCE, TCS, INFY) all returned 3,623 rows, so the probe itself
works.

**This is the important conclusion: the survivorship gap here is mostly a
UNIVERSE CONSTRUCTION problem, not a data availability one.** These names are
absent from the study because the membership list is a snapshot, not because the
vendor cannot serve them. That is fixable — and until it is fixed, the verdict
stays NOT VALIDATED.

`DHFL`, `ALOKTEXT` and `JPASSOCIAT` were genuinely unreachable and remain
permanently absent.

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
