# AlgoDollar — Data Integrity Report

**Purpose:** state exactly what data exists, exactly what is wrong with it, and exactly what each defect prevents. A research result is only as trustworthy as the inventory behind it.

**Generated from:** `app/data/inventory.py`, run against the live cache. Every figure below is computed from the data, not transcribed.

---

## 1. Dataset inventory — exact

A previous report described this dataset as "99 NSE symbols, 4,262 daily bars" without stating whether 4,262 was per instrument or in total. That was ambiguous. The precise figures:

```
Number of securities        : 99
Total observations          : 434,998   (non-null security-date pairs)
Observations per security   : mean 4,394,  min 1,292,  max 4,685
Distinct trading dates      : 4,685
Earliest date               : 2006-01-02
Latest date                 : 2024-12-31
Currently listed securities : 99
Delisted securities         : 0
Incomplete histories        : 17
Complete histories          : 82
Number of sectors           : 8
Point-in-time constituents  : False
```

**Reading these numbers correctly:**

- **4,685** is the number of distinct trading dates in the panel. It is *not* the observation count.
- **434,998** is the total number of non-null (security, date) observations.
- The research run in `REAL_DATA_VALIDATION_REPORT.md` used a **4,262-date** subset — the panel trimmed to where the NIFTY 50 benchmark exists (2007-09-17 onward). That is the number previously quoted, and it referred to dates, not bars.
- **Delisted securities: 0** is not a claim that no NSE company was delisted in 19 years. It is a statement that this dataset contains none — which is the defect described in §3.
- **17 incomplete histories** are securities whose data begins more than a year after the panel start or ends more than 30 days before its end. The effective universe grows over time.

---

## 2. Dataset provenance

| Field | Value |
|---|---|
| Source | Yahoo Finance via `yfinance`, NSE `.NS` tickers |
| Acquisition date | 2026-09-03 |
| Frequency | Daily |
| Timezone | Naive dates representing Asia/Kolkata trading sessions |
| Adjustment method | Vendor adjusted close (splits + dividends), applied retroactively |
| Corporate-action policy | Vendor adjustment, plus masking of 12 detected unadjusted events |
| Survivorship policy | **FILTERED** — see §3 |
| Point-in-time policy | **NONE** — today's universe applied to all history |
| Checksum | `be05d2ccd470044973b23c044832780ada21d7832aba6f7224d60d5f387893fe` |
| License | Unspecified — vendor terms must be verified before any redistribution or commercial use |

Machine-readable manifest: `research/data_manifest.json`. `app/data/inventory.py:verify_manifest()` recomputes the content checksum and reports a mismatch, so a research run can prove which data version produced it.

---

## 3. Survivorship status — FILTERED, direction indeterminate

### 3.1 The bias is present. This was measured, not assumed.

Each company below was queried over a window in which it was actively listed and trading:

| Company | Event | Window queried | Rows returned |
|---|---|---|---|
| SATYAMCOMP | 2009 accounting fraud | 2006-01 → 2008-12 | **0 — absent** |
| DHFL | 2019 collapse | 2016-01 → 2019-01 | **0 — absent** |
| VIDEOIND | insolvency | 2010-01 → 2016-01 | **0 — absent** |
| RELIANCE (control) | still listed | 2006-01 → 2008-12 | 720 — present |

The control returning full history rules out an unreachable source. The vendor retains survivors and drops failures.

### 3.2 CORRECTION: the direction of this bias is NOT determinate

An earlier version of the validation report claimed *"results are biased upward, therefore a negative result is robust."* **That reasoning was wrong** and is corrected here and in `REAL_DATA_VALIDATION_REPORT.md` §2.3.

Survivorship inflates the **absolute** return of a long-only book, because companies that went to zero are missing. But the research measured **excess return of a top quintile over an equal-weight benchmark of the same universe**. Both legs share the filtering, so the bias does not cleanly survive the subtraction.

For a momentum signal it plausibly runs the other way. A company heading for delisting has collapsing momentum long before it disappears: Satyam through 2008, DHFL through 2019. A momentum rule would have ranked such a name in its **bottom** quintile and avoided it, while the equal-weight benchmark held it all the way down. Removing those companies deletes episodes in which the signal was **right**, biasing its measured excess return **downward**.

**Correct statement:** this dataset cannot establish true historical performance of any strategy tested on it, in either direction.

### 3.3 What this does and does not invalidate

| Conclusion | Still valid? | Why |
|---|---|---|
| The ten candidates failed **on this dataset** | **Yes** | A fact about this measurement |
| No exploitable edge exists in Indian equities | **NO** | Never established; the dataset cannot support it |
| Turnover inverts the IC-to-profit ranking | **Yes** | Property of the return series, independent of universe composition |
| LightGBM lost to momentum on net return | **Yes** | Both measured on the same universe |
| Performance concentrated in 4% of rebalances | **Yes** | Property of the return series itself |
| Edge disappears below the 200-day MA | **Yes** | Property of the return series itself |
| Absolute CAGR / Sharpe figures | **No** | Depend directly on universe composition |

---

## 4. Point-in-time status — ABSENT

| Requirement | Status | What it blocks |
|---|---|---|
| Point-in-time index constituents | **Absent** | Survivorship cannot be corrected. Using today's membership across all history additionally selects companies *for having become large* — a second look-ahead layered on the first. |
| Point-in-time fundamentals with publication dates | **Absent** | **The long-term engine cannot be validated at all.** |
| Historical sector membership | **Absent** | Sector concentration analysis uses today's classification retroactively. |

The current sector map (8 sectors, used in `swing_robustness.py`) is a hand-maintained present-day classification. Sector reclassifications over 19 years are not captured.

---

## 5. Corporate-action coverage — PARTIAL, with detected gaps

Vendor adjustment is applied to **98 of 99** symbols (adjusted close differs from raw close).

**12 unadjusted events were detected and masked.** They were identified by a specific test: when a vendor fails to apply an adjustment, the adjusted and raw series move by the *identical* ratio on that date, because the factor was never applied to either.

| Symbol | Date | Log return | adj_ratio | raw_ratio |
|---|---|---:|---:|---:|
| BAJAJFINSV | 2008-05-26 | −2.6102 | 0.0735 | 0.0735 |
| NESTLEIND | 2010-01-08 | −1.4411 | 0.2367 | 0.2367 |
| ABBOTINDIA | 2010-01-08 | +1.0582 | 2.8811 | 2.8811 |
| BAJAJFINSV | 2008-03-14 | −1.0157 | 0.3621 | 0.3621 |
| DABUR | 2007-01-25 | −0.7545 | 0.4702 | 0.4702 |
| GLENMARK | 2007-09-10 | +0.7285 | 2.0719 | 2.0719 |
| LT | 2006-09-27 | −0.7046 | 0.4943 | 0.4943 |
| DABUR | 2007-01-29 | +0.6868 | 1.9873 | 1.9873 |
| ZYDUSLIFE | 2006-08-30 | −0.6746 | 0.5093 | 0.5093 |
| LT | 2006-09-28 | +0.6683 | 1.9510 | 1.9510 |
| BAJAJ-AUTO | 2008-05-26 | −0.5522 | 0.5757 | 0.5757 |
| ADANIENT | 2015-06-03 | −0.4902 | 0.6125 | 0.6125 |

`adj_ratio == raw_ratio` in all twelve cases confirms the diagnosis. BAJAJFINSV's −261% is the 2008 Bajaj demerger.

**Why this matters:** left in place, an unadjusted 1:5 split reads as an 80% single-day crash. Any mean-reversion model learns to buy it and books an enormous imaginary profit on a "recovery" that never happened — the shareholder simply held five times as many shares throughout.

After masking: maximum daily move falls from a fake **261%** to a plausible **38%**, at a cost of **12 / 434,998 = 0.003%** of observations.

**Remaining risk:** the detector finds events above a 40% threshold. Smaller unadjusted actions (a 5:4 bonus, say) would not trigger it and may remain. Dividend adjustment correctness has not been independently verified against a corporate-actions source.

---

## 6. Fundamental-data coverage — NONE

No point-in-time fundamentals were obtained. The long-term engine currently sources from `_MockFundamentalProvider`, which generates **synthetic random values**.

Guards now in place (from the previous phase): the mock provider is deterministic per symbol, is flagged `IS_MOCK`, and **raises `MockDataInLiveModeError`** at construction, at `generate_signals`, and at `calculate_position_size` when not in paper mode.

**No mock-derived output appears anywhere in this repository as a research result.** The long-term engine has produced no validated numbers, and none are quoted.

Required fields per fundamental observation, none of which are currently available:

```
security identifier
observation date        (the period the value describes)
publication date        (when it became public — this is the critical field)
effective period
value
source
```

Without the publication date, a fundamental feature cannot be built causally: annual results for FY2015 were not knowable in April 2015.

---

## 7. Intraday-data coverage — INSUFFICIENT FOR RESEARCH

Measured vendor limits:

| Interval | History available | Bars retrieved (RELIANCE) |
|---|---|---|
| 1 minute | ~7 days | 2,526 |
| 5 minutes | ~60 days | 4,374 |
| 15 minutes | ~60 days | 1,472 |
| 1 hour | ~730 days | 5,050 |

**Sixty days of 5-minute data cannot support intraday strategy research.** It spans one market regime at best, offers no bear-market or crisis observations, and provides far too few independent periods for any multiple-testing-corrected claim. Bid/ask spreads are not available at any resolution.

The intraday engine is therefore **unvalidatable on current data** — a data blocker, not a code defect.

---

## 8. Data-quality testing

`app/data/quality.py` runs before any modelling and blocks on CRITICAL findings.

Current result on this panel: **0 CRITICAL, 6 WARNING, 3 INFO → USABLE**

| Severity | Finding |
|---|---|
| INFO | 272 weekdays absent (5.5%) — consistent with NSE holidays (expected 4–7%) |
| WARNING | 400,363 bars where close lies outside [low, high] — expected when comparing *adjusted* close against *unadjusted* high/low |
| WARNING | 12 extreme moves > 40% — the corporate actions of §5, now masked |
| WARNING | 4 symbols with ≥10 consecutive identical closes — volatility understated for these names |
| WARNING | 1 date with fewer than 10 live symbols |
| WARNING | 31,403 zero-volume bars (6.77%) — a fill cannot be assumed on a bar with no trading |
| WARNING | 1 symbol with median daily turnover below ₹1 crore |
| INFO | 98/99 symbols show adjustment applied |
| INFO | 17 symbols with staggered listing dates |

The adjusted-vs-unadjusted OHLC mismatch is a real limitation, not a cosmetic one: it means intrabar high/low cannot be used together with adjusted close without care, and ATR computed across the two is approximate.

---

## 9. Compliance with the required criteria

| Criterion | Status |
|---|---|
| No dataset incorrectly described as survivorship-free | **Met** — filtering is stated in the provider docstring, the manifest, the quality probe and every report |
| Point-in-time status explicitly known | **Met** — absent, and documented as such |
| Corporate-action treatment verified | **Met** — 12 unadjusted events detected, diagnosed and masked; residual risk stated |
| Historical universe construction documented | **Met** — hand-maintained present-day list, explicitly not point-in-time |
| Delisted securities explicitly accounted for | **Met** — count is 0 and the reason is documented |
| Missing data identified | **Met** — §4, §6, §7 |
| Data-quality tests exist | **Met** — `app/data/quality.py`, run before modelling, blocks on CRITICAL |
| No mock fundamental data presented as real research | **Met** — guarded by a runtime raise; no mock-derived number is quoted anywhere |
| No synthetic data presented as evidence of profitability | **Met** — synthetic results are labelled as machinery verification in `AUDIT_REPORT.md` §0 |

---

## 10. Verdict

**The current dataset is adequate for engineering verification and inadequate for any performance claim.**

It supports: causality testing, cost modelling, turnover measurement, concentration analysis, regime analysis, and end-to-end pipeline exercise.

It does not support: any statement about the true historical performance of any strategy, in either direction.

Required to change that: point-in-time index constituents and delisted-security price history. See `DATA_ACQUISITION_PLAN.md`.
