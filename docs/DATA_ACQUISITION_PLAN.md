# AlgoDollar — Data Acquisition Plan

**Purpose:** every research component that is currently blocked, the exact data that would unblock it, and the next concrete step.

**Rule observed throughout:** missing data is documented, never fabricated or substituted with synthetic values.

---

## 1. Requirements matrix

| Dataset | Required? | Why | Minimum history | Frequency | Candidate sources | Current status |
|---|---|---|---|---|---|---|
| **Point-in-time index constituents** | **YES — blocking everything** | Without membership as of date *t*, the universe silently selects companies for having survived and grown. Corrupts every horizon. | 15+ years | Monthly / on-change | NSE Indices official index-change circulars; Refinitiv/LSEG; Bloomberg; CMIE Prowess; academic PIT datasets | **ABSENT** |
| **Delisted security prices** | **YES — blocking performance claims** | Needed to quantify survivorship bias rather than merely detect it. Yahoo drops these entirely. | Same span as the panel | Daily OHLCV | BSE/NSE historical archives; CMIE Prowess; Refinitiv; Bhavcopy archives | **ABSENT** |
| **Corporate actions (authoritative)** | **YES** | Currently inferred by detecting anomalies. 12 unadjusted events were found; smaller ones may remain undetected. | Same span | Event-based | NSE/BSE corporate-action feeds; exchange bhavcopy | **INFERRED ONLY** |
| **Daily OHLCV** | Yes | Core price data. | 15+ years | Daily | Yahoo (current); NSE bhavcopy; broker historical API | **PRESENT** (survivorship-filtered) |
| **Point-in-time fundamentals** | **YES — blocks long-term engine** | Quality/growth/valuation scoring is meaningless without values *and their publication dates*. | 10+ years | Quarterly, with publication dates | CMIE Prowess; Capitaline; Refinitiv Fundamentals; screener.in (verify PIT); company filings | **ABSENT — currently mock** |
| **Intraday granular data** | **YES — blocks intraday engine** | 60 days spans one regime; cannot support a corrected claim. | 3–5 years minimum | 1-min or 5-min | Zerodha Kite historical API (subscription); TrueData; GDFL; NSE data products | **INSUFFICIENT (~60 days)** |
| **Bid/ask / spread data** | Yes for intraday, desirable for swing | Slippage is currently modelled, never measured. | 2+ years | Tick or snapshot | Broker tick feed; exchange depth products | **ABSENT** |
| **Sector / industry classification (PIT)** | Yes | Sector concentration analysis currently applies today's classification retroactively. | Same span | On-change | NSE/AMFI classification history; vendor sector history | **PRESENT-DAY ONLY** |
| **Market/index reference series** | Yes | Benchmark and market-relative features. | Same span | Daily | Yahoo `^NSEI` (current); NSE official | **PRESENT** (from 2007-09-17) |

---

## 2. What each acquisition unblocks

### 2.1 Point-in-time constituents + delisted prices → the swing horizon

**Currently blocked because:** the universe is today's membership; delisted names are absent; the direction of the resulting bias on excess return is indeterminate.

**Unblocks:** the ability to make *any* statement about true historical performance. Until then, results are measurements of a filtered sample.

**Expected effect on results:** unknown, and worth stating honestly. Two opposing forces:
- Adding failed companies to the benchmark lowers the benchmark, which *helps* a strategy that avoided them.
- Adding them to the tradeable universe creates opportunities to hold them, which *hurts* a strategy that would have bought them.

For a momentum signal the first effect is more likely to dominate, since collapsing momentum would have kept those names out of the long book. **This is a hypothesis, not a prediction, and it must be measured rather than assumed.**

**Next step:** obtain NSE index-change circulars (publicly published) and reconstruct NIFTY 100/500 membership by date. This is the single highest-value acquisition and is partly achievable from public sources with effort.

### 2.2 Point-in-time fundamentals → the long-term engine

**Currently blocked because:** fundamentals are synthetic random values. The engine is guarded to refuse to run outside paper mode, and produces no quoted results.

**Required schema per observation:**

```
security_id
observation_date      the fiscal period the value describes
publication_date      when it became publicly known   <-- the critical field
effective_period      e.g. FY2015-Q3
value
source
```

**Why `publication_date` is the whole problem:** FY2015 annual results were not knowable in April 2015. A dataset carrying only the fiscal-period date will silently create look-ahead in every fundamental feature, and the causality tests in `tests/test_lookahead_causality.py` cannot catch it because the data itself is mis-stamped.

**Next step:** evaluate CMIE Prowess or Capitaline for genuine PIT coverage with publication dates. Confirm before purchase that publication dates are included — many "historical fundamentals" products are restated and therefore useless for backtesting.

**Interim recommendation:** disable the long-term engine outright rather than leaving it running on mock data. A guard that raises is good; not shipping the code path at all is better.

### 2.3 Intraday history → the intraday engine

**Currently blocked because:** ~60 days of 5-minute data.

**Statistical justification for the minimum:** an intraday strategy trading a few times per day generates a few hundred independent observations per year. To distinguish a Sharpe of 1.0 from zero at conventional confidence needs roughly 3–4 years; to survive a multiple-testing correction across even ten candidate configurations needs more. Sixty days provides neither the sample size nor any bear-market or crisis regime. **Sixty days is not a smaller version of the right dataset; it is the wrong dataset.**

**Minimum viable:** 3 years of 1-minute or 5-minute bars across the tradeable universe, plus spread data.

**Next step:** Zerodha Kite Connect historical API provides intraday history for subscribers. This is the natural source since it is also the execution venue, which removes a data/execution mismatch. Requires the ₹2,000/month subscription.

### 2.4 Authoritative corporate actions

**Currently:** actions are *inferred* by detecting anomalies where adjusted and raw ratios coincide. This caught 12 events above a 40% threshold. A 5:4 bonus would not trigger it.

**Next step:** ingest NSE/BSE corporate-action feeds and reconcile against the detector. The detector becomes a cross-check rather than the primary source.

---

## 3. Sequencing

Ordered by value per unit of effort:

1. **NSE index-change circulars → point-in-time constituents.** Public, free, laborious. Unblocks the most.
2. **Delisted price history.** Bhavcopy archives are public but require assembly. Enables quantifying the bias.
3. **Authoritative corporate actions.** Public feeds; converts inference into verification.
4. **Kite historical intraday.** Paid but modest; also removes the data/execution venue mismatch.
5. **Point-in-time fundamentals.** Most expensive and hardest to verify. Defer until 1–3 are done, and consider disabling the long-term engine in the interim.

---

## 4. What must NOT happen while data is missing

- Do not substitute synthetic data for unavailable real data and report the output as research.
- Do not treat present-day universe results as historical performance estimates.
- Do not present mock-fundamental output as a long-term result.
- Do not lower a validation standard because the data is inconvenient.
- Do not re-run the existing candidates against the holdout hoping for a different answer.

---

## 5. Status

Every blocked component has a defined requirement and a next step. **No missing dataset has been fabricated, approximated, or substituted.**

The honest position: the platform can be made trustworthy without this data — that is the current phase's goal — but it cannot produce a defensible performance claim until at least items 1 and 2 are acquired.
