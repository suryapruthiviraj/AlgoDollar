# Quant Research Methodology

This directory contains structured research documents, literature reviews, and methodology notes
that underpin AlgoDollar's strategy development. Unlike the interactive notebooks in `../notebooks/`,
the files here are static reference documents — decision logs, academic summaries, and design choices.

---

## Research Philosophy

### Evidence-Based Development

Every strategy element must have empirical or theoretical support. Intuition alone is not sufficient.

> "If you can't cite a paper or provide your own out-of-sample evidence, it's a guess."

Before implementing any signal:
1. Find at least one academic paper documenting the effect in equity markets
2. Understand the proposed mechanism (why does this edge exist?)
3. Understand the documented risks (when does this effect fail?)
4. Replicate the effect in the NSE universe before assuming it transfers

### The Replication Standard

A hypothesis is considered validated when:
- It replicates in the NSE universe using the same methodology as the source paper
- The Sharpe ratio is statistically significant (t-stat > 2.0)
- It survives out-of-sample testing
- It remains profitable after realistic transaction costs

If it does not replicate, document why and what was learned — negative results are valuable.

---

## Academic Foundations

### Factor Investing (Long-term Strategy)

#### Momentum
- **Jegadeesh & Titman (1993)**: *Returns to Buying Winners and Selling Losers*.
  Journal of Finance. Foundational paper on cross-sectional momentum.
  12-1 momentum (skip last month) is the standard construction.

- **Carhart (1997)**: *On Persistence in Mutual Fund Performance*.
  Journal of Finance. Added momentum as the 4th factor in the Carhart model.

- **Asness, Moskowitz & Pedersen (2013)**: *Value and Momentum Everywhere*.
  Journal of Finance. Momentum works across asset classes and geographies.

**NSE-specific note**: Momentum has been documented in Indian equities (see Sehgal & Jain 2011,
and various NSE working papers). The skip-last-month construction is standard.

#### Quality
- **Novy-Marx (2013)**: *The Other Side of Value: The Gross Profitability Premium*.
  Journal of Financial Economics. Gross profitability (gross profit / assets) predicts returns.

- **Asness, Frazzini & Pedersen (2019)**: *Quality Minus Junk*.
  Review of Accounting Studies. Composite quality score using profitability, growth, safety, payout.

**Implementation note**: Point-in-time fundamental data is required to avoid look-ahead bias.
Reported earnings dates are NOT the same as the data available date. Use announcement dates.

#### Value
- **Fama & French (1993)**: *Common Risk Factors in the Returns on Stocks and Bonds*.
  Journal of Financial Economics. Book-to-market (value factor) in the three-factor model.

- **Fama & French (2015)**: *A Five-Factor Asset Pricing Model*.
  Journal of Financial Economics. Adds profitability and investment factors.

**NSE-specific note**: Value effects in India differ from US due to different sectoral composition.
Financial sector stocks (banks, NBFCs) require special handling of book value.

### Technical Signals (Swing Strategy)

#### Momentum Continuation
- **Chan (1996)**: *Momentum Strategies*. Post-earnings announcement drift (PEAD) is a well-documented
  anomaly. Strong earnings surprises tend to be followed by continued price appreciation.

- **Jegadeesh & Titman (2001)**: *Profitability of Momentum Strategies: An Evaluation of Alternative
  Explanations*. Journal of Finance. Confirms the momentum effect is not simply risk compensation.

#### Volume as a Signal
- **Gervais, Kaniel & Mingelgrin (2001)**: *The High-Volume Return Premium*.
  Journal of Finance. High trading volume is associated with subsequent price continuation.

- **Lee & Swaminathan (2000)**: *Price Momentum and Trading Volume*.
  Journal of Finance. Volume helps predict the duration and magnitude of momentum.

#### Technical Indicators
- **Lo, Mamaysky & Wang (2000)**: *Foundations of Technical Analysis*.
  Journal of Finance. Provides rigorous statistical tests for common chart patterns.

**Caution**: Technical indicators tested on historical US data may not generalise to NSE.
Always validate on NSE data specifically and after transaction costs.

### Mean Reversion (Intraday Strategy)

- **Lo & MacKinlay (1990)**: *When Are Contrarian Profits Due to Stock Market Overreaction?*
  Review of Financial Studies. Short-term mean reversion at the daily level.

- **Lehmann (1990)**: *Fads, Martingales, and Market Efficiency*.
  Quarterly Journal of Economics. Short-horizon return reversals.

**NSE-specific note**: Intraday mean reversion depends heavily on tick structure and
liquidity. The VWAP deviation reversion strategy requires sufficient depth in the order book.
Liquid large-caps (Nifty 50) are more suitable than small-caps for intraday strategies.

---

## Data Sources

### Market Data (OHLCV)

| Source | Type | Coverage | Notes |
|---|---|---|---|
| Zerodha Kite Connect | Live ticks + historical OHLCV | NSE equities | Primary source |
| NSE India website | EOD barter data | NSE full history | Free download |
| Yahoo Finance (`yfinance`) | Historical OHLCV | Global | Adjusted prices, gaps possible |
| Quandl / Nasdaq Data Link | Historical | Various | Paid for premium data |

**Preferred**: Zerodha Kite historical API for data consistency with live trading.
The same data feed is used for backtesting and live trading, reducing implementation risk.

### Fundamental Data

| Source | Type | Notes |
|---|---|---|
| Screener.in | Financials, ratios | Good for NSE, manual scraping or paid API |
| Tijori Finance | NSE fundamentals | Paid |
| BSE India XBRL | Regulatory filings | Free, requires parsing |
| NSE India | Corporate announcements | Free |

**Critical**: Always use filing date (not fiscal year end) for point-in-time fundamental data.
Using year-end data as if it were available at fiscal year end introduces look-ahead bias.

### Corporate Events

| Event Type | Source |
|---|---|
| Earnings dates | NSE corporate actions page |
| Dividends | Zerodha Kite `kite.instruments()` |
| Splits / bonuses | Zerodha Kite historical data (adjusted) |
| Index rebalancing | NSE announcement page |

---

## Feature Engineering Philosophy

### Point-in-Time Rules

Every feature must obey strict point-in-time rules:

```
At time T (e.g., end of day D):
  - May use: all OHLCV data up to and including end of day D
  - May use: fundamental data filed on or before market close of day D
  - May NOT use: day D+1 prices (look-ahead)
  - May NOT use: fiscal year data released after day D (even if the fiscal
                 year ended before D)
```

### Feature Categories

**Price-based** (no look-ahead if computed from close prices up to T):
- Momentum: returns over N periods, skip last M
- Volatility: rolling standard deviation of returns
- RSI, MACD, ADX, Bollinger Bands
- VWAP and deviations from VWAP (intraday only)

**Volume-based**:
- Volume ratio (today vs N-day average)
- Dollar volume (price * volume) — proxy for liquidity
- On-Balance Volume (OBV) trend

**Fundamental-based** (requires point-in-time careful handling):
- Return on Equity (ROE) — use most recently filed quarterly
- Debt/Equity ratio
- P/E, P/B relative to sector median
- Earnings surprise magnitude (actual vs consensus)

**Regime-based** (computed from market-level data):
- Nifty 50 20-day vs 60-day SMA ratio
- Market-wide realised volatility (VIX proxy)
- Advance-Decline ratio

### Normalisation

- **Cross-sectional z-score**: subtract cross-sectional mean, divide by std (within date)
- **Winsorisation**: clip extreme values at ±3σ before normalisation
- **Rank normalisation**: convert raw values to percentile ranks (robust to outliers)

---

## Research Records

### Completed Studies

Document completed studies here with links to notebooks and conclusions:

| Study | Date | Notebooks | Conclusion |
|---|---|---|---|
| (add studies as completed) | | | |

### Rejected Hypotheses

Document rejected hypotheses to avoid re-investigating the same dead ends:

| Hypothesis | Date | Why Rejected |
|---|---|---|
| (add as encountered) | | |

### Open Questions

| Question | Date Opened | Priority |
|---|---|---|
| Does momentum persist after adjusting for Nifty beta? | | Medium |
| Does quality factor add value beyond momentum in NSE? | | High |
| What is the optimal look-back window for regime detection on Nifty? | | High |

---

## Statistical Guidelines

### Minimum Evidence Requirements

| Test | Minimum Threshold |
|---|---|
| Sharpe ratio (annualised, net of costs) | > 0.5 |
| t-statistic of Sharpe | > 2.0 |
| OOS Sharpe / IS Sharpe | > 0.5 (degradation < 50%) |
| Minimum sample period | 5 years |
| Minimum number of trades | 100 |
| Walk-forward OOS windows | >= 5 non-overlapping windows |

### Multiple Testing Adjustment

When testing N parameter combinations or N strategies:

```
Bonferroni threshold: p < 0.05 / N

Deflated Sharpe (López de Prado 2018):
  DSR = SR * sqrt(1 - skewness * SR + (kurtosis - 1)/4 * SR^2)
  Adjusted for N independent tests
```

Always report the number of variants tested, even when reporting only the best result.

### Regime Segmentation

Disaggregate all reported results by market regime:

| Regime | Definition | Expected Strategy Behaviour |
|---|---|---|
| BULL | Nifty 20-SMA > 60-SMA by >1% | Momentum strategies should outperform |
| BEAR | Nifty 20-SMA < 60-SMA by >1% | Defensive strategies; reduce intraday exposure |
| NEUTRAL | Between -1% and +1% | Mixed; rely on stock-specific signals |
| HIGH_VOL | VIX > 25 | All strategies should reduce size |

A strategy that only works in BULL regimes is not suitable for year-round deployment.
