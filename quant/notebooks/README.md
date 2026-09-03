# Research Notebooks

This directory contains Jupyter notebooks for quantitative research and strategy development.

## Research Workflow

Every strategy follows this mandatory pipeline before any capital is allocated — in paper mode or live. Steps cannot be skipped or reordered.

```
DATA
  │  Load and validate historical OHLCV + fundamental data.
  │  Check for data quality issues: gaps, outliers, survivorship bias.
  │  Document the universe, date range, and data source.
  ▼
FEATURES
  │  Engineer features from raw data (momentum, volatility, volume, etc.).
  │  Verify all features are point-in-time (no look-ahead).
  │  Normalise and winsorise as appropriate.
  ▼
HYPOTHESIS
  │  State the trading hypothesis clearly and concisely.
  │  Reference supporting academic or empirical evidence.
  │  Define the expected mechanism: why should this edge exist?
  │  Define what would falsify the hypothesis.
  ▼
BASELINE
  │  Implement a simple baseline (e.g., equal-weight, buy-and-hold, random).
  │  Document baseline performance as the floor to beat.
  │  All subsequent results are compared to this baseline.
  ▼
MODEL
  │  Implement the signal model (rule-based or ML).
  │  Train ONLY on the training set. Never touch the test set during development.
  │  Document hyperparameters and their selection method.
  ▼
BACKTEST
  │  Run event-driven backtest with realistic costs (Zerodha cost model).
  │  Include: brokerage cap, STT, exchange charges, GST, stamp duty.
  │  Include slippage estimates appropriate to the instrument's liquidity.
  │  Report: net Sharpe, max drawdown, win rate, avg hold, turnover.
  ▼
VALIDATION
  │  Statistical significance: is the Sharpe > 0 with p < 0.05?
  │  Is the result robust to minor parameter changes (sensitivity analysis)?
  │  Is the distribution of returns consistent with the hypothesis?
  │  Are there suspicious patterns (e.g., perfect entries at exact highs/lows)?
  ▼
WALK-FORWARD
  │  Slide a fixed training window forward through the full dataset.
  │  At each step: train on window T, test on next N months (OOS).
  │  Report consistency of OOS performance across all windows.
  │  A strategy that only works in-sample is overfit — discard it.
  ▼
OOS
  │  Reserve the most recent 20% of data as a completely held-out test set.
  │  This data is touched ONCE, at the end of the research pipeline.
  │  The OOS result is the "honest" estimate of forward performance.
  │  If OOS significantly underperforms IS, investigate before proceeding.
  ▼
COST TEST
  │  Stress-test with higher costs (2x and 5x the base cost model).
  │  Identify the break-even cost rate — above which the strategy loses money.
  │  Is the edge large enough to survive realistic cost uncertainty?
  ▼
SLIPPAGE TEST
  │  Stress-test with higher slippage (0.1%, 0.3%, 0.5%, 1.0%).
  │  For intraday strategies: slippage is often the dominant cost.
  │  Is the edge large enough to survive realistic slippage?
  ▼
MONTE CARLO
  │  Simulate 1,000+ bootstrap resamples of the trade sequence.
  │  Report distribution of Sharpe ratios and max drawdowns.
  │  Report the 5th percentile drawdown — your realistic worst case.
  │  This quantifies the range of outcomes, not just the point estimate.
  ▼
MULTIPLE TESTING
  │  Have you tested multiple variants of this strategy? Apply a correction.
  │  Bonferroni correction (conservative): p_threshold = 0.05 / n_tests
  │  Or report the deflated Sharpe ratio (López de Prado 2018).
  │  Acknowledge if the result is one of many tested — this inflates false positives.
  ▼
REGIME TEST
  │  Segment results by market regime (BULL / BEAR / NEUTRAL / HIGH_VOL).
  │  Does the strategy hold up in BEAR markets?
  │  Does it degrade under high volatility?
  │  Regime-specific performance informs the allocator's regime adjustment rules.
  ▼
PAPER
  │  Run the strategy in paper trading mode for at least 30 trading days.
  │  Compare paper performance to backtest expectations.
  │  Investigate any significant divergence (execution differences, data issues).
  │  Document paper P&L, fill rates, and any implementation surprises.
  ▼
LIVE
     Only after all previous steps are completed and reviewed.
     Start with a small fraction of the strategy's target allocation.
     Monitor closely for the first 60 trading days.
     Compare live performance to paper and backtest.
```

---

## Notebook Naming Convention

```
YYYY-MM-DD_NNN_<description>.ipynb
```

- `YYYY-MM-DD`: Date the notebook was created
- `NNN`: Sequential number within a research project (001, 002, ...)
- `<description>`: Short snake_case description

Examples:
```
2024-01-15_001_momentum_feature_engineering.ipynb
2024-01-18_002_momentum_baseline_backtest.ipynb
2024-01-22_003_momentum_lgbm_model.ipynb
2024-01-28_004_momentum_walkforward_validation.ipynb
2024-02-03_005_momentum_oos_test.ipynb
```

---

## Directory Structure

```
notebooks/
├── README.md                  (this file)
├── longterm/                  Long-term factor strategy research
│   ├── 001_universe_setup.ipynb
│   ├── 002_feature_engineering.ipynb
│   └── ...
├── swing/                     Swing strategy research
│   ├── 001_catalyst_signals.ipynb
│   └── ...
├── intraday/                  Intraday strategy research
│   ├── 001_opening_range.ipynb
│   └── ...
└── shared/                    Utilities shared across strategies
    ├── data_loader.ipynb
    ├── cost_model_validation.ipynb
    └── regime_analysis.ipynb
```

---

## Required Notebook Template Structure

Every research notebook must include:

1. **Header cell (Markdown)**:
   - Hypothesis statement
   - Data source and date range
   - Expected mechanism
   - What falsifies the hypothesis

2. **Data loading and validation**

3. **Feature engineering** (with look-ahead verification)

4. **Model / signal implementation**

5. **Backtest** (using AlgoDollar's backtesting engine, not ad-hoc loops)

6. **Results summary** table

7. **Conclusion cell (Markdown)**:
   - Is the hypothesis supported by the evidence?
   - What are the limitations?
   - What are the next steps?
   - Decision: proceed to paper / reject / needs more research

---

## Common Pitfalls to Avoid

- **Look-ahead bias**: Using data at time T that was not available at T
  (e.g., end-of-day close in a signal that triggers at the open)
- **Survivorship bias**: Testing only on companies that are currently listed
- **Overfitting**: Testing many parameter combinations and reporting the best
- **Ignoring costs**: Gross returns look great; net returns are what matter
- **Data snooping**: Each "independent" test of OOS data contaminates it
- **Ignoring market impact**: A strategy that works for small positions may not
  scale because it moves the market

---

## Useful References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Jansen, S. (2020). *Machine Learning for Algorithmic Trading*. Packt.
- Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley.
- Carhart, M. (1997). On persistence in mutual fund performance. *Journal of Finance*.
- Fama, E. & French, K. (1993). Common risk factors in stock returns. *Journal of Financial Economics*.
- Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*.
