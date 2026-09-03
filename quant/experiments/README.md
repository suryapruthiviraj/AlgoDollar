# Experiments

This directory contains structured experiment records for systematic strategy development.

An "experiment" in AlgoDollar is a single, well-defined test of a hypothesis against data.
Each experiment has a defined purpose, defined inputs, defined outputs, and a recorded conclusion.
This discipline prevents the most common mistake in quant research: running many
tests and remembering only the good ones.

---

## Why Structured Experiments?

Without structure, quant research suffers from:

- **HARKing (Hypothesising After Results are Known)**: looking at results, then inventing
  a hypothesis that matches — this is the opposite of the scientific method
- **File-drawer problem**: discarding negative results and only reporting positive ones
- **Parameter overfitting**: testing 100 parameter combinations, reporting the best 1
- **Cumulative look-ahead**: unconsciously using knowledge of future events to design features
- **Recency bias**: strategies that look good in the recent past but have no theoretical basis

A structured log forces you to define the hypothesis BEFORE running the test.

---

## Experiment Log Format

Each experiment is recorded as a markdown file in this directory.

### File Naming

```
YYYY-MM-DD_EXP-NNN_<short_description>.md
```

Examples:
```
2024-01-15_EXP-001_momentum_12_1_baseline.md
2024-01-22_EXP-002_momentum_lgbm_v1.md
2024-01-28_EXP-003_momentum_lgbm_v2_with_volume.md
2024-02-05_EXP-004_swing_earnings_surprise.md
```

### Required Sections in Each Experiment File

```markdown
# EXP-NNN: <Title>

**Date**: YYYY-MM-DD
**Author**: <name>
**Strategy**: longterm | swing | intraday
**Status**: planned | in-progress | completed | rejected

## 1. Hypothesis

State the specific hypothesis being tested.
Example: "12-month momentum (skipping last month) predicts next-month returns
in the NSE 200 universe."

## 2. Motivation

Why test this? What evidence supports trying it?
Cite papers or prior experiments if applicable.

## 3. Setup

- Universe: <e.g., NSE 200 by market cap>
- Date range: <e.g., 2015-01-01 to 2023-12-31>
- Training set: <e.g., 2015–2020>
- OOS set: <e.g., 2021–2023>
- Feature(s) under test: <specific feature names and construction>
- Signal construction: <how features map to buy/sell signals>
- Data source: <e.g., Zerodha Kite historical>
- Cost model: <e.g., ZerodhaCostModel defaults>

## 4. What Would Falsify This

Define upfront what result would cause you to reject this hypothesis.
Example: "Annualised net Sharpe < 0.3, or t-stat < 1.5, or OOS performance
< 50% of IS performance."

## 5. Results

### In-Sample (Training Set)
| Metric | Value |
|---|---|
| Annualised return (net) | |
| Annualised Sharpe (net) | |
| Max drawdown | |
| Win rate | |
| Number of trades | |
| Average hold (trading days) | |

### Out-of-Sample (Test Set)
| Metric | Value |
|---|---|
| Annualised return (net) | |
| Annualised Sharpe (net) | |
| Max drawdown | |
| Win rate | |
| Number of trades | |

### Walk-Forward Windows
| Window | IS Sharpe | OOS Sharpe |
|---|---|---|
| 1 | | |
| 2 | | |
| ... | | |

### Regime Breakdown
| Regime | Sharpe | Trades |
|---|---|---|
| BULL | | |
| BEAR | | |
| NEUTRAL | | |
| HIGH_VOL | | |

### Cost Sensitivity
| Cost Multiple | Net Sharpe |
|---|---|
| 1x (base) | |
| 2x | |
| 5x | |

## 6. Interpretation

What do the results mean? Is the hypothesis supported?
Are there notable sub-period or regime-specific behaviours?
What are the limitations of this test?

## 7. Conclusion

- **Decision**: [proceed to paper / more research needed / reject]
- **Reason**: <brief justification>
- **Next experiment (if any)**: EXP-NNN+1 — <description>

## 8. Artifacts

- Notebook: `../notebooks/<path>`
- Data: `<path to any saved data files>`
- Model: `<path to saved model if applicable>`
```

---

## Running Experiments

### Setup

```bash
# Create a new experiment notebook from template
cd /path/to/AlgoDollar
cp quant/notebooks/template.ipynb quant/notebooks/<strategy>/$(date +%Y-%m-%d)_NNN_<description>.ipynb

# Create experiment log entry
cp quant/experiments/template.md quant/experiments/$(date +%Y-%m-%d)_EXP-NNN_<description>.md
# Fill in sections 1–4 BEFORE running any code
```

### Discipline Rules

1. **Write hypothesis before running**: Fill in sections 1–4 of the experiment log
   before executing a single notebook cell.

2. **Record all experiments**: Even if the result is negative. Especially if negative.

3. **Do not cherry-pick parameters**: Report the performance of the first parameter
   set that had theoretical justification, not the best-performing one found by search.

4. **One hypothesis per experiment**: If you want to test multiple things, create
   multiple experiments.

5. **Commit experiment logs**: Experiment logs are committed to git so there is a
   permanent record of what was tried and when.

---

## Comparing Experiments

When comparing two experiments on the same hypothesis with different implementations:

```python
# Use AlgoDollar's experiment comparison utility (when implemented)
from quant.utils.experiment_compare import compare_experiments

compare_experiments(
    "EXP-001",
    "EXP-002",
    metrics=["net_sharpe", "max_drawdown", "oos_degradation"],
)
```

Or manually tabulate in a comparison log:

```markdown
# Comparison: EXP-001 vs EXP-002 vs EXP-003

| Experiment | Description | IS Sharpe | OOS Sharpe | Max DD | Trades |
|---|---|---|---|---|---|
| EXP-001 | Momentum 12-1, equal weight | 0.82 | 0.54 | -18% | 320 |
| EXP-002 | Momentum + volume filter | 0.91 | 0.71 | -15% | 210 |
| EXP-003 | Momentum + volume + quality | 0.95 | 0.73 | -13% | 175 |

Winner: EXP-003 — proceed to walk-forward validation
```

---

## Experiment Index

Update this table as experiments are completed:

| ID | Date | Description | Strategy | Status | Conclusion |
|---|---|---|---|---|---|
| EXP-001 | (pending) | Momentum 12-1 baseline | longterm | planned | — |

---

## Common Experiment Failure Modes

### "It worked in the notebook but not in production"

Likely causes:
- Feature calculation in production uses different data (e.g., adjusted vs unadjusted prices)
- Signal timing differs (end-of-day signal executed next day open — this is correct; same-bar execution is look-ahead)
- Cost model used in notebook differs from production cost model

**Fix**: Validate feature parity between notebook and production pipeline before paper trading.

### "OOS is much worse than IS"

Likely causes:
- Overfitting (too many parameters, too little data)
- Data snooping (OOS was accidentally touched during development)
- Regime change between IS and OOS periods

**Fix**: Increase regularisation, expand IS period, or reject the hypothesis.

### "Good results but I can't explain why it works"

This is a warning sign, not a feature. A strategy you cannot explain is more likely
to be spurious than one with a clear mechanism.

**Fix**: Dig deeper into the results. If no mechanism emerges, treat with skepticism
and require stronger evidence (longer OOS, more walk-forward windows, stricter t-stat threshold).

### "The best parameters are at the edge of the search space"

If the optimal momentum window is 252 days and that is the maximum you tested, you do
not know if 300 or 400 days would be better. The edge-of-grid result suggests more
exploration is needed — and the parameter choice is fragile.

**Fix**: Expand the parameter search range and test sensitivity around the optimum.
