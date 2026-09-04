# AlgoDollar

## Personal Quantitative Portfolio Management + Automated Trading Platform

> **Important Disclaimer**: This platform is for research and educational purposes only.
> Trading involves substantial risk of loss of capital. No guarantee of profit is expressed or
> implied. Past performance does not guarantee future results. Always consult a qualified
> financial advisor before making investment decisions. Paper trading mode is the default and
> the recommended starting point — live trading requires explicit manual opt-in.

---

## Current status: NOT VALIDATED — live trading is BLOCKED

Research was run on **real NSE daily data**: 108 symbols, 362,697 observations,
3,623 sessions, 2012-01-02 → 2026-09-03, benchmarked against `^NSEI`.

Six textbook baselines (parameters fixed before any result was seen) were evaluated under
purged, embargoed walk-forward, with selection done inside each fold on training data only.

| Out-of-sample (stitched test windows) | |
|---|---:|
| Sharpe (net of 25bps/side) | **1.256** |
| CAGR | 16.84% |
| Benchmark CAGR | 12.03% |
| Excess | **+4.81%** |
| Deflated Sharpe Ratio (6 trials) | 0.9982 |
| Probability of Backtest Overfitting | 0.014 |
| Bootstrap Sharpe 90% CI | [0.68, 1.89] |

**Seven of eight acceptance criteria pass. The eighth fails, and it is decisive.**

| Criterion | Observed | Threshold | |
|---|---:|---:|---|
| OOS Sharpe | 1.2559 | ≥ 0.50 | PASS |
| Bootstrap 5th percentile | 0.6836 | > 0 | PASS |
| Deflated Sharpe Ratio | 0.9982 | ≥ 0.95 | PASS |
| PBO | 0.0143 | ≤ 0.50 | PASS |
| Excess CAGR vs benchmark | +4.81% | > 0 | PASS |
| Survives 50bps costs | 1.1119 | ≥ 0.50 | PASS |
| Positive excess years | 76.9% | ≥ 55% | PASS |
| **Point-in-time universe** | **present-day snapshot** | point-in-time | **FAIL** |

The universe is a present-day NIFTY 500 snapshot, so it contains **no company that failed**.
A probe of 16 known Indian corporate failures found **13 reachable from the same vendor**
with full history — RCOM −98.9%, GTLINFRA −87.2%, RELINFRA −81.1%, UNITECH −80.7%,
JETAIRWAYS −80.2%. They are missing because of how the universe is built, not because the
data is unavailable.

The selected signal is low-volatility, which is *specifically* exposed to this: a company
heading for insolvency is frequently calm right up until it is not. A statistically
convincing result measured on a survivor-only universe is a convincing description of the
bias.

**Machine learning: REJECTED.** LightGBM out-of-sample Sharpe 0.809 against the baseline's
1.256 (−0.447), mean cross-sectional IC 0.0084 decaying to 0.0004 in the most recent fold.
It does not earn its complexity.

| | Status |
|---|---|
| Production model | **None** |
| Live trading | **BLOCKED_INSUFFICIENT_DATA** — 22 of 31 gates failing |
| Paper trading | **Working end to end** (see below) |
| Long-term sleeve | **Unvalidatable** — no point-in-time fundamentals |
| Intraday sleeve | **Unvalidatable** — daily bars only, no intraday feed |

Full evidence: **[docs/RESEARCH_VALIDATION.md](docs/RESEARCH_VALIDATION.md)**

### What *is* verified

Paper trading runs end to end through the production dependency graph: market data → signal
→ allocation → risk → eligibility → order manager → broker → persistence → reconciliation →
API → frontend. Restart, idempotency, reconciliation, the kill switch and every risk gate are
covered by tests that assert refusals, not just successes.

Earlier adversarial audits found and fixed defects that would have lost money — regime
detection that had no effect on allocation, user risk caps that did not bind, an intraday
book that would never have squared off on a UTC server, long-term decisions driven by an
unseeded RNG, and an original test suite that imported **zero** production modules.

Read **[docs/AUDIT_REPORT.md](docs/AUDIT_REPORT.md)** before interpreting any number this
codebase produces.

---

## What is AlgoDollar?

AlgoDollar is a personal quantitative trading platform for the Indian equity markets (NSE).
It applies evidence-based, systematic investment principles across three complementary
strategy horizons simultaneously:

- **Long-term** (weeks to months): Factor-based investing — momentum, quality, value.
- **Swing** (days to weeks): Technical and fundamental catalyst-driven setups.
- **Intraday** (minutes to hours): Volatility capture with strict risk controls.

Capital is allocated dynamically across these three buckets based on market regime,
recent strategy performance, and risk budget — never as a fixed percentage split.

---

## Architecture

```
Browser  <-->  Next.js 15 (TypeScript + Tailwind)
                      |
               FastAPI (Python 3.11)
              /        |         \
     PostgreSQL     Redis      Celery Workers
          |                         |
      SQLAlchemy              Strategy Runners
          |                    /    |    \
      Alembic           LongTerm  Swing  Intraday
                              \    |    /
                           Risk Engine
                                 |
                        Zerodha Kite Connect
                        (paper mode default)
```

| Layer | Technology |
|---|---|
| Frontend | Next.js 15, TypeScript, Tailwind CSS |
| Backend API | FastAPI 0.115, Python 3.11 |
| Database | PostgreSQL 16 (primary), SQLite (tests) |
| Cache / Broker | Redis 7 |
| Task Queue | Celery + Redis |
| Quant Engine | pandas, numpy, scipy, scikit-learn, LightGBM |
| Broker API | Zerodha Kite Connect v5 |

---

## Key Features

### Dynamic Capital Allocation
Capital allocation is never a fixed 30/30/40 split. The allocator adjusts
bucket weights daily based on:
- **Market regime** (bull, bear, neutral, high-volatility)
- **Recent strategy Sharpe ratios** (poor performance reduces allocation)
- **Risk budget consumption** (approaching max drawdown reduces deployment)
- **Strategy confidence** (model uncertainty gates position sizing)

### Three Independent Strategies
Each strategy operates in its own capital bucket with independent risk limits:

| Strategy | Horizon | Instruments | Typical Positions |
|---|---|---|---|
| Long-term | 1 week – 6 months | NSE-listed equities | 10–25 |
| Swing | 3 days – 4 weeks | NSE equities + F&O | 5–15 |
| Intraday | Same day | NSE equities (MIS) | 1–8 |

### Risk-Adjusted Position Sizing
Position sizes are calculated using a Kelly-fraction approximation with a
conservative multiplier (0.25x full Kelly by default). The `RiskEngine` enforces:
- Maximum 10% of portfolio in any single stock
- Maximum 30% in any single sector
- Portfolio-level variance cap
- Dynamic risk scaling as drawdown grows

### Event-Driven Backtester
- Real Zerodha cost model (brokerage cap, STT, exchange charges, GST, stamp duty)
- Slippage modelling
- Walk-forward validation: training window slides forward, OOS is never contaminated
- Mandatory look-ahead bias checks

### Paper Trading with Live Prices
The `PaperBroker` subscribes to live Zerodha WebSocket ticks and simulates
order fills using bid-ask spread estimates — giving realistic paper P&L without
risking real capital.

### Emergency Kill Switch
A single API call or environment variable flip halts ALL order submission
immediately. The kill switch is checked as the first validation gate.

### 16-Page Professional Dashboard
- Portfolio overview with real-time P&L
- Strategy-level allocation and performance
- Trade journal with execution metrics
- Risk dashboard (drawdown waterfall, VaR, regime indicator)
- Backtest results and walk-forward chart

---

## Philosophy: Capital is Not Risk

A common mistake is treating "how much capital I deploy" and "how much risk I take"
as the same decision. They are separate:

> Deploying ₹1,00,000 into a diversified 10-stock basket of large-caps is lower
> risk than deploying ₹50,000 into a concentrated 2-stock position with high
> leverage.

AlgoDollar separates these decisions:
- **Capital allocation**: how much goes into each strategy bucket
- **Position sizing**: how much of that bucket goes into each trade (risk-adjusted)
- **Leverage**: explicitly zero for most strategies (no margin on long-term)

---

## Cash is a Valid Decision

The capital allocator can and does choose to keep a portion — or all — of new
contributions as cash. This happens when:
- Market regime is BEAR or HIGH_VOL
- All strategies are below their confidence threshold
- Portfolio drawdown is above the warning threshold
- The kill switch has been activated

Cash in the reserve earns Zerodha's liquid fund returns (optional integration).

---

## Live Trading: DISABLED by Default

Live trading requires ALL of the following to be explicitly configured:

1. `TRADING_MODE=live` in `.env` (not `paper`)
2. `ZERODHA_API_KEY` and `ZERODHA_API_SECRET` set to real credentials
3. A valid daily access token (obtained via the `/api/v1/auth/kite/login` flow)
4. `LIVE_TRADING_ENABLED=true` (separate flag — both must be set)
5. Static IP whitelisted in your Kite Connect app

**DO NOT enable live trading without thoroughly understanding the risks.**

---

## Zerodha Setup

1. Create a Kite Connect app at [developers.kite.trade](https://developers.kite.trade)
2. Note your **API key** and **API secret**
3. Set the redirect URL to `http://localhost:8000/api/v1/auth/kite/callback`
4. Add to `.env`:
   ```
   ZERODHA_API_KEY=your_api_key
   ZERODHA_API_SECRET=your_api_secret
   ```
5. Daily token refresh: GET `/api/v1/auth/kite/login` -> follow Kite login -> token stored automatically

See [docs/zerodha_setup.md](docs/zerodha_setup.md) for full instructions.

---

## Running paper trading

Paper is the default and is the only mode that runs without an explicit opt-in.

```bash
# 1. Configuration — no credentials are needed for paper.
cp .env.example .env            # set POSTGRES_PASSWORD and REDIS_PASSWORD
                                # TRADING_MODE defaults to "paper"; leave it.

# 2. Bring up Postgres, Redis, backend and frontend.
docker compose up -d
docker compose ps               # every service should report healthy

# 3. Confirm the system came up able to trade.
curl -s localhost:8000/api/v1/health | jq
```

Startup runs reconciliation **before** trading is permitted. If it does not reach
`RECONCILIATION_OK`, the service still starts and every order is refused with an audited
reason — a stack that exists but declines to trade is far more diagnosable than a missing
one.

```bash
# 4. Run one signal cycle. Dry run first: this produces the target and every
#    reason without submitting anything.
curl -s -X POST localhost:8000/api/v1/allocation/rebalance \
  -H 'Content-Type: application/json' \
  -d '{"total_capital": 1000000, "dry_run": true}' | jq

# 5. Ask why nothing traded. This is the endpoint that matters.
curl -s 'localhost:8000/api/v1/audit?rejected_only=true' | jq '.entries[].headline'
```

Which answers with the specific gate, never with an absence:

```
"RELIANCE BUY x10 rejected — daily risk limit"
"RELIANCE SELL x40 rejected by broker — short selling is not supported"
"RELIANCE BUY x5 blocked — kill switch engaged"
```

The dashboard at `localhost:3000` shows the same feed above the charts, because when
nothing is trading the first question is *why*, and an empty P&L chart cannot answer it.

### Expect NO TRADE

With no validated alpha model the swing sleeve scores on its 12-1 momentum prior, whose
expected 5-day return tops out near 20bps against a 40bps threshold set to clear the ~34bps
round-trip cost. **It correctly proposes nothing.** That is the strategy declining a
negative expected net return, not a malfunction — see
[docs/RESEARCH_VALIDATION.md](docs/RESEARCH_VALIDATION.md).

### Data for research

```bash
cd backend
python scripts/acquire_data.py --symbols 120 --start 2012-01-01   # ~17MB, not committed
python scripts/audit_data.py          # integrity report + machine-readable manifest
python scripts/probe_survivorship.py  # measures the survivorship gap
python scripts/run_research.py        # walk-forward study -> VALIDATED / NOT VALIDATED
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourhandle/algodollar.git
cd algodollar

# 2. Configure environment
cp .env.example .env
# Edit .env — at minimum set SECRET_KEY to a random string

# 3. Start all services (Docker Compose)
make dev-up

# 4. Run database migrations
make migrate

# 5. Access the dashboard
open http://localhost:3000

# 6. Access the API docs
open http://localhost:8000/docs
```

### Running Tests

```bash
cd backend
pip install -e ".[test]"
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | (required) | PostgreSQL connection string |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `SECRET_KEY` | (required) | JWT signing secret — generate with `openssl rand -hex 32` |
| `TRADING_MODE` | `paper` | `paper` or `live` |
| `LIVE_TRADING_ENABLED` | `false` | Secondary live trading gate |
| `ZERODHA_API_KEY` | `` | Kite Connect API key |
| `ZERODHA_API_SECRET` | `` | Kite Connect API secret |
| `ZERODHA_REDIRECT_URL` | `http://localhost:8000/api/v1/auth/kite/callback` | OAuth callback |
| `MAX_PORTFOLIO_LOSS_PCT` | `0.15` | Kill at 15% portfolio drawdown |
| `MAX_SINGLE_POSITION_PCT` | `0.10` | Max 10% in one stock |
| `KILL_SWITCH` | `false` | Master trading halt |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `ENVIRONMENT` | `development` | `development`, `staging`, `production` |

---

## Regulatory Considerations

> **Important**: Algorithmic trading in India is regulated. The following notes
> are informational only and may not reflect the current regulatory state.
> **Always verify current SEBI and NSE requirements before going live.**

- SEBI has issued guidelines on algorithmic trading (circular CIR/MRD/DP/09/2012 and subsequent updates).
- Retail investors using algos via API (e.g., Kite Connect) may be subject to
  broker-specific risk management rules.
- A static IP address is required by Zerodha for production API usage.
- Audit trails and order logs must be maintained.
- Zerodha reserves the right to disable API access for accounts that violate
  their API usage policy.

---

## Limitations and Known Caveats

- **Point-in-time fundamental data**: Requires an external data provider (e.g., Screener.in, Equitymaster, or a paid fundamental data feed). Historical fundamental data is NOT included.
- **Survivorship bias**: The historical stock universe includes only currently listed companies by default. Delisted stocks are excluded, which can overstate historical strategy returns.
- **Backtests may not reflect future performance**: Model parameters are estimated from historical data that may not repeat.
- **Model degradation**: Machine learning models can degrade as market structure changes. Scheduled retraining is built in but not a guarantee of continued performance.
- **Liquidity assumptions**: Backtests assume fills at the simulated price. Real fills in illiquid stocks can be significantly worse.
- **Single broker dependency**: Currently only Zerodha Kite Connect is supported. Broker API outages will halt all automated activity.

---

## Contributing

This is a personal project. Issues and PRs are welcome but may not be reviewed promptly.

---

## License

Proprietary — personal use only. Not licensed for redistribution or commercial use.
