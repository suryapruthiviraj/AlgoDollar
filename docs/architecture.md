# AlgoDollar System Architecture

## Table of Contents
1. [System Components](#1-system-components)
2. [Data Flow](#2-data-flow)
3. [Database Schema Overview](#3-database-schema-overview)
4. [Capital Allocation Algorithm](#4-capital-allocation-algorithm)
5. [Risk Engine](#5-risk-engine)
6. [Strategy Overview](#6-strategy-overview)
7. [Deployment Architecture](#7-deployment-architecture)

---

## 1. System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Browser / Client                           │
│                     Next.js 15 + TypeScript                         │
│              Dashboard / Trade Journal / Settings UI                │
└─────────────────────┬───────────────────────────────────────────────┘
                      │ HTTP / WebSocket
┌─────────────────────▼───────────────────────────────────────────────┐
│                        FastAPI Backend                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  REST API    │  │  WebSocket   │  │  Background Scheduler    │  │
│  │  /api/v1/    │  │  /ws/live    │  │  (Celery Beat)           │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬──────────────┘  │
│         │                 │                       │                 │
│  ┌──────▼─────────────────▼───────────────────────▼──────────────┐ │
│  │                    Core Services Layer                         │ │
│  │  PortfolioManager │ CapitalAllocator │ RiskEngine              │ │
│  │  ExecutionSafety  │ ZerodhaCostModel │ RegimeDetector          │ │
│  └──────────────────────────────┬─────────────────────────────────┘ │
│                                 │                                   │
│  ┌──────────────────────────────▼─────────────────────────────────┐ │
│  │                     Strategy Runners                           │ │
│  │   LongtermStrategy    SwingStrategy    IntradayStrategy        │ │
│  │   (FeatureBuilder)   (SignalEngine)   (ExecutionEngine)        │ │
│  └──────┬────────────────────────────────────────┬───────────────┘ │
│         │                                        │                 │
│  ┌──────▼────────────┐              ┌────────────▼───────────────┐ │
│  │  Data Layer       │              │  Broker Layer              │ │
│  │  - PostgreSQL     │              │  - KiteConnect (paper)     │ │
│  │  - Redis Cache    │              │  - KiteConnect (live)      │ │
│  │  - Tick Buffer    │              │  - PaperBroker (stub)      │ │
│  └───────────────────┘              └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Zerodha Kite Connect  │
                    │   - REST API (orders)   │
                    │   - WebSocket (ticks)   │
                    └─────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|---|---|
| `FastAPI` | REST API, WebSocket broadcaster, authentication, rate limiting |
| `Celery Beat` | Scheduled jobs: data refresh, rebalance triggers, daily PnL |
| `CapitalAllocator` | Computes INR amounts to deploy per strategy bucket |
| `RiskEngine` | Position sizing, variance limits, kill switch, regime detection |
| `ExecutionSafety` | 12-point pre-flight checklist before every order submission |
| `ZerodhaCostModel` | Brokerage, STT, exchange charges, GST, stamp duty calculation |
| `PaperBroker` | Simulates Kite order fills using live tick data |
| `LongtermStrategy` | Factor model (momentum + quality + value) signal generation |
| `SwingStrategy` | Catalyst-driven technical setups with fundamental filters |
| `IntradayStrategy` | Volatility capture with intraday mean-reversion signals |
| `RegimeDetector` | Classifies market as BULL/BEAR/NEUTRAL/HIGH_VOL |

---

## 2. Data Flow

### A. Market Data Ingestion

```
Zerodha WebSocket (live ticks)
         │
         ▼
    TickHandler
         │ filters, validates, enriches
         ▼
    Redis Tick Buffer (ring buffer per symbol, last 1000 ticks)
         │
         ├──► Real-time P&L update (WebSocket broadcast)
         ├──► Intraday signal evaluation (per tick)
         └──► PostgreSQL tick storage (async batch write)
```

### B. Daily Signal Pipeline

```
Celery Beat (06:00 IST daily)
         │
         ▼
  HistoricalDataLoader
         │ loads OHLCV from PostgreSQL
         ▼
  FeatureBuilder
         │ computes: momentum_N, volatility_N, RSI, MACD, ADX,
         │           volume_ratio, close_z_52w, sector_momentum,
         │           quality_scores, valuation_ratios
         ▼
  SignalGenerator (LightGBM / rule-based)
         │ outputs: signal_strength [-1, +1], confidence [0, 1]
         ▼
  RegimeDetector
         │ outputs: MarketRegime enum
         ▼
  CapitalAllocator
         │ inputs: new contribution, regime, strategy performance
         │ outputs: AllocationResult (longterm, swing, intraday, cash amounts)
         ▼
  RiskEngine.validate_allocation()
         │ checks: variance budget, max drawdown, kill switch
         ▼
  PortfolioManager.rebalance()
         │ diffs current vs target positions
         ▼
  ExecutionSafety.validate() for each order
         │ 12-point check
         ▼
  Broker.place_order() (paper or live)
         │
         ▼
  PostgreSQL (orders, trades, positions tables)
```

### C. Intraday Execution Loop

```
Every tick (via WebSocket):
  ┌─────────────────────────────────────────┐
  │  For each monitored symbol:             │
  │    1. Update tick buffer                │
  │    2. Run intraday signal               │
  │    3. If signal triggers:               │
  │       - ExecutionSafety.validate()      │
  │       - RiskEngine.size_position()      │
  │       - Broker.place_order()            │
  │    4. Update open position mark-to-mkt  │
  │    5. Check stop-loss / target levels   │
  └─────────────────────────────────────────┘
```

---

## 3. Database Schema Overview

### Core Tables

```sql
-- Users / authentication
users (id, email, hashed_password, is_active, created_at)

-- Portfolio snapshots
portfolio_snapshots (id, timestamp, total_value, cash, invested_value,
                     unrealised_pnl, realised_pnl, peak_value, drawdown_pct)

-- Open and historical positions
positions (id, symbol, strategy, quantity, average_price, current_price,
           market_value, unrealised_pnl, side, entry_time, last_updated,
           is_open)

-- Order lifecycle
orders (id, symbol, strategy, quantity, price, order_type, product,
        side, status, broker_order_id, placed_at, filled_at,
        fill_price, cost_breakdown_json)

-- Completed trades (aggregated from orders)
trades (id, symbol, strategy, entry_date, exit_date, entry_price,
        exit_price, quantity, gross_pnl, net_pnl, total_cost,
        hold_duration_days)

-- Strategy allocation history
allocations (id, timestamp, strategy, allocated_amount, deployed_amount,
             cash_amount, regime, config_snapshot_json)

-- Market data
ohlcv (id, symbol, date, open, high, low, close, volume, exchange)

-- Feature store
features (id, symbol, date, feature_name, feature_value)

-- Signal log
signals (id, symbol, strategy, date, signal_value, confidence,
         model_version, features_snapshot_json)

-- Risk events
risk_events (id, timestamp, event_type, details_json)
```

### Index Strategy

- `ohlcv`: composite index on `(symbol, date)` — primary query pattern
- `positions`: index on `(symbol, strategy, is_open)` for quick lookups
- `trades`: index on `(symbol, strategy, entry_date)`
- All timestamp columns are indexed for time-series queries

---

## 4. Capital Allocation Algorithm

### Inputs
- `contribution`: new capital to deploy (INR)
- `regime`: current market regime (BULL/BEAR/NEUTRAL/HIGH_VOL)
- `strategy_performance`: rolling Sharpe ratio per strategy (trailing 60 days)
- `portfolio_state`: current drawdown, remaining risk budget

### Algorithm

```
Step 1: Determine base target percentages
  longterm_pct   = config.longterm_target_pct    # e.g. 0.40
  swing_pct      = config.swing_target_pct       # e.g. 0.35
  intraday_pct   = config.intraday_target_pct    # e.g. 0.15
  cash_pct       = 1 - longterm_pct - swing_pct - intraday_pct

Step 2: Apply regime adjustments
  if regime == BEAR:
    intraday_pct *= 0.5       # cut intraday in half
  if regime == HIGH_VOL:
    intraday_pct *= 0.0       # zero out intraday
    swing_pct    *= 0.5

Step 3: Apply performance scaling
  for each strategy s:
    if rolling_sharpe[s] < 0:
      target_pct[s] *= 0.5    # halve allocation for negative Sharpe
    if rolling_sharpe[s] < -0.5:
      target_pct[s] = 0.0     # zero out deeply negative strategies

Step 4: Apply disabled-strategy zeroing
  for each disabled strategy:
    target_pct[strategy] = 0.0

Step 5: Enforce cash floor
  if sum(invested_pcts) > (1 - cash_floor):
    scale each invested_pct proportionally to respect floor

Step 6: Normalise to sum to 1.0

Step 7: Convert to INR amounts
  amounts = contribution * target_pcts

Step 8: Fix floating-point rounding
  assign remainder to cash bucket
```

### Properties
- Total always equals contribution exactly (modulo 1 paisa rounding)
- Cash is always >= cash_floor (default 10%)
- Disabled strategies always receive exactly ₹0
- Monotone with regime: BEAR allocates less intraday than NEUTRAL

---

## 5. Risk Engine

### Position Sizing

```
max_position_value = portfolio_value * max_single_stock_pct
max_shares         = floor(max_position_value / price)
kelly_fraction     = win_rate - (1 - win_rate) / (avg_win / avg_loss)
conservative_kelly = kelly_fraction * 0.25      # use 25% of full Kelly
risk_adjusted_size = min(max_shares, floor(capital * conservative_kelly / price))
```

### Dynamic Risk Scaling

As portfolio drawdown increases toward the maximum allowed drawdown, all
position sizes are scaled linearly:

```
scale = 1 - (current_drawdown / max_drawdown)
scale = max(0.0, min(1.0, scale))
```

At 0% drawdown: scale = 1.0 (full sizing)
At 50% of max drawdown: scale = 0.5 (half sizing)
At max drawdown: scale = 0.0 (no new positions)

### Portfolio Variance Budget

```
portfolio_variance = w.T @ Sigma @ w

where:
  w     = vector of portfolio weights
  Sigma = covariance matrix (estimated from rolling 60-day returns)

Constraint: portfolio_variance <= target_annual_variance
```

Covariance estimation uses a shrinkage estimator (Ledoit-Wolf) to handle
the ill-conditioning of sample covariance matrices with many assets.

### Regime Detection

A dual-SMA crossover on the Nifty 50 (or portfolio universe average):

```
short_sma = mean(prices[-20:])
long_sma  = mean(prices[-60:])
ratio     = (short_sma - long_sma) / long_sma

if ratio > +0.01: BULL
if ratio < -0.01: BEAR
else:             NEUTRAL

VIX level overlay:
if VIX > 25: HIGH_VOL (overrides BULL/BEAR)
```

---

## 6. Strategy Overview

### 6.1 Long-term Strategy

**Objective**: Capture equity risk premium and factor premia over weeks to months.

**Signal Generation**:
- Compute momentum (returns over 1, 3, 6, 12 months, exclude last month)
- Quality score (ROE, debt/equity, earnings stability)
- Valuation (P/E, P/B relative to sector)
- Composite score ranks NSE 500 universe; top quintile is investable

**Position Management**:
- Monthly rebalance
- Trailing stop at 8% below entry (or 20-day SMA breach)
- Maximum hold period: 9 months

**Risk Limits**:
- Max 25 positions
- No more than 30% in any single sector
- Minimum market cap: ₹2,000 crore (large-cap bias)

### 6.2 Swing Strategy

**Objective**: Capture short-to-medium-term price dislocations around catalysts.

**Signal Generation**:
- Earnings surprise + price momentum continuation
- Technical breakout from multi-week consolidation
- Volume confirmation (1.5x+ above 20-day average)
- LightGBM classifier trained on features: RSI, MACD, ADX, volume ratio, sector strength

**Position Management**:
- Entry at breakout close or next-day open
- Fixed risk: 1.5x ATR trailing stop
- Profit target: 2:1 risk-reward minimum

**Risk Limits**:
- Max 15 positions
- No single position > 8% of swing bucket
- Stop-loss mandatory on entry

### 6.3 Intraday Strategy

**Objective**: Capture intraday volatility with mean-reversion and momentum signals.

**Signal Generation**:
- Opening range breakout (first 15-minute candle)
- VWAP deviation reversion
- Relative strength vs Nifty (first hour)

**Position Management**:
- MIS product (auto-squared off by broker at 15:15 IST)
- Max hold: market close
- Hard stop: 0.5% below entry price

**Risk Limits**:
- Max 8 simultaneous positions
- Total intraday exposure: max 2x intraday bucket (no leverage beyond bucket)
- Suspended on HIGH_VOL regime days

---

## 7. Deployment Architecture

### Development (Docker Compose)

```
docker-compose.yml:
  - postgres:16    (port 5432)
  - redis:7        (port 6379)
  - backend        (port 8000, hot-reload)
  - celery-worker  (auto-restart)
  - celery-beat    (scheduler)
  - frontend       (port 3000, hot-reload)
```

### Production Considerations

```
┌──────────────────────────────────────────────────────┐
│                   Load Balancer                      │
│                  (nginx or Caddy)                    │
└───────────────────────┬──────────────────────────────┘
                        │
          ┌─────────────▼──────────────┐
          │      FastAPI (uvicorn)     │
          │    2+ workers recommended  │
          │   STATIC IP REQUIRED for   │
          │   Zerodha production API   │
          └─────────────┬──────────────┘
                        │
        ┌───────────────┼──────────────────┐
        │               │                  │
        ▼               ▼                  ▼
  PostgreSQL 16     Redis 7          Celery Workers
  (RDS or VM)    (ElastiCache        (2–4 workers)
                  or VM)
```

**Critical**: Zerodha Kite Connect requires a **static IP address** for production
API usage. Dynamic IPs will result in rejected requests. Use an Elastic IP (AWS)
or a VM with a static IP.

### Security Requirements

- `SECRET_KEY`: minimum 32 random bytes, never committed to git
- `DATABASE_URL`: must not be publicly accessible; use VPC security groups
- API keys: stored in environment variables or a secrets manager
- Never commit `.env` to version control
- Enable PostgreSQL SSL in production (`sslmode=require`)
- Rate limiting is enabled on all API endpoints via SlowAPI

### Monitoring Hooks

- Structlog for structured JSON logging
- Health check endpoint: `GET /health`
- `/metrics` endpoint compatible with Prometheus (optional)
- Failed order alerts via configured webhook (Slack, email)
- Daily P&L summary via Celery Beat task

### NEVER in CI/CD

- Never auto-enable live trading on deploy
- Never auto-run database migrations with `--run-sync` in production
- Always use `alembic upgrade head` explicitly after review
