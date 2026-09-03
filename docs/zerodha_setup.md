# Zerodha Kite Connect Setup Guide

> **Note**: Kite Connect APIs, rate limits, and SEBI/NSE requirements for
> algorithmic trading can change. Always verify the current documentation at
> [kite.trade/docs](https://kite.trade/docs) and check with Zerodha support
> before going live with real capital.

---

## Table of Contents
1. [Create a Kite Connect App](#1-create-a-kite-connect-app)
2. [API Key and Secret Configuration](#2-api-key-and-secret-configuration)
3. [Daily Access Token Flow](#3-daily-access-token-flow)
4. [WebSocket Subscription Setup](#4-websocket-subscription-setup)
5. [Rate Limits](#5-rate-limits)
6. [Static IP Requirements](#6-static-ip-requirements)
7. [Paper Trading Mode](#7-paper-trading-mode)
8. [SEBI and NSE Algo Requirements](#8-sebi-and-nse-algo-requirements)
9. [Common Issues](#9-common-issues)

---

## 1. Create a Kite Connect App

1. Log in to [developers.kite.trade](https://developers.kite.trade) with your Zerodha account.
2. Click **Create new app**.
3. Fill in the details:
   - **App name**: AlgoDollar (or any personal name)
   - **Type**: Personal (for personal use)
   - **Redirect URL**: `http://localhost:8000/api/v1/auth/kite/callback` (for development)
   - **Postback URL**: (optional; leave blank for now)
   - **Description**: Personal algo trading system
4. Note your **API key** and **API secret** — the secret is shown only once on creation.

> **Production Redirect URL**: Change to your production domain, e.g.,
> `https://yourdomain.com/api/v1/auth/kite/callback`.

---

## 2. API Key and Secret Configuration

Add to your `.env` file:

```env
ZERODHA_API_KEY=your_api_key_here
ZERODHA_API_SECRET=your_api_secret_here
ZERODHA_REDIRECT_URL=http://localhost:8000/api/v1/auth/kite/callback
```

**Security rules**:
- Never commit `.env` to version control (it is already in `.gitignore`)
- Never log the API secret
- Rotate the secret immediately if it is ever exposed
- In production, use a secrets manager (AWS Secrets Manager, HashiCorp Vault) rather than `.env`

---

## 3. Daily Access Token Flow

Zerodha Kite Connect access tokens **expire at the end of each trading day**
(approximately 06:00 IST of the next day). A new token must be obtained before
every trading session.

### Manual Flow (Development)

1. **Initiate login**: Open or GET `http://localhost:8000/api/v1/auth/kite/login`.
   The backend returns a Kite login URL.
2. **Browser redirect**: Open the login URL in a browser. Log in with your
   Zerodha credentials and complete 2FA (TOTP or SMS OTP).
3. **Callback**: Zerodha redirects to your redirect URL with a `request_token`
   query parameter. The backend captures this automatically.
4. **Token exchange**: The backend calls `kite.generate_session(request_token, api_secret)`
   to obtain the `access_token`.
5. **Storage**: The access token is stored in Redis with a TTL matching its
   expiry (end of trading day). All subsequent API calls use this token.

### Automated Flow (Production)

Automate the login using your Zerodha TOTP secret:

```python
import pyotp

totp = pyotp.TOTP(your_totp_secret)
otp = totp.now()  # Use this OTP in the browser automation
```

A Celery Beat task can trigger automated login at 08:45 IST daily, giving
15 minutes buffer before market open at 09:00 IST (NSE pre-market).

> **Warning**: Storing your TOTP secret requires careful security. Use
> environment variables or a secrets manager — never hardcode it.

### Token Verification

```bash
# Verify the stored token is valid
GET /api/v1/auth/kite/status

# Response:
{
  "token_valid": true,
  "expires_at": "2024-01-15T06:00:00+05:30",
  "user_id": "YOUR_ZERODHA_CLIENT_ID"
}
```

---

## 4. WebSocket Subscription Setup

Kite Ticker streams real-time market data (ticks) over WebSocket.

### Subscription Modes

| Mode | Data | Use Case |
|---|---|---|
| `LTP` | Last Traded Price only | Monitoring, alerts |
| `QUOTE` | LTP + OHLC + depth (5 levels) | Intraday signals |
| `FULL` | QUOTE + additional OI, market depth | Options, futures |

### Implementation Notes

```python
from kiteconnect import KiteTicker

ticker = KiteTicker(api_key, access_token)

def on_ticks(ws, ticks):
    for tick in ticks:
        # tick contains: instrument_token, last_price, volume, etc.
        process_tick(tick)

def on_connect(ws, response):
    # Subscribe to instruments after connection
    tokens = [instrument_token_1, instrument_token_2, ...]
    ws.subscribe(tokens)
    ws.set_mode(ws.MODE_QUOTE, tokens)

ticker.on_ticks = on_ticks
ticker.on_connect = on_connect
ticker.connect(threaded=True)
```

### Instrument Tokens

- Each NSE symbol has a unique integer `instrument_token`
- Fetch the full instrument dump daily: `kite.instruments("NSE")`
- Cache the token-to-symbol mapping in Redis (refreshed daily after 06:00 IST)
- The instrument list is updated by Zerodha for corporate actions, new listings, etc.

### WebSocket Limitations

- Maximum **3,000 instruments** per WebSocket connection
- One active WebSocket connection per access token
- Reconnection with exponential backoff is recommended (the KiteTicker class handles this)

---

## 5. Rate Limits

> **Important**: Rate limits are set by Zerodha and may change. Always check
> the current limits at [kite.trade/docs](https://kite.trade/docs/connect/v3/).

As of the last update, approximate limits include:

| Endpoint Category | Approximate Limit |
|---|---|
| Order placement / modification | ~10 orders/second |
| Historical data (OHLCV) | ~3 requests/second |
| Instruments dump | Once per day recommended |
| Quote (REST) | ~1 request/second |
| WebSocket ticks | No per-tick limit; 3,000 instruments max |

AlgoDollar implements:
- Redis-backed rate limiter before each Kite REST call
- Celery task queuing to smooth out historical data fetches
- Exponential backoff with jitter on HTTP 429 responses

---

## 6. Static IP Requirements

For production API usage, Zerodha typically requires a **static IP address**
to be whitelisted in your Kite Connect app settings.

### Setting Up Static IP

**AWS (recommended for cloud deployment)**:
```
1. Launch an EC2 instance in your target region
2. Allocate an Elastic IP address (EIP)
3. Associate the EIP with your EC2 instance
4. Add the EIP to your Kite Connect app at developers.kite.trade
5. All API calls to Zerodha must originate from this IP
```

**Alternative: NAT Gateway**:
- Route all outbound traffic through a NAT Gateway with an EIP
- Allows multiple instances behind a single static IP

**Development (dynamic IP)**:
- For local development with a dynamic IP, temporarily whitelist your current IP
- Use `curl ifconfig.me` to check your current public IP
- Dynamic IPs are NOT suitable for unattended production trading

---

## 7. Paper Trading Mode

Paper trading is the default and recommended mode for development and testing.

### How Paper Trading Works

The `PaperBroker` class:
1. Subscribes to live Zerodha WebSocket ticks (requires a valid access token)
2. Simulates order fills using the following rules:
   - **MARKET orders**: Filled at the current LTP + half the bid-ask spread estimate
   - **LIMIT orders**: Filled when LTP crosses the limit price
3. Tracks virtual positions, P&L, and available capital in Redis
4. Exposes the same interface as the live `KiteConnect` order API

### Enabling Paper Mode

```env
TRADING_MODE=paper
# LIVE_TRADING_ENABLED defaults to false — do not change this
```

### Paper vs Live Differences

| Behaviour | Paper | Live |
|---|---|---|
| Order fills | Simulated at LTP | Actual market fill |
| Slippage | Estimated | Real |
| Brokerage | Calculated (not charged) | Actually deducted |
| Position data | Stored in Redis | Stored in Zerodha account |
| Capital | Virtual | Real |
| Taxes | Calculated (not filed) | Actually applicable |

---

## 8. SEBI and NSE Algo Requirements

> **Disclaimer**: This section provides general information only. Regulatory
> requirements change frequently. **Always verify current requirements with
> SEBI, NSE, and Zerodha compliance teams before deploying live trading.**

General considerations:

- **SEBI Circular on Algo Trading** (CIR/MRD/DP/09/2012 and subsequent updates):
  Algorithmic trading in India is regulated. Retail investors using broker APIs
  are typically classified under the broker's algo framework.

- **Broker-Level Controls**: Zerodha applies its own risk management controls
  (such as order throttling and exposure limits) even for API orders. These
  controls may differ from what AlgoDollar's own risk engine computes.

- **Order Types**: Not all order types available in the API are approved for
  all account types. Verify which order types your account is approved for.

- **Audit Trail**: Maintain logs of all orders placed, signals generated, and
  risk decisions made. AlgoDollar logs these to PostgreSQL and structured logs.

- **No Automated Login Without Review**: The daily token refresh is semi-automated.
  The final order of magnitude of orders in a day should be reviewed regularly.

---

## 9. Common Issues

### "Invalid API key or secret"
- Verify the API key and secret in `.env` match exactly what is shown in
  the Kite developer console (no extra spaces or newlines)

### "Invalid access token"
- The token has expired (they expire daily around 06:00 IST)
- Run the login flow again: GET `/api/v1/auth/kite/login`

### "Access token not found" on startup
- The backend starts before a token exists — this is expected on first run
- Paper trading needs a valid token to receive live ticks
- Without a token, the system falls back to simulated ticks (no live data)

### WebSocket disconnects frequently
- Check internet stability
- Kite WebSocket reconnects automatically with the KiteTicker class
- Add `reconnect=True` and `reconnect_max_delay=30` to ticker config

### Historical data returns empty
- NSE instruments are unavailable during certain maintenance windows (06:00–08:00 IST)
- Verify `instrument_token` is correct for the symbol
- Check the date range — weekends and holidays return empty

### Order rejected: "Insufficient funds"
- Paper mode: virtual capital may have been depleted; reset via API
- Live mode: actual account margin/funds insufficient — check Zerodha Console

### Rate limit (HTTP 429)
- AlgoDollar's built-in rate limiter should prevent this
- If it occurs, increase the delay multiplier in `KITE_RATE_LIMIT_DELAY_SEC`
