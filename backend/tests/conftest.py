"""
pytest configuration and shared fixtures for AlgoDollar backend tests.

All database fixtures use SQLite in-memory so tests run without PostgreSQL.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# ---------------------------------------------------------------------------
# Engine / Session
# ---------------------------------------------------------------------------

SQLITE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


@pytest_asyncio.fixture(scope="function")
async def async_session() -> AsyncGenerator[AsyncSession, None]:
    """Async SQLite in-memory session, fresh per test function."""
    engine = create_async_engine(
        SQLITE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Import Base lazily so models can be discovered at fixture time.
    try:
        from app.database.base import Base  # noqa: PLC0415
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except ImportError:
        pass  # Base not yet wired; tests that need tables will import explicitly

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


# ---------------------------------------------------------------------------
# Settings override
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_settings():
    """Return a settings-like namespace with test overrides."""
    settings = MagicMock()
    settings.DATABASE_URL = SQLITE_URL
    settings.TRADING_MODE = "paper"
    settings.REDIS_URL = "redis://localhost:6379/15"
    settings.SECRET_KEY = "test-secret-key-not-for-production"
    settings.ENVIRONMENT = "test"
    settings.LOG_LEVEL = "DEBUG"
    settings.ZERODHA_API_KEY = "test_api_key"
    settings.ZERODHA_API_SECRET = "test_api_secret"
    settings.MAX_PORTFOLIO_LOSS_PCT = 0.15
    settings.MAX_SINGLE_POSITION_PCT = 0.10
    settings.KILL_SWITCH = False
    return settings


# ---------------------------------------------------------------------------
# Paper broker
# ---------------------------------------------------------------------------


@pytest.fixture
def paper_broker():
    """Return a minimal PaperBroker instance suitable for unit tests."""
    try:
        from app.broker.paper import PaperBroker  # noqa: PLC0415
        return PaperBroker(initial_capital=1_000_000.0)
    except ImportError:
        # Inline stub so tests depending only on interface still run
        broker = MagicMock()
        broker.cash = 1_000_000.0
        broker.positions = {}
        broker.orders = []
        broker.place_order = MagicMock(return_value={"order_id": "PAPER-001", "status": "complete"})
        broker.cancel_order = MagicMock(return_value={"order_id": "PAPER-001", "status": "cancelled"})
        return broker


# ---------------------------------------------------------------------------
# Synthetic market data
# ---------------------------------------------------------------------------

SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
_DAYS = 500
_SEED = 42


@pytest.fixture(scope="session")
def sample_prices_df() -> pd.DataFrame:
    """
    500 trading days of synthetic OHLCV data for 5 NSE symbols.

    Returns a DataFrame with a MultiIndex (date, symbol) and columns:
    open, high, low, close, volume.
    """
    rng = np.random.default_rng(_SEED)

    base_date = datetime(2022, 1, 3, tzinfo=timezone.utc)
    dates = pd.bdate_range(base_date, periods=_DAYS)

    records = []
    for sym in SYMBOLS:
        # Each symbol starts at a different seed price
        start_price = rng.uniform(200, 3000)
        log_returns = rng.normal(0.0003, 0.015, size=_DAYS)
        closes = start_price * np.exp(np.cumsum(log_returns))

        for i, date in enumerate(dates):
            c = closes[i]
            spread = rng.uniform(0.001, 0.005)
            o = c * (1 + rng.uniform(-spread, spread))
            h = max(o, c) * (1 + rng.uniform(0, spread))
            lo = min(o, c) * (1 - rng.uniform(0, spread))
            vol = int(rng.integers(50_000, 5_000_000))
            records.append(
                {
                    "date": date,
                    "symbol": sym,
                    "open": round(o, 2),
                    "high": round(h, 2),
                    "low": round(lo, 2),
                    "close": round(c, 2),
                    "volume": vol,
                }
            )

    df = pd.DataFrame(records).set_index(["date", "symbol"]).sort_index()
    return df


@pytest.fixture(scope="session")
def sample_features_df(sample_prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Synthetic feature matrix derived from sample_prices_df.

    Columns: momentum_20, momentum_60, volatility_20, volume_ratio,
             rsi_14, macd_signal, adx_14, close_z_52w.
    """
    rng = np.random.default_rng(_SEED + 1)

    # Pivot closes to wide format for rolling calculations
    closes = sample_prices_df["close"].unstack("symbol")

    rows = []
    for date in closes.index:
        for sym in closes.columns:
            c = closes.loc[date, sym]
            rows.append(
                {
                    "date": date,
                    "symbol": sym,
                    "momentum_20": rng.normal(0, 0.05),
                    "momentum_60": rng.normal(0, 0.08),
                    "volatility_20": abs(rng.normal(0.015, 0.005)),
                    "volume_ratio": abs(rng.lognormal(0, 0.4)),
                    "rsi_14": rng.uniform(20, 80),
                    "macd_signal": rng.normal(0, 2),
                    "adx_14": rng.uniform(10, 50),
                    "close_z_52w": rng.normal(0, 1),
                    "close": c,
                }
            )

    df = pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()
    return df


# ---------------------------------------------------------------------------
# Sample positions
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_positions():
    """Return a list of mock Position-like objects."""
    now = datetime.now(tz=timezone.utc)

    def _pos(symbol, qty, avg_price, strategy, side="long"):
        p = MagicMock()
        p.symbol = symbol
        p.quantity = qty
        p.average_price = avg_price
        p.current_price = avg_price * 1.02  # 2% unrealised gain
        p.market_value = qty * p.current_price
        p.unrealised_pnl = qty * (p.current_price - avg_price)
        p.strategy = strategy
        p.side = side
        p.entry_time = now - timedelta(days=5)
        p.last_updated = now
        return p

    return [
        _pos("RELIANCE", 10, 2450.00, "longterm"),
        _pos("TCS", 5, 3800.00, "swing"),
        _pos("INFY", 20, 1550.00, "swing"),
        _pos("HDFCBANK", 15, 1620.00, "intraday"),
        _pos("ICICIBANK", 25, 920.00, "longterm"),
    ]
