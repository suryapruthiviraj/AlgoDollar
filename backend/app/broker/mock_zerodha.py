"""
Clearly-labelled MOCK of the Zerodha KiteConnect *read* surface.

WHY IT EXISTS
-------------
The health and market-overview routes talk to Zerodha only through two SDK
calls: ``profile()`` and ``quote()``. Without real credentials those routes
report "not_configured" or fall back to hardcoded numbers. ``ZERODHA_MOCK_MODE``
gives a local development build a stable, deterministic stand-in for exactly
those two calls so the rest of the stack can be exercised.

What it is NOT:
  * Not a fake broker. There is no order surface, no positive, no margin.
  * Not a route to live trading. Live still requires `trading_mode=live`,
    explicit authorization, a real live broker, and a passing eligibility gate.
  * Not a substitute for data. The quotes are synthetic and every value is
    tagged ``source: "mock"`` so no layer of the system can mistake them for a
    market price.
"""

from __future__ import annotations

import random
from typing import Any

#: A stable, recognisable profile so logs and health pages are unambiguous.
MOCK_PROFILE: dict[str, Any] = {
    "user_name": "MOCK_USER",
    "user_id": "MOCK0000",
    "display_name": "Mock Mode — no real Zerodha credentials",
    "broker": "Zerodha",
    "exchange": ["NSE", "BSE"],
    "products": ["CNC", "MIS", "NRML"],
    "order_types": ["MARKET", "LIMIT", "SL", "SL-M"],
    "source": "mock",
}

#: Reference levels so index mock quotes are within a plausible NSE band.
_BASE_LEVELS: dict[str, float] = {
    "NSE:NIFTY 50": 24500.0,
    "NSE:NIFTY BANK": 52000.0,
    "NSE:INDIA VIX": 14.5,
}

_MOCK_ORDER_SURFACE = (
    "place_order", "cancel_order", "modify_order",
    "positions", "holdings", "orders", "trades", "margins",
    "instruments", "historical_data",
)


def _mock_quote(symbol: str, seed: int = 7) -> dict[str, Any]:
    """Deterministic quote for one symbol, tagged as mock."""
    rng = random.Random(f"{symbol}:{seed}")
    base = _BASE_LEVELS.get(symbol, 1800.0)
    drift = rng.uniform(-0.015, 0.015)
    last = round(base * (1.0 + drift), 2)
    ohlc_open = round(base * (1.0 + rng.uniform(-0.004, 0.004)), 2)
    ohlc_high = round(max(ohlc_open, last) * (1.0 + rng.uniform(0.0, 0.003)), 2)
    ohlc_low = round(min(ohlc_open, last) * (1.0 - rng.uniform(0.0, 0.003)), 2)
    net_change = round(last - ohlc_open, 2)
    return {
        "last_price": last,
        "net_change": net_change,
        "change": round(net_change / ohlc_open * 100.0 if ohlc_open else 0.0, 2),
        "ohlc": {
            "open": ohlc_open,
            "high": ohlc_high,
            "low": ohlc_low,
            "close": last,
        },
        "depth": {
            "buy": [{"price": round(last * 0.999, 2), "quantity": rng.randint(50, 500)}],
            "sell": [{"price": round(last * 1.001, 2), "quantity": rng.randint(50, 500)}],
        },
        "volume": rng.randint(100_000, 3_000_000),
        "oi": 0,
        "timestamp": None,
        "source": "mock",
    }


class MockKiteClient:
    """
    Read-only stand-in for `kiteconnect.KiteConnect`.

    Exposes only the two calls the application's non-trading routes use:
    ``profile()`` and ``quote()``. Every other SDK method raises, because a
    mock credential must be exposed as one the moment anything tries to trade
    or to read a real account through it.
    """

    def __init__(
        self,
        api_key: str = "MOCK_API_KEY",
        access_token: str = "MOCK_ACCESS_TOKEN",
    ) -> None:
        self._api_key = api_key
        self._access_token = access_token

    @property
    def is_mock(self) -> bool:
        return True

    # -- the only real application uses -------------------------------- #

    def profile(self) -> dict[str, Any]:
        return dict(MOCK_PROFILE)

    def quote(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        return {sym: _mock_quote(sym) for sym in symbols}

    # -- everything else refuses ---------------------------------------- #

    def __getattr__(self, name: str) -> Any:
        if name in _MOCK_ORDER_SURFACE:
            def _refuse(*_args: Any, **_kwargs: Any) -> Any:
                raise NotImplementedError(
                    f"MockKiteClient.{name}() is not implemented. Mock "
                    "credentials have no trading or account surface by design; "
                    "they exist so non-trading routes run without real Zerodha "
                    "credentials."
                )
            return _refuse
        raise AttributeError(
            f"MockKiteClient has no attribute {name!r}. Only profile() and "
            "quote() exist on the mock read surface."
        )
