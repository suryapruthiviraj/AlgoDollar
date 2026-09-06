"""
Tests for the ZERODHA_MOCK_MODE credential stand-in.

The rule this file enforces: mock credentials are a *labelled, opt-in*
development convenience. They must (a) never report the broker as "connected",
(b) never claim market hooks are real, and (c) never enable live trading — the
mock client has no order surface and live still requires trading_mode=live plus
explicit authorization plus a passing eligibility gate.
"""

from __future__ import annotations

import pytest

from app.api.routes.health import _check_broker, health_detailed
from app.api.routes.markets import _fetch_kite_quote, market_overview
from app.broker.mock_zerodha import MOCK_PROFILE, MockKiteClient
from app.core.config import settings

_MOCK_INDEX = {"NSE:NIFTY 50", "NSE:NIFTY BANK", "NSE:INDIA VIX"}


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from the compiled-in defaults."""
    monkeypatch.setattr(settings, "zerodha_mock_mode", False)


# --------------------------------------------------------------------------- #
#  health                                                                      #
# --------------------------------------------------------------------------- #

async def test_health_reports_mock_not_connected() -> None:
    settings.zerodha_mock_mode = True
    status = await _check_broker()
    assert status == "mock"
    assert status != "connected"  # the string must stay distinct


async def test_health_without_mock_and_without_creds_is_not_configured() -> None:
    status = await _check_broker()
    assert status == "not_configured"


async def test_health_detailed_exposes_mock_mode() -> None:
    settings.zerodha_mock_mode = True
    resp = await health_detailed(session=None)  # type: ignore[arg-type]
    assert resp.components["broker"]["status"] == "mock"
    assert resp.components["broker"]["zerodha_mock_mode"] is True
    assert resp.is_live_trading_enabled is False
    assert resp.trading_mode == "paper"


# --------------------------------------------------------------------------- #
#  market overview                                                             #
# --------------------------------------------------------------------------- #

async def test_market_overview_uses_mock_quotes_and_labels_them() -> None:
    settings.zerodha_mock_mode = True
    quotes = MockKiteClient().quote(sorted(_MOCK_INDEX))
    overview = await market_overview(current_user=object())  # type: ignore[arg-type]

    assert overview.data_source == "mock"
    nq = quotes["NSE:NIFTY 50"]
    assert overview.nifty.last_price == nq["last_price"]
    assert overview.nifty.high == nq["ohlc"]["high"]
    assert overview.banknifty.last_price == quotes["NSE:NIFTY BANK"]["last_price"]
    assert overview.vix == quotes["NSE:INDIA VIX"]["last_price"]


async def test_market_overview_without_feed_is_labelled_fallback() -> None:
    # No mock mode, no real credentials -> hardcoded placeholder, explicitly
    # labelled rather than pretending to be the market.
    overview = await market_overview(current_user=object())  # type: ignore[arg-type]
    assert overview.data_source == "fallback_demo"
    assert overview.nifty.last_price == 24500.0


async def test_fetch_kite_quote_mock_is_deterministic_and_tagged() -> None:
    settings.zerodha_mock_mode = True
    first = _fetch_kite_quote(["NSE:NIFTY 50"])
    second = _fetch_kite_quote(["NSE:NIFTY 50"])
    assert first == second
    assert first["NSE:NIFTY 50"]["source"] == "mock"
    assert "last_price" in first["NSE:NIFTY 50"]


async def test_fetch_kite_quote_not_configured_is_empty() -> None:
    assert _fetch_kite_quote(["NSE:NIFTY 50"]) == {}


# --------------------------------------------------------------------------- #
#  the mock client is read-only                                                #
# --------------------------------------------------------------------------- #

async def test_mock_client_profile_and_quote_only() -> None:
    client = MockKiteClient()
    assert client.profile() == MOCK_PROFILE
    assert client.profile()["source"] == "mock"
    assert client.is_mock is True
    assert isinstance(client.quote(["RELIANCE"]), dict)


async def test_mock_client_has_no_order_or_account_surface() -> None:
    client = MockKiteClient()
    for method in (
        "place_order", "cancel_order", "modify_order",
        "positions", "holdings", "orders", "trades", "margins",
    ):
        with pytest.raises(NotImplementedError):
            getattr(client, method)()
    with pytest.raises(AttributeError):
        client.get_quote(["NSE:RELIANCE"])  # not part of the mock surface


# --------------------------------------------------------------------------- #
#  mock mode never enables live                                                #
# --------------------------------------------------------------------------- #

async def test_mock_mode_cannot_enable_live_trading() -> None:
    settings.zerodha_mock_mode = True
    assert settings.is_live_trading_enabled is False
    assert settings.trading_mode == "paper"
    assert settings.kite_credentials_configured is False  # mock != configured
