"""Abstract base class and enums for broker interface."""

from __future__ import annotations

import abc
from enum import Enum
from typing import Optional

import pandas as pd


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"          # stop-loss with limit
    SL_M = "SL-M"      # stop-loss market


class TransactionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Product(str, Enum):
    CNC = "CNC"         # delivery / cash-and-carry
    MIS = "MIS"         # intraday margin
    NRML = "NRML"       # normal (F&O overnight)
    CO = "CO"           # cover order
    BO = "BO"           # bracket order


class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"         # NSE F&O
    BFO = "BFO"         # BSE F&O
    MCX = "MCX"
    CDS = "CDS"


class BrokerInterface(abc.ABC):
    """Abstract base class that every broker adapter must implement."""

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish connection / authenticate with the broker."""

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Gracefully disconnect from the broker."""

    # ------------------------------------------------------------------ #
    #  Account                                                             #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    async def get_profile(self) -> dict:
        """Return account profile information."""

    @abc.abstractmethod
    async def get_holdings(self) -> list[dict]:
        """Return all delivery holdings."""

    @abc.abstractmethod
    async def get_positions(self) -> list[dict]:
        """Return open intraday / overnight positions."""

    @abc.abstractmethod
    async def get_orders(self) -> list[dict]:
        """Return all orders for the current session."""

    @abc.abstractmethod
    async def get_trades(self) -> list[dict]:
        """Return executed trades for the current session."""

    @abc.abstractmethod
    async def get_funds(self) -> dict:
        """Return fund summary: cash, margin_available, margin_used."""

    # ------------------------------------------------------------------ #
    #  Market data                                                         #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    async def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        """
        Return full quote for each symbol.

        Parameters
        ----------
        symbols : list[str]
            Strings of the form "EXCHANGE:TRADINGSYMBOL", e.g. "NSE:RELIANCE".

        Returns
        -------
        dict mapping the same key to a quote dict with keys:
        last_price, bid, ask, ohlc, volume, oi, timestamp, ...
        """

    @abc.abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        """
        Return OHLCV candles as a DataFrame.

        Parameters
        ----------
        symbol    : trading symbol e.g. "RELIANCE"
        exchange  : "NSE" / "BSE" / …
        interval  : "minute", "3minute", "5minute", "15minute",
                    "30minute", "60minute", "day", "week", "month"
        from_date : "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
        to_date   : same format

        Returns DataFrame with columns: date, open, high, low, close, volume
        """

    # ------------------------------------------------------------------ #
    #  Order management                                                    #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    async def place_order(
        self,
        symbol: str,
        exchange: str,
        txn_type: TransactionType,
        qty: int,
        price: float,
        order_type: OrderType,
        product: Product,
        tag: str = "",
    ) -> str:
        """
        Place an order.

        Returns
        -------
        str
            broker-assigned order_id
        """

    @abc.abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True on success."""

    @abc.abstractmethod
    async def modify_order(
        self,
        order_id: str,
        qty: Optional[int] = None,
        price: Optional[float] = None,
    ) -> bool:
        """Modify qty/price of a pending order. Returns True on success."""

    @abc.abstractmethod
    async def get_order_status(self, order_id: str) -> dict:
        """Return current status dict for a given order_id."""

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """True if broker session is live."""

    @property
    @abc.abstractmethod
    def trading_mode(self) -> str:
        """'paper' or 'live'."""

    @abc.abstractmethod
    def instrument_token(self, symbol: str, exchange: str) -> int:
        """Return the numeric instrument token for a symbol/exchange pair."""
