"""Paper-trading broker that mirrors the BrokerInterface using live Zerodha prices."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from .base import (
    BrokerInterface,
    Exchange,
    OrderType,
    Product,
    TransactionType,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Transaction cost constants (Zerodha 2024 rates)                             #
# --------------------------------------------------------------------------- #

BROKERAGE_INTRADAY_RATE = 0.0003          # 0.03 % per order
BROKERAGE_INTRADAY_MAX = 20.0             # ₹20 per order cap
BROKERAGE_DELIVERY = 0.0                  # ₹0 delivery

STT_INTRADAY_SELL = 0.00025              # 0.025 % sell side
STT_DELIVERY = 0.001                     # 0.1 % both sides

NSE_EXCHANGE_CHARGE = 0.0000322          # 0.00322 %
BSE_EXCHANGE_CHARGE = 0.0000375

GST_RATE = 0.18
STAMP_DUTY_BUY = 0.00003                 # 0.003 % on buy side
SEBI_CHARGE = 0.000001                   # ₹10 per crore = 0.0001% ≈ 1e-6 of turnover


def compute_transaction_costs(
    symbol: str,
    qty: int,
    price: float,
    txn_type: TransactionType,
    product: Product,
    exchange: str = "NSE",
) -> dict[str, float]:
    """Return a breakdown of all transaction costs for an equity order."""
    turnover = qty * price
    is_intraday = product == Product.MIS

    # Brokerage
    if is_intraday:
        brokerage = min(BROKERAGE_INTRADAY_RATE * turnover, BROKERAGE_INTRADAY_MAX)
    else:
        brokerage = BROKERAGE_DELIVERY

    # STT
    if is_intraday:
        stt = STT_INTRADAY_SELL * turnover if txn_type == TransactionType.SELL else 0.0
    else:
        stt = STT_DELIVERY * turnover  # both sides

    # Exchange charges
    exc_rate = NSE_EXCHANGE_CHARGE if exchange.upper() == "NSE" else BSE_EXCHANGE_CHARGE
    exchange_charges = exc_rate * turnover

    # SEBI
    sebi = SEBI_CHARGE * turnover

    # GST on (brokerage + exchange_charges + sebi)
    gst = GST_RATE * (brokerage + exchange_charges + sebi)

    # Stamp duty: only on buy side
    stamp = STAMP_DUTY_BUY * turnover if txn_type == TransactionType.BUY else 0.0

    total = brokerage + stt + exchange_charges + sebi + gst + stamp
    return {
        "brokerage": round(brokerage, 4),
        "stt": round(stt, 4),
        "exchange_charges": round(exchange_charges, 4),
        "sebi": round(sebi, 4),
        "gst": round(gst, 4),
        "stamp_duty": round(stamp, 4),
        "total": round(total, 4),
    }


# --------------------------------------------------------------------------- #
#  Internal data structures                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class PaperOrder:
    order_id: str
    symbol: str
    exchange: str
    txn_type: str
    qty: int
    price: float
    order_type: str
    product: str
    tag: str
    status: str = "OPEN"           # OPEN / COMPLETE / CANCELLED / REJECTED
    filled_qty: int = 0
    average_price: float = 0.0
    placed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    filled_at: Optional[str] = None
    costs: dict = field(default_factory=dict)


@dataclass
class PaperPosition:
    symbol: str
    exchange: str
    product: str
    quantity: int
    average_price: float
    last_price: float = 0.0
    pnl: float = 0.0
    realised: float = 0.0
    unrealised: float = 0.0


# --------------------------------------------------------------------------- #
#  PaperBroker                                                                 #
# --------------------------------------------------------------------------- #

class PaperBroker(BrokerInterface):
    """
    Full paper-trading simulation.

    Requires a *connected* ZerodhaBroker for live price data.
    State is persisted to Redis so it survives process restarts.
    """

    REDIS_KEY_PREFIX = "paper_broker:"

    def __init__(
        self,
        data_broker: BrokerInterface,
        initial_cash: float = 1_000_000.0,
        slippage_pct: float = 0.0005,
        redis_client=None,
        account_id: str = "default",
    ) -> None:
        self._data_broker = data_broker
        self._initial_cash = initial_cash
        self._slippage_pct = slippage_pct
        self._redis = redis_client
        self._account_id = account_id
        self._connected = False

        # In-memory state
        self._cash: float = initial_cash
        self._orders: dict[str, PaperOrder] = {}
        self._positions: dict[str, PaperPosition] = {}  # key: f"{symbol}:{product}"
        self._trades: list[dict] = []

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        """Restore persisted state from Redis if available."""
        await self._load_state()
        self._connected = True
        logger.info(
            "PaperBroker connected | cash=₹%.2f | positions=%d",
            self._cash,
            len(self._positions),
        )

    async def disconnect(self) -> None:
        await self._save_state()
        self._connected = False
        logger.info("PaperBroker disconnected and state saved.")

    # ------------------------------------------------------------------ #
    #  Persistence helpers                                                 #
    # ------------------------------------------------------------------ #

    def _key(self, suffix: str) -> str:
        return f"{self.REDIS_KEY_PREFIX}{self._account_id}:{suffix}"

    async def _save_state(self) -> None:
        if self._redis is None:
            return
        try:
            payload = {
                "cash": self._cash,
                "orders": {k: asdict(v) for k, v in self._orders.items()},
                "positions": {k: asdict(v) for k, v in self._positions.items()},
                "trades": self._trades,
            }
            self._redis.set(self._key("state"), json.dumps(payload))
        except Exception as exc:
            logger.warning("Could not save paper state to Redis: %s", exc)

    async def _load_state(self) -> None:
        if self._redis is None:
            return
        try:
            raw = self._redis.get(self._key("state"))
            if raw is None:
                return
            data = json.loads(raw)
            self._cash = data.get("cash", self._initial_cash)
            self._orders = {
                k: PaperOrder(**v) for k, v in data.get("orders", {}).items()
            }
            self._positions = {
                k: PaperPosition(**v) for k, v in data.get("positions", {}).items()
            }
            self._trades = data.get("trades", [])
            logger.info("Paper state loaded from Redis.")
        except Exception as exc:
            logger.warning("Could not load paper state from Redis: %s", exc)

    # ------------------------------------------------------------------ #
    #  Price helper                                                        #
    # ------------------------------------------------------------------ #

    async def _get_live_price(self, symbol: str, exchange: str) -> float:
        key = f"{exchange}:{symbol}"
        quotes = await self._data_broker.get_quote([key])
        price = quotes.get(key, {}).get("last_price")
        if price is None:
            raise RuntimeError(f"Cannot get live price for {key}")
        return float(price)

    def _apply_slippage(self, price: float, txn_type: TransactionType) -> float:
        slip = self._slippage_pct
        if txn_type == TransactionType.BUY:
            return price * (1 + slip)
        return price * (1 - slip)

    def _simulate_partial_fill(self, qty: int, volume: int) -> int:
        """Probabilistic partial fill based on order vs. available volume."""
        if volume <= 0:
            return qty
        fill_probability = min(1.0, (volume * 0.05) / max(qty, 1))
        if random.random() < fill_probability:
            return qty
        # partial fill between 50%–100% of qty
        return max(1, int(qty * random.uniform(0.5, 1.0)))

    # ------------------------------------------------------------------ #
    #  Account methods                                                     #
    # ------------------------------------------------------------------ #

    async def get_profile(self) -> dict:
        return {
            "user_name": f"PaperTrader_{self._account_id}",
            "user_type": "paper",
            "email": "",
            "broker": "PAPER",
        }

    async def get_holdings(self) -> list[dict]:
        holdings = []
        for pos in self._positions.values():
            if pos.product == Product.CNC.value and pos.quantity > 0:
                holdings.append(asdict(pos))
        return holdings

    async def get_positions(self) -> list[dict]:
        result = []
        for pos in self._positions.values():
            if pos.quantity != 0:
                d = asdict(pos)
                d["pnl"] = pos.pnl
                result.append(d)
        return result

    async def get_orders(self) -> list[dict]:
        return [asdict(o) for o in self._orders.values()]

    async def get_trades(self) -> list[dict]:
        return list(self._trades)

    async def get_funds(self) -> dict:
        return {
            "cash": round(self._cash, 2),
            "margin_available": round(self._cash, 2),
            "margin_used": 0.0,
        }

    # ------------------------------------------------------------------ #
    #  Market data (delegated to live data broker)                         #
    # ------------------------------------------------------------------ #

    async def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        return await self._data_broker.get_quote(symbols)

    async def get_historical_data(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        return await self._data_broker.get_historical_data(
            symbol, exchange, interval, from_date, to_date
        )

    # ------------------------------------------------------------------ #
    #  Order placement / simulation                                        #
    # ------------------------------------------------------------------ #

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
        order_id = str(uuid.uuid4())

        # For MARKET orders, get current live price
        if order_type == OrderType.MARKET:
            market_price = await self._get_live_price(symbol, exchange)
        else:
            market_price = price  # LIMIT: use specified price

        fill_price = self._apply_slippage(market_price, txn_type)

        # Simulate volume-based partial fill
        key = f"{exchange}:{symbol}"
        try:
            quote = await self._data_broker.get_quote([key])
            volume = quote.get(key, {}).get("volume", 0) or 0
        except Exception:
            volume = 0
        filled_qty = self._simulate_partial_fill(qty, int(volume))

        # Transaction costs
        costs = compute_transaction_costs(
            symbol, filled_qty, fill_price, txn_type, product, exchange
        )

        # Capital check (BUY side)
        turnover = filled_qty * fill_price
        if txn_type == TransactionType.BUY:
            required = turnover + costs["total"]
            if required > self._cash:
                order = PaperOrder(
                    order_id=order_id,
                    symbol=symbol,
                    exchange=exchange,
                    txn_type=txn_type.value,
                    qty=qty,
                    price=price,
                    order_type=order_type.value,
                    product=product.value,
                    tag=tag,
                    status="REJECTED",
                )
                self._orders[order_id] = order
                logger.warning(
                    "Paper order REJECTED (insufficient funds): need ₹%.2f, have ₹%.2f",
                    required, self._cash,
                )
                return order_id

        # Build and record order
        filled_at = datetime.now(timezone.utc).isoformat()
        order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            exchange=exchange,
            txn_type=txn_type.value,
            qty=qty,
            price=price,
            order_type=order_type.value,
            product=product.value,
            tag=tag,
            status="COMPLETE",
            filled_qty=filled_qty,
            average_price=round(fill_price, 4),
            filled_at=filled_at,
            costs=costs,
        )
        self._orders[order_id] = order

        # Update cash
        if txn_type == TransactionType.BUY:
            self._cash -= (filled_qty * fill_price + costs["total"])
        else:
            self._cash += (filled_qty * fill_price - costs["total"])

        # Update positions
        pos_key = f"{symbol}:{product.value}"
        pos = self._positions.get(pos_key)
        if txn_type == TransactionType.BUY:
            if pos is None:
                self._positions[pos_key] = PaperPosition(
                    symbol=symbol,
                    exchange=exchange,
                    product=product.value,
                    quantity=filled_qty,
                    average_price=fill_price,
                    last_price=fill_price,
                )
            else:
                total_qty = pos.quantity + filled_qty
                pos.average_price = (
                    (pos.average_price * pos.quantity + fill_price * filled_qty) / total_qty
                )
                pos.quantity = total_qty
                pos.last_price = fill_price
        else:  # SELL
            if pos is not None:
                realised = (fill_price - pos.average_price) * filled_qty - costs["total"]
                pos.realised += realised
                pos.quantity -= filled_qty
                pos.last_price = fill_price
                if pos.quantity == 0:
                    del self._positions[pos_key]

        # Record trade
        self._trades.append({
            "order_id": order_id,
            "symbol": symbol,
            "exchange": exchange,
            "txn_type": txn_type.value,
            "qty": filled_qty,
            "price": fill_price,
            "product": product.value,
            "costs": costs,
            "timestamp": filled_at,
            "tag": tag,
        })

        await self._save_state()
        logger.info(
            "Paper order COMPLETE: %s %s %s x%d @ ₹%.2f (cost ₹%.2f) id=%s",
            txn_type.value, symbol, exchange, filled_qty, fill_price,
            costs["total"], order_id,
        )
        return order_id

    async def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None:
            logger.warning("Cancel: order %s not found.", order_id)
            return False
        if order.status != "OPEN":
            logger.warning("Cancel: order %s is already %s.", order_id, order.status)
            return False
        order.status = "CANCELLED"
        await self._save_state()
        return True

    async def modify_order(
        self,
        order_id: str,
        qty: Optional[int] = None,
        price: Optional[float] = None,
    ) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status != "OPEN":
            return False
        if qty is not None:
            order.qty = qty
        if price is not None:
            order.price = price
        await self._save_state()
        return True

    async def get_order_status(self, order_id: str) -> dict:
        order = self._orders.get(order_id)
        if order is None:
            return {}
        return asdict(order)

    # ------------------------------------------------------------------ #
    #  Paper-specific analytics                                            #
    # ------------------------------------------------------------------ #

    async def get_paper_performance(self) -> dict:
        """Return P&L metrics for the paper account."""
        total_realised = sum(
            t["qty"] * t["price"] * (1 if t["txn_type"] == "SELL" else -1)
            - t["costs"]["total"]
            for t in self._trades
        )
        unrealised = 0.0
        for pos in self._positions.values():
            if pos.quantity > 0:
                try:
                    lp = await self._get_live_price(pos.symbol, pos.exchange)
                except Exception:
                    lp = pos.last_price
                unrealised += (lp - pos.average_price) * pos.quantity

        total_costs = sum(t["costs"]["total"] for t in self._trades)
        portfolio_value = self._cash + sum(
            pos.average_price * pos.quantity for pos in self._positions.values()
        )
        return {
            "initial_cash": self._initial_cash,
            "current_cash": round(self._cash, 2),
            "portfolio_value": round(portfolio_value, 2),
            "total_trades": len(self._trades),
            "total_transaction_costs": round(total_costs, 2),
            "unrealised_pnl": round(unrealised, 2),
            "return_pct": round(
                (portfolio_value - self._initial_cash) / self._initial_cash * 100, 4
            ),
        }

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def trading_mode(self) -> str:
        return "paper"

    def instrument_token(self, symbol: str, exchange: str) -> int:
        return self._data_broker.instrument_token(symbol, exchange)
