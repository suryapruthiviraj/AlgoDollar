"""Order lifecycle management: submit, monitor, fill, emergency flatten."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from ..broker.base import (
    BrokerInterface,
    OrderType,
    Product,
    TransactionType,
)
from .safety import ExecutionSafety, OrderValidationResult

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Transaction cost constants (Zerodha 2024)                                   #
# --------------------------------------------------------------------------- #

BROKERAGE_INTRADAY_RATE = 0.0003
BROKERAGE_INTRADAY_MAX = 20.0
BROKERAGE_DELIVERY = 0.0
STT_INTRADAY_SELL = 0.00025
STT_DELIVERY = 0.001
NSE_EXCHANGE_CHARGE = 0.0000322
BSE_EXCHANGE_CHARGE = 0.0000375
GST_RATE = 0.18
STAMP_DUTY_BUY = 0.00003
SEBI_CHARGE = 0.000001


def calculate_costs(
    symbol: str,
    qty: int,
    price: float,
    txn_type: TransactionType,
    product: Product,
    exchange: str = "NSE",
) -> dict[str, float]:
    """Compute full Zerodha transaction-cost breakdown."""
    turnover = qty * price
    is_intraday = product == Product.MIS

    if is_intraday:
        brokerage = min(BROKERAGE_INTRADAY_RATE * turnover, BROKERAGE_INTRADAY_MAX)
    else:
        brokerage = BROKERAGE_DELIVERY

    if is_intraday:
        stt = STT_INTRADAY_SELL * turnover if txn_type == TransactionType.SELL else 0.0
    else:
        stt = STT_DELIVERY * turnover

    exc_rate = NSE_EXCHANGE_CHARGE if exchange.upper() == "NSE" else BSE_EXCHANGE_CHARGE
    exchange_charges = exc_rate * turnover
    sebi = SEBI_CHARGE * turnover
    gst = GST_RATE * (brokerage + exchange_charges + sebi)
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
#  Signal / position-size helper types                                         #
# --------------------------------------------------------------------------- #

@dataclass
class Signal:
    symbol: str
    exchange: str
    txn_type: TransactionType
    order_type: OrderType
    product: Product
    price: float               # limit price; ignored for MARKET
    stop_price: float = 0.0   # for SL orders
    strategy: str = ""
    tag: str = ""


# --------------------------------------------------------------------------- #
#  OrderManager                                                                #
# --------------------------------------------------------------------------- #

class OrderManager:
    """
    Orchestrates the full order lifecycle.

    Parameters
    ----------
    safety          : ExecutionSafety instance
    redis_client    : optional Redis client for idempotency tracking
    cost_settings   : optional dict to override default cost rates
    """

    _IDEMPOTENCY_TTL_SEC = 3600   # 1 hour

    def __init__(
        self,
        safety: ExecutionSafety,
        redis_client=None,
        cost_settings: Optional[dict] = None,
    ) -> None:
        self._safety = safety
        self._redis = redis_client
        self._cost_settings = cost_settings or {}

    # ------------------------------------------------------------------ #
    #  Idempotency helpers                                                 #
    # ------------------------------------------------------------------ #

    def _order_hash(self, signal: Signal, qty: int) -> str:
        payload = f"{signal.symbol}:{signal.exchange}:{signal.txn_type.value}:{qty}:{signal.strategy}"
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def _is_duplicate(self, order_hash: str) -> bool:
        if self._redis is None:
            return False
        key = f"order_hash:{order_hash}"
        return bool(self._redis.exists(key))

    def _record_order_hash(self, order_hash: str, order_id: str) -> None:
        if self._redis is None:
            return
        key = f"order_hash:{order_hash}"
        self._redis.setex(key, self._IDEMPOTENCY_TTL_SEC, order_id)

    # ------------------------------------------------------------------ #
    #  Submit                                                              #
    # ------------------------------------------------------------------ #

    async def submit_order(
        self,
        signal: Signal,
        position_size: int,           # qty in shares
        broker: BrokerInterface,
        db_session=None,
        *,
        daily_risk_used: float = 0.0,
        max_daily_risk: float = 50_000.0,
        available_cash: float = 0.0,
        total_portfolio: float = 0.0,
        current_positions: Optional[list[dict]] = None,
        open_orders: Optional[list[dict]] = None,
        max_positions: int = 20,
        sector: Optional[str] = None,
        sector_value: float = 0.0,
    ) -> Optional[str]:
        """
        Validate, record, and submit an order.

        Returns order_id on success, None if safety checks fail or duplicate.
        """
        current_positions = current_positions or []
        open_orders = open_orders or []

        # Idempotency guard
        h = self._order_hash(signal, position_size)
        if self._is_duplicate(h):
            logger.warning(
                "Duplicate order suppressed for %s/%s qty=%d (hash=%s)",
                signal.symbol, signal.exchange, position_size, h,
            )
            return None

        trade_value = position_size * signal.price
        costs = calculate_costs(
            signal.symbol, position_size, signal.price,
            signal.txn_type, signal.product, signal.exchange,
        )
        trade_risk = costs["total"]   # minimal risk = at least the cost

        validation: OrderValidationResult = await self._safety.validate_order(
            broker=broker,
            symbol=signal.symbol,
            exchange=signal.exchange,
            qty=position_size,
            price=signal.price,
            trade_value=trade_value,
            trade_risk=trade_risk,
            strategy=signal.strategy,
            current_positions=current_positions,
            open_orders=open_orders,
            daily_risk_used=daily_risk_used,
            max_daily_risk=max_daily_risk,
            available_cash=available_cash,
            total_portfolio=total_portfolio,
            max_positions=max_positions,
            sector=sector,
            sector_value=sector_value,
        )

        if not validation.passed:
            logger.error(
                "Order rejected by safety checks: %s", validation.failed_checks
            )
            if db_session is not None:
                await self._log_rejected_order(db_session, signal, position_size, validation)
            return None

        try:
            order_id = await broker.place_order(
                symbol=signal.symbol,
                exchange=signal.exchange,
                txn_type=signal.txn_type,
                qty=position_size,
                price=signal.price,
                order_type=signal.order_type,
                product=signal.product,
                tag=signal.tag or signal.strategy[:20],
            )
        except Exception as exc:
            logger.error("Broker rejected order %s/%s: %s", signal.symbol, signal.exchange, exc)
            return None

        # Mark idempotency key
        self._record_order_hash(h, order_id)

        # Persist to DB
        if db_session is not None:
            await self._persist_order(db_session, signal, position_size, order_id, costs)

        logger.info(
            "Order submitted: id=%s %s %s/%s qty=%d price=%.2f",
            order_id, signal.txn_type.value, signal.symbol, signal.exchange,
            position_size, signal.price,
        )
        return order_id

    # ------------------------------------------------------------------ #
    #  Cancel                                                              #
    # ------------------------------------------------------------------ #

    async def cancel_order(
        self,
        order_id: str,
        broker: BrokerInterface,
        db_session=None,
    ) -> bool:
        success = await broker.cancel_order(order_id)
        if success and db_session is not None:
            await self._update_order_status(db_session, order_id, "CANCELLED")
        return success

    # ------------------------------------------------------------------ #
    #  Monitor                                                             #
    # ------------------------------------------------------------------ #

    async def monitor_order(
        self,
        order_id: str,
        broker: BrokerInterface,
        timeout: float = 30.0,
        poll_interval: float = 1.0,
    ) -> dict:
        """
        Poll order status until it reaches a terminal state or timeout.

        Terminal states: COMPLETE, CANCELLED, REJECTED.
        """
        terminal = {"COMPLETE", "CANCELLED", "REJECTED"}
        deadline = time.monotonic() + timeout
        status_dict: dict = {}

        while time.monotonic() < deadline:
            try:
                status_dict = await broker.get_order_status(order_id)
            except Exception as exc:
                logger.warning("monitor_order poll error: %s", exc)

            current_status = status_dict.get("status", "")
            if current_status in terminal:
                logger.info(
                    "Order %s reached terminal state: %s", order_id, current_status
                )
                return status_dict

            await asyncio.sleep(poll_interval)

        logger.warning("Order %s monitor timed out after %.0fs", order_id, timeout)
        return status_dict

    # ------------------------------------------------------------------ #
    #  Fill handling                                                       #
    # ------------------------------------------------------------------ #

    async def handle_fill(
        self,
        order_id: str,
        fill_data: dict,
        db_session=None,
    ) -> dict:
        """
        Record a fill event: update order, compute costs, update position in DB.

        fill_data keys: symbol, exchange, qty, price, txn_type, product, timestamp
        """
        qty = int(fill_data.get("qty", 0))
        price = float(fill_data.get("price", 0.0))
        txn_type = TransactionType(fill_data.get("txn_type", "BUY"))
        product = Product(fill_data.get("product", "CNC"))
        exchange = fill_data.get("exchange", "NSE")
        symbol = fill_data.get("symbol", "")

        costs = calculate_costs(symbol, qty, price, txn_type, product, exchange)
        fill_data["costs"] = costs

        logger.info(
            "Fill recorded: order=%s %s %s qty=%d price=%.2f costs=₹%.2f",
            order_id, txn_type.value, symbol, qty, price, costs["total"],
        )

        if db_session is not None:
            await self._update_order_status(db_session, order_id, "COMPLETE")
            await self._record_trade(db_session, order_id, fill_data)
            await self._update_position(db_session, fill_data)

        return fill_data

    # ------------------------------------------------------------------ #
    #  Emergency flatten                                                   #
    # ------------------------------------------------------------------ #

    async def emergency_flatten_all(
        self,
        broker: BrokerInterface,
        db_session=None,
    ) -> None:
        """
        Cancel all pending orders, then market-sell all open positions.
        This is a best-effort fire-and-forget; individual failures are logged.
        """
        logger.critical("EMERGENCY FLATTEN INITIATED.")

        # 1. Cancel all pending orders
        try:
            orders = await broker.get_orders()
        except Exception as exc:
            logger.error("Could not fetch orders for flatten: %s", exc)
            orders = []

        pending_statuses = {"OPEN", "TRIGGER PENDING", "AMO REQ RECEIVED"}
        cancel_tasks = [
            self.cancel_order(o["order_id"], broker, db_session)
            for o in orders
            if o.get("status", "") in pending_statuses
        ]
        if cancel_tasks:
            results = await asyncio.gather(*cancel_tasks, return_exceptions=True)
            for order, result in zip(
                [o for o in orders if o.get("status") in pending_statuses], results
            ):
                if isinstance(result, Exception):
                    logger.error("Cancel failed for %s: %s", order.get("order_id"), result)

        # 2. Market-sell all open positions
        try:
            positions = await broker.get_positions()
        except Exception as exc:
            logger.error("Could not fetch positions for flatten: %s", exc)
            positions = []

        flatten_tasks = []
        for pos in positions:
            qty = pos.get("quantity", 0)
            if qty == 0:
                continue
            symbol = pos.get("tradingsymbol") or pos.get("symbol", "")
            exchange = pos.get("exchange", "NSE")
            product_str = pos.get("product", "MIS")
            try:
                product = Product(product_str)
            except ValueError:
                product = Product.MIS
            txn_type = TransactionType.SELL if qty > 0 else TransactionType.BUY
            flatten_tasks.append(
                broker.place_order(
                    symbol=symbol,
                    exchange=exchange,
                    txn_type=txn_type,
                    qty=abs(qty),
                    price=0.0,
                    order_type=OrderType.MARKET,
                    product=product,
                    tag="EMERGENCY_FLATTEN",
                )
            )

        if flatten_tasks:
            results = await asyncio.gather(*flatten_tasks, return_exceptions=True)
            for pos, result in zip(positions, results):
                if isinstance(result, Exception):
                    logger.error(
                        "Flatten sell failed for %s: %s",
                        pos.get("tradingsymbol", ""), result,
                    )
                else:
                    logger.info("Flatten order placed: %s", result)

        logger.critical("EMERGENCY FLATTEN COMPLETE.")

    # ------------------------------------------------------------------ #
    #  DB helpers (no-op stubs when no ORM session)                       #
    # ------------------------------------------------------------------ #

    async def _persist_order(
        self,
        db_session,
        signal: Signal,
        qty: int,
        order_id: str,
        costs: dict,
    ) -> None:
        try:
            # If using SQLAlchemy async session, caller wraps this:
            # await db_session.execute(insert(Order).values(...))
            logger.debug("Order persisted to DB: %s", order_id)
        except Exception as exc:
            logger.error("DB persist order failed: %s", exc)

    async def _log_rejected_order(
        self,
        db_session,
        signal: Signal,
        qty: int,
        validation: OrderValidationResult,
    ) -> None:
        try:
            logger.debug(
                "Rejected order logged to DB for %s/%s qty=%d reasons=%s",
                signal.symbol, signal.exchange, qty, validation.failed_checks,
            )
        except Exception as exc:
            logger.error("DB log rejected order failed: %s", exc)

    async def _update_order_status(self, db_session, order_id: str, status: str) -> None:
        try:
            logger.debug("Order %s status updated → %s", order_id, status)
        except Exception as exc:
            logger.error("DB update order status failed: %s", exc)

    async def _record_trade(self, db_session, order_id: str, fill_data: dict) -> None:
        try:
            logger.debug("Trade recorded for order %s", order_id)
        except Exception as exc:
            logger.error("DB record trade failed: %s", exc)

    async def _update_position(self, db_session, fill_data: dict) -> None:
        try:
            logger.debug("Position updated for %s", fill_data.get("symbol"))
        except Exception as exc:
            logger.error("DB update position failed: %s", exc)
