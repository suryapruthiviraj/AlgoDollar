"""Reconciliation engine — compare broker state vs local DB on startup and periodically."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..broker.base import BrokerInterface

logger = logging.getLogger(__name__)


class ReconciliationStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Discrepancy:
    kind: str              # missing_local | missing_broker | mismatched_qty | mismatched_price
    symbol: str
    order_id: Optional[str]
    broker_value: Any
    local_value: Any
    details: str = ""


@dataclass
class ReconciliationResult:
    status: ReconciliationStatus
    discrepancies: list[Discrepancy] = field(default_factory=list)
    recommendation: str = ""


class ReconciliationError(RuntimeError):
    """Raised when a critical mismatch is detected."""


class ReconciliationEngine:
    """
    Compares broker-reported positions / orders / trades against the local DB.

    Should run:
    - At application startup before any trading begins.
    - Periodically (e.g. every 15 minutes) during the trading session.
    """

    def __init__(self, kill_switch_store=None) -> None:
        """
        Parameters
        ----------
        kill_switch_store
            Redis client (or any object with .set(key, value)).
            On a critical mismatch the kill switch is activated via
            store.set("kill_switch", "1").
        """
        self._store = kill_switch_store

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    async def reconcile(
        self,
        broker: BrokerInterface,
        db_session=None,
    ) -> ReconciliationResult:
        """
        Full reconciliation pass.

        Steps
        -----
        1. Fetch positions from broker and DB.
        2. Fetch open orders from broker and DB.
        3. Fetch trades from broker and DB.
        4. Identify discrepancies across all three.
        5. Classify severity and decide kill-switch action.
        """
        logger.info("Reconciliation started.")
        discrepancies: list[Discrepancy] = []

        # --- positions -------------------------------------------------
        try:
            broker_positions = await broker.get_positions()
        except Exception as exc:
            logger.error("Cannot fetch broker positions: %s", exc)
            broker_positions = []

        db_positions = await self._fetch_db_positions(db_session)
        discrepancies.extend(
            self._reconcile_positions(broker_positions, db_positions)
        )

        # --- orders ----------------------------------------------------
        try:
            broker_orders = await broker.get_orders()
        except Exception as exc:
            logger.error("Cannot fetch broker orders: %s", exc)
            broker_orders = []

        db_orders = await self._fetch_db_orders(db_session)
        discrepancies.extend(
            self._reconcile_orders(broker_orders, db_orders)
        )

        # --- trades ----------------------------------------------------
        try:
            broker_trades = await broker.get_trades()
        except Exception as exc:
            logger.error("Cannot fetch broker trades: %s", exc)
            broker_trades = []

        db_trades = await self._fetch_db_trades(db_session)
        discrepancies.extend(
            self._reconcile_trades(broker_trades, db_trades)
        )

        # --- classify --------------------------------------------------
        result = self._classify(discrepancies)

        # Log everything
        for d in discrepancies:
            level = logging.WARNING if result.status == ReconciliationStatus.WARNING else logging.ERROR
            logger.log(
                level,
                "Discrepancy [%s] %s oid=%s broker=%s local=%s — %s",
                d.kind, d.symbol, d.order_id, d.broker_value, d.local_value, d.details,
            )

        # Activate kill switch on critical mismatch
        if result.status == ReconciliationStatus.CRITICAL:
            self._activate_kill_switch()
            raise ReconciliationError(
                f"Critical reconciliation mismatch: {len(discrepancies)} discrepancies. "
                "Kill switch activated."
            )

        logger.info(
            "Reconciliation complete: status=%s discrepancies=%d",
            result.status, len(discrepancies),
        )
        return result

    # ------------------------------------------------------------------ #
    #  Position reconciliation                                             #
    # ------------------------------------------------------------------ #

    def _reconcile_positions(
        self,
        broker_positions: list[dict],
        db_positions: list[dict],
    ) -> list[Discrepancy]:
        discrepancies: list[Discrepancy] = []

        broker_map: dict[str, dict] = {
            self._pos_key(p): p for p in broker_positions
        }
        db_map: dict[str, dict] = {
            self._pos_key(p): p for p in db_positions
        }

        # In broker but not in DB
        for key, bp in broker_map.items():
            if bp.get("quantity", 0) == 0:
                continue
            if key not in db_map:
                discrepancies.append(Discrepancy(
                    kind="missing_local",
                    symbol=key,
                    order_id=None,
                    broker_value=bp.get("quantity"),
                    local_value=None,
                    details="Position exists at broker but not in local DB.",
                ))

        # In DB but not in broker
        for key, dp in db_map.items():
            if dp.get("quantity", 0) == 0:
                continue
            if key not in broker_map or broker_map[key].get("quantity", 0) == 0:
                discrepancies.append(Discrepancy(
                    kind="missing_broker",
                    symbol=key,
                    order_id=None,
                    broker_value=None,
                    local_value=dp.get("quantity"),
                    details="Position exists in local DB but not at broker.",
                ))

        # Qty mismatch
        for key in broker_map.keys() & db_map.keys():
            bq = broker_map[key].get("quantity", 0)
            dq = db_map[key].get("quantity", 0)
            if bq != dq:
                discrepancies.append(Discrepancy(
                    kind="mismatched_qty",
                    symbol=key,
                    order_id=None,
                    broker_value=bq,
                    local_value=dq,
                    details=f"Quantity mismatch: broker={bq}, local={dq}.",
                ))

        return discrepancies

    # ------------------------------------------------------------------ #
    #  Order reconciliation                                                #
    # ------------------------------------------------------------------ #

    def _reconcile_orders(
        self,
        broker_orders: list[dict],
        db_orders: list[dict],
    ) -> list[Discrepancy]:
        discrepancies: list[Discrepancy] = []

        open_statuses = {"OPEN", "TRIGGER PENDING", "AMO REQ RECEIVED"}

        broker_open = {
            o["order_id"]: o
            for o in broker_orders
            if o.get("status", "") in open_statuses
        }
        db_open = {
            o["order_id"]: o
            for o in db_orders
            if o.get("status", "") in open_statuses
        }

        for oid, bo in broker_open.items():
            if oid not in db_open:
                discrepancies.append(Discrepancy(
                    kind="missing_local",
                    symbol=bo.get("tradingsymbol", bo.get("symbol", "")),
                    order_id=oid,
                    broker_value=bo.get("status"),
                    local_value=None,
                    details="Open order at broker not found in local DB.",
                ))

        for oid, do in db_open.items():
            if oid not in broker_open:
                discrepancies.append(Discrepancy(
                    kind="missing_broker",
                    symbol=do.get("symbol", ""),
                    order_id=oid,
                    broker_value=None,
                    local_value=do.get("status"),
                    details="Open order in local DB not found at broker.",
                ))

        return discrepancies

    # ------------------------------------------------------------------ #
    #  Trade reconciliation                                                #
    # ------------------------------------------------------------------ #

    def _reconcile_trades(
        self,
        broker_trades: list[dict],
        db_trades: list[dict],
    ) -> list[Discrepancy]:
        discrepancies: list[Discrepancy] = []

        # Use order_id as the trade key (one fill per order in equity)
        broker_trade_ids = {t.get("order_id", t.get("order_id", "")): t for t in broker_trades}
        db_trade_ids = {t.get("order_id", ""): t for t in db_trades}

        for oid, bt in broker_trade_ids.items():
            if oid and oid not in db_trade_ids:
                discrepancies.append(Discrepancy(
                    kind="missing_local",
                    symbol=bt.get("tradingsymbol", bt.get("symbol", "")),
                    order_id=oid,
                    broker_value=bt.get("quantity"),
                    local_value=None,
                    details="Trade recorded by broker not found in local DB.",
                ))

        for oid, dt in db_trade_ids.items():
            if oid and oid not in broker_trade_ids:
                discrepancies.append(Discrepancy(
                    kind="missing_broker",
                    symbol=dt.get("symbol", ""),
                    order_id=oid,
                    broker_value=None,
                    local_value=dt.get("qty"),
                    details="Trade in local DB not found in broker records.",
                ))

        # Price mismatch check
        for oid in broker_trade_ids.keys() & db_trade_ids.keys():
            bp = float(broker_trade_ids[oid].get("average_price", 0) or 0)
            dp = float(db_trade_ids[oid].get("price", 0) or 0)
            if bp > 0 and dp > 0 and abs(bp - dp) / bp > 0.001:  # >0.1% difference
                discrepancies.append(Discrepancy(
                    kind="mismatched_price",
                    symbol=broker_trade_ids[oid].get("tradingsymbol", ""),
                    order_id=oid,
                    broker_value=bp,
                    local_value=dp,
                    details=f"Fill price mismatch: broker={bp:.4f}, local={dp:.4f}.",
                ))

        return discrepancies

    # ------------------------------------------------------------------ #
    #  Classification                                                      #
    # ------------------------------------------------------------------ #

    def _classify(self, discrepancies: list[Discrepancy]) -> ReconciliationResult:
        if not discrepancies:
            return ReconciliationResult(
                status=ReconciliationStatus.OK,
                recommendation="All positions, orders, and trades match.",
            )

        # Critical: position missing on either side, or qty mismatch
        critical_kinds = {"missing_local", "missing_broker", "mismatched_qty"}
        has_critical = any(d.kind in critical_kinds for d in discrepancies)

        if has_critical:
            return ReconciliationResult(
                status=ReconciliationStatus.CRITICAL,
                discrepancies=discrepancies,
                recommendation=(
                    "CRITICAL: Halt trading immediately. "
                    "Manually verify positions and reconcile DB."
                ),
            )

        return ReconciliationResult(
            status=ReconciliationStatus.WARNING,
            discrepancies=discrepancies,
            recommendation=(
                "Minor discrepancies detected (e.g. price differences). "
                "Review and update DB records."
            ),
        )

    # ------------------------------------------------------------------ #
    #  Kill switch                                                         #
    # ------------------------------------------------------------------ #

    def _activate_kill_switch(self) -> None:
        logger.critical("RECONCILIATION: activating kill switch.")
        if self._store is not None:
            try:
                self._store.set("kill_switch", "1")
            except Exception as exc:
                logger.error("Could not set kill switch in store: %s", exc)

    # ------------------------------------------------------------------ #
    #  DB fetch stubs (replace with real ORM queries)                     #
    # ------------------------------------------------------------------ #

    async def _fetch_db_positions(self, db_session) -> list[dict]:
        """Return all non-zero positions from local DB."""
        if db_session is None:
            return []
        try:
            # Example SQLAlchemy:
            # result = await db_session.execute(select(Position).where(Position.quantity != 0))
            # return [r._asdict() for r in result.scalars()]
            return []
        except Exception as exc:
            logger.error("DB fetch positions failed: %s", exc)
            return []

    async def _fetch_db_orders(self, db_session) -> list[dict]:
        if db_session is None:
            return []
        try:
            return []
        except Exception as exc:
            logger.error("DB fetch orders failed: %s", exc)
            return []

    async def _fetch_db_trades(self, db_session) -> list[dict]:
        if db_session is None:
            return []
        try:
            return []
        except Exception as exc:
            logger.error("DB fetch trades failed: %s", exc)
            return []

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pos_key(position: dict) -> str:
        symbol = (
            position.get("tradingsymbol")
            or position.get("symbol", "UNKNOWN")
        )
        exchange = position.get("exchange", "NSE")
        product = position.get("product", "CNC")
        return f"{exchange}:{symbol}:{product}"
