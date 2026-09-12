"""
File-backed durable stores — the O2 "survive restart" slice.

Two stores must survive a process restart for a running paper campaign to have
a *persistent* risk state:

* :class:`FileKillSwitchStore` — an engaged halt (operator, reconciliation, or
  the campaign's own risk enforcement) survives the process, and survives the
  per-day stack rebuilds the campaign performs.  An in-memory switch was set,
  forgotten, and the next day kept trading.
* :class:`FileOrderStore` — the order lifecycle (especially the idempotency
  claim in ``reserve``) survives, so a resumed session watching the same bars
  can never double-submit an intent whose client order id was already claimed.

CONVENTIONS
-----------
* **Atomic writes.**  Every mutation is written to a temp file and
  ``os.replace``d into place (the same pattern ``PaperBroker`` uses for its
  book), so a crash mid-write can never leave a half-written store.
* **Fail closed.**  A store that cannot be read at construction raises.  For
  the kill switch that is the safe direction anyway (``KillSwitch`` treats an
  unreadable source as ACTIVE); for the order store it means a restart cannot
  silently forget an in-flight order.
* **Single writer.**  These stores assume one process at a time (the campaign
  and the API service already do).  Cross-process writes to the same file are
  not serialised — that is what the Redis-backed stores are for.

These are the durable *default*p for the paper campaign; ``RedisOrderStore`` /
``RedisKillSwitchStore`` remain the multi-writer live-account answer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .lifecycle import OrderRecord, PersistenceError

logger = logging.getLogger(__name__)


def atomic_write(path: Path, blob: str) -> None:
    """Write ``blob`` to ``path`` atomically (temp file + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(blob)
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    if not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text())
    except Exception as exc:  # noqa: BLE001 -- re-raised as a store error
        raise PersistenceError(
            f"{Path(path).name} store is corrupt or unreadable ({exc!r}); "
            "refusing to fall back to empty state. A safety store that cannot "
            "be read must never be treated as blank."
        ) from exc


# --------------------------------------------------------------------------- #
#  Kill switch                                                                #
# --------------------------------------------------------------------------- #


class FileKillSwitchStore:
    """
    Kill-switch store persisted to a JSON file.

    Implements the ``get``/``set``/``delete`` surface ``ExecutionSafety``,
    ``ReconciliationEngine`` and the API kill-switch route expect, plus the
    ``engage``/``release`` convenience the operator path uses.  Survives
    process restarts and the campaign's per-day stack rebuilds.
    """

    durable = True

    def __init__(self, path: Any) -> None:
        self._path = Path(path)
        raw = _read_json(self._path)
        self._data: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}

    # -- kv surface ----------------------------------------------------- #

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[str(key)] = value
        self._flush()

    def delete(self, key: str) -> None:
        self._data.pop(str(key), None)
        self._flush()

    # -- operator convenience ------------------------------------------- #

    def engage(self, reason: str = "manual") -> None:
        self._data["kill_switch"] = "1"
        self._data["kill_switch_reason"] = reason
        self._flush()
        logger.critical("KILL SWITCH ENGAGED (durable store %s): %s", self._path, reason)

    def release(self) -> None:
        self._data.pop("kill_switch", None)
        self._flush()
        logger.warning("KILL SWITCH RELEASED (durable store %s)", self._path)

    @property
    def active(self) -> bool:
        return bool(self._data.get("kill_switch"))

    def reason(self) -> Optional[str]:
        return self._data.get("kill_switch_reason")

    # -- persistence ---------------------------------------------------- #

    def _flush(self) -> None:
        atomic_write(self._path, json.dumps(self._data, separators=(",", ":"), default=str))


# --------------------------------------------------------------------------- #
#  Order store                                                                #
# --------------------------------------------------------------------------- #


class FileOrderStore:
    """
    :class:`OrderStore` implementation persisted to one JSON file.

    Semantics are identical to :class:`InMemoryOrderStore` (which is the
    reference implementation this mirrors) — the only difference is that every
    mutation is durably flushed, and a fresh instance on the same path
    reconstructs the previous state.  ``reserve`` therefore stays
    set-if-not-exists across a restart: a resumed campaign cannot claim an
    intent that a crashed run already reserved.
    """

    durable = True

    def __init__(self, path: Any) -> None:
        self._path = Path(path)
        loaded = _read_json(self._path) or {}
        self._records: dict[str, str] = dict(loaded.get("records", {}))
        self._trades: dict[str, list[dict]] = {
            str(k): list(v) for k, v in dict(loaded.get("trades", {})).items()
        }
        self._trade_ids: set[str] = set(loaded.get("trade_ids", []))
        self._positions: dict[str, dict] = dict(loaded.get("positions", {}))
        logger.info(
            "file order store restored (%s): records=%d trades=%d positions=%d",
            self._path,
            len(self._records),
            len(self._trade_ids),
            len(self._positions),
        )

    # -- protocol ------------------------------------------------------- #

    async def reserve(self, record: OrderRecord) -> bool:
        if record.client_order_id in self._records:
            return False
        self._records[record.client_order_id] = record.to_json()
        self._flush()
        return True

    async def get(self, client_order_id: str) -> Optional[OrderRecord]:
        raw = self._records.get(client_order_id)
        return OrderRecord.from_json(raw) if raw is not None else None

    async def save(self, record: OrderRecord) -> None:
        if record.client_order_id not in self._records:
            raise PersistenceError(f"save() before reserve() for {record.client_order_id}")
        self._records[record.client_order_id] = record.to_json()
        self._flush()

    async def list_open(self) -> list[OrderRecord]:
        return [r for r in await self.list_all() if not r.is_terminal]

    async def list_all(self) -> list[OrderRecord]:
        return [OrderRecord.from_json(r) for r in list(self._records.values())]

    async def record_trade(self, client_order_id: str, fill: dict) -> bool:
        fill_id = str(fill.get("fill_id") or "")
        if not fill_id:
            raise ValueError("fill must carry a fill_id")
        if fill_id in self._trade_ids:
            return False
        self._trade_ids.add(fill_id)
        self._trades.setdefault(client_order_id, []).append(dict(fill))
        self._flush()
        return True

    async def list_trades(self, client_order_id: str) -> list[dict]:
        return list(self._trades.get(client_order_id, []))

    async def apply_position_delta(
        self,
        symbol: str,
        exchange: str,
        product: str,
        qty_delta: int,
        price: float,
    ) -> dict:
        key = f"{exchange}:{symbol}:{product}"
        pos = self._positions.setdefault(
            key,
            {
                "symbol": symbol,
                "exchange": exchange,
                "product": product,
                "quantity": 0,
                "average_price": 0.0,
            },
        )
        old_qty = pos["quantity"]
        new_qty = old_qty + qty_delta
        if old_qty >= 0 and qty_delta > 0:  # adding to a long
            notional = pos["average_price"] * old_qty + price * qty_delta
            pos["average_price"] = notional / new_qty if new_qty else 0.0
        elif new_qty == 0:
            pos["average_price"] = 0.0
        pos["quantity"] = new_qty
        self._flush()
        return dict(pos)

    async def get_position(self, symbol: str, exchange: str, product: str) -> dict:
        key = f"{exchange}:{symbol}:{product}"
        return dict(
            self._positions.get(
                key,
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "product": product,
                    "quantity": 0,
                    "average_price": 0.0,
                },
            )
        )

    # -- persistence ---------------------------------------------------- #

    def _flush(self) -> None:
        atomic_write(
            self._path,
            json.dumps(
                {
                    "version": 1,
                    "records": self._records,
                    "trades": self._trades,
                    "trade_ids": sorted(self._trade_ids),
                    "positions": self._positions,
                },
                separators=(",", ":"),
            ),
        )


__all__ = [
    "FileKillSwitchStore",
    "FileOrderStore",
    "atomic_write",
]
