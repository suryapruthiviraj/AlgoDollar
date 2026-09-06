"""
Channel-based in-process pub/sub for real-time updates.

WHY A CHANNEL BUS AT ALL
------------------------
The old ``/ws`` endpoint broadcast every message to every connection. That is
a fan-out the frontend cannot filter — a heartbeat wakes every dashboard
subscriber, and an intraday tick is delivered to a client that only subscribed
for long-term portfolio updates. A channel name per message family ("ticks",
"orders", "portfolio", "risk", "system", ...) lets a client ask for exactly
the streams it renders, and lets a publisher address exactly the audience it
has.

FAIL CLOSED ON UNKNOWN CHANNELS
-------------------------------
``subscribe`` accepts only channels on the registered allowlist; an unknown
name is *refused* rather than silently subscribed. A dashboard that subscribes
to a misspelt "orderz" must learn immediately, not discover the hole by never
receiving anything.

NO OWN CLOCK, NO FAN-OUT RIGHT
------------------------------
The hub only relays what a publisher wrote. Publishing is explicit and driven
by application events (ticks, order outcomes, risk alerts); nothing here
schedules anything. There is no back-pressure or queue — a slow client is
dropped so one dashboard cannot wedge the event loop that feeds everyone.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

#: The authorised message families. Adding a family here is a contract change
#: (frontend/ws types) and must stay reviewed like any schema.
KNOWN_CHANNELS = frozenset(
    {
        "ticks",  # per-tick quote feed (mock paper sessions)
        "bars",  # intraday 1-minute bar closes
        "orders",  # order lifecycle events
        "portfolio",  # portfolio / cash / equity snapshots
        "risk",  # risk events and limit breaches
        "strategy",  # strategy signals / health
        "audit",  # execution audit records
        "system",  # service lifecycle (started, stopping, gate states)
    }
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChannelHub:
    """
    Known-channel registry mapping connected sockets to their subscriptions.

    The socket type is duck-typed: anything with an awaitable ``send_text``
    and identity usable as a dict key will do (FastAPI WebSocket in
    production, stub objects in tests).
    """

    def __init__(self, known_channels: Optional[Iterable[str]] = None) -> None:
        self._known = frozenset(known_channels) if known_channels is not None else KNOWN_CHANNELS
        self._clients: dict[Any, set[str]] = {}

    # ── introspection ───────────────────────────────────────────────────── #

    @property
    def known_channels(self) -> frozenset:
        return self._known

    def known_channel(self, channel: str) -> bool:
        return channel in self._known

    def count(self) -> int:
        return len(self._clients)

    def has_subscribers(self, channel: str) -> bool:
        return channel in self._known and any(channel in subs for subs in self._clients.values())

    def subscribed(self, ws: Any) -> list[str]:
        return sorted(self._clients.get(ws, set()))

    # ── lifecycle ───────────────────────────────────────────────────────── #

    def connect(self, ws: Any) -> int:
        """
        Register a newly-accepted socket with no subscriptions.
        """
        self._clients.setdefault(ws, set())
        return len(self._clients)

    def subscribe(self, ws: Any, channels: Sequence[str]) -> list[str]:
        """
        Add the socket to the given channels, returning what was actually
        granted. Unknown channels are refused, not silently swallowed.
        """
        subs = self._clients.setdefault(ws, set())
        granted: list[str] = []
        for channel in channels:
            if channel in self._known:
                subs.add(channel)
                granted.append(channel)
        return sorted(granted)

    def unsubscribe(self, ws: Any, channels: Optional[Sequence[str]] = None) -> list[str]:
        """
        Remove the socket from the given channels (all if ``None``) and
        return what was removed.
        """
        subs = self._clients.get(ws)
        if subs is None:
            return []
        removed = sorted(subs) if channels is None else [c for c in sorted(subs) if c in channels]
        for channel in removed:
            subs.discard(channel)
        if not subs:
            self._clients.pop(ws, None)
        return removed

    def disconnect(self, ws: Any) -> None:
        self._clients.pop(ws, None)

    # ── publish ─────────────────────────────────────────────────────────── #

    async def publish(self, channel: str, payload: dict[str, Any]) -> int:
        """
        Send ``payload`` to every socket subscribed to ``channel``.

        Unknown channels publish to nobody and report 0. A socket that fails
        to send is dropped, so a dead dashboard cannot wedge the loop or
        starve its neighbours. Returns the number of sockets addressed.
        """
        if channel not in self._known:
            logger.warning("publish to unknown channel %r refused", channel)
            return 0
        message = json.dumps({"type": channel, "timestamp": _utcnow(), **payload})
        dead: list[Any] = []
        count = 0
        for ws, subs in tuple(self._clients.items()):
            if channel not in subs:
                continue
            count += 1
            try:
                await ws.send_text(message)
            except Exception as exc:  # noqa: BLE001
                logger.warning("dropping dead ws client: %s", exc)
                dead.append(ws)
        for ws in dead:
            self._clients.pop(ws, None)
        return count

    async def broadcast(self, payload: dict[str, Any]) -> int:
        """
        Send ``payload`` to every connected socket regardless of channel.
        """
        message = json.dumps({"type": "broadcast", "timestamp": _utcnow(), **payload})
        dead: list[Any] = []
        count = 0
        for ws in tuple(self._clients):
            count += 1
            try:
                await ws.send_text(message)
            except Exception as exc:  # noqa: BLE001
                logger.warning("dropping dead ws client: %s", exc)
                dead.append(ws)
        for ws in dead:
            self._clients.pop(ws, None)
        return count


class TickPublisher:
    """
    Adapts the synchronous ``on_tick(symbol, ts, price, volume)`` callback
    shape of the mock tick feed into an async ``ChannelHub.publish`` fan-out.

    It is a no-op when nobody subscribes to the target channel, so arming it
    next to the aggregator costs nothing until a dashboard actually asks for
    ticks. The publish is scheduled, never awaited, keeping the feed's
    synchronous critical path intact.
    """

    def __init__(self, hub: ChannelHub, channel: str = "ticks") -> None:
        if channel not in (hub.known_channels or KNOWN_CHANNELS):
            raise ValueError(f"TickPublisher target channel {channel!r} is not a known channel")
        self._hub = hub
        self._channel = channel

    def __call__(self, symbol: str, ts: datetime, price: float, volume: float) -> None:
        if not self._hub.has_subscribers(self._channel):
            return
        payload = {
            "symbol": str(symbol),
            "timestamp": ts.isoformat(),
            "last_price": float(price),
            "volume": int(max(0.0, float(volume or 0.0))),
            "source": "mock",
            "synthetic": True,
        }
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._hub.publish(self._channel, payload))
