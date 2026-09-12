"""
Channel-based WebSocket pub/sub (O2).

The old `/ws` endpoint broadcast every heartbeat to every connection. This
suite pins the replacement: a known-channel hub the frontend subscribes to
explicitly, publishers that fan out only to their audience, and an endpoint
whose message contract survives without httpx/TestClient (which the CI venv
deliberately lacks).

Guarantees exercised:

1. Subscribe grants known channels and REFUSES unknown ones (fail closed —
   a misspelt channel must be seen, not silently subscribed to nothing).
2. Publishing fans out to exactly the subscribers of that channel; the
   payload arrives as {"type": <channel>, "timestamp", ...}.
3. A dead socket is dropped without blocking or poisoning its neighbours.
4. Disconnect/unsubscribe removes a socket from every channel.
5. The /ws endpoint (driven with a stub socket, no server) answers
   ping/pong, subscribe/subscribed, list/subscriptions and unsubscribe, and
   self-cleans on disconnect.
6. TickPublisher bridges the sync tick-feed callback into the async hub
   without touching the aggregator's critical path, and is a no-op while
   nobody watches.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.realtime import ChannelHub, TickPublisher, websocket_endpoint


class FakeSocket:
    """A stand-in for FastAPI's WebSocket core: accept, send, receive."""

    def __init__(self, incoming: list[str] | None = None) -> None:
        self._incoming = list(incoming or [])
        self.sent: list[str] = []
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, data: str) -> None:
        self.sent.append(data)

    async def receive_text(self) -> str:
        if not self._incoming:
            from fastapi import WebSocketDisconnect

            raise WebSocketDisconnect
        return self._incoming.pop(0)

    def send_payloads(self) -> list[dict]:
        return [json.loads(x) for x in self.sent]


class EndlessSocket(FakeSocket):
    """Like FakeSocket, but blocks (instead of disconnecting) once drained,
    so a test can publish into a still-subscribed connection."""

    async def receive_text(self) -> str:
        while not self._incoming:
            await asyncio.sleep(0.01)
        return self._incoming.pop(0)


def _subscribe_ws(hub: ChannelHub) -> dict:
    return {"type": "subscribe", "channels": ["ticks", "orders", "nonsense-channel"]}


# ---------------------------------------------------------------------------
# ChannelHub units
# ---------------------------------------------------------------------------


def test_known_channels_are_allowlisted():
    hub = ChannelHub()
    assert "ticks" in hub.known_channels
    assert "orders" in hub.known_channels
    assert "risk" in hub.known_channels
    assert hub.known_channel("ticks")
    assert not hub.known_channel("nope")


def test_subscribe_returns_known_channels_and_refuses_unknown():
    hub = ChannelHub()
    ws = FakeSocket()
    hub.connect(ws)
    granted = hub.subscribe(ws, ["ticks", "orders", "nonsense"])
    assert granted == ["orders", "ticks"]
    assert hub.subscribed(ws) == ["orders", "ticks"]
    assert "nonsense" not in hub.known_channels
    assert "nonsense" not in hub.subscribed(ws)


def test_publish_fans_out_by_channel_only():
    hub = ChannelHub()
    ticks = FakeSocket()
    orders = FakeSocket()
    idle = FakeSocket()
    hub.connect(ticks)
    hub.connect(orders)
    hub.connect(idle)
    hub.subscribe(ticks, ["ticks"])
    hub.subscribe(orders, ["orders"])

    asyncio.run(hub.publish("ticks", {"symbol": "RELIANCE", "last_price": 2400.0}))

    tick_msgs = ticks.send_payloads()
    assert len(tick_msgs) == 1
    assert tick_msgs[0]["type"] == "ticks"
    assert tick_msgs[0]["symbol"] == "RELIANCE"
    assert tick_msgs[0]["timestamp"]
    assert orders.send_payloads() == []
    assert idle.send_payloads() == []


def test_publish_to_unknown_channel_is_a_noop():
    hub = ChannelHub()
    ws = FakeSocket()
    hub.connect(ws)
    hub.subscribe(ws, ["system"])
    assert asyncio.run(hub.publish("does-not-exist", {"x": 1})) == 0
    assert ws.send_payloads() == []


def test_dead_client_is_dropped_without_upsetting_neighbours():
    hub = ChannelHub()

    async def scenario() -> None:
        dead = FakeSocket()

        async def failing_send(data: str) -> None:
            raise RuntimeError("socket closed")

        dead.send_text = failing_send  # type: ignore[method-assign]
        alive = FakeSocket()
        hub.connect(dead)
        hub.connect(alive)
        hub.subscribe(dead, ["system"])
        hub.subscribe(alive, ["system"])

        count = await hub.publish("system", {"event": "x"})
        assert count == 2  # both were addressed
        assert hub.count() == 1  # dead one removed
        assert alive.send_payloads()[0]["event"] == "x"

    asyncio.run(scenario())


def test_unsubscribe_and_disconnect_cleanup():
    hub = ChannelHub()
    ws = FakeSocket()
    hub.connect(ws)
    hub.subscribe(ws, ["ticks", "orders"])

    assert hub.unsubscribe(ws, ["ticks"]) == ["ticks"]
    assert hub.subscribed(ws) == ["orders"]

    hub.disconnect(ws)
    assert hub.count() == 0
    assert ws not in hub._clients


def test_has_subscribers_tracks_channel_occupancy():
    hub = ChannelHub()
    ws = FakeSocket()
    hub.connect(ws)
    assert not hub.has_subscribers("ticks")
    hub.subscribe(ws, ["ticks"])
    assert hub.has_subscribers("ticks")
    assert not hub.has_subscribers("orders")
    hub.disconnect(ws)
    assert not hub.has_subscribers("ticks")


# ---------------------------------------------------------------------------
# /ws endpoint (stub-driven, no server, no httpx)
# ---------------------------------------------------------------------------


def test_endpoint_ping_pong_and_subscribe_contract():
    hub = ChannelHub()
    ws = FakeSocket(incoming=[json.dumps({"type": "ping"}), json.dumps(_subscribe_ws(hub))])

    asyncio.run(websocket_endpoint(ws, hub))

    assert ws.accepted
    payloads = ws.send_payloads()
    assert payloads[0]["type"] == "pong"
    subscribed = payloads[1]
    assert subscribed["type"] == "subscribed"
    assert subscribed["channels"] == ["orders", "ticks"]
    assert subscribed["rejected"] == ["nonsense-channel"]


def test_endpoint_list_returns_current_subscriptions():
    hub = ChannelHub()
    ws = FakeSocket(incoming=[json.dumps({"type": "list"})])
    hub.connect(ws)
    hub.subscribe(ws, ["risk"])

    asyncio.run(websocket_endpoint(ws, hub))
    last = ws.send_payloads()[-1]
    assert last["type"] == "subscriptions"
    assert last["channels"] == ["risk"]


def test_endpoint_disconnect_cleans_up_hub():
    hub = ChannelHub()
    ws = FakeSocket()  # no incoming -> immediately disconnects
    asyncio.run(websocket_endpoint(ws, hub))
    assert hub.count() == 0


def test_endpoint_receives_published_frames_on_subscribed_channel():
    hub = ChannelHub()

    async def scenario() -> list[dict]:
        ws = EndlessSocket(incoming=[json.dumps({"type": "ping"}), json.dumps(_subscribe_ws(hub))])
        task = asyncio.create_task(websocket_endpoint(ws, hub))
        for _ in range(10):
            await asyncio.sleep(0)  # let the endpoint absorb ping + subscribe
        await hub.publish("orders", {"client_order_id": "SIG-1"})
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return ws.send_payloads()

    frames = asyncio.run(scenario())
    published = [f for f in frames if f["type"] == "orders"]
    assert len(published) == 1
    assert published[0]["client_order_id"] == "SIG-1"


# ---------------------------------------------------------------------------
# TickPublisher bridge
# ---------------------------------------------------------------------------


def test_tick_publisher_is_noop_without_subscribers():
    hub = ChannelHub()
    publisher = TickPublisher(hub, channel="ticks")
    # Called synchronously outside any loop — must not raise.
    publisher("RELIANCE", None, 1.0, 2.0)


def test_tick_publisher_fans_out_to_subscribers():
    hub = ChannelHub()
    ws = FakeSocket()
    hub.connect(ws)
    hub.subscribe(ws, ["ticks"])
    publisher = TickPublisher(hub, channel="ticks")

    from datetime import datetime, timezone

    ts = datetime(2026, 9, 6, 11, 30, 0, tzinfo=timezone.utc)

    async def drive() -> None:
        publisher("RELIANCE", ts, 2401.25, 500)
        await asyncio.sleep(0)  # let the scheduled publish task deliver

    asyncio.run(drive())

    ticks = [m for m in ws.send_payloads() if m["type"] == "ticks"]
    assert len(ticks) == 1
    assert ticks[0]["symbol"] == "RELIANCE"
    assert ticks[0]["last_price"] == 2401.25
    assert ticks[0]["source"] == "mock"
    assert ticks[0]["synthetic"] is True


def test_tick_publisher_unknown_channel_rejected_at_construction():
    hub = ChannelHub()
    with pytest.raises(ValueError):
        TickPublisher(hub, channel="not-a-channel")
