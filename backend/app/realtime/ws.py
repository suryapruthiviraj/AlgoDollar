"""
The ``/ws`` endpoint handler, extracted from the app factory so the tests can
drive it against a stub socket without booting the whole application (and
without httpx/TestClient, which the CI venv deliberately lacks).

Message contract (JSON text frames sent by the client):
    {"type": "ping"}                 -> {"type": "pong", ...}
    {"type": "subscribe", "channels": [...]}  -> {"type": "subscribed", ...}
    {"type": "unsubscribe", "channels": [...]}-> {"type": "unsubscribed", ...}
    {"type": "list"}                 -> {"type": "subscriptions", ...}
Anything else is treated as a ping (matches the pre-existing /ws behaviour).

Server -> client frames from publish() carry ``{"type": <channel>, ...}``;
``broadcast()`` carries ``{"type": "broadcast", ...}``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def websocket_endpoint(websocket: WebSocket, hub: Any) -> None:
    """
    Serve one WebSocket connection against a ``ChannelHub``.

    ``hub`` is passed in rather than imported so tests can hand this handler
    a stub socket and a fresh hub. Subscriptions are scoped to the socket:
    when it disconnects, ``disconnect`` removes it from every channel.
    """
    await websocket.accept()
    hub.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            if not raw:
                break
            try:
                msg = json.loads(raw)
                msg_type = msg.get("type", "ping")
            except (json.JSONDecodeError, TypeError, AttributeError):
                msg = {}
                msg_type = "ping"

            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": _now()}))
            elif msg_type == "subscribe":
                requested = msg.get("channels") or []
                granted = hub.subscribe(websocket, requested)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "subscribed",
                            "channels": granted,
                            "requested": list(requested),
                            "rejected": [c for c in requested if c not in granted],
                            "timestamp": _now(),
                        }
                    )
                )
            elif msg_type == "unsubscribe":
                removed = hub.unsubscribe(websocket, msg.get("channels") or None)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "unsubscribed",
                            "channels": removed,
                            "timestamp": _now(),
                        }
                    )
                )
            elif msg_type == "list":
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "subscriptions",
                            "channels": hub.subscribed(websocket),
                            "timestamp": _now(),
                        }
                    )
                )
    except WebSocketDisconnect:
        pass
    finally:
        hub.disconnect(websocket)
