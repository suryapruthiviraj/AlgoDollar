"""In-process channel-based pub/sub for real-time UI updates.

```text
                          ┌───────────────┐
  ticks ── TickPublisher ─▶               │
  orders ◀───────────────▶  ChannelHub    │  ──▶  /ws  (frontend)
  risk  ─────────────────▶               │
                          └───────────────┘
```

The hub is deliberately broker-less: it runs in the API process and has no
clock of its own, so a client sees only what a publisher wrote.  Redis
fan-out across multiple API replicas is a later concern — the platform
today runs a single API worker (documented in architecture.md), and this
module's contract is sized to that.
"""

from .channels import (
    KNOWN_CHANNELS,
    ChannelHub,
    TickPublisher,
)
from .ws import websocket_endpoint

__all__ = [
    "KNOWN_CHANNELS",
    "ChannelHub",
    "TickPublisher",
    "websocket_endpoint",
]
