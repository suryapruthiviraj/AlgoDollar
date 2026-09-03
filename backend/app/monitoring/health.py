"""System health monitor — checks all components and stores status to Redis."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ComponentStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    name: str
    status: ComponentStatus
    message: str = ""
    latency_ms: Optional[float] = None


@dataclass
class SystemHealth:
    overall: ComponentStatus
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    components: list[ComponentHealth] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "overall": self.overall.value,
            "timestamp": self.timestamp,
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                }
                for c in self.components
            ],
        }


class HealthMonitor:
    """
    Checks health of all system components and persists results to Redis.

    Parameters
    ----------
    db_session_factory  : async callable returning an SQLAlchemy session, or None
    redis_client        : synchronous or async Redis client, or None
    broker              : BrokerInterface instance, or None
    """

    REDIS_KEY = "system_health"
    REDIS_TTL = 300   # 5 minutes

    def __init__(
        self,
        db_session_factory=None,
        redis_client=None,
        broker=None,
        strategy_registry: Optional[dict] = None,
    ) -> None:
        self._db_factory = db_session_factory
        self._redis = redis_client
        self._broker = broker
        self._strategy_registry = strategy_registry or {}

    # ------------------------------------------------------------------ #
    #  Component checks                                                    #
    # ------------------------------------------------------------------ #

    async def check_database_connection(self) -> ComponentHealth:
        import time
        name = "database"
        if self._db_factory is None:
            return ComponentHealth(name, ComponentStatus.UNKNOWN, "No DB configured.")
        try:
            t0 = time.monotonic()
            async with self._db_factory() as session:
                await session.execute("SELECT 1")
            latency = (time.monotonic() - t0) * 1000
            return ComponentHealth(name, ComponentStatus.OK, latency_ms=round(latency, 2))
        except Exception as exc:
            logger.error("DB health check failed: %s", exc)
            return ComponentHealth(name, ComponentStatus.DOWN, str(exc))

    async def check_redis_connection(self) -> ComponentHealth:
        import time
        name = "redis"
        if self._redis is None:
            return ComponentHealth(name, ComponentStatus.UNKNOWN, "No Redis configured.")
        try:
            t0 = time.monotonic()
            ping = getattr(self._redis, "ping", None)
            if asyncio.iscoroutinefunction(ping):
                await ping()
            elif ping is not None:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, ping)
            latency = (time.monotonic() - t0) * 1000
            return ComponentHealth(name, ComponentStatus.OK, latency_ms=round(latency, 2))
        except Exception as exc:
            logger.error("Redis health check failed: %s", exc)
            return ComponentHealth(name, ComponentStatus.DOWN, str(exc))

    async def check_broker_connection(self, broker=None) -> ComponentHealth:
        name = "broker"
        b = broker or self._broker
        if b is None:
            return ComponentHealth(name, ComponentStatus.UNKNOWN, "No broker configured.")
        try:
            connected = b.is_connected
            if connected:
                return ComponentHealth(name, ComponentStatus.OK, "Broker connected.")
            return ComponentHealth(name, ComponentStatus.DOWN, "Broker not connected.")
        except Exception as exc:
            return ComponentHealth(name, ComponentStatus.DOWN, str(exc))

    async def check_websocket_connection(self) -> ComponentHealth:
        name = "websocket"
        if self._broker is None:
            return ComponentHealth(name, ComponentStatus.UNKNOWN, "No broker configured.")
        try:
            kws = getattr(self._broker, "_kws", None)
            if kws is None:
                return ComponentHealth(name, ComponentStatus.DEGRADED, "WebSocket not initialised.")
            if hasattr(kws, "is_connected") and kws.is_connected():
                return ComponentHealth(name, ComponentStatus.OK)
            return ComponentHealth(name, ComponentStatus.DEGRADED, "WebSocket not connected.")
        except Exception as exc:
            return ComponentHealth(name, ComponentStatus.DOWN, str(exc))

    async def check_strategy_health(self) -> ComponentHealth:
        name = "strategies"
        try:
            if not self._strategy_registry:
                return ComponentHealth(name, ComponentStatus.UNKNOWN, "No strategies registered.")
            statuses = {n: s.get("status", "UNKNOWN") for n, s in self._strategy_registry.items()}
            active = [n for n, s in statuses.items() if s == "ACTIVE"]
            paused = [n for n, s in statuses.items() if s in ("PAUSED", "STOPPED")]
            msg = f"Active: {len(active)}, Paused/Stopped: {len(paused)}"
            if not active:
                return ComponentHealth(name, ComponentStatus.DEGRADED, msg)
            return ComponentHealth(name, ComponentStatus.OK, msg)
        except Exception as exc:
            return ComponentHealth(name, ComponentStatus.DOWN, str(exc))

    async def check_risk_limits(self) -> ComponentHealth:
        name = "risk_limits"
        if self._redis is None:
            return ComponentHealth(name, ComponentStatus.UNKNOWN, "No Redis to query risk state.")
        try:
            import json
            key = "risk_state"
            get_fn = getattr(self._redis, "get", None)
            if asyncio.iscoroutinefunction(get_fn):
                raw = await get_fn(key)
            else:
                loop = asyncio.get_event_loop()
                raw = await loop.run_in_executor(None, lambda: self._redis.get(key))
            if raw is None:
                return ComponentHealth(name, ComponentStatus.UNKNOWN, "No risk state in Redis.")
            state = json.loads(raw)
            breaches = state.get("breaches", [])
            if breaches:
                return ComponentHealth(
                    name, ComponentStatus.DEGRADED, f"Active risk breaches: {breaches}"
                )
            return ComponentHealth(name, ComponentStatus.OK, "All risk limits within bounds.")
        except Exception as exc:
            return ComponentHealth(name, ComponentStatus.UNKNOWN, str(exc))

    # ------------------------------------------------------------------ #
    #  Composite check                                                     #
    # ------------------------------------------------------------------ #

    async def check_system_health(self, broker=None) -> SystemHealth:
        """Run all component checks and return aggregated SystemHealth."""
        checks = await asyncio.gather(
            self.check_database_connection(),
            self.check_redis_connection(),
            self.check_broker_connection(broker),
            self.check_websocket_connection(),
            self.check_strategy_health(),
            self.check_risk_limits(),
            return_exceptions=True,
        )

        components: list[ComponentHealth] = []
        for check in checks:
            if isinstance(check, Exception):
                components.append(ComponentHealth("unknown", ComponentStatus.DOWN, str(check)))
            else:
                components.append(check)

        # Aggregate overall status
        statuses = [c.status for c in components]
        if any(s == ComponentStatus.DOWN for s in statuses):
            overall = ComponentStatus.DOWN
        elif any(s == ComponentStatus.DEGRADED for s in statuses):
            overall = ComponentStatus.DEGRADED
        elif all(s == ComponentStatus.OK for s in statuses):
            overall = ComponentStatus.OK
        else:
            overall = ComponentStatus.UNKNOWN

        health = SystemHealth(overall=overall, components=components)

        # Persist to Redis
        await self._store_health(health)
        return health

    async def _store_health(self, health: SystemHealth) -> None:
        if self._redis is None:
            return
        import json
        try:
            data = json.dumps(health.as_dict())
            set_fn = getattr(self._redis, "setex", None)
            if set_fn is None:
                return
            if asyncio.iscoroutinefunction(set_fn):
                await set_fn(self.REDIS_KEY, self.REDIS_TTL, data)
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, lambda: self._redis.setex(self.REDIS_KEY, self.REDIS_TTL, data)
                )
        except Exception as exc:
            logger.warning("Could not store health to Redis: %s", exc)

    # ------------------------------------------------------------------ #
    #  Background monitoring loop                                          #
    # ------------------------------------------------------------------ #

    async def start_monitoring_loop(self, interval: float = 60.0) -> None:
        """
        Start an infinite background loop that checks system health every `interval` seconds.

        Should be run as an asyncio task:
            asyncio.create_task(monitor.start_monitoring_loop())
        """
        logger.info("Health monitoring loop started (interval=%.0fs).", interval)
        while True:
            try:
                health = await self.check_system_health()
                logger.info(
                    "System health: %s | %s",
                    health.overall.value,
                    ", ".join(f"{c.name}={c.status.value}" for c in health.components),
                )
            except Exception as exc:
                logger.error("Health monitoring loop error: %s", exc)
            await asyncio.sleep(interval)
