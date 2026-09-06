from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings

logger = logging.getLogger(__name__)

# The alembic.ini lives at the backend package root, three directories up
# from this file:  backend/app/database/session.py -> backend/alembic.ini.
# This path is used in the container (where everything sits under /app) and in
# the repo checkout alike.
ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"

engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=settings.database_echo,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

async_session_maker: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def apply_migrations(engine: AsyncEngine | None = None) -> str:
    """
    Take the database to the latest migration revision. Raise on failure.

    This is the ONLY thing that creates or alters schema. ``Base.metadata
    .create_all`` is no longer a boot path: it would silently create missing
    tables and never have altered existing ones, which is exactly the drift
    ``create_all_tables`` used to name in a log line and then leave behind.

    Runs inside the *caller's* engine connection, so a fresh in-memory SQLite
    (StaticPool) is migrated in the same database the caller then reads —
    an engine Alembic opened on its own would be a different, throwaway
    database. Intended for explicit operator action and for tests; the app
    itself only ever *checks* migration state (see ``migration_status``) in
    line with the repo rule that migrations are reviewed and run manually.

    Returns the head revision applied.
    """
    from alembic import command
    from alembic.config import Config

    bind = _default_engine() if engine is None else engine
    cfg = Config(str(ALEMBIC_INI))

    def _upgrade(sync_conn: Any) -> None:
        cfg.attributes["connection"] = sync_conn
        command.upgrade(cfg, "head")

    async with bind.begin() as conn:
        await conn.run_sync(_upgrade)

    status = await migration_status(bind)
    if not status["at_head"]:
        raise RuntimeError(
            "migrations did not reach head: "
            f"database is at {status['current']!r}, head is {status['head']!r}"
        )
    return str(status["head"])


async def migration_status(engine: AsyncEngine | None = None) -> dict[str, Any]:
    """Return the database's migration state vs the head revision.

    Read-only and never raises on a fresh or un-migrated database: a missing
    ``alembic_version`` table is reported, not an error — it is the state of
    "never migrated", which startup logs and an operator fixes with
    ``alembic -c backend/alembic.ini upgrade head``.
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from sqlalchemy import inspect, text

    bind = engine if engine is not None else _default_engine()
    cfg = Config(str(ALEMBIC_INI))
    script = ScriptDirectory.from_config(cfg)
    head = script.get_current_head()
    state: dict[str, Any] = {
        "head": head,
        "current": None,
        "version_table_exists": False,
        "at_head": False,
    }

    def _read(sync_conn: Any) -> None:
        inspector = inspect(sync_conn)
        if not inspector.has_table("alembic_version"):
            return
        state["version_table_exists"] = True
        version_num = sync_conn.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar()
        state["current"] = version_num

    async with bind.connect() as conn:
        try:
            await conn.run_sync(_read)
        except Exception:  # noqa: BLE001
            # Unreachable database (tests, partial startup, db still cold).
            # The caller decides whether that is fatal; this only reports.
            return state

    state["at_head"] = (
        state["version_table_exists"] and state["current"] == state["head"]
    )
    return state


def _default_engine() -> AsyncEngine:
    return engine


async def verify_schema(engine: AsyncEngine | None = None) -> list[str]:
    """
    Return a list of schema items the models declare and the database lacks.

    Empty means the database can serve every model. Non-empty is reported
    rather than raised: refusing to start would take down a read-only API that
    is otherwise fine, while starting silently is how the drift stays hidden.
    The execution path already fails closed on a persistence error.
    """
    from sqlalchemy import inspect

    from app.database.models import Base

    bind = _default_engine() if engine is None else engine

    problems: list[str] = []

    def _inspect(sync_conn: Any) -> None:
        inspector = inspect(sync_conn)
        existing_tables = set(inspector.get_table_names())
        for table_name, table in Base.metadata.tables.items():
            if table_name not in existing_tables:
                problems.append(f"missing table '{table_name}'")
                continue
            existing_cols = {c["name"] for c in inspector.get_columns(table_name)}
            for column in table.columns:
                if column.name not in existing_cols:
                    problems.append(
                        f"missing column '{table_name}.{column.name}'"
                    )

    try:
        async with bind.begin() as conn:
            await conn.run_sync(_inspect)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"schema could not be inspected: {exc}")
    return problems
