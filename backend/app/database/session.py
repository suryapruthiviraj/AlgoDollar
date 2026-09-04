from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings

logger = logging.getLogger(__name__)

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


async def create_all_tables() -> None:
    """
    Create any missing tables, then VERIFY the schema matches the models.

    `create_all` creates tables that do not exist. It does NOT alter tables
    that do — a column added to a model never appears on an existing database.
    That is fine for a fresh deployment and silently wrong for an upgraded one:
    the process starts happily and then fails at the first query touching the
    new column, far from the cause.

    So the schema is checked here and a drift is reported with the exact
    missing tables and columns. This is not a migration system; it is the
    honest boundary of not having one, and the check names the repair.
    """
    from app.database.models import Base  # noqa: F401 – side-effect import

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    drift = await verify_schema()
    if drift:
        logger.error(
            "SCHEMA DRIFT: the database is missing %d item(s) the models "
            "declare. create_all() adds missing TABLES but never alters "
            "existing ones, so an upgraded deployment needs these applied by "
            "hand or by a migration tool. Details: %s",
            len(drift), "; ".join(drift),
        )
    return None


async def verify_schema() -> list[str]:
    """
    Return a list of schema items the models declare and the database lacks.

    Empty means the database can serve every model. Non-empty is reported
    rather than raised: refusing to start would take down a read-only API that
    is otherwise fine, while starting silently is how the drift stays hidden.
    The execution path already fails closed on a persistence error.
    """
    from sqlalchemy import inspect

    from app.database.models import Base

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
        async with engine.begin() as conn:
            await conn.run_sync(_inspect)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"schema could not be inspected: {exc}")
    return problems
