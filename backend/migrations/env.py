"""Alembic migrations entrypoint.

Supports two ways of running, so the same script list serves the CLI and the
tests:
  * CLI (`alembic upgrade head`): spins up its own async engine from the
    configured ``DATABASE_URL``, exactly like the app's own ``app.database``
    engine. Used by operators after review (deployment.md) — the app never
    auto-runs migrations.
  * Programmatic (tests and the migration-parity suite): a call site passes a
    live connection via ``config.attributes["connection"]`` and Alembic runs
    the scripts inside THAT connection/transaction. This is how a fresh
    in-memory SQLite is taken to head under test without the "engine opens its
    own connection" trap.

``target_metadata`` is the ORM metadata. Autogeneration therefore diffes the
declarative models (``app.database.models``) against the live schema, and a
migrated database is only "at head" when it matches the models.
"""

from __future__ import annotations

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# The migrations tree lives inside backend/, but `alembic` may be launched from
# the repo root (`alembic -c backend/alembic.ini`) or from anywhere else. The
# `app` package has to be importable no matter the CWD.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.database.models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

DATABASE_URL = config.get_main_option("sqlalchemy.url")
if not DATABASE_URL:
    DATABASE_URL = settings.database_url
# configparser %-interpolation would corrupt URL passwords containing `%`.
config.set_main_option("sqlalchemy.url", DATABASE_URL.replace("%", "%%"))

target_metadata = Base.metadata


def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=False,
    )


def run_migrations_offline() -> None:
    """Emit migration SQL without a database connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_with_connection(connection: Connection) -> None:
    """Run the migration scripts inside a caller-owned connection."""
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        run_migrations_with_connection(connection)
        return

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async def go() -> None:
        async with connectable.connect() as connection:
            await connection.run_sync(run_migrations_with_connection)
        await connectable.dispose()

    asyncio.run(go())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
