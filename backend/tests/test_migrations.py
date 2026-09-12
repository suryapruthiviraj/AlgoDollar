"""
Alembic migrations — the schema's single source of truth.

Until this suite existed, the schema came from ``Base.metadata.create_all``:
it created missing tables, never altered existing ones, and a database that
had once been created could silently drift from the models until the first
query touched a missing column. CI_SECURITY_AUDIT.md limitation #5 ("No
migration validation. Alembic is a dependency but no migration is run in CI")
was that gap.

These tests close it with four guarantees, all against SQLite:

1. A fresh database taken to head by the migrations is exactly the database
   ``create_all`` would have made — column for column, constraint for
   constraint — so the migration file is not a second, drifting idea of the
   schema. The reverse guarantee, ``alembic check``, is exercised too: it
   reports nothing on a migrated database and reports drift the moment a
   table is dropped.
2. The boot path no longer contains ``create_all_tables`` at all; startup
   checks ``migration_status`` and reports rather than silently creating or
   mutating schema.
3. Applying migrations twice is a no-op and preserves data.
4. Downgrade is reversible; upgrade after downgrade works again.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from app.database.models import Base

MAIN_INI = Path("alembic.ini").resolve()


def _make_engine(tmp_path: Path, name: str):
    return create_async_engine(f"sqlite+aiosqlite:///{tmp_path / name}")


def _in_memory_engine():
    return create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


async def _upgrade(engine, cfg: Config) -> None:
    """Run `alembic upgrade head` inside the caller's engine connection."""

    def _go(sync_conn):
        cfg.attributes["connection"] = sync_conn
        command.upgrade(cfg, "head")

    async with engine.begin() as conn:
        await conn.run_sync(_go)


def _describe(sync_conn) -> dict:
    """A dialect-fair structural description of a database, keyed by table."""
    inspector = inspect(sync_conn)
    out: dict = {}
    for table in inspector.get_table_names():
        pk = inspector.get_pk_constraint(table) or {}
        out[table] = {
            "cols": sorted(
                (c["name"], str(c["type"]), c["nullable"])
                for c in inspector.get_columns(table)
            ),
            "pk": sorted(pk.get("constrained_columns", [])),
            "uniques": sorted(
                (u["name"], sorted(u["column_names"]))
                for u in inspector.get_unique_constraints(table)
                if u.get("name")
            ),
            "indexes": sorted(
                (ix["name"], sorted(ix["column_names"]))
                for ix in inspector.get_indexes(table)
                if ix.get("name")
            ),
            "fks": sorted(
                (
                    fk["referred_table"],
                    sorted(fk["constrained_columns"]),
                    fk.get("options", {}).get("ondelete"),
                )
                for fk in inspector.get_foreign_keys(table)
            ),
        }
    return out


async def _describe_async(engine) -> dict:
    box: dict = {}

    async with engine.connect() as conn:
        await conn.run_sync(lambda c: box.update(_describe(c)))
    return box


@pytest.fixture
def cfg() -> Config:
    config = Config(str(MAIN_INI))
    config.attributes["connection"] = None
    return config


class TestFreshDatabaseToHead:
    async def test_upgrade_reaches_head_with_no_model_drift(
        self, tmp_path, cfg
    ):
        engine = _make_engine(tmp_path, "up.db")
        try:
            from app.database.session import (
                apply_migrations,
                migration_status,
                verify_schema,
            )

            head = await apply_migrations(engine)
            assert head  # a revision was applied

            status = await migration_status(engine)
            assert status["version_table_exists"] is True
            assert status["current"] == status["head"]
            assert status["at_head"] is True

            # The models, looking at the migrated database, see every table
            # and every column. This is the same check startup runs — it has
            # to be empty for the deployment to be healthy.
            assert await verify_schema(engine) == []
        finally:
            await engine.dispose()

    async def test_migrated_schema_matches_create_all_exactly(
        self, tmp_path, cfg
    ):
        migrated = _make_engine(tmp_path, "migrated.db")
        reference = _in_memory_engine()
        try:
            await _upgrade(migrated, cfg)

            async with reference.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            got = await _describe_async(migrated)
            want = await _describe_async(reference)

            performed = set(want)  # every model table must exist identically
            for table in performed:
                assert got.get(table) == want[table], (
                    f"migrations diverged from the models on table "
                    f"'{table}'\n  create_all: {want[table]}\n  migrated: "
                    f"{got.get(table)}"
                )
            # Exactly the model tables, plus only Alembic's own bookkeeping.
            extra = set(got) - performed
            assert extra == {"alembic_version"}, f"unexpected tables: {extra}"
        finally:
            await migrated.dispose()
            await reference.dispose()

    async def test_alembic_check_guards_drift_both_ways(
        self, tmp_path, cfg
    ):
        engine = _make_engine(tmp_path, "check.db")
        try:
            await _upgrade(engine, cfg)

            def _check(sync_conn) -> None:
                cfg.attributes["connection"] = sync_conn
                command.check(cfg)

            # A migrated database has no pending autogenerate differences.
            async with engine.connect() as conn:
                await conn.run_sync(_check)

            # Drop a table: the same command must now report pending ops.
            async with engine.begin() as conn:
                await conn.execute(text("DROP TABLE account_cash"))
            async with engine.connect() as conn:
                with pytest.raises(Exception, match="[Uu]pgrade operations"):
                    await conn.run_sync(_check)
        finally:
            await engine.dispose()


class TestBootPathHasNoCreateAll:
    def test_create_all_tables_was_dropped(self):
        """Startup must never silently create or alter schema."""
        import app.database.session as sess

        assert not hasattr(sess, "create_all_tables"), (
            "create_all_tables must not return: the boot path would silently "
            "re-create missing tables as 'always create missing tables, never "
            "alter existing' instead of surfacing migration debt"
        )
        src = Path("app/main.py").read_text()
        assert "create_all_tables" not in src

    async def test_migration_status_names_the_repair_on_unmigrated_db(
        self, tmp_path, cfg
    ):
        """A never-migrated database is reported, not crashed over."""
        from app.database.session import migration_status

        engine = _make_engine(tmp_path, "unmigrated.db")
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            status = await migration_status(engine)
            assert status["version_table_exists"] is False
            assert status["current"] is None
            assert status["at_head"] is False
            assert status["head"]

            status = await migration_status(_in_memory_engine())
            assert status["version_table_exists"] is False
        finally:
            await engine.dispose()


class TestIdempotencyAndReversibility:
    async def test_reupgrade_is_a_noop_and_preserves_data(
        self, tmp_path, cfg
    ):
        engine = _make_engine(tmp_path, "idem.db")
        try:
            await _upgrade(engine, cfg)

            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO users (email, hashed_password, is_active) "
                        "VALUES ('pinned@example.com', 'x', 1)"
                    )
                )

            await _upgrade(engine, cfg)

            async with engine.connect() as conn:
                rows = await conn.execute(
                    text("SELECT email FROM users WHERE email='pinned@example.com'")
                )
                assert rows.scalar() == "pinned@example.com"
                versions = await conn.execute(
                    text("SELECT COUNT(*) FROM alembic_version")
                )
                assert versions.scalar() == 1
        finally:
            await engine.dispose()

    async def test_downgrade_to_base_is_reversible(self, tmp_path, cfg):
        engine = _make_engine(tmp_path, "downgrade.db")
        try:
            await _upgrade(engine, cfg)
            created = {
                t for t in await _describe_async(engine) if t != "alembic_version"
            }
            assert "users" in created

            def _go_base(sync_conn):
                cfg.attributes["connection"] = sync_conn
                command.downgrade(cfg, "base")

            async with engine.begin() as conn:
                await conn.run_sync(_go_base)

            assert "users" not in {
                t for t in (await _describe_async(engine)) if t != "alembic_version"
            }

            await _upgrade(engine, cfg)
            again = set(await _describe_async(engine))
            assert "users" in again and created == {
                t for t in again if t != "alembic_version"
            }
        finally:
            await engine.dispose()
