"""tests for app/execution/file_stores.py — O2: durable kill switch + order store.

The durable stores give a *running* paper campaign a risk state that survives a
process restart: an engaged halt stays engaged, and a reserved order claim
stays claimed.  These tests pin the durability contract (atomic write, fail
closed on unreadable state, reserve set-if-not-exists across re-instantiation)
and the campaign integration that turns a recorded breach into an acted-on one.
"""

from __future__ import annotations

import asyncio
import os
from datetime import date, timedelta

import pytest

from app.engine.campaign import PaperCampaignConfig, run_campaign
from app.engine.replay import _build_stack
from app.execution.bootstrap import InMemoryKillSwitchStore
from app.execution.file_stores import FileKillSwitchStore, FileOrderStore, atomic_write
from app.execution.lifecycle import (
    InMemoryOrderStore,
    OrderRecord,
    OrderState,
    PersistenceError,
)


def _record(cid: str, **kw) -> OrderRecord:
    defaults = dict(
        client_order_id=cid,
        symbol="RELIANCE",
        exchange="NSE",
        side="BUY",
        qty=100,
        order_type="MARKET",
        product="MIS",
        strategy="intraday_test",
    )
    defaults.update(kw)
    return OrderRecord(**defaults)


# ── atomic writes ────────────────────────────────────────────────────────────


def test_atomic_write_replaces_in_place(tmp_path):
    p = tmp_path / "sub" / "f.json"  # parent does not exist yet
    atomic_write(p, "{}")
    assert p.read_text() == "{}"
    assert not p.with_suffix(".json.tmp").exists(), "temp file leaked"
    atomic_write(p, '{"a": 1}')
    assert p.read_text() == '{"a": 1}'


# ── kill switch ──────────────────────────────────────────────────────────────


def test_kill_switch_engage_survives_reinstantiation(tmp_path):
    path = str(tmp_path / "kill_switch.json")
    FileKillSwitchStore(path).engage("daily loss breached")
    store = FileKillSwitchStore(path)
    assert store.active is True
    assert store.reason() == "daily loss breached"
    assert bool(store.get("kill_switch")) is True

    store.release()
    reloaded = FileKillSwitchStore(path)
    assert reloaded.active is False  # released; the reason is retained as evidence


def test_kill_switch_kv_surface_and_durable_flag(tmp_path):
    path = str(tmp_path / "kill_switch.json")
    store = FileKillSwitchStore(path)
    assert store.durable is True
    assert store.get("missing") is None
    store.set("some_key", "1")
    assert FileKillSwitchStore(path).get("some_key") == "1"
    store.delete("some_key")
    assert FileKillSwitchStore(path).get("some_key") is None


def test_durable_kill_switch_fails_closed_on_corrupt_file(tmp_path):
    path = tmp_path / "kill_switch.json"
    path.write_text("{not json")
    with pytest.raises(PersistenceError, match="corrupt or unreadable"):
        FileKillSwitchStore(str(path))


# ── order store: parity with the in-memory reference ─────────────────────────


async def test_file_order_store_matches_in_memory_head_for_head(tmp_path):
    file = FileOrderStore(str(tmp_path / "orders.json"))
    mem = InMemoryOrderStore()
    cid = "TESTPARITY000001"

    rec = _record(cid)
    assert await file.reserve(rec) is True
    assert await mem.reserve(rec) is True
    assert await file.reserve(_record(cid)) is False  # set-if-not-exists
    assert await mem.reserve(_record(cid)) is False

    assert await file.get(cid) == await mem.get(cid)
    assert await file.list_open() == await mem.list_open()

    moved = rec.transition(OrderState.RISK_CHECK_PENDING, reason="queued")
    moved.transition(OrderState.RISK_APPROVED, reason="cleared")
    moved.transition(OrderState.SUBMITTED, reason="sent")
    await file.save(moved)
    await mem.save(moved)
    assert await file.get(cid) == await mem.get(cid)

    fill = {"fill_id": "f-a", "qty": 100, "price": 10.0}
    assert await file.record_trade(cid, fill) is True
    assert await mem.record_trade(cid, fill) is True
    assert await file.record_trade(cid, dict(fill)) is False  # dedupe
    assert await mem.record_trade(cid, dict(fill)) is False
    assert await file.list_trades(cid) == await mem.list_trades(cid)

    await file.apply_position_delta("INFY", "NSE", "MIS", 50, 101.0)
    await mem.apply_position_delta("INFY", "NSE", "MIS", 50, 101.0)
    await file.apply_position_delta("INFY", "NSE", "MIS", -50, 102.0)
    await mem.apply_position_delta("INFY", "NSE", "MIS", -50, 102.0)
    assert await file.get_position("INFY", "NSE", "MIS") == await mem.get_position(
        "INFY", "NSE", "MIS"
    )
    assert (await file.get_position("INFY", "NSE", "MIS"))["quantity"] == 0
    assert file.durable is True
    assert mem.durable is False


def test_save_before_reserve_raises(tmp_path):
    store = FileOrderStore(str(tmp_path / "orders.json"))
    with pytest.raises(PersistenceError, match="save\\(\\) before reserve"):
        asyncio.run(store.save(_record("NOPE0000000001")))


# ── order store: durability across re-instantiation (the restart) ────────────


async def test_reserved_claim_survives_restart(tmp_path):
    path = str(tmp_path / "orders.json")
    cid = "TESTRESTART001"
    assert await FileOrderStore(path).reserve(_record(cid)) is True

    store = FileOrderStore(path)  # "process restarted"
    loaded = await store.get(cid)
    assert loaded is not None
    assert loaded.client_order_id == cid
    assert loaded.state is OrderState.INTENT_CREATED
    assert await store.reserve(_record(cid)) is False, (
        "a claim must stay claimed across a restart — this is the double-submit defence"
    )


async def test_state_transitions_and_fills_survive_restart(tmp_path):
    path = str(tmp_path / "orders.json")
    rec = _record("TESTRESTART002")
    rec.transition(OrderState.RISK_CHECK_PENDING, reason="queued")
    rec.transition(OrderState.RISK_APPROVED, reason="cleared")
    rec.transition(OrderState.SUBMITTED, reason="sent")
    rec.transition(OrderState.FILLED, reason="executed", reconciled=True)
    await FileOrderStore(path).reserve(rec)
    await FileOrderStore(path).record_trade(
        "TESTRESTART002", {"fill_id": "f-1", "qty": 100, "price": 50.0}
    )

    store = FileOrderStore(path)
    loaded = await store.get("TESTRESTART002")
    assert loaded.state is OrderState.FILLED
    assert loaded.is_terminal
    assert [t["fill_id"] for t in await store.list_trades("TESTRESTART002")] == ["f-1"]
    assert await store.list_open() == []
    assert (
        await store.record_trade("TESTRESTART002", {"fill_id": "f-1", "qty": 1, "price": 1.0})
        is False
    ), "a fill recorded before the crash must not be re-applied"


def test_file_order_store_fails_closed_on_corrupt_file(tmp_path):
    path = tmp_path / "orders.json"
    path.write_text("{not json")
    with pytest.raises(PersistenceError, match="corrupt or unreadable"):
        FileOrderStore(str(path))


# ── default stack stays in-memory (non-campaign paths) ───────────────────────


def test_default_stack_still_uses_in_memory_stores():
    from app.broker.paper import PaperBroker
    from app.broker.tickdata import TickBarAggregator, TickQuoteSource
    from app.engine.replay import ReplayClock

    clock = ReplayClock(date(2025, 6, 10))
    broker = PaperBroker(
        data_broker=TickQuoteSource(TickBarAggregator()), initial_cash=1_000_000.0, clock=clock
    )
    stack = _build_stack(broker, cash=1_000_000.0)
    assert isinstance(stack["kill_store"], InMemoryKillSwitchStore)
    assert isinstance(stack["service"].order_manager._store, InMemoryOrderStore)


# ── campaign integration: a recorded breach becomes an acted-on halt ─────────


def _cfg(total: int, risk_limits=None) -> PaperCampaignConfig:
    return PaperCampaignConfig(
        symbols=["RELIANCE", "TCS", "INFY"],
        seed=0,
        sessions=[date(2025, 6, 10) + timedelta(days=i) for i in range(total)],
        risk_limits=risk_limits,
    )


async def test_campaign_breach_engages_durable_kill_switch(tmp_path):
    res = await run_campaign(
        3,
        _cfg(3, {"max_daily_loss": 50.0}),
        ledger_path=str(tmp_path / "l.jsonl"),
        paper_state_path=str(tmp_path / "book.json"),
    )
    assert res.completed is False
    assert res.kill_switch_engaged is True
    assert "daily_loss" in (res.kill_switch_reason or "")
    assert res.stop_reason and "durable kill switch engaged" in res.stop_reason
    assert res.records, "the halted day must still be recorded"
    halted_day = res.records[-1]
    assert halted_day["risk"]["enforced_halted"] is True
    assert halted_day["risk"]["enforced_halt_reason"] == "daily_loss"
    assert res.summary()["paper_risk_halts"] == 1
    # The durable switch file (per-book, next to the book) is engaged for real.
    assert FileKillSwitchStore(str(tmp_path / "book.json.kill_switch.json")).active is True


async def test_engaged_kill_switch_blocks_a_fresh_run_in_the_same_dir(tmp_path):
    ledger = str(tmp_path / "l.jsonl")
    book = str(tmp_path / "book.json")
    first = await run_campaign(
        3, _cfg(3, {"max_daily_loss": 50.0}), ledger_path=ledger, paper_state_path=book
    )
    assert first.kill_switch_engaged

    second = await run_campaign(
        3, _cfg(3, {"max_daily_loss": 50.0}), ledger_path=ledger, paper_state_path=book
    )
    assert second.completed is False
    assert second.records == first.records, "no extra day may run while engaged"
    assert "durable kill switch is engaged" in (second.stop_reason or "")
    assert second.kill_switch_engaged is True

    # Releasing the durable switch makes the same book resumable again.  Run
    # the remainder recording-only so a fresh breach cannot re-engage it.
    FileKillSwitchStore(str(tmp_path / "book.json.kill_switch.json")).release()
    resumed = await run_campaign(3, _cfg(3), ledger_path=ledger, paper_state_path=book)
    assert resumed.completed is True
    assert len(resumed.records) == 3


async def test_default_campaign_records_without_acting(tmp_path):
    res = await run_campaign(
        2,
        _cfg(2),
        ledger_path=str(tmp_path / "l.jsonl"),
        paper_state_path=str(tmp_path / "book.json"),
    )
    assert res.completed is True
    assert res.kill_switch_engaged is False
    assert res.stop_reason is None
    assert res.summary()["paper_risk_halts"] == 0
    assert all(r["risk"]["enforced_halted"] is False for r in res.records), (
        "recording-only campaigns must never claim they halted"
    )
    store = FileKillSwitchStore(str(tmp_path / "book.json.kill_switch.json"))
    assert store.active is False


def test_campaign_limits_rejects_unknown_fields():
    from app.engine.campaign import _campaign_limits

    with pytest.raises(ValueError, match="unknown risk limit"):
        _campaign_limits({"max_daily_loss": 100.0, "bogus": 1})
    assert _campaign_limits(None) is None
    assert _campaign_limits({"max_daily_loss": 100.0}).max_daily_loss == 100.0


async def test_durable_stores_are_isolated_per_book(tmp_path):
    """Two books in one directory are two campaigns: no cross-book bleed."""
    base = str(tmp_path)
    breach = {"max_daily_loss": 50.0}
    a = await run_campaign(
        2, _cfg(2, breach), ledger_path=f"{base}/a.jsonl", paper_state_path=f"{base}/book_a.json"
    )
    b = await run_campaign(
        2, _cfg(2, breach), ledger_path=f"{base}/b.jsonl", paper_state_path=f"{base}/book_b.json"
    )

    # Book A's engaged switch must not pre-block book B: both halted on their
    # OWN breach (enforcement message), each with its own switch file.
    for res in (a, b):
        assert res.kill_switch_engaged and not res.completed
        assert "durable kill switch engaged by" in (res.stop_reason or "")
    switch_a = str(tmp_path / "book_a.json.kill_switch.json")
    switch_b = str(tmp_path / "book_b.json.kill_switch.json")
    assert FileKillSwitchStore(switch_a).active is True
    assert FileKillSwitchStore(switch_b).active is True
    assert set(os.listdir(base)) >= {
        "book_a.json.kill_switch.json",
        "book_a.json.orders.json",
        "book_b.json.kill_switch.json",
        "book_b.json.orders.json",
    }

    # Releasing B (while A stays engaged) lets B resume and A stay blocked.
    FileKillSwitchStore(switch_b).release()
    b_resumed = await run_campaign(
        2, _cfg(2), ledger_path=f"{base}/b.jsonl", paper_state_path=f"{base}/book_b.json"
    )
    assert b_resumed.completed
    a_retry = await run_campaign(
        2, _cfg(2, breach), ledger_path=f"{base}/a.jsonl", paper_state_path=f"{base}/book_a.json"
    )
    assert "durable kill switch is engaged" in (a_retry.stop_reason or "")
    assert FileKillSwitchStore(switch_a).active is True
