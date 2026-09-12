"""Multi-day offline paper campaign (O1) — persistence, restart/recovery,
and fail-closed behaviour."""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from app.engine.campaign import (
    CAMPAIGN_NOTE,
    PaperCampaignConfig,
    _append_record,
    day_records,
    load_ledger,
    run_campaign,
)

IST = ZoneInfo("Asia/Kolkata")
_TRADING_DAY = date(2025, 6, 10)  # Tuesday, not an NSE holiday


def _next_minute(minute: time) -> time:
    return time(minute.hour + 1, 0) if minute.minute == 59 else time(minute.hour, minute.minute + 1)


def _sessions(n: int) -> List[date]:
    return [_TRADING_DAY + timedelta(days=i) for i in range(n)]


def _config(n: int, seed: int = 0, **kw) -> PaperCampaignConfig:
    return PaperCampaignConfig(
        symbols=["RELIANCE", "TCS", "INFY"],
        seed=seed,
        sessions=_sessions(n),
        **kw,
    )


def _climb_then_flat(day: date = _TRADING_DAY) -> dict[str, pd.DataFrame]:
    """Strong monotonic climb 09:31..10:16 then flat — guarantees a fill."""
    rows = []
    minute = time(9, 31)
    px = 100.0
    end_climb = time(10, 16)
    while minute <= end_climb:
        rows.append({
            "time": datetime.combine(day, minute, tzinfo=IST),
            "open": round(px, 2), "high": round(px * 1.002, 2),
            "low": round(px * 0.998, 2), "close": round(px * 1.002, 2),
            "volume": 200_000,
        })
        px *= 1.002
        minute = _next_minute(minute)
    while minute <= time(15, 30):
        rows.append({
            "time": datetime.combine(day, minute, tzinfo=IST),
            "open": round(px, 2), "high": round(px, 2),
            "low": round(px, 2), "close": round(px, 2),
            "volume": 200_000,
        })
        minute = _next_minute(minute)
    return {"RELIANCE": pd.DataFrame(rows)}


async def test_three_day_synthetic_campaign(tmp_path) -> None:
    ledger = str(tmp_path / "ledger.jsonl")
    book = str(tmp_path / "book.json")
    cfg = _config(3)
    res = await run_campaign(3, cfg, ledger_path=ledger, paper_state_path=book)

    assert res.completed
    assert res.stop_reason is None
    assert res.synthetic is True
    assert len(res.records) == 3
    assert res.summary()["source"] == "synthetic"
    assert len(day_records(load_ledger(ledger))) == 3
    assert res.summary()["restarts_resumed"] == 0
    assert res.summary()["runs_total"] == 1
    assert res.run_starts == 1
    for r in res.records:
        assert r["source"] == "synthetic"
        assert r["recovery_ok"] is True
        assert r["reconciliation"] == "OK"
        assert r["cash_start"] > 0
    for prev, nxt in zip(res.records, res.records[1:]):
        assert nxt["cash_start"] == pytest.approx(prev["cash_end"], abs=0.01)
    assert res.summary()["net_pnl_rupees"] == pytest.approx(
        sum(r["net_pnl_rupees"] for r in res.records)
    )
    assert res.summary()["days_with_errors"] == 0
    assert CAMPAIGN_NOTE in res.summary()["note"]


async def test_resume_after_restart(tmp_path) -> None:
    ledger = str(tmp_path / "ledger.jsonl")
    book = str(tmp_path / "book.json")
    cfg = _config(3)

    first = await run_campaign(2, cfg, ledger_path=ledger, paper_state_path=book)
    assert first.completed
    assert len(first.records) == 2

    resume = await run_campaign(3, cfg, ledger_path=ledger, paper_state_path=book)
    assert resume.completed
    assert len(resume.records) == 3
    assert resume.records[2]["resume_after"] == 2
    assert resume.summary()["restarts_resumed"] == 1
    assert resume.run_starts == 2
    assert resume.records[1]["resume_after"] == 1
    assert resume.records[0]["resume_after"] == 0
    assert resume.records[2]["cash_start"] == pytest.approx(
        resume.records[1]["cash_end"], abs=0.01
    )
    assert len(day_records(load_ledger(ledger))) == 3

    full = await run_campaign(
        3,
        _config(3),
        ledger_path=str(tmp_path / "b" / "ledger.jsonl"),
        paper_state_path=str(tmp_path / "b" / "book.json"),
    )
    assert full.summary()["net_pnl_rupees"] == pytest.approx(
        resume.summary()["net_pnl_rupees"], abs=0.01
    )


async def test_deterministic_runs(tmp_path) -> None:
    a_paths = (str(tmp_path / "a" / "ledger.jsonl"), str(tmp_path / "a" / "book.json"))
    b_paths = (str(tmp_path / "b" / "ledger.jsonl"), str(tmp_path / "b" / "book.json"))
    a = await run_campaign(
        4, _config(4, seed=7), ledger_path=a_paths[0], paper_state_path=a_paths[1]
    )
    b = await run_campaign(
        4, _config(4, seed=7), ledger_path=b_paths[0], paper_state_path=b_paths[1]
    )
    pa = [(r["day"], r["cash_end"], r["net_pnl_rupees"]) for r in a.records]
    pb = [(r["day"], r["cash_end"], r["net_pnl_rupees"]) for r in b.records]
    assert pa == pb
    assert a.summary()["net_pnl_rupees"] == b.summary()["net_pnl_rupees"]


async def test_fail_closed_on_ledger_disagreement(tmp_path) -> None:
    ledger = str(tmp_path / "ledger.jsonl")
    book = str(tmp_path / "book.json")
    cfg = _config(3)
    await run_campaign(1, cfg, ledger_path=ledger, paper_state_path=book)

    real = load_ledger(ledger)[-1]
    forged = dict(real)
    forged["day"] = "2000-01-03"
    forged["resume_after"] = 1
    forged["positions"] = [{
        "symbol": "RELIANCE", "exchange": "NSE", "quantity": 10,
        "average_price": 100.0, "product": "MIS", "strategy": "intraday",
    }]
    forged["open_at_end"] = ["RELIANCEx10"]
    _append_record(ledger, forged)

    res = await run_campaign(3, cfg, ledger_path=ledger, paper_state_path=book)
    assert not res.completed
    assert res.stop_reason is not None
    # the phantom claim must never trade a reconciliation-failed day
    assert len(res.records) == 2
    assert len(day_records(load_ledger(ledger))) == 2
    assert res.summary()["stopped"] is True
    assert res.summary()["planned_days"] == 3


async def test_fail_closed_on_corrupt_paper_state(tmp_path) -> None:
    ledger = str(tmp_path / "ledger.jsonl")
    book = str(tmp_path / "book.json")
    cfg = _config(3)
    await run_campaign(1, cfg, ledger_path=ledger, paper_state_path=book)

    raw = Path(book).read_text()
    envelope = json.loads(raw)
    body = envelope["body"]
    idx = body.index('"cash_paise":') + len('"cash_paise":')
    body = body[:idx] + ("9" if body[idx] != "9" else "0") + body[idx + 1:]
    envelope["body"] = body
    Path(book).write_text(json.dumps(envelope))

    res = await run_campaign(3, cfg, ledger_path=ledger, paper_state_path=book)
    assert not res.completed
    assert res.stop_reason is not None
    assert "state" in res.stop_reason.lower()
    # the corrupt book must never be silently replaced by a fresh account
    assert len(day_records(load_ledger(ledger))) == 1
    assert "corrupt" in res.stop_reason.lower() or "checksum" in res.stop_reason.lower()


async def test_custom_trending_bars_fill_trades(tmp_path) -> None:
    cfg = PaperCampaignConfig(
        symbols=["RELIANCE"],
        seed=0,
        source="custom",
        frames=_climb_then_flat(),
        sessions=[_TRADING_DAY],
        opening_cash=1_000_000.0,
    )
    ledger = str(tmp_path / "ledger.jsonl")
    book = str(tmp_path / "book.json")
    res = await run_campaign(1, cfg, ledger_path=ledger, paper_state_path=book)
    assert res.completed
    r = res.records[0]
    assert r["recovery_ok"] is True
    assert r["trades"] >= 1
    assert r["costs_rupees"] > 0


async def test_plan_rejects_insufficient_sessions(tmp_path) -> None:
    """A campaign whose session list is too short fails at plan time."""
    cfg = _config(2, source="custom", frames=_climb_then_flat())
    ledger = str(tmp_path / "ledger.jsonl")
    book = str(tmp_path / "book.json")
    with pytest.raises(ValueError):
        await run_campaign(3, cfg, ledger_path=ledger, paper_state_path=book)
