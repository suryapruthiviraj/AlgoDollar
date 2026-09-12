"""tests for app/engine/equity.py — O2: persisted intraday P&L / high-water mark."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.engine.campaign import PaperCampaignConfig, run_campaign
from app.engine.equity import (
    EquitySnapshot,
    EquityTracker,
    risk_breaches_for,
    worst_risk,
)
from app.risk.limits import RiskLimits

IST = ZoneInfo("Asia/Kolkata")


def _ts(minute: str) -> datetime:
    return datetime(2025, 6, 10, int(minute[:2]), int(minute[2:]), tzinfo=IST)


def _positions(symbol: str = "RELIANCE", qty: int = 100, price: float = 100.0):
    return [{"symbol": symbol, "quantity": qty, "last_price": price}]


def _prices(symbol: str = "RELIANCE", price: float = 100.0) -> dict:
    return {symbol: price}


# ── tracker maths ────────────────────────────────────────────────────────────

def test_equity_is_cash_plus_marked_positions():
    t = EquityTracker()
    snap = t.record(_ts("0931"), cash=90000.0, positions=_positions(qty=100),
                    prices=_prices(price=110.0))
    assert snap.cash == 90000.0
    assert snap.position_value == 11000.0
    assert snap.equity == 101000.0
    assert snap.daily_loss_rupees == 0.0
    assert t.base_equity == 101000.0


def test_high_water_mark_and_drawdown():
    t = EquityTracker()
    t.record(_ts("0931"), 100000.0, [], {})
    t.record(_ts("0932"), 102000.0, [], {})   # new peak
    low = t.record(_ts("0933"), 98000.0, [], {})
    assert low.peak_equity == 102000.0
    assert low.equity == 98000.0
    assert low.drawdown_pct == pytest.approx((102000 - 98000) / 102000)
    assert low.daily_loss_rupees == pytest.approx(2000.0)
    assert t.peak_equity == 102000.0
    assert t.max_drawdown_pct == pytest.approx(low.drawdown_pct, abs=1e-9)
    assert t.min_equity == 98000.0


def test_daily_loss_is_a_positive_magnitude():
    t = EquityTracker()
    t.record(_ts("0931"), 100000.0, [], {})
    down = t.record(_ts("0932"), 94000.0, [], {})
    up = t.record(_ts("0933"), 103000.0, [], {})
    assert down.daily_loss_rupees == 6000.0
    assert up.daily_loss_rupees == 0.0          # profitable minute is not a loss
    assert t.max_daily_loss_rupees == 6000.0


def test_fails_closed_on_open_position_without_live_price():
    t = EquityTracker()
    with pytest.raises(ValueError, match="No live price"):
        t.record(_ts("0931"), 90000.0,
                 _positions(symbol="TCS", qty=50), _prices("RELIANCE"))


def test_snapshots_roundtrip():
    t = EquityTracker()
    t.record(_ts("0931"), 100000.0, [], {})
    d = t.snapshots[0].to_dict()
    restored = EquitySnapshot.from_dict(d)
    assert restored == t.snapshots[0]


# ── risk-limit evaluation ────────────────────────────────────────────────────

def test_daily_loss_limit_breach_is_detected():
    limits = RiskLimits(max_daily_loss=10_000.0)
    t = EquityTracker()
    t.record(_ts("0931"), 1_000_000.0, [], {})
    t.record(_ts("0932"), 995_000.0, [], {})   # 5k loss → below limit
    t.record(_ts("0933"), 985_000.0, [], {})   # 15k loss → breach
    names = {b.limit_name for b in risk_breaches_for(t.snapshots[-1], limits)}
    assert "daily_loss" in names

    summary = worst_risk(t.snapshots, limits)
    assert summary["breached"] is True
    assert "daily_loss" in summary["breaches"]
    assert summary["daily_loss_max_rupees"] == 15_000.0
    assert summary["limit_daily_loss_rupees"] == limits.max_daily_loss


def test_drawdown_limit_breach_is_detected():
    limits = RiskLimits(max_drawdown_pct=0.10)
    t = EquityTracker()
    t.record(_ts("0931"), 1_000_000.0, [], {})
    t.record(_ts("0932"), 1_050_000.0, [], {})                       # peak
    t.record(_ts("0933"), 920_000.0, [], {})                          # -12.4%
    summary = worst_risk(t.snapshots, limits)
    assert "drawdown" in summary["breaches"]


def test_worst_risk_is_empty_for_flat_unbreached_session():
    t = EquityTracker()
    t.record(_ts("0931"), 1_000_000.0, [], {})
    t.record(_ts("0932"), 1_001_000.0, [], {})
    summary = worst_risk(t.snapshots)
    assert summary["breached"] is False
    assert summary["breaches"] == []


# ── campaign integration ─────────────────────────────────────────────────────

def _cfg(n: int):
    return PaperCampaignConfig(
        symbols=["RELIANCE", "TCS", "INFY"],
        seed=0,
        sessions=[date(2025, 6, 10) + timedelta(days=i) for i in range(n)],
    )


async def _run(n: int, ledger: str, book: str):
    return await run_campaign(n, _cfg(n), ledger_path=ledger, paper_state_path=book)


async def test_campaign_records_intraday_equity_per_day(tmp_path):
    res = await _run(3, str(tmp_path / "l.jsonl"), str(tmp_path / "b.json"))
    assert res.completed
    assert len(res.records) == 3
    for rec in res.records:
        eq = rec["equity"]
        assert eq["start"] > 0 and eq["end"] > 0
        assert eq["peak"] >= eq["end"]
        assert eq["max_drawdown_pct"] is not None
        assert 0.0 <= eq["max_drawdown_pct"] <= 1.0
        assert eq["daily_loss_rupees"] >= 0.0
        assert eq["snapshots"], "no intraday snapshots recorded"
        assert len(eq["snapshots"]) > 100
        assert rec["risk"]["drawdown_max_pct"] <= eq["max_drawdown_pct"] + 1e-9


async def test_campaign_summary_exposes_paper_metrics(tmp_path):
    res = await _run(3, str(tmp_path / "l.jsonl"), str(tmp_path / "b.json"))
    summary = res.summary()
    assert summary["paper_trading_days"] == 3
    assert summary["paper_peak_equity_rupees"] > 0
    assert summary["paper_max_drawdown_pct"] is not None
    assert 0.0 <= summary["paper_max_drawdown_pct"] <= 1.0
    assert summary["paper_sharpe"] is not None
    assert summary["synthetic"] is True


async def test_equity_curve_is_identical_across_resume(tmp_path):
    full = await run_campaign(3, _cfg(3),
                              ledger_path=str(tmp_path / "l.jsonl"),
                              paper_state_path=str(tmp_path / "b_full.json"))

    await run_campaign(2, _cfg(3),
                       ledger_path=str(tmp_path / "l2.jsonl"),
                       paper_state_path=str(tmp_path / "b2.json"))
    resumed = await run_campaign(3, _cfg(3),
                                 ledger_path=str(tmp_path / "l2.jsonl"),
                                 paper_state_path=str(tmp_path / "b2.json"))

    assert full.records[-1]["equity"]["end"] == resumed.records[-1]["equity"]["end"]
    assert full.records[-1]["equity"]["snapshots"] == \
        resumed.records[-1]["equity"]["snapshots"]


async def test_campaign_day_without_trades_still_records_equity(tmp_path):
    res = await _run(3, str(tmp_path / "l.jsonl"), str(tmp_path / "b.json"))
    assert all(rec["equity"]["snapshots"] for rec in res.records)
