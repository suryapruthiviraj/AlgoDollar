"""
Tests for the offline auto-trader replay harness (app/engine/replay.py).

The harness feeds historical 1-minute bars through the REAL production stack —
TickBarAggregator -> TickQuoteSource -> PaperBroker -> IntradayAutoTrader with a
replayed IST clock — so these tests assert the same honesty properties the
Phase A suites assert for live simulation:

* determinism (same seed -> same outcome),
* trade-log/pnl arithmetic consistency against the broker's real fills,
* session gating (square-off clears the book; flat bars never trade),
* explicit synthetic/offline labelling (never mistakable for market data),
* bar-source validation and CSV loading.

Every timestamp is aware IST.  Bars are pushed minute-by-minute exactly as a
live tick feed would present them, once per minute of the 09:15-15:30 window.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from app.engine.replay import (
    _SESSION_CLOSE,
    _SESSION_OPEN,
    REPLAY_WARN,
    load_bars_from_csv,
    run_replay,
    sessions_in,
    synthetic_bar_source,
)

IST = ZoneInfo("Asia/Kolkata")

_TRADING_DAY = date(2025, 6, 10)          # Tuesday, not an NSE holiday


def _at(day: date, hour: int, minute: int) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=IST)


def _next_minute(minute: time) -> time:
    return time(minute.hour + 1, 0) if minute.minute == 59 else time(minute.hour, minute.minute + 1)


def _flat_source(day: date = _TRADING_DAY, px: float = 100.0) -> dict[str, pd.DataFrame]:
    rows = []
    minute = _SESSION_OPEN
    while minute <= _SESSION_CLOSE:
        rows.append({
            "time": datetime.combine(day, minute, tzinfo=IST),
            "open": px, "high": px, "low": px, "close": px, "volume": 200_000,
        })
        minute = _next_minute(minute)
    return {"RELIANCE": pd.DataFrame(rows)}


def _climb_then_flat_by(day: date = _TRADING_DAY) -> dict[str, pd.DataFrame]:
    rows = []
    minute = time(9, 31)
    px = 100.0
    end_climb = time(10, 16)                                 # 09:31..10:16 climb
    while minute <= end_climb:
        rows.append({
            "time": datetime.combine(day, minute, tzinfo=IST),
            "open": round(px, 2), "high": round(px * 1.002, 2),
            "low": round(px * 0.998, 2), "close": round(px * 1.002, 2),
            "volume": 200_000,
        })
        px *= 1.002
        minute = _next_minute(minute)
    while minute <= _SESSION_CLOSE:                          # flat afterwards: only square-off can exit
        rows.append({
            "time": datetime.combine(day, minute, tzinfo=IST),
            "open": round(px, 2), "high": round(px, 2),
            "low": round(px, 2), "close": round(px, 2),
            "volume": 200_000,
        })
        minute = _next_minute(minute)
    return {"RELIANCE": pd.DataFrame(rows)}


async def test_empty_and_unreadable_sources_raise() -> None:
    with pytest.raises(ValueError):
        await run_replay({"RELIANCE": pd.DataFrame()})
    with pytest.raises(ValueError):
        await run_replay({"RELIANCE": pd.DataFrame({"close": [1.0]})})


async def test_synthetic_deterministic() -> None:
    sessions = [_TRADING_DAY, _TRADING_DAY + timedelta(days=1)]
    a = synthetic_bar_source(["TCS"], sessions, seed=7)
    b = synthetic_bar_source(["TCS"], sessions, seed=7)
    c = synthetic_bar_source(["TCS"], sessions, seed=8)
    assert a["TCS"].equals(b["TCS"])
    assert not a["TCS"].equals(c["TCS"])

    ra = await run_replay(b, synthetic=True)
    rb = await run_replay(a, synthetic=True)
    assert ra.summary == rb.summary


async def test_run_replay_structure_and_labelling() -> None:
    days = [_TRADING_DAY + timedelta(days=i) for i in range(3)]
    bars = synthetic_bar_source(["RELIANCE", "TCS"], days, seed=1)
    result = await run_replay(bars, symbols=["RELIANCE", "TCS"], synthetic=True)

    assert result.synthetic is True
    assert result.symbols == ["RELIANCE", "TCS"]
    assert result.summary["days"] == 3
    assert len(result.sessions) == 3
    assert len(result.equity) == 3
    day_iso = {d.isoformat() for d in days}
    for s in result.sessions:
        assert s.cycles > 0
        assert s.session in day_iso
        assert any(e["session"] == s.session for e in result.equity)
    assert result.summary["trades"] == len(result.trades)


async def test_trade_math_matches_broker_fills() -> None:
    days = [_TRADING_DAY + timedelta(days=i) for i in range(5)]
    bars = synthetic_bar_source(["RELIANCE", "TCS", "INFY"], days, seed=3)
    result = await run_replay(bars, synthetic=True)

    for t in result.trades:
        assert t.qty > 0
        assert t.entry_price > 0 and t.exit_price > 0
        assert t.entry_ts < t.exit_ts
        assert t.costs_rupees >= 0
        assert t.net_pnl_rupees == pytest.approx(t.pnl_rupees - t.costs_rupees, abs=0.02)
        assert t.exit_reason in {
            "round_trip", "broker_exit", "stop", "target",
            "vwap_reversal", "square_off", "stop_placement_failed",
        }

    expected = round(sum(t.net_pnl_rupees for t in result.trades), 2)
    assert result.summary["net_pnl_rupees"] == expected
    assert result.summary["trades_per_day"] == pytest.approx(
        len(result.trades) / result.summary["days"], abs=0.01)
    assert 0.0 <= result.summary["win_rate_pct"] <= 100.0
    assert result.summary["max_drawdown_pct"] >= 0.0
    sharpe = result.summary["sharpe_annualized"]
    assert sharpe is None or isinstance(sharpe, float)


async def test_flat_bars_produce_no_trades() -> None:
    result = await run_replay(_flat_source(), synthetic=True)
    assert result.summary["days"] == 1
    assert result.summary["trades"] == 0
    assert result.summary["net_pnl_rupees"] == 0.0
    assert result.sessions[0].cycles > 0


async def test_square_off_sweeps_an_open_position() -> None:
    bars = _climb_then_flat_by()
    result = await run_replay(bars, symbols=["RELIANCE"], synthetic=True)

    session = result.sessions[0]
    assert session.entries >= 1
    assert session.stops_placed >= 1
    assert session.open_at_end == []
    assert len(session.trades) == 1
    trade = session.trades[0]
    assert trade.exit_reason == "square_off"
    assert trade.exit_ts.endswith("15:15:00+05:30")
    assert result.summary["trades"] == 1


def test_build_strategy_config_validates_the_grid_surface() -> None:
    from app.engine.replay import build_strategy_config

    assert build_strategy_config({"min_net_edge": 0.002}) == {"min_net_edge": 0.002}
    with pytest.raises(ValueError, match="unknown strategy override"):
        build_strategy_config({"min_net_edge": 0.002, "_MIN_NET_EDGE": 0.999})
    with pytest.raises(ValueError, match="unknown strategy override"):
        build_strategy_config({"continuation_coef": 0.99})


async def test_strategy_config_gate_is_honest() -> None:
    bars = _climb_then_flat_by()
    base = await run_replay(bars, symbols=["RELIANCE"], synthetic=True)
    gated = await run_replay(
        bars, symbols=["RELIANCE"], synthetic=True,
        strategy_config={"min_net_edge": 0.10},
    )
    assert base.summary["trades"] == 1
    assert gated.summary["trades"] == 0


async def test_csv_loader(tmp_path) -> None:
    csv = tmp_path / "TCS.csv"
    csv.write_text(
        "time,open,high,low,close,volume\n"
        "2025-06-10 09:31:00,100,101,99,100.5,1000\n"
        "2025-06-10 09:32:00,100.5,102,100,101,1200\n"
    )
    bars = load_bars_from_csv(str(tmp_path))
    assert list(bars) == ["TCS"]
    df = bars["TCS"]
    assert df["time"].dt.tz is not None
    assert sessions_in(bars) == [_TRADING_DAY]

    bad = tmp_path / "BAD.csv"
    bad.write_text("time,close\n2025-06-10 09:31:00,1\n")
    with pytest.raises(ValueError, match="missing column"):
        load_bars_from_csv(str(tmp_path), symbols=["BAD"])


def test_sessions_in_is_sorted_unique() -> None:
    days = [_TRADING_DAY + timedelta(days=i) for i in range(4)]
    bars = synthetic_bar_source(["TCS"], days, seed=0)
    assert sessions_in(bars) == sorted(days)


def test_warning_surface() -> None:
    assert "NOT live trading" in REPLAY_WARN
    assert "synthetic" in REPLAY_WARN.lower()
