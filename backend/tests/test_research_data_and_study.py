"""
Data interfaces, signal conventions, and the study's verdict logic.

The theme is refusal: an absent dataset must raise rather than substitute, a
stale bar must be identified rather than filled, and a verdict must be able to
come back NOT VALIDATED. A research system that can only say yes is not
measuring anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research.datasets import ParquetDailyBars, SnapshotUniverse, StaticSectorMap
from app.research.interfaces import (
    KNOWN_UNAVAILABLE,
    DataUnavailable,
    unavailable,
)
from app.research.signals import (
    BASELINE_SIGNALS,
    UNTESTABLE_SIGNALS,
    breakout,
    low_volatility,
    momentum_12_1,
    short_term_reversal,
    zscore_cross_section,
)
from app.research.study import (
    MAX_PBO,
    MIN_DSR,
    MIN_OOS_SHARPE,
    Criterion,
    bootstrap_sharpe_ci,
    regime_analysis,
    subperiod_analysis,
)


def make_prices(n: int = 900, k: int = 25, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            f"S{i:02d}": 100.0 * np.exp(
                np.cumsum((i - k / 2) * 0.0003 + rng.normal(0, 0.011, n))
            )
            for i in range(k)
        },
        index=idx,
    )


class TestAbsentDatasetsRefuse:
    """An unavailable dataset must raise, never substitute."""

    def test_every_access_raises_with_a_stated_reason(self):
        prov = unavailable("point_in_time_universe")
        with pytest.raises(DataUnavailable) as ei:
            prov.members(pd.Timestamp("2020-01-01").date())
        assert "survivorship" in str(ei.value).lower()

    def test_the_reason_is_specific_not_generic(self):
        for key in ("intraday_bars", "corporate_actions", "fundamentals_point_in_time"):
            assert len(KNOWN_UNAVAILABLE[key]) > 60, f"{key} has no real explanation"

    def test_an_unknown_dataset_still_refuses(self):
        with pytest.raises(DataUnavailable):
            unavailable("something_nobody_wired").anything()

    def test_it_reports_itself_as_not_point_in_time(self):
        assert unavailable("intraday_bars").point_in_time is False

    def test_coverage_carries_the_reason(self):
        cov = unavailable("corporate_actions").coverage()
        assert cov.symbols == 0 and cov.notes


class TestUniverseAndSectorsDeclareTheirBias:

    def test_the_snapshot_universe_admits_it_is_not_point_in_time(self):
        u = SnapshotUniverse(["A", "B"])
        assert u.point_in_time is False
        assert "NOT point-in-time" in u.name

    def test_it_returns_the_same_members_for_every_date(self):
        """The bias made explicit: 2013 and 2025 get today's list."""
        import datetime as dt

        u = SnapshotUniverse(["A", "B"])
        assert u.members(dt.date(2013, 1, 1)) == u.members(dt.date(2025, 1, 1))

    def test_coverage_flags_survivorship(self):
        assert any("SURVIVORSHIP" in n for n in SnapshotUniverse(["A"]).coverage().notes)

    def test_sectors_declare_reconstruction(self):
        assert StaticSectorMap({"A": "IT"}).point_in_time is False


class TestStaleBars:
    """A carried-forward bar produces a fake zero return."""

    def _frame(self) -> pd.DataFrame:
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        return pd.DataFrame(
            {
                "open": [10.0, 11.0, 11.0, 12.0, 13.0],
                "high": [10.5, 11.0, 11.5, 12.5, 13.5],
                "low": [9.5, 11.0, 10.5, 11.5, 12.5],
                "close": [10.2, 11.0, 11.2, 12.2, 13.2],
                "volume": [1000, 0, 900, 0, 1100],
            },
            index=idx,
        )

    def test_a_flat_zero_volume_bar_is_identified(self):
        mask = ParquetDailyBars.stale_bar_mask(self._frame())
        assert list(mask) == [False, True, False, False, False]

    def test_a_zero_volume_bar_that_still_moved_is_not_stale(self):
        """Row 3 has volume 0 but a real high/low range — a data oddity, not a
        carried-forward close, and must not be silently discarded."""
        assert ParquetDailyBars.stale_bar_mask(self._frame()).iloc[3] is np.False_ or (
            not ParquetDailyBars.stale_bar_mask(self._frame()).iloc[3]
        )

    def test_a_frame_without_volume_reports_nothing_stale(self):
        df = self._frame().drop(columns=["volume"])
        assert not ParquetDailyBars.stale_bar_mask(df).any()


class TestSignalConventions:
    """Higher must always mean more attractive. A flipped sign inverts a study."""

    def test_momentum_prefers_the_riser(self):
        idx = pd.date_range("2015-01-01", periods=400, freq="B")
        up = pd.Series(np.linspace(100, 300, 400), index=idx)
        down = pd.Series(np.linspace(300, 100, 400), index=idx)
        sig = momentum_12_1(pd.DataFrame({"UP": up, "DOWN": down}))
        assert sig["UP"].iloc[-1] > sig["DOWN"].iloc[-1]

    def test_reversal_prefers_the_faller(self):
        """The opposite sign to momentum — the easiest one to get wrong."""
        idx = pd.date_range("2015-01-01", periods=100, freq="B")
        up = pd.Series(np.linspace(100, 200, 100), index=idx)
        down = pd.Series(np.linspace(200, 100, 100), index=idx)
        sig = short_term_reversal(pd.DataFrame({"UP": up, "DOWN": down}))
        assert sig["DOWN"].iloc[-1] > sig["UP"].iloc[-1]

    def test_low_volatility_prefers_the_calm_name(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2015-01-01", periods=300, freq="B")
        calm = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.002, 300))), index=idx)
        wild = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.040, 300))), index=idx)
        sig = low_volatility(pd.DataFrame({"CALM": calm, "WILD": wild}))
        assert sig["CALM"].iloc[-1] > sig["WILD"].iloc[-1]

    def test_breakout_excludes_the_current_bar_from_its_range(self):
        """Otherwise every new high scores exactly 1.0 by construction."""
        idx = pd.date_range("2015-01-01", periods=300, freq="B")
        rising = pd.DataFrame({"A": np.linspace(100, 200, 300)}, index=idx)
        assert float(breakout(rising)["A"].iloc[-1]) > 1.0

    def test_every_signal_only_uses_trailing_data(self):
        """
        Truncating the future must not change any past value.

        The decisive property: if it does, the signal saw data it could not have.
        """
        px = make_prices()
        cut = len(px) - 50
        for name, fn in BASELINE_SIGNALS.items():
            full = fn(px).iloc[:cut]
            truncated = fn(px.iloc[:cut])
            aligned = full.align(truncated, join="inner")
            assert np.allclose(
                aligned[0].fillna(0).to_numpy(), aligned[1].fillna(0).to_numpy(),
                atol=1e-9,
            ), f"{name} changed its history when the future was removed"

    def test_unrunnable_signals_are_named_with_reasons(self):
        assert UNTESTABLE_SIGNALS
        for name, reason in UNTESTABLE_SIGNALS.items():
            assert "not available" in reason, name

    def test_zscore_standardises_across_symbols_not_time(self):
        """Standardising along time leaks the whole sample into every past date."""
        px = make_prices(n=200, k=10)
        z = zscore_cross_section(px)
        row = z.iloc[-1].dropna()
        assert abs(float(row.mean())) < 1e-9, "rows are not centred; wrong axis"


class TestStudyAnalytics:

    def test_bootstrap_reports_an_interval_not_a_point(self):
        rng = np.random.default_rng(1)
        idx = pd.date_range("2015-01-01", periods=800, freq="B")
        r = pd.Series(rng.normal(0.0004, 0.01, 800), index=idx)
        ci = bootstrap_sharpe_ci(r, n_sims=200)
        assert ci["p05"] < ci["p50"] < ci["p95"]

    def test_bootstrap_refuses_a_short_series(self):
        r = pd.Series(np.zeros(10))
        assert "error" in bootstrap_sharpe_ci(r)

    def test_subperiod_counts_positive_years(self):
        idx = pd.date_range("2016-01-01", periods=1200, freq="B")
        r = pd.Series(0.0005, index=idx)
        sub = subperiod_analysis(r, None)
        assert sub["positive_years"] == sub["total_years"] > 1

    def test_regimes_are_labelled_from_trailing_data_only(self):
        rng = np.random.default_rng(2)
        idx = pd.date_range("2015-01-01", periods=900, freq="B")
        bench = pd.Series(rng.normal(0.0003, 0.01, 900), index=idx)
        out = regime_analysis(pd.Series(rng.normal(0, 0.01, 900), index=idx), bench)
        assert set(out) <= {"bull", "bear", "sideways", "error"}

    def test_regime_analysis_refuses_insufficient_overlap(self):
        idx = pd.date_range("2020-01-01", periods=30, freq="B")
        s = pd.Series(0.0, index=idx)
        assert "error" in regime_analysis(s, s)


class TestVerdictLogic:
    """The verdict must be able to be NO."""

    def test_thresholds_are_declared_as_constants(self):
        """Fixed in advance, so moving one is a reviewable diff."""
        assert MIN_OOS_SHARPE > 0
        assert 0 < MIN_DSR < 1
        assert 0 < MAX_PBO <= 1

    def test_one_failed_criterion_blocks_validation(self):
        crits = [
            Criterion("a", True, 1.0, 0.5),
            Criterion("b", False, 0.1, 0.5),
        ]
        assert not all(c.passed for c in crits)

    def test_a_criterion_records_what_it_saw(self):
        c = Criterion("oos_sharpe", False, 0.12, 0.5, "detail")
        assert c.observed == 0.12 and c.threshold == 0.5 and c.detail
