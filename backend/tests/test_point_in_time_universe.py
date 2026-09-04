"""
Adversarial tests for survivorship and temporal leakage.

Each of the seven properties Phase 4 requires has a test that FAILS if the
property is violated, built on a synthetic panel with known entry and exit dates
so the expected answer is not a matter of opinion.

The construction is deliberately hostile: a future entrant, an early exit, a
mid-period failure, and a name that is never liquid enough all coexist, so a
provider that simply returns everything cannot pass.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.research.universe import (
    MIN_OBSERVATIONS,
    PointInTimeUniverse,
    UniverseUnavailable,
)

# --------------------------------------------------------------------------- #
#  A panel with known lifetimes                                                 #
# --------------------------------------------------------------------------- #

DATES = pd.bdate_range("2015-01-01", "2020-12-31")

#: symbol -> (first tradeable date, last tradeable date or None, liquid?)
LIFETIMES: dict[str, tuple[str, str | None, bool]] = {
    "ALWAYS":     ("2015-01-01", None,         True),   # present throughout
    "LATE":       ("2019-06-03", None,         True),   # lists mid-panel
    "EARLYEXIT":  ("2015-01-01", "2017-03-15", True),   # leaves mid-panel
    "FAILED":     ("2015-01-01", "2018-09-28", True),   # collapses mid-panel
    "ILLIQUID":   ("2015-01-01", None,         False),  # never liquid enough
    "PENNY":      ("2015-01-01", None,         True),   # priced below the floor
}


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    px = pd.DataFrame(index=DATES, columns=list(LIFETIMES), dtype=float)
    vol = pd.DataFrame(index=DATES, columns=list(LIFETIMES), dtype=float)
    rng = np.random.default_rng(11)

    for sym, (start, end, liquid) in LIFETIMES.items():
        mask = DATES >= pd.Timestamp(start)
        if end is not None:
            mask &= DATES <= pd.Timestamp(end)
        n = int(mask.sum())
        base = 2.0 if sym == "PENNY" else 500.0
        prices = base * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        px.loc[mask, sym] = prices
        # Liquid names clear both thresholds comfortably; ILLIQUID clears
        # neither. Volume is set so turnover follows price without being tuned.
        vol.loc[mask, sym] = 2_000_000.0 if liquid else 1_000.0

    return px, vol


@pytest.fixture
def universe() -> PointInTimeUniverse:
    px, vol = build_panel()
    return PointInTimeUniverse(
        px, vol,
        listing_dates={
            "ALWAYS": date(2010, 1, 1),
            "LATE": date(2019, 6, 3),
            "EARLYEXIT": date(2009, 5, 1),
            "FAILED": date(2007, 2, 1),
            "ILLIQUID": date(2008, 1, 1),
            "PENNY": date(2008, 1, 1),
        },
    )


# =========================================================================== #
#  1. A future entrant cannot appear before its entry date                    #
# =========================================================================== #

class TestFutureEntrant:

    def test_a_2019_listing_is_absent_from_2016(self, universe):
        """The defining survivorship failure: tomorrow's winner in yesterday's book."""
        assert "LATE" not in universe.get_members(date(2016, 6, 1))

    def test_it_is_absent_the_day_before_it_lists(self, universe):
        assert "LATE" not in universe.get_members(date(2019, 6, 2))

    def test_it_appears_only_after_its_liquidity_window_fills(self, universe):
        """
        Present later, absent earlier — and never on the listing day itself,
        because the trailing window has no observations yet.
        """
        assert "LATE" not in universe.get_members(date(2019, 6, 3))
        assert "LATE" in universe.get_members(date(2020, 6, 1))

    def test_the_nse_listing_date_wins_when_it_is_later_than_the_first_bar(self):
        """
        A bar before the published listing date is a data error.

        Taking the LATER of the two keeps such a bar from admitting a symbol on
        a day it could not have been bought.
        """
        px, vol = build_panel()
        u = PointInTimeUniverse(
            px, vol, listing_dates={"ALWAYS": date(2018, 1, 1)}
        )
        assert "ALWAYS" not in u.get_members(date(2017, 6, 1))
        assert "ALWAYS" in u.get_members(date(2019, 6, 1))


# =========================================================================== #
#  2. An exited constituent disappears after its exit date                    #
# =========================================================================== #

class TestExit:

    def test_it_is_present_while_trading(self, universe):
        assert "EARLYEXIT" in universe.get_members(date(2016, 6, 1))

    def test_it_is_gone_after_it_stops(self, universe):
        assert "EARLYEXIT" not in universe.get_members(date(2018, 6, 1))

    def test_it_is_gone_the_year_after(self, universe):
        assert "EARLYEXIT" not in universe.get_members(date(2019, 1, 2))

    def test_its_exit_date_is_the_last_observed_bar(self, universe):
        listing = universe.listings["EARLYEXIT"]
        assert listing.stopped_trading
        assert listing.exit_date == date(2017, 3, 15)

    def test_a_name_still_trading_has_no_exit_date(self, universe):
        assert universe.listings["ALWAYS"].exit_date is None


# =========================================================================== #
#  3. A failed constituent remains while it was historically eligible         #
# =========================================================================== #

class TestFailedConstituentRemains:
    """
    The other half of survivorship, and the half usually missed.

    Removing a company that later failed is as wrong as adding one that later
    listed: the strategy must be charged for the chance it would have bought it.
    """

    def test_it_is_present_throughout_its_tradeable_life(self, universe):
        for when in (date(2015, 6, 1), date(2016, 6, 1),
                     date(2017, 6, 1), date(2018, 6, 1)):
            assert "FAILED" in universe.get_members(when), when

    def test_it_disappears_only_after_it_stopped(self, universe):
        assert "FAILED" in universe.get_members(date(2018, 9, 3))
        assert "FAILED" not in universe.get_members(date(2019, 6, 3))

    def test_it_is_not_removed_retroactively(self, universe):
        """Knowing it failed in 2018 must not erase it from 2015."""
        assert "FAILED" in universe.get_members(date(2015, 12, 1))


# =========================================================================== #
#  4. Unknown membership fails closed                                         #
# =========================================================================== #

class TestFailsClosed:

    def test_a_date_before_the_data_raises(self, universe):
        with pytest.raises(UniverseUnavailable):
            universe.get_members(date(2010, 1, 1))

    def test_a_date_after_the_data_raises(self, universe):
        with pytest.raises(UniverseUnavailable):
            universe.get_members(date(2030, 1, 1))

    def test_the_liquidity_warm_up_period_raises_rather_than_guessing(
        self, universe
    ):
        """
        Partial data is not a small version of full data.

        Answering during the warm-up would judge liquidity on a handful of bars
        and admit names on evidence too thin to support them.
        """
        with pytest.raises(UniverseUnavailable, match="UNAVAILABLE"):
            universe.get_members(DATES[MIN_OBSERVATIONS - 5].date())

    def test_an_unanswerable_date_does_not_return_an_empty_universe(
        self, universe
    ):
        """
        Empty reads as "no opportunities today"; the truth is "no answer".

        A backtest cannot tell those apart, so the provider must raise.
        """
        try:
            members = universe.get_members(date(2010, 1, 1))
        except UniverseUnavailable:
            return
        pytest.fail(f"returned {members!r} instead of refusing")

    def test_an_empty_panel_is_refused_at_construction(self):
        with pytest.raises(UniverseUnavailable):
            PointInTimeUniverse(pd.DataFrame(), pd.DataFrame())


# =========================================================================== #
#  5. No current-universe leakage                                             #
# =========================================================================== #

class TestNoCurrentUniverseLeakage:

    def test_membership_is_not_the_final_day_repeated(self, universe):
        """The bug being guarded: today's list served for every date."""
        early = universe.get_members(date(2016, 6, 1))
        late = universe.get_members(date(2020, 6, 1))
        assert early != late, (
            "every date returned the same members, which is what using a "
            "snapshot looks like"
        )

    def test_a_symbol_absent_at_the_end_is_present_in_the_middle(self, universe):
        """Impossible if membership were derived from the final day."""
        assert "FAILED" not in universe.get_members(date(2020, 6, 1))
        assert "FAILED" in universe.get_members(date(2016, 6, 1))

    def test_truncating_the_future_does_not_change_the_past(self):
        """
        The decisive leakage test.

        If membership on date D changes when data after D is removed, then
        computing it used information from after D.
        """
        px, vol = build_panel()
        full = PointInTimeUniverse(px, vol)

        cut = pd.Timestamp("2018-01-02")
        truncated = PointInTimeUniverse(px.loc[:cut], vol.loc[:cut])

        for when in (date(2016, 6, 1), date(2017, 6, 1)):
            assert full.get_members(when) == truncated.get_members(when), (
                f"membership on {when} changed when the future was removed"
            )

    def test_an_illiquid_name_never_enters(self, universe):
        assert "ILLIQUID" not in universe.get_members(date(2018, 6, 1))

    def test_a_sub_floor_price_never_enters(self, universe):
        """A tick is a large fraction of a Rs 2 price; the cost model lies there."""
        assert "PENNY" not in universe.get_members(date(2018, 6, 1))


# =========================================================================== #
#  6. The universe is deterministic                                           #
# =========================================================================== #

class TestDeterminism:

    def test_the_same_date_gives_the_same_answer(self, universe):
        assert universe.get_members(date(2018, 6, 1)) == \
               universe.get_members(date(2018, 6, 1))

    def test_two_providers_over_the_same_data_agree(self):
        px, vol = build_panel()
        a = PointInTimeUniverse(px, vol)
        b = PointInTimeUniverse(px, vol)
        assert a.get_members(date(2018, 6, 1)) == b.get_members(date(2018, 6, 1))
        assert a.fingerprint() == b.fingerprint()

    def test_members_are_sorted(self, universe):
        m = universe.get_members(date(2018, 6, 1))
        assert list(m) == sorted(m)

    def test_a_threshold_change_changes_the_fingerprint(self):
        """A cached validation result must not survive a universe change."""
        px, vol = build_panel()
        a = PointInTimeUniverse(px, vol)
        b = PointInTimeUniverse(px, vol, min_avg_turnover=1e9)
        assert a.fingerprint() != b.fingerprint()


# =========================================================================== #
#  7. Universe membership is logged in the experiment manifest                #
# =========================================================================== #

class TestManifest:

    def test_the_manifest_states_the_definition(self, universe):
        m = universe.manifest()
        assert "liquidity" in m["definition"].lower()
        assert m["fingerprint"]
        assert m["thresholds"]["min_avg_volume"] == 500_000

    def test_it_records_that_this_is_not_index_membership(self, universe):
        assert "NIFTY 500" in universe.manifest()["not_index_membership"]

    def test_it_reports_pool_completeness_including_the_limitation(self, universe):
        pc = universe.manifest()["pool_completeness"]
        assert pc["stopped_trading"] >= 2, (
            "no stopped-trading names, so the universe is still survivor-only"
        )
        assert "limitation" in pc

    def test_it_names_the_period_it_cannot_answer_for(self, universe):
        assert universe.manifest()["unavailable_before"]

    def test_coverage_counts_names_that_stopped_trading(self, universe):
        cov = universe.coverage()
        assert cov.n_stopped_trading == 2   # EARLYEXIT and FAILED
        assert cov.pool_size == len(LIFETIMES)
