"""
longterm.py — Quantitative long-term stock screening strategy.

WARNING: FUNDAMENTAL DATA DEPENDENCY
-------------------------------------
This strategy requires fundamental financial data (ROE, ROCE, revenue growth,
earnings CAGR, FCF, P/E, P/B, EV/EBITDA, etc.) which is NOT provided by
the broker's market data feed.  You must integrate one of:
  - NSE Corporate Action / Financial Results API
  - Screener.in API (www.screener.in/api/)
  - Tickertape / Trendlyne API
  - A paid data vendor (Bloomberg, Refinitiv, etc.)

Until real fundamental data is wired in, this module operates with a MOCK
data provider that returns plausible but SYNTHETIC fundamentals.  Every
method that touches fundamentals emits a prominent warning log.

Composite scoring
-----------------
score = 0.30 * quality + 0.25 * growth + 0.20 * valuation
       + 0.15 * momentum + 0.10 * risk

The composite is an average of five components that are each centred at 50
with capped z-scores, so averaging SHRINKS its dispersion: empirically
mean ~49, sd ~6, max ~68 over 1500 draws.  Absolute cut-offs on that scale are
therefore not interchangeable with intuition about "a score out of 100" —
a 65 buy threshold admitted 0.2% of names while a 40 sell threshold fired on
6.5%, i.e. a structurally one-way liquidating book.  Thresholds are now
CROSS-SECTIONAL PERCENTILES of the universe actually being scored:

BUY    if composite is in the top decile of the current universe
       AND valuation_score is not in the "expensive" tail.
SELL   if composite falls into the bottom quintile of the current universe.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.strategies.base import (
    DELIVERY_ROUND_TRIP_COST, MAX_GROSS_EXPOSURE,
    BaseStrategy, Signal, SignalDirection,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Composite score weights
# ---------------------------------------------------------------------------
_W_QUALITY    = 0.30
_W_GROWTH     = 0.25
_W_VALUATION  = 0.20
_W_MOMENTUM   = 0.15
_W_RISK       = 0.10

# Cross-sectional percentile thresholds (see module docstring).
_BUY_PERCENTILE   = 90.0   # buy the top decile of the scored universe
_SELL_PERCENTILE  = 20.0   # sell out of the bottom quintile

# Absolute fallbacks, used ONLY when there is no cross-section to rank
# against (e.g. should_exit() on a single position with no universe context).
_BUY_THRESHOLD    = 65.0
_SELL_THRESHOLD   = 40.0

# A percentile cut needs a real cross-section behind it.  Below this many
# scored names the strategy emits no BUYs rather than ranking noise.
_MIN_UNIVERSE_FOR_PERCENTILE = 20

_MAX_EXPENSIVE_VALUATION_SCORE = 20.0  # reject if valuation_score < 20 (top decile expensive)

# Expected 1-year excess return attributed to a top-decile composite.  Used to
# turn the (dimensionless) composite percentile into Signal.expected_return so
# that the number carries the same units as holding_period_days=252.
_EXPECTED_ANNUAL_SPREAD = 0.08

# Round-trip cost for a CNC (delivery) trade — see base.py.
_COST_ESTIMATE_DELIVERY = DELIVERY_ROUND_TRIP_COST

_LONGTERM_HOLDING_DAYS = 252


class MockDataInLiveModeError(RuntimeError):
    """
    Raised when a strategy backed by MOCK/synthetic fundamentals is asked to
    act outside paper mode.

    Synthetic fundamentals must never be able to drive a real order.  This is
    a hard runtime guard rather than a comment because the previous
    ``metadata={"data_source": "MOCK"}`` marker had no consumer anywhere in
    the system.
    """

# ---------------------------------------------------------------------------
# Mock fundamental data — replace with real data provider in production
# ---------------------------------------------------------------------------

class _MockFundamentalProvider:
    """
    Synthetic fundamental data for development and testing.

    ALL VALUES ARE FAKE.  This provider exists solely to allow integration
    tests and UI development without a paid data feed.

    NEVER use mock data for real capital allocation decisions — the
    IS_MOCK flag below is checked at runtime and blocks non-paper use.

    DETERMINISM
    -----------
    The generator is seeded PER SYMBOL from a stable hash of the symbol, so
    get_fundamentals("RELIANCE") returns the same dict every time, in every
    process.  The previous class-level RNG advanced on every call, so the same
    stock scored 39.3 / 50.6 / 37.7 on three consecutive calls and straddled
    the sell threshold: buy/sell decisions were coin flips.
    """

    #: Consumed by LongtermStrategy._assert_data_source_safe().
    IS_MOCK = True
    DATA_SOURCE = "MOCK"

    _SEED_SALT = 42

    @staticmethod
    def _seed_for(key: str) -> int:
        """
        Stable 64-bit seed derived from a key.

        Uses blake2b rather than the builtin hash(), which is randomised per
        process by PYTHONHASHSEED and would silently break reproducibility
        across restarts.
        """
        digest = hashlib.blake2b(
            f"{_MockFundamentalProvider._SEED_SALT}:{key}".encode("utf-8"),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "big")

    @classmethod
    def get_fundamentals(cls, symbol: str) -> Dict[str, float]:
        warnings.warn(
            f"[MOCK DATA] Returning synthetic fundamentals for {symbol}. "
            "Do NOT use for real trading.",
            UserWarning,
            stacklevel=4,
        )
        rng = np.random.default_rng(cls._seed_for(f"fundamentals:{symbol}"))
        return {
            # Quality
            "roe":                  rng.uniform(5, 35),        # %
            "roce":                 rng.uniform(8, 40),        # %
            "margin_std_pct":       rng.uniform(1, 15),        # std of net margin
            "margin_mean_pct":      rng.uniform(5, 25),        # mean net margin
            # Growth
            "revenue_cagr_3yr":     rng.uniform(-5, 30),       # %
            "earnings_cagr_3yr":    rng.uniform(-10, 40),      # %
            "fcf_cagr_3yr":         rng.uniform(-15, 35),      # %
            "fwd_growth_estimate":  rng.uniform(0, 25),        # %
            # Valuation
            "pe_ratio":             rng.uniform(8, 80),
            "pb_ratio":             rng.uniform(0.5, 15),
            "ev_ebitda":            rng.uniform(4, 40),
            "fcf_yield_pct":        rng.uniform(0, 8),         # %
            "pb_percentile":        rng.uniform(0, 100),       # own history
            # Risk
            "debt_to_equity":       rng.uniform(0, 3),
            "max_drawdown_1yr":     rng.uniform(0, 50),        # %
            # Market data (duplicated here for convenience)
            "market_cap_crore":     rng.uniform(500, 200_000),
            "sector":               "Unknown",
        }

    @classmethod
    def get_sector_medians(cls, sector: str) -> Dict[str, float]:
        """Synthetic sector median multiples — deterministic per sector."""
        warnings.warn(
            f"[MOCK DATA] Returning synthetic sector medians for {sector}.",
            UserWarning,
            stacklevel=4,
        )
        rng = np.random.default_rng(cls._seed_for(f"sector:{sector}"))
        return {
            "pe_median":      rng.uniform(15, 35),
            "roe_median":     rng.uniform(10, 25),
            "roce_median":    rng.uniform(12, 28),
            "ev_ebitda_median": rng.uniform(8, 20),
        }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _safe_zscore(value: float, mean: float, std: float, cap: float = 3.0) -> float:
    """Z-score capped at ±cap."""
    if std < 1e-6:
        return 0.0
    return float(np.clip((value - mean) / std, -cap, cap))


def _map_to_0_100(z: float, cap: float = 3.0) -> float:
    """Map a z-score in [-cap, cap] to [0, 100]."""
    return float(np.clip((z + cap) / (2 * cap) * 100, 0, 100))


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class LongtermStrategy(BaseStrategy):
    """
    Quantitative long-term stock screening and portfolio construction.

    Holdings are intended to be held months to years.  Signals represent
    screening recommendations, not high-frequency trades.  The strategy
    uses a composite multi-factor score:

      Composite = 0.30 * Quality + 0.25 * Growth + 0.20 * Valuation
                + 0.15 * Momentum + 0.10 * Risk

    NOTE: Fundamental data (quality, growth, valuation) must come from a
    real data provider.  The built-in fallback uses MOCK (synthetic) data.
    """

    name = "LongtermQuality"
    holding_period = "months_to_years"

    def __init__(
        self,
        fundamental_provider=None,
        paper_mode: bool = True,
        max_positions: int = 20,
        buy_threshold: float = _BUY_THRESHOLD,
        sell_threshold: float = _SELL_THRESHOLD,
        buy_percentile: float = _BUY_PERCENTILE,
        sell_percentile: float = _SELL_PERCENTILE,
    ):
        """
        Parameters
        ----------
        fundamental_provider
            Object with .get_fundamentals(symbol) → dict and
            .get_sector_medians(sector) → dict.  If None, uses the MOCK
            provider — which then makes the strategy refuse to run outside
            paper mode (see _assert_data_source_safe).
        paper_mode : bool
        max_positions : int
        buy_percentile : float
            Cross-sectional percentile of the CURRENT universe above which to
            BUY (90 = top decile).  This is the operative buy rule.
        sell_percentile : float
            Cross-sectional percentile below which a HELD name is sold
            (20 = bottom quintile).  This is the operative sell rule.
        buy_threshold, sell_threshold : float
            Absolute composite fallbacks, used only when no cross-section is
            available to rank against.  Retained for API compatibility; the
            percentile rules take precedence in generate_signals().
        """
        super().__init__(paper_mode=paper_mode)
        if fundamental_provider is None:
            logger.warning(
                "LongtermStrategy: no fundamental_provider supplied; "
                "using MOCK data.  Do NOT use for real trading."
            )
            fundamental_provider = _MockFundamentalProvider()
        self._fund = fundamental_provider
        self.max_positions = max_positions
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.buy_percentile = float(buy_percentile)
        self.sell_percentile = float(sell_percentile)
        #: Absolute composite cut implied by sell_percentile at the last
        #: generate_signals() call; consumed by should_exit().
        self._last_sell_cut: Optional[float] = None

        # Fail fast: a mock-backed strategy may not even be CONSTRUCTED live.
        self._assert_data_source_safe("construction")

    # ------------------------------------------------------------------
    # Data-source guard
    # ------------------------------------------------------------------

    @property
    def uses_mock_data(self) -> bool:
        """True if the wired fundamental provider is synthetic."""
        return bool(getattr(self._fund, "IS_MOCK", False))

    @property
    def data_source(self) -> str:
        """Provider-declared data source, e.g. "MOCK" or "screener.in"."""
        return str(getattr(self._fund, "DATA_SOURCE", "UNKNOWN"))

    def _assert_data_source_safe(self, context: str) -> None:
        """
        Refuse to operate on synthetic fundamentals outside paper mode.

        Called on construction AND on every signal-generating / sizing entry
        point, because paper_mode is a mutable attribute — checking once at
        construction would be trivially defeated by flipping it afterwards.
        """
        if self.uses_mock_data and not self.paper_mode:
            raise MockDataInLiveModeError(
                f"LongtermStrategy refused at {context}: fundamental provider "
                f"{type(self._fund).__name__} is MOCK (synthetic) data and "
                "paper_mode is False.  Synthetic fundamentals must never size "
                "or trigger a real order.  Wire a real fundamental provider, "
                "or run with paper_mode=True."
            )

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        universe: List[str],
        features_df: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame],
        existing_positions: Optional[Dict[str, dict]] = None,
        current_date: Optional[datetime] = None,
        regime: Optional[str] = None,
    ) -> List[Signal]:
        """
        Screen the universe and return BUY signals.

        Also returns SELL signals for existing positions whose score has
        deteriorated below the sell threshold.

        Parameters
        ----------
        universe : list[str]
        features_df : pd.DataFrame (daily features, for momentum computation).
        market_data : dict[str, pd.DataFrame] (daily OHLCV per symbol).
        existing_positions : dict[str, dict] (currently held positions).
        current_date : datetime
        regime : str (optional, for context logging).

        Returns
        -------
        list[Signal]
        """
        # Hard guard: synthetic fundamentals may never reach a live order.
        self._assert_data_source_safe("generate_signals")

        if not self._is_operational():
            logger.info("LongtermStrategy is %s; no signals.", self.health)
            return []

        now = current_date or datetime.now()
        existing_syms = set((existing_positions or {}).keys())
        signals: List[Signal] = []

        # Score all universe symbols
        all_scores: List[Tuple[str, float, dict]] = []
        for sym in universe:
            try:
                score_dict = self._compute_composite(sym, features_df, market_data)
                all_scores.append((sym, score_dict["composite"], score_dict))
            except Exception as exc:
                logger.debug("Scoring failed for %s: %s", sym, exc)

        if not all_scores:
            return []

        # ---- Cross-sectional cuts (defect: 65/40 were absolute magic numbers
        # on a shrunken scale — P(buy)=0.2%, P(sell)=6.5%, a one-way book) ----
        buy_cut, sell_cut = self._threshold_cuts(all_scores)
        self._last_sell_cut = sell_cut

        buy_signals: List[Signal] = []

        # --- BUY signals for non-held stocks ---
        available_slots = self.max_positions - len(existing_syms)
        if available_slots <= 0:
            logger.info(
                "Longterm at position limit (%d held / %d max): no BUY signals.",
                len(existing_syms), self.max_positions,
            )
        elif buy_cut is None:
            logger.info(
                "Only %d names scored (< %d): no cross-section to rank "
                "against, so no BUY signals.",
                len(all_scores), _MIN_UNIVERSE_FOR_PERCENTILE,
            )
        else:
            buy_candidates = [
                (sym, score, sd)
                for sym, score, sd in all_scores
                if sym not in existing_syms
                and score >= buy_cut
                and sd.get("valuation_score", 50) >= _MAX_EXPENSIVE_VALUATION_SCORE
            ]
            buy_candidates.sort(key=lambda x: x[1], reverse=True)

            # NOTE: available_slots is guaranteed > 0 here.  Without that
            # guard, buy_candidates[:-10] (Python's negative slice) returned
            # items instead of [] and emitted BUYs while over the limit.
            comps = np.array([s for _, s, _ in all_scores], dtype=float)
            for sym, comp_score, score_dict in buy_candidates[:available_slots]:
                ohlcv = market_data.get(sym)
                if ohlcv is None or ohlcv.empty:
                    continue
                last_price = ohlcv["close"].iloc[-1]
                atr = self._compute_atr(ohlcv)
                stop_loss_pct = max((2.5 * atr) / last_price, 0.05) if last_price > 0 else 0.05
                target_pct    = stop_loss_pct * 3.0  # 3:1 for long-term

                vol = self._compute_realized_vol(ohlcv, 63)
                pct_rank = float((comps <= comp_score).mean())
                expected_ret = self._expected_annual_return(pct_rank)
                edge = expected_ret - _COST_ESTIMATE_DELIVERY
                if edge <= 0:
                    continue

                signal = Signal(
                    symbol=sym,
                    direction=SignalDirection.LONG,
                    strategy_name=self.name,
                    timestamp=now,
                    signal_date=now,
                    edge_score=edge,
                    expected_return=expected_ret,
                    expected_return_std=max(vol, 1e-6),
                    stop_loss_pct=stop_loss_pct,
                    target_pct=target_pct,
                    holding_period_days=_LONGTERM_HOLDING_DAYS,
                    feature_snapshot={
                        "composite_score": comp_score,
                        "composite_pct_rank": pct_rank,
                        "quality_score":    score_dict.get("quality_score", 0),
                        "growth_score":     score_dict.get("growth_score", 0),
                        "valuation_score":  score_dict.get("valuation_score", 0),
                        "momentum_score":   score_dict.get("momentum_score", 0),
                        "risk_score":       score_dict.get("risk_score", 0),
                        "realized_vol_63d": vol,
                    },
                    metadata={
                        "regime": regime or "UNKNOWN",
                        "data_source": self.data_source,
                        "return_units": (
                            f"simple_return_over_{_LONGTERM_HOLDING_DAYS}"
                            "_trading_days"
                        ),
                        "buy_cut": buy_cut,
                        "buy_percentile": self.buy_percentile,
                    },
                )
                buy_signals.append(signal)

            # Portfolio budget across the emitted BUY set (20 x 10% cap was
            # 200% gross with no cross-signal normalisation).
            budget = self._entry_budget(self.max_positions, len(existing_syms))
            self._stamp_target_weights(
                buy_signals,
                [self._raw_target_weight(s) for s in buy_signals],
                budget=budget,
            )
            signals.extend(buy_signals)

        # --- SELL signals for held positions ---
        for sym in existing_syms:
            try:
                score_dict = next(
                    sd for s, _, sd in all_scores if s == sym
                )
                comp = score_dict["composite"]
            except StopIteration:
                # Held name not scoreable this cycle — treat as deteriorated.
                logger.warning("Held name %s could not be scored; flagging exit.", sym)
                comp = float("-inf")

            effective_sell_cut = (
                sell_cut if sell_cut is not None else self.sell_threshold
            )
            if comp < effective_sell_cut:
                logger.info(
                    "SELL signal for %s: composite=%.1f < cut=%.1f (p%.0f of universe)",
                    sym, comp, effective_sell_cut, self.sell_percentile,
                )
                ohlcv = market_data.get(sym)
                last_price = 0.0
                if ohlcv is not None and not ohlcv.empty:
                    last_price = ohlcv["close"].iloc[-1]
                signals.append(Signal(
                    symbol=sym,
                    direction=SignalDirection.EXIT,
                    strategy_name=self.name,
                    timestamp=now,
                    signal_date=now,
                    edge_score=1.0,  # exit is always actionable
                    expected_return=0.0,
                    expected_return_std=0.0,
                    stop_loss_pct=0.0,
                    target_pct=0.0,
                    holding_period_days=0,
                    feature_snapshot={"composite_score": comp},
                    metadata={
                        "reason": "composite_below_cross_sectional_cut",
                        "sell_cut": effective_sell_cut,
                        "sell_percentile": self.sell_percentile,
                        "data_source": self.data_source,
                    },
                ))

        logger.info(
            "LongtermStrategy: %d universe, %d buy signals, %d sell signals "
            "(buy_cut=%s, sell_cut=%s)",
            len(universe),
            sum(1 for s in signals if s.direction == SignalDirection.LONG),
            sum(1 for s in signals if s.direction == SignalDirection.EXIT),
            f"{buy_cut:.1f}" if buy_cut is not None else "n/a",
            f"{sell_cut:.1f}" if sell_cut is not None else "n/a",
        )
        return signals

    # ------------------------------------------------------------------
    # Cross-sectional thresholds
    # ------------------------------------------------------------------

    def _threshold_cuts(
        self,
        all_scores: List[Tuple[str, float, dict]],
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Convert the buy/sell PERCENTILES into absolute composite cuts using
        the realised cross-sectional distribution of this scoring run.

        Returns (buy_cut, sell_cut).  Both are None when the universe is too
        small for a percentile to mean anything, in which case the caller
        falls back to no BUYs and the absolute sell threshold.
        """
        comps = np.array([s for _, s, _ in all_scores], dtype=float)
        comps = comps[np.isfinite(comps)]
        if comps.size < _MIN_UNIVERSE_FOR_PERCENTILE:
            return None, None
        buy_cut = float(np.percentile(comps, self.buy_percentile))
        sell_cut = float(np.percentile(comps, self.sell_percentile))
        return buy_cut, sell_cut

    @staticmethod
    def _expected_annual_return(pct_rank: float) -> float:
        """
        Map a composite PERCENTILE RANK (0-1) to a 1-year expected excess
        return, so that Signal.expected_return carries the same units as
        holding_period_days=252.

        Linear in rank around the median:
            E[r_1y] = _EXPECTED_ANNUAL_SPREAD * (2 * rank - 1)
        A top-decile name (rank ~0.95) is therefore worth ~7.2% over a year,
        the median name 0%.  This replaces `(composite - 50)/100`, which read
        a shrunken, uncalibrated 0-100 scale as if it were a percentage.
        """
        rank = float(np.clip(pct_rank, 0.0, 1.0))
        return float(_EXPECTED_ANNUAL_SPREAD * (2.0 * rank - 1.0))

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        signal: Signal,
        available_capital: float,
        risk_engine,
    ) -> float:
        """
        Long-term position sizing: volatility-scaled equal weight, bounded by
        the portfolio budget.

        Uses the BATCH-NORMALISED weight stamped by generate_signals() so the
        emitted set can never intend more than 100% of the sleeve (20 names at
        the 10% per-position cap was 200% gross).  Un-stamped signals fall
        back to the raw weight, hard-capped at MAX_GROSS_EXPOSURE /
        max_positions.

        Raises
        ------
        MockDataInLiveModeError : mock fundamentals outside paper mode.
        RiskEngineError         : risk engine unavailable or raising.
        """
        self._assert_data_source_safe("calculate_position_size")

        if available_capital <= 0 or not signal.is_valid():
            return 0.0

        multiplier = self._position_size_multiplier()
        if multiplier == 0.0:
            return 0.0

        stamped = signal.metadata.get("target_weight")
        if stamped is not None:
            target_weight = float(stamped)
        else:
            target_weight = min(
                self._raw_target_weight(signal),
                MAX_GROSS_EXPOSURE / max(1, self.max_positions),
            )

        size = available_capital * max(0.0, target_weight) * multiplier

        if not self._risk_engine_approves(risk_engine, signal.symbol, size, "longterm"):
            return 0.0

        return max(0.0, size)

    def _raw_target_weight(self, signal: Signal) -> float:
        """Pre-normalisation vol-scaled weight (an intent, not an allocation)."""
        realized_vol = signal.feature_snapshot.get("realized_vol_63d", 0.25)
        if realized_vol is None or not np.isfinite(realized_vol) or realized_vol <= 0:
            realized_vol = 0.25

        target_vol = 0.15  # 15% portfolio vol target
        target_weight = target_vol / (realized_vol * self.max_positions ** 0.5)
        target_weight = min(target_weight, 1.0 / self.max_positions * 2)
        return max(target_weight, 0.02)

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def should_exit(
        self,
        position: dict,
        current_data: Dict[str, Any],
    ) -> bool:
        """
        Exit if:
        1. Stop-loss hit (wide stop for long-term: 3× ATR).
        2. Fundamental deterioration, measured cross-sectionally where
           possible:
             - current_data["composite_percentile"] (0-100) below
               sell_percentile, else
             - current_data["composite_score"] below the cut implied by the
               last generate_signals() run, else
             - the absolute sell_threshold fallback.
        3. Holding period > 2 years (force review).
        """
        current_price = float(current_data.get("price", 0))

        if current_price > 0 and self._hit_stop(position, current_price):
            return True

        # Fundamental deterioration check (if caller provides current score)
        current_pct = current_data.get("composite_percentile")
        if current_pct is not None and float(current_pct) < self.sell_percentile:
            logger.info(
                "Fundamental deterioration exit for %s (percentile=%.1f < p%.0f).",
                position["symbol"], float(current_pct), self.sell_percentile,
            )
            return True

        current_composite = current_data.get("composite_score")
        if current_composite is not None:
            cut = self._last_sell_cut
            if cut is None:
                cut = self.sell_threshold
            if float(current_composite) < cut:
                logger.info(
                    "Fundamental deterioration exit for %s (composite=%.1f < %.1f).",
                    position["symbol"], float(current_composite), cut,
                )
                return True

        # Time limit: 2 years
        entry_date = position.get("entry_date")
        current_date = current_data.get("date")
        if entry_date and current_date:
            days_held = (current_date - entry_date).days
            if days_held > 730:
                logger.info(
                    "Long-term holding >730 days for %s — force review.",
                    position["symbol"],
                )
                return True

        return False

    # ------------------------------------------------------------------
    # Scoring components
    # ------------------------------------------------------------------

    def _compute_composite(
        self,
        symbol: str,
        features_df: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Compute the composite multi-factor score for one symbol.

        Returns dict with keys: composite, quality_score, growth_score,
        valuation_score, momentum_score, risk_score.
        """
        fund = self._fund.get_fundamentals(symbol)
        sector = fund.get("sector", "Unknown")
        sector_med = self._fund.get_sector_medians(sector)

        quality  = self._quality_score(fund, sector_med)
        growth   = self._growth_score(fund)
        valuation = self._valuation_score(fund, sector_med)
        momentum = self._momentum_score(symbol, features_df, market_data)
        risk     = self._risk_score(fund, market_data.get(symbol))

        composite = (
            _W_QUALITY   * quality
            + _W_GROWTH  * growth
            + _W_VALUATION * valuation
            + _W_MOMENTUM * momentum
            + _W_RISK    * risk
        )
        return {
            "composite":       composite,
            "quality_score":   quality,
            "growth_score":    growth,
            "valuation_score": valuation,
            "momentum_score":  momentum,
            "risk_score":      risk,
        }

    @staticmethod
    def _quality_score(fund: dict, sector_med: dict) -> float:
        """
        Quality score (0–100).

        Components:
        - ROE vs sector median (z-score)
        - ROCE vs sector median
        - Profit margin consistency (lower std = better)
        """
        roe  = fund.get("roe", 0)
        roce = fund.get("roce", 0)
        margin_std  = fund.get("margin_std_pct", 10)
        margin_mean = fund.get("margin_mean_pct", 10)

        roe_med  = sector_med.get("roe_median", 15)
        roce_med = sector_med.get("roce_median", 18)

        roe_z  = _safe_zscore(roe,  roe_med,  roe_med  * 0.5)
        roce_z = _safe_zscore(roce, roce_med, roce_med * 0.5)

        # Margin consistency: lower coefficient of variation = better
        cv = margin_std / (abs(margin_mean) + 1e-6)
        consistency_score = _map_to_0_100(-_safe_zscore(cv, 0.5, 0.3))

        score = (
            0.40 * _map_to_0_100(roe_z)
            + 0.35 * _map_to_0_100(roce_z)
            + 0.25 * consistency_score
        )
        return float(np.clip(score, 0, 100))

    @staticmethod
    def _growth_score(fund: dict) -> float:
        """
        Growth score (0–100).

        Components:
        - 3yr revenue CAGR
        - 3yr earnings CAGR
        - 3yr FCF CAGR
        - Forward growth estimate (weighted lower, less certain)
        """
        rev_cagr = fund.get("revenue_cagr_3yr", 0)
        eps_cagr = fund.get("earnings_cagr_3yr", 0)
        fcf_cagr = fund.get("fcf_cagr_3yr", 0)
        fwd      = fund.get("fwd_growth_estimate", 0)

        # Map each CAGR to 0-100 (0% → 50, 20% → ~80, -10% → ~30)
        def cagr_to_score(cagr: float) -> float:
            z = _safe_zscore(cagr, 10.0, 15.0)  # mean ~10%, std ~15%
            return _map_to_0_100(z)

        score = (
            0.30 * cagr_to_score(rev_cagr)
            + 0.35 * cagr_to_score(eps_cagr)
            + 0.25 * cagr_to_score(fcf_cagr)
            + 0.10 * cagr_to_score(fwd)
        )
        return float(np.clip(score, 0, 100))

    @staticmethod
    def _valuation_score(fund: dict, sector_med: dict) -> float:
        """
        Valuation score (0–100).  Cheaper = higher score (inverse).

        Components:
        - P/E vs sector median (z-score, inverted)
        - P/B vs own history percentile (inverted: lower percentile = cheaper)
        - EV/EBITDA vs sector median (inverted)
        - FCF yield (higher = better)
        """
        pe       = fund.get("pe_ratio", 25)
        pb_pct   = fund.get("pb_percentile", 50)  # 0-100, own history
        ev_ebitda = fund.get("ev_ebitda", 12)
        fcf_yield = fund.get("fcf_yield_pct", 2)

        pe_med = sector_med.get("pe_median", 22)
        ev_med = sector_med.get("ev_ebitda_median", 12)

        # Inverted: cheaper stock → higher score
        pe_z  = -_safe_zscore(pe, pe_med, pe_med * 0.4)
        ev_z  = -_safe_zscore(ev_ebitda, ev_med, ev_med * 0.4)
        pb_score = 100 - pb_pct  # lower percentile = cheaper
        fcf_score = _map_to_0_100(_safe_zscore(fcf_yield, 2.0, 2.0))

        score = (
            0.35 * _map_to_0_100(pe_z)
            + 0.25 * pb_score
            + 0.25 * _map_to_0_100(ev_z)
            + 0.15 * fcf_score
        )
        return float(np.clip(score, 0, 100))

    @staticmethod
    def _momentum_score(
        symbol: str,
        features_df: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame],
    ) -> float:
        """
        Momentum score (0–100).

        12-1 month momentum, relative strength vs NIFTY, 52-week high proximity.
        Uses market_data for computation to avoid features_df unavailability.
        """
        ohlcv = market_data.get(symbol)
        if ohlcv is None or len(ohlcv) < 252:
            return 50.0

        px = ohlcv["close"]

        # 12-1 momentum
        if len(px) >= 252:
            mom_12_1 = float(np.log(px.iloc[-21] / px.iloc[-252]))
        else:
            mom_12_1 = 0.0

        # 52-week high proximity (0 = at low, 100 = at high)
        high_52w = px.tail(252).max()
        low_52w  = px.tail(252).min()
        last     = px.iloc[-1]
        if high_52w > low_52w:
            proximity = (last - low_52w) / (high_52w - low_52w)
        else:
            proximity = 0.5

        # Map 12-1 momentum to score
        mom_z = _safe_zscore(mom_12_1, 0.05, 0.25)
        mom_score = _map_to_0_100(mom_z)

        score = (
            0.60 * mom_score
            + 0.40 * (proximity * 100)
        )
        return float(np.clip(score, 0, 100))

    @staticmethod
    def _risk_score(fund: dict, ohlcv: Optional[pd.DataFrame]) -> float:
        """
        Risk score (0–100).  Lower risk = higher score (inverse).

        Components:
        - 1-year realized volatility (lower = better)
        - Max 1-year drawdown (lower = better)
        - Debt-to-equity vs sector (lower = better)
        """
        d_e = fund.get("debt_to_equity", 0.5)
        max_dd = fund.get("max_drawdown_1yr", 20)  # percent

        # Realized vol from market data
        if ohlcv is not None and len(ohlcv) >= 63:
            px = ohlcv["close"].tail(253)
            log_rets = np.log(px / px.shift(1)).dropna()
            realized_vol_annual = float(log_rets.std() * np.sqrt(252) * 100)  # percent
        else:
            realized_vol_annual = 30.0  # default assumption

        # Invert: lower vol/dd/debt → higher score
        vol_z = -_safe_zscore(realized_vol_annual, 25.0, 15.0)
        dd_z  = -_safe_zscore(max_dd, 20.0, 15.0)
        de_z  = -_safe_zscore(d_e, 0.5, 0.5)

        score = (
            0.40 * _map_to_0_100(vol_z)
            + 0.35 * _map_to_0_100(dd_z)
            + 0.25 * _map_to_0_100(de_z)
        )
        return float(np.clip(score, 0, 100))

    # ------------------------------------------------------------------
    # Market data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_atr(ohlcv: pd.DataFrame, period: int = 14) -> float:
        if len(ohlcv) < period:
            return 0.0
        high  = ohlcv["high"].tail(period + 1)
        low   = ohlcv["low"].tail(period + 1)
        close = ohlcv["close"].tail(period + 1)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1).dropna()
        return float(tr.mean()) if len(tr) > 0 else 0.0

    @staticmethod
    def _compute_realized_vol(ohlcv: pd.DataFrame, window: int = 63) -> float:
        px = ohlcv["close"].tail(window + 1)
        if len(px) < 10:
            return 0.25
        log_rets = np.log(px / px.shift(1)).dropna()
        return float(log_rets.std() * np.sqrt(252))
