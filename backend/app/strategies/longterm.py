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

BUY if composite > 65 AND valuation_score NOT in bottom decile (not expensive).
REDUCE/SELL if composite < 40 OR fundamental deterioration detected.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.app.strategies.base import (
    BaseStrategy, Signal, SignalDirection, StrategyHealth
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

_BUY_THRESHOLD    = 65.0
_SELL_THRESHOLD   = 40.0
_MAX_EXPENSIVE_VALUATION_SCORE = 20.0  # reject if valuation_score < 20 (top decile expensive)

# ---------------------------------------------------------------------------
# Mock fundamental data — replace with real data provider in production
# ---------------------------------------------------------------------------

class _MockFundamentalProvider:
    """
    Synthetic fundamental data for development and testing.

    ALL VALUES ARE FAKE AND RANDOMLY GENERATED.  This provider exists solely
    to allow integration tests and UI development without a paid data feed.

    NEVER use mock data for real capital allocation decisions.
    """

    _RNG = np.random.default_rng(seed=42)

    @classmethod
    def get_fundamentals(cls, symbol: str) -> Dict[str, float]:
        warnings.warn(
            f"[MOCK DATA] Returning synthetic fundamentals for {symbol}. "
            "Do NOT use for real trading.",
            UserWarning,
            stacklevel=4,
        )
        rng = cls._RNG
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
        """Synthetic sector median multiples."""
        warnings.warn(
            f"[MOCK DATA] Returning synthetic sector medians for {sector}.",
            UserWarning,
            stacklevel=4,
        )
        return {
            "pe_median":      cls._RNG.uniform(15, 35),
            "roe_median":     cls._RNG.uniform(10, 25),
            "roce_median":    cls._RNG.uniform(12, 28),
            "ev_ebitda_median": cls._RNG.uniform(8, 20),
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
    ):
        """
        Parameters
        ----------
        fundamental_provider
            Object with .get_fundamentals(symbol) → dict and
            .get_sector_medians(sector) → dict.  If None, uses MOCK provider.
        paper_mode : bool
        max_positions : int
        buy_threshold : float, composite score above which to BUY.
        sell_threshold : float, composite score below which to SELL.
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

        # --- BUY signals for non-held stocks ---
        buy_candidates = [
            (sym, score, sd)
            for sym, score, sd in all_scores
            if sym not in existing_syms
            and score >= self.buy_threshold
            and sd.get("valuation_score", 50) >= _MAX_EXPENSIVE_VALUATION_SCORE
        ]
        buy_candidates.sort(key=lambda x: x[1], reverse=True)

        available_slots = self.max_positions - len(existing_syms)
        for sym, comp_score, score_dict in buy_candidates[:available_slots]:
            ohlcv = market_data.get(sym)
            if ohlcv is None or ohlcv.empty:
                continue
            last_price = ohlcv["close"].iloc[-1]
            atr = self._compute_atr(ohlcv)
            stop_loss_pct = max((2.5 * atr) / last_price, 0.05) if last_price > 0 else 0.05
            target_pct    = stop_loss_pct * 3.0  # 3:1 for long-term

            vol = self._compute_realized_vol(ohlcv, 63)
            expected_ret = (comp_score - 50) / 100.0  # rough proxy

            signal = Signal(
                symbol=sym,
                direction=SignalDirection.LONG,
                strategy_name=self.name,
                timestamp=now,
                signal_date=now,
                edge_score=max(0.0, expected_ret - 0.002),
                expected_return=expected_ret,
                expected_return_std=expected_ret * 0.8,
                stop_loss_pct=stop_loss_pct,
                target_pct=target_pct,
                holding_period_days=252,
                feature_snapshot={
                    "composite_score": comp_score,
                    "quality_score":    score_dict.get("quality_score", 0),
                    "growth_score":     score_dict.get("growth_score", 0),
                    "valuation_score":  score_dict.get("valuation_score", 0),
                    "momentum_score":   score_dict.get("momentum_score", 0),
                    "risk_score":       score_dict.get("risk_score", 0),
                    "realized_vol_63d": vol,
                },
                metadata={"regime": regime or "UNKNOWN", "data_source": "MOCK"},
            )
            signals.append(signal)

        # --- SELL signals for held positions ---
        for sym in existing_syms:
            try:
                score_dict = next(
                    sd for s, _, sd in all_scores if s == sym
                )
                comp = score_dict["composite"]
            except StopIteration:
                comp = 0.0

            if comp < self.sell_threshold:
                logger.info(
                    "SELL signal for %s: composite=%.1f < threshold=%.1f",
                    sym, comp, self.sell_threshold,
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
                    metadata={"reason": "composite_below_threshold"},
                ))

        logger.info(
            "LongtermStrategy: %d universe, %d buy signals, %d sell signals",
            len(universe),
            sum(1 for s in signals if s.direction == SignalDirection.LONG),
            sum(1 for s in signals if s.direction == SignalDirection.EXIT),
        )
        return signals

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
        Long-term position sizing: volatility-scaled equal weight.

        Each position targets an equal volatility contribution.
        max_weight = 1/max_positions * health_multiplier, scaled by vol.
        """
        if available_capital <= 0 or not signal.is_valid():
            return 0.0

        multiplier = self._position_size_multiplier()
        if multiplier == 0.0:
            return 0.0

        realized_vol = signal.feature_snapshot.get("realized_vol_63d", 0.25)
        if realized_vol <= 0:
            realized_vol = 0.25

        target_vol = 0.15  # 15% portfolio vol target
        target_weight = target_vol / (realized_vol * self.max_positions ** 0.5)
        target_weight = min(target_weight, 1.0 / self.max_positions * 2)
        target_weight = max(target_weight, 0.02)

        size = available_capital * target_weight * multiplier

        try:
            if hasattr(risk_engine, "approve_trade"):
                if not risk_engine.approve_trade(signal.symbol, size, "longterm"):
                    return 0.0
        except Exception:
            pass

        return max(0.0, size)

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
        2. Fundamental deterioration: composite score < sell threshold.
        3. Holding period > 2 years (force review).
        """
        current_price = float(current_data.get("price", 0))

        if current_price > 0 and self._hit_stop(position, current_price):
            return True

        # Fundamental deterioration check (if caller provides current score)
        current_composite = current_data.get("composite_score")
        if current_composite is not None and current_composite < self.sell_threshold:
            logger.info(
                "Fundamental deterioration exit for %s (composite=%.1f).",
                position["symbol"],
                current_composite,
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
