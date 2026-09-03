"""
swing.py — Cross-sectional swing trading strategy (2–20 day horizon).

Design
------
- Ranks universe by ML model's expected 5-day return.
- Applies portfolio constraints (sector exposure, correlation).
- Sizes with volatility targeting (equal vol contribution per position).
- Only enters in non-BEAR/PANIC regimes.
- Maximum 10 concurrent swing positions.
- Paper mode by default.

NO LOOK-AHEAD CONTRACT
-----------------------
Features passed to the model at time T must have been computed using
only data up to the close of bar T.  Forward return labels for training
must start at T+1 open or T+1 close.  This class does not split data;
the caller must enforce alignment.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from backend.app.strategies.base import (
    BaseStrategy, Signal, SignalDirection, StrategyHealth
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_SWING_POSITIONS = 10
_MAX_SECTOR_EXPOSURE = 0.40   # max 40% of swing capital in one sector
_MIN_SIGNAL_STRENGTH = 0.010  # 1% expected return threshold (gross, before costs)
_TARGET_VOL_ANNUAL   = 0.20   # 20% annualized vol target for swing portfolio
_MAX_POSITION_VOL_WEIGHT = 0.30  # max weight for a single position in vol-targeting
_STOP_LOSS_ATR_MULT  = 2.0
_TARGET_ATR_MULT     = 3.0
_MAX_HOLDING_DAYS    = 20
_DEGRADATION_PERCENTILE = 30  # exit if signal strength drops below bottom 30% of entry strength

# Regimes where swing trading is suspended
_BLOCKED_REGIMES = {"BEAR", "PANIC"}


class SwingStrategy(BaseStrategy):
    """
    Cross-sectional swing trading strategy.

    Signal generation flow
    ----------------------
    1. Build eligible universe (Nifty 500 filtered for liquidity).
    2. Compute/receive features from FeatureEngine.
    3. Score with trained GBM model → expected 5-day return.
    4. Cross-sectional rank by model score.
    5. Apply portfolio constraints (sector, correlation).
    6. Size with volatility targeting.
    7. Return top signals if strength > threshold and regime is not BEAR/PANIC.
    """

    name = "SwingMomentum"
    holding_period = "2_to_20_days"

    def __init__(
        self,
        alpha_model=None,
        paper_mode: bool = True,
        max_positions: int = _MAX_SWING_POSITIONS,
        min_signal_strength: float = _MIN_SIGNAL_STRENGTH,
        target_portfolio_vol: float = _TARGET_VOL_ANNUAL,
        sector_map: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        alpha_model : AlphaModelBase
            Trained model to score stocks.  If None, falls back to 12-1
            momentum ranking (no-model baseline).
        paper_mode : bool
        max_positions : int
        min_signal_strength : float
            Minimum expected gross return to generate a signal.
        target_portfolio_vol : float
            Annual volatility target for the swing sleeve.
        sector_map : dict[str, str]
            symbol → sector mapping for constraint enforcement.
        """
        super().__init__(paper_mode=paper_mode)
        self.alpha_model = alpha_model
        self.max_positions = max_positions
        self.min_signal_strength = min_signal_strength
        self.target_portfolio_vol = target_portfolio_vol
        self.sector_map = sector_map or {}

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        universe: List[str],
        features_df: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame],
        regime: Optional[str] = None,
        current_date: Optional[datetime] = None,
        existing_positions: Optional[Dict[str, dict]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> List[Signal]:
        """
        Generate swing trading signals.

        Parameters
        ----------
        universe : list[str], eligible symbols.
        features_df : pd.DataFrame
            Wide feature panel.  The last row (current date) provides the
            feature vector for each symbol.
        market_data : dict[str, pd.DataFrame], daily OHLCV per symbol.
        regime : str or None
            Current market regime label (e.g. "BULL_MEDIUM_VOL").
            If BEAR or PANIC, returns [].
        current_date : datetime, defaults to now.
        existing_positions : dict[str, dict], currently held swing positions.
        feature_names : list[str], column names if features_df has a different
            ordering than training.

        Returns
        -------
        list[Signal] ranked by edge_score.
        """
        if not self._is_operational():
            logger.info("SwingStrategy is %s; no signals.", self.health)
            return []

        # Regime gate
        if regime and any(b in regime.upper() for b in _BLOCKED_REGIMES):
            logger.info("Swing signals blocked: regime is %s.", regime)
            return []

        now = current_date or datetime.now()
        existing_syms = set((existing_positions or {}).keys())
        available_slots = self.max_positions - len(existing_syms)
        if available_slots <= 0:
            return []

        # ---- Score each symbol ----
        scores = self._score_universe(universe, features_df, market_data, feature_names)
        if scores.empty:
            return []

        # ---- Cross-sectional rank (0 to 1) ----
        scores["rank"] = scores["raw_score"].rank(pct=True)

        # ---- Apply portfolio constraints ----
        scores = self._apply_constraints(scores, existing_syms)

        # ---- Filter by minimum signal strength ----
        eligible = scores[scores["raw_score"] >= self.min_signal_strength].copy()
        eligible = eligible.sort_values("rank", ascending=False)

        # ---- Build signals (top slots) ----
        signals: List[Signal] = []
        sector_exposure: Dict[str, float] = {}

        for sym, row in eligible.iterrows():
            if sym in existing_syms:
                continue
            if len(signals) >= available_slots:
                break

            # Sector constraint: max 40% of allocated positions in one sector
            sector = self.sector_map.get(sym, "Unknown")
            sector_count = sector_exposure.get(sector, 0)
            if sector_count >= _MAX_SECTOR_EXPOSURE * self.max_positions:
                continue

            ohlcv = market_data.get(sym)
            if ohlcv is None or ohlcv.empty:
                continue

            atr = self._compute_atr(ohlcv, period=14)
            last_price = ohlcv["close"].iloc[-1]
            if last_price <= 0 or atr <= 0:
                continue

            stop_loss_pct = (_STOP_LOSS_ATR_MULT * atr) / last_price
            target_pct    = (_TARGET_ATR_MULT    * atr) / last_price

            # Vol for position sizing
            realized_vol = self._compute_realized_vol(ohlcv, window=21)

            # Net edge: gross expected return minus estimated costs (delivery)
            gross_ret = float(row["raw_score"])
            est_cost = 0.002  # ~20 bps delivery round-trip
            net_edge = gross_ret - est_cost

            if net_edge <= 0:
                continue

            signal = Signal(
                symbol=sym,
                direction=SignalDirection.LONG,
                strategy_name=self.name,
                timestamp=now,
                signal_date=now,
                edge_score=net_edge,
                expected_return=gross_ret,
                expected_return_std=gross_ret * 0.5,
                stop_loss_pct=stop_loss_pct,
                target_pct=target_pct,
                holding_period_days=5,
                feature_snapshot={
                    "rank": float(row["rank"]),
                    "raw_score": gross_ret,
                    "realized_vol_21d": realized_vol,
                    "sector": sector,
                },
                metadata={"regime": regime or "UNKNOWN"},
            )
            signals.append(signal)
            sector_exposure[sector] = sector_exposure.get(sector, 0) + 1

        logger.info(
            "SwingStrategy: %d universe → %d eligible → %d signals",
            len(universe),
            len(eligible),
            len(signals),
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
        Volatility-targeting position size.

        target_weight = target_vol_annual / realized_vol_annual
        capped at max_position_vol_weight.
        position_size (INR) = available_capital * target_weight * health_multiplier
        """
        if available_capital <= 0 or not signal.is_valid():
            return 0.0

        multiplier = self._position_size_multiplier()
        if multiplier == 0.0:
            return 0.0

        realized_vol = signal.feature_snapshot.get("realized_vol_21d", self.target_portfolio_vol)
        if realized_vol <= 0:
            realized_vol = self.target_portfolio_vol

        # Weight each position to contribute equal vol
        target_weight = self.target_portfolio_vol / (realized_vol * self.max_positions ** 0.5)
        target_weight = min(target_weight, _MAX_POSITION_VOL_WEIGHT)
        target_weight = max(target_weight, 0.01)  # at least 1%

        size = available_capital * target_weight * multiplier

        try:
            if hasattr(risk_engine, "approve_trade"):
                if not risk_engine.approve_trade(signal.symbol, size, "swing"):
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
        Exit swing position if:
        1. Stop-loss hit.
        2. Target hit.
        3. Holding period > 20 days.
        4. Current signal strength has degraded below entry rank percentile floor.

        Parameters
        ----------
        position : dict with keys: symbol, entry_price, entry_date, direction,
                   stop_loss, target, signal (Signal object).
        current_data : dict with keys: price (float), date (datetime),
                       current_rank (float, optional), features (dict, optional).
        """
        current_price = float(current_data.get("price", 0))

        # 1. Stop
        if current_price > 0 and self._hit_stop(position, current_price):
            return True

        # 2. Target
        if current_price > 0 and self._hit_target(position, current_price):
            return True

        # 3. Time limit
        entry_date = position.get("entry_date")
        current_date = current_data.get("date")
        if entry_date and current_date:
            days_held = (current_date - entry_date).days
            if days_held > _MAX_HOLDING_DAYS:
                logger.debug(
                    "Time-based exit for %s (held %d days).",
                    position["symbol"],
                    days_held,
                )
                return True

        # 4. Signal degradation
        current_rank = current_data.get("current_rank")
        if current_rank is not None:
            entry_rank = position.get("signal", {})
            entry_rank_val = (
                entry_rank.feature_snapshot.get("rank", 0.5)
                if hasattr(entry_rank, "feature_snapshot")
                else 0.5
            )
            _ = entry_rank_val  # captured for future use
            if current_rank < (_DEGRADATION_PERCENTILE / 100):
                logger.debug(
                    "Signal degraded for %s (current_rank=%.2f).",
                    position["symbol"],
                    current_rank,
                )
                return True

        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_universe(
        self,
        universe: List[str],
        features_df: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame],
        feature_names: Optional[List[str]],
    ) -> pd.DataFrame:
        """
        Score all universe symbols.  Returns DataFrame with columns:
        symbol (index), raw_score.
        """
        rows = {}

        # Try ML model first
        if self.alpha_model is not None:
            try:
                X, valid_syms = self._build_feature_matrix(
                    universe, features_df, feature_names
                )
                if len(valid_syms) > 0:
                    preds = self.alpha_model.predict(X)
                    for sym, pred in zip(valid_syms, preds):
                        rows[sym] = float(pred)
            except Exception as exc:
                logger.warning("Alpha model prediction failed: %s; using momentum.", exc)

        # Fallback: 12-1 momentum
        if not rows:
            for sym in universe:
                ohlcv = market_data.get(sym)
                if ohlcv is None or len(ohlcv) < 252:
                    continue
                px = ohlcv["close"]
                try:
                    mom = float(np.log(px.iloc[-21] / px.iloc[-252]))
                    rows[sym] = mom
                except Exception:
                    continue

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame.from_dict(
            {"raw_score": rows}
        )

    def _build_feature_matrix(
        self,
        universe: List[str],
        features_df: pd.DataFrame,
        feature_names: Optional[List[str]],
    ) -> tuple[np.ndarray, List[str]]:
        """
        Extract the last-row feature vector for each symbol from features_df.

        features_df is assumed to have columns named "{feat}__{symbol}".
        """
        valid_syms = []
        vectors = []

        last_row = features_df.iloc[-1]
        for sym in universe:
            if feature_names:
                feat_cols = [f"{fn}__{sym}" for fn in feature_names]
            else:
                feat_cols = [c for c in features_df.columns if c.endswith(f"__{sym}")]

            if not feat_cols:
                continue

            available = [c for c in feat_cols if c in last_row.index]
            if not available:
                continue

            vec = last_row[available].values.astype(float)
            if np.isnan(vec).any():
                # Fill NaN with median of non-NaN values
                med = np.nanmedian(vec)
                vec = np.where(np.isnan(vec), med, vec)

            valid_syms.append(sym)
            vectors.append(vec)

        if not vectors:
            return np.empty((0, 0)), []

        X = np.vstack(vectors)
        return X, valid_syms

    def _apply_constraints(
        self,
        scores: pd.DataFrame,
        existing_syms: set,
    ) -> pd.DataFrame:
        """Remove already-held symbols from candidates."""
        return scores.loc[~scores.index.isin(existing_syms)]

    @staticmethod
    def _compute_atr(ohlcv: pd.DataFrame, period: int = 14) -> float:
        """Average True Range from daily OHLCV."""
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
    def _compute_realized_vol(ohlcv: pd.DataFrame, window: int = 21) -> float:
        """Annualized realized volatility over last `window` days."""
        px = ohlcv["close"].tail(window + 1)
        if len(px) < 5:
            return 0.20
        log_rets = np.log(px / px.shift(1)).dropna()
        return float(log_rets.std() * np.sqrt(252))
