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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.strategies.base import (
    DELIVERY_ROUND_TRIP_COST, MAX_GROSS_EXPOSURE,
    BaseStrategy, Signal, SignalDirection,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_SWING_POSITIONS = 10
_MAX_SECTOR_EXPOSURE = 0.40   # max 40% of swing capital in one sector
_TARGET_VOL_ANNUAL   = 0.20   # 20% annualized vol target for swing portfolio
_MAX_POSITION_VOL_WEIGHT = 0.30  # max weight for a single position in vol-targeting
_STOP_LOSS_ATR_MULT  = 2.0
_TARGET_ATR_MULT     = 3.0
_MAX_HOLDING_DAYS    = 20

# ---------------------------------------------------------------------------
# Horizon and units
# ---------------------------------------------------------------------------
# EVERY score this strategy produces is a SIMPLE EXPECTED RETURN OVER
# _HORIZON_DAYS TRADING DAYS.  Nothing in this module may emit a number on a
# different horizon into Signal.expected_return / Signal.edge_score.
_HORIZON_DAYS = 5

# 12-1 momentum spans close[-252] .. close[-21] = 231 trading days.  Feeding
# that number straight through as a 5-day expected return overstated the edge
# by ~46x (231/5) and made a zero-drift random walk look like a 40%+ 5-day
# opportunity.
_MOMENTUM_LOOKBACK_DAYS = 231

# Information coefficient assumed for cross-sectional 12-1 momentum, i.e. the
# correlation between the momentum z-score and the subsequent 5-day return.
# Published cross-sectional momentum ICs sit in the 0.02-0.05 range; 0.03 is a
# deliberately conservative mid-point.  E[r_5d | z] = IC * z * sigma_5d.
_MOMENTUM_IC = 0.03

# Minimum number of names needed before a cross-sectional z-score is credible.
_MIN_XS_FOR_ZSCORE = 10

# Sanity bound on any model's 5-day return prediction.  A model that predicts
# more than this is almost certainly emitting a different horizon (annual
# returns, percent instead of fraction, ...) — a units bug, not an opinion.
_MAX_PLAUSIBLE_HORIZON_RETURN = 0.25

# Minimum GROSS expected 5-day return required to consider a signal.
# 0.004 = 40 bps over 5 trading days (~20% annualized gross), which clears the
# ~34 bps delivery round-trip hurdle with margin.  The old value (1% per 5 days
# ~ 65% annualized) was never met by an honestly-scaled score, and was met by
# 35/50 names when the score was accidentally a 12-month number.
_MIN_SIGNAL_STRENGTH = 0.004

# Round-trip cost hurdle for a CNC (delivery) swing trade — see base.py.
_COST_ESTIMATE_DELIVERY = DELIVERY_ROUND_TRIP_COST

# Signal-degradation exit (see should_exit).
_DEGRADATION_PERCENTILE = 30   # absolute floor: bottom 30% of the cross-section
_DEGRADATION_ENTRY_RATIO = 0.70  # ...or 70% of the ENTRY rank, whichever is higher

# Regimes where swing trading is suspended
_BLOCKED_REGIMES = {"BEAR", "PANIC"}


class FeatureContractError(ValueError):
    """
    Raised when the feature matrix handed to the alpha model does not match
    the contract the model was trained on (missing features, inconsistent
    widths, unknown column ordering).

    Silently shortening a feature vector or reordering columns produces
    predictions that are numerically fine and semantically garbage, so this
    is fatal by design.
    """


class SignalUnitsError(ValueError):
    """Raised when a score is not plausibly a _HORIZON_DAYS-day return."""


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
    6. Size with volatility targeting, normalised across the emitted batch so
       gross exposure never exceeds the sleeve capital.
    7. Return top signals if strength > threshold and regime is not BEAR/PANIC.

    UNITS
    -----
    raw_score, expected_return and edge_score are all SIMPLE EXPECTED RETURNS
    OVER 5 TRADING DAYS (0.004 = 40 bps over the holding period), matching
    Signal.holding_period_days = 5.  The alpha model is contractually required
    to predict on the same horizon; the no-model momentum fallback converts
    12-1 momentum to that horizon explicitly (see _score_universe).
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
        horizon_days: int = _HORIZON_DAYS,
    ):
        """
        Parameters
        ----------
        alpha_model : AlphaModelBase
            Trained model to score stocks.  Must predict the simple return
            over `horizon_days` TRADING DAYS.  If None, falls back to a 12-1
            momentum prior (no-model baseline) converted to the same horizon.
        paper_mode : bool
        max_positions : int
        min_signal_strength : float
            Minimum expected GROSS return, over `horizon_days`, to generate a
            signal.  Must be interpreted on the same horizon as the score;
            the default (40 bps / 5 days) clears the ~34 bps delivery
            round-trip hurdle.
        target_portfolio_vol : float
            Annual volatility target for the swing sleeve.
        sector_map : dict[str, str]
            symbol → sector mapping for constraint enforcement.
        horizon_days : int
            Forecast horizon in TRADING days.  This is the single knob that
            defines the units of raw_score / expected_return / edge_score and
            it is copied verbatim into Signal.holding_period_days.  The class
            docstring's "2 to 20 days" band means 5 (default) through 20 are
            all legitimate; costs are horizon-independent, so a longer horizon
            has more room to clear them.
        """
        super().__init__(paper_mode=paper_mode)
        if horizon_days <= 0:
            raise ValueError("horizon_days must be positive.")
        self.alpha_model = alpha_model
        self.max_positions = max_positions
        self.min_signal_strength = min_signal_strength
        self.target_portfolio_vol = target_portfolio_vol
        self.sector_map = sector_map or {}
        self.horizon_days = int(horizon_days)

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

            # Net edge: gross expected 5-day return minus round-trip costs.
            # gross_ret is a 5-DAY return (see _score_universe); est_cost is a
            # round-trip fraction of notional.  Same units, so the subtraction
            # is meaningful.
            gross_ret = float(row["raw_score"])
            net_edge = gross_ret - _COST_ESTIMATE_DELIVERY

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
                # Residual dispersion of the 5-day return around the forecast.
                # (The old `gross_ret * 0.5` went negative for negative scores
                # and had nothing to do with forecast uncertainty.)
                expected_return_std=float(row.get(
                    "sigma_horizon", realized_vol * np.sqrt(self.horizon_days / 252.0)
                )),
                stop_loss_pct=stop_loss_pct,
                target_pct=target_pct,
                holding_period_days=self.horizon_days,
                feature_snapshot={
                    "rank": float(row["rank"]),
                    "raw_score": gross_ret,
                    "realized_vol_21d": realized_vol,
                    "sector": sector,
                },
                metadata={
                    "regime": regime or "UNKNOWN",
                    "return_units": f"simple_return_over_{self.horizon_days}_trading_days",
                    "score_source": str(row.get("score_source", "unknown")),
                    "cost_estimate": _COST_ESTIMATE_DELIVERY,
                },
            )
            signals.append(signal)
            sector_exposure[sector] = sector_exposure.get(sector, 0) + 1

        # ---- Portfolio budget: normalise weights across the emitted set ----
        # Per-position caps do not bound gross exposure (10 x 30% = 300%), so
        # the batch is scaled to the sleeve's remaining budget here, and
        # calculate_position_size() consumes the stamped weight.
        budget = self._entry_budget(self.max_positions, len(existing_syms))
        self._stamp_target_weights(
            signals,
            [self._raw_target_weight(s) for s in signals],
            budget=budget,
        )

        logger.info(
            "SwingStrategy: %d universe → %d eligible → %d signals "
            "(gross intent %.1f%% of sleeve, budget %.0f%%)",
            len(universe),
            len(eligible),
            len(signals),
            sum(s.metadata.get("target_weight", 0.0) for s in signals) * 100,
            budget * 100,
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
        Volatility-targeting position size, bounded by the portfolio budget.

        The weight used is the BATCH-NORMALISED weight stamped onto the signal
        by generate_signals(), so that the sum over the emitted signal set can
        never exceed 100% of the sleeve capital.  For signals that did not
        come from generate_signals() (hand-built, replayed, or from another
        producer) the raw vol-target weight is used but hard-capped at
        MAX_GROSS_EXPOSURE / max_positions, which bounds a full book at 100%.

        position_size (INR) = available_capital * target_weight * health_multiplier

        Raises
        ------
        RiskEngineError : if a supplied risk engine cannot approve the trade.
                          Blocking is deliberate; this call never fails open.
        """
        if available_capital <= 0 or not signal.is_valid():
            return 0.0

        multiplier = self._position_size_multiplier()
        if multiplier == 0.0:
            return 0.0

        stamped = signal.metadata.get("target_weight")
        if stamped is not None:
            target_weight = float(stamped)
        else:
            # Un-stamped signal: fall back to the raw vol-target weight but
            # cap it so max_positions such signals still fit inside the sleeve.
            target_weight = min(
                self._raw_target_weight(signal),
                MAX_GROSS_EXPOSURE / max(1, self.max_positions),
            )

        size = available_capital * max(0.0, target_weight) * multiplier

        if not self._risk_engine_approves(risk_engine, signal.symbol, size, "swing"):
            return 0.0

        return max(0.0, size)

    def _raw_target_weight(self, signal: Signal) -> float:
        """
        Pre-normalisation vol-target weight for one signal.

        Equal volatility contribution across max_positions slots, capped at
        _MAX_POSITION_VOL_WEIGHT and floored at 1%.  This is an INTENT, not an
        allocation — see _stamp_target_weights for the budget that binds it.
        """
        realized_vol = signal.feature_snapshot.get("realized_vol_21d", self.target_portfolio_vol)
        if realized_vol is None or not np.isfinite(realized_vol) or realized_vol <= 0:
            realized_vol = self.target_portfolio_vol

        target_weight = self.target_portfolio_vol / (realized_vol * self.max_positions ** 0.5)
        target_weight = min(target_weight, _MAX_POSITION_VOL_WEIGHT)
        return max(target_weight, 0.01)  # at least 1%

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
        4. Signal degradation, measured AGAINST THE ENTRY RANK: exit when the
           current cross-sectional rank falls below
               max(_DEGRADATION_PERCENTILE/100,
                   entry_rank * _DEGRADATION_ENTRY_RATIO)
           i.e. below an absolute floor (bottom 30% of the cross-section) or
           below 70% of the rank the position was entered at, whichever binds
           first.  A name entered at rank 0.95 exits at 0.665; a name entered
           near the floor exits at the floor.

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

        # 4. Signal degradation, relative to the ENTRY rank
        current_rank = current_data.get("current_rank")
        if current_rank is not None:
            entry_signal = position.get("signal")
            entry_rank_val = 0.5
            if hasattr(entry_signal, "feature_snapshot"):
                entry_rank_val = float(
                    entry_signal.feature_snapshot.get("rank", 0.5) or 0.5
                )

            floor = _DEGRADATION_PERCENTILE / 100.0
            exit_rank = max(floor, entry_rank_val * _DEGRADATION_ENTRY_RATIO)
            if float(current_rank) < exit_rank:
                logger.debug(
                    "Signal degraded for %s (current_rank=%.2f < %.2f; "
                    "entry_rank=%.2f).",
                    position["symbol"],
                    float(current_rank),
                    exit_rank,
                    entry_rank_val,
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
        Score all universe symbols.

        Returns
        -------
        pd.DataFrame indexed by symbol with columns:
            raw_score      : EXPECTED SIMPLE RETURN OVER _HORIZON_DAYS (5)
                             TRADING DAYS.  Not an annual number, not a
                             ranking score — a return on the signal's own
                             horizon.
            sigma_horizon  : 1-sigma dispersion of that 5-day return.
            score_source   : "alpha_model" or "momentum_prior".

        Alpha-model failures are NOT silently downgraded to the momentum
        heuristic: a model that raises is a broken contract and must be seen.
        The momentum path is used only when no model is configured.
        """
        if self.alpha_model is not None:
            X, valid_syms = self._build_feature_matrix(
                universe, features_df, feature_names
            )
            if not valid_syms:
                logger.warning("No symbol produced a complete feature vector.")
                return pd.DataFrame()

            preds = np.asarray(self.alpha_model.predict(X), dtype=float).ravel()
            if preds.shape[0] != len(valid_syms):
                raise FeatureContractError(
                    f"Alpha model returned {preds.shape[0]} predictions for "
                    f"{len(valid_syms)} symbols."
                )

            worst = float(np.nanmax(np.abs(preds))) if preds.size else 0.0
            if worst > _MAX_PLAUSIBLE_HORIZON_RETURN:
                raise SignalUnitsError(
                    f"Alpha model predicted |return| up to {worst:.3f} for a "
                    f"{self.horizon_days}-trading-day horizon (bound "
                    f"{_MAX_PLAUSIBLE_HORIZON_RETURN}).  The model is almost "
                    "certainly emitting a different horizon or unit than the "
                    "Signal contract requires."
                )

            sigma = np.array(
                [self._horizon_sigma(market_data.get(s)) for s in valid_syms]
            )
            return pd.DataFrame(
                {
                    "raw_score": preds,
                    "sigma_horizon": sigma,
                    "score_source": "alpha_model",
                },
                index=valid_syms,
            )

        return self._momentum_prior_scores(universe, market_data)

    def _momentum_prior_scores(
        self,
        universe: List[str],
        market_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        No-model baseline: convert 12-1 momentum into a genuine
        _HORIZON_DAYS-day expected return.

        12-1 momentum is a 231-trading-day log return.  It is a RANKING
        signal, not a forecast, so it is turned into a forecast the standard
        way::

            z          = cross-sectional z-score of the momentum (clipped +/-3)
            sigma_5d   = 21-day realised daily vol * sqrt(5)
            E[r_5d|z]  = IC * z * sigma_5d          with IC = 0.03

        On a zero-drift random walk the cross-sectional z-scores still span
        +/-2 (dispersion always exists), but the resulting expected returns
        are single basis points rather than the 40%+ "5-day return" produced
        by passing the raw 231-day momentum straight through.

        If the cross-section is too small for a z-score (< _MIN_XS_FOR_ZSCORE
        names), momentum is instead scaled to the horizon arithmetically
        (mom * 5/231) and shrunk by the same IC — cruder, same units.
        """
        moms: Dict[str, float] = {}
        sigmas: Dict[str, float] = {}
        for sym in universe:
            ohlcv = market_data.get(sym)
            if ohlcv is None or len(ohlcv) < 252:
                continue
            px = ohlcv["close"]
            try:
                mom = float(np.log(px.iloc[-21] / px.iloc[-252]))
            except Exception:
                continue
            if not np.isfinite(mom):
                continue
            moms[sym] = mom
            sigmas[sym] = self._horizon_sigma(ohlcv)

        if not moms:
            return pd.DataFrame()

        syms = list(moms.keys())
        raw = np.array([moms[s] for s in syms], dtype=float)
        sig = np.array([sigmas[s] for s in syms], dtype=float)

        if len(syms) >= _MIN_XS_FOR_ZSCORE:
            sd = float(np.std(raw, ddof=1))
            if sd > 1e-9:
                z = np.clip((raw - float(np.mean(raw))) / sd, -3.0, 3.0)
            else:
                z = np.zeros_like(raw)
            expected = _MOMENTUM_IC * z * sig
        else:
            # Arithmetic horizon rescale of the log momentum, then IC shrinkage.
            expected = _MOMENTUM_IC * raw * (self.horizon_days / _MOMENTUM_LOOKBACK_DAYS)

        # ---- Absolute-trend filter (dual momentum) ----
        # The z-score is purely RELATIVE: in a market where every name is down
        # 50%, the least-bad names still score positively.  A long is only
        # taken when the name's own trend is up as well, which is what kept the
        # strategy flat through a crashing universe.  (The regime gate remains
        # the primary defence; this is the belt to its braces.)
        expected = np.where(raw > 0.0, expected, np.minimum(expected, 0.0))

        return pd.DataFrame(
            {
                "raw_score": expected,
                "sigma_horizon": sig,
                "score_source": "momentum_prior",
            },
            index=syms,
        )

    def _horizon_sigma(self, ohlcv: Optional[pd.DataFrame]) -> float:
        """1-sigma dispersion of the horizon_days-day return for one symbol."""
        annual_vol = 0.25
        if ohlcv is not None and not ohlcv.empty:
            annual_vol = SwingStrategy._compute_realized_vol(ohlcv, window=21)
        if not np.isfinite(annual_vol) or annual_vol <= 0:
            annual_vol = 0.25
        return float(annual_vol * np.sqrt(self.horizon_days / 252.0))

    def _canonical_feature_order(
        self,
        universe: List[str],
        features_df: pd.DataFrame,
        feature_names: Optional[List[str]],
    ) -> List[str]:
        """
        Determine the canonical feature ORDER for the model input.

        If the caller supplies feature_names (the training order), that is
        authoritative.  Otherwise the stems are derived from the panel and
        SORTED, so the ordering is at least deterministic and reproducible
        across runs — DataFrame column order is not a contract.
        """
        if feature_names:
            return list(feature_names)

        stems = {
            c.rsplit("__", 1)[0]
            for c in features_df.columns
            if "__" in c and c.rsplit("__", 1)[1] in set(universe)
        }
        order = sorted(stems)
        logger.warning(
            "No feature_names supplied; falling back to a sorted canonical "
            "order of %d features derived from the panel. Pass the training "
            "feature order explicitly to guarantee train/serve alignment.",
            len(order),
        )
        return order

    def _build_feature_matrix(
        self,
        universe: List[str],
        features_df: pd.DataFrame,
        feature_names: Optional[List[str]],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract the last-row feature vector for each symbol from features_df.

        features_df is assumed to have columns named "{feat}__{symbol}".

        Guarantees
        ----------
        - EVERY row has exactly len(canonical_order) entries, in the canonical
          order.  A symbol missing any required feature is DROPPED with a
          warning; the vector is never silently shortened (which previously
          produced a ragged np.vstack and a ValueError that was swallowed
          upstream).
        - NaNs are imputed PER FEATURE, using the cross-sectional median of
          that feature across symbols — never with the median of a single
          stock's own heterogeneous features (which turned a missing volume
          into 35.01 given [rsi=70, ret5=0.02, NaN]).  A feature that is NaN
          for every symbol is imputed with 0.0 and flagged.
        """
        if features_df is None or features_df.empty:
            return np.empty((0, 0)), []

        order = self._canonical_feature_order(universe, features_df, feature_names)
        if not order:
            raise FeatureContractError(
                "Could not determine a canonical feature order for the model; "
                "pass feature_names explicitly."
            )

        last_row = features_df.iloc[-1]
        index_set = set(last_row.index)

        valid_syms: List[str] = []
        vectors: List[np.ndarray] = []
        dropped: Dict[str, List[str]] = {}

        for sym in universe:
            cols = [f"{fn}__{sym}" for fn in order]
            missing = [c for c in cols if c not in index_set]
            if missing:
                # Never shorten the vector — drop the symbol instead.
                dropped[sym] = missing
                continue
            vec = pd.to_numeric(last_row[cols], errors="coerce").to_numpy(dtype=float)
            if vec.shape[0] != len(order):
                raise FeatureContractError(
                    f"{sym}: built a {vec.shape[0]}-feature vector but the "
                    f"canonical order has {len(order)} features."
                )
            valid_syms.append(sym)
            vectors.append(vec)

        if dropped:
            logger.warning(
                "Dropped %d/%d symbols with incomplete feature vectors "
                "(e.g. %s missing %s).",
                len(dropped), len(universe),
                next(iter(dropped)), dropped[next(iter(dropped))][:3],
            )

        if not vectors:
            return np.empty((0, len(order))), []

        X = np.vstack(vectors)
        if X.shape[1] != len(order):
            raise FeatureContractError(
                f"Feature matrix has {X.shape[1]} columns, canonical order "
                f"has {len(order)}."
            )

        # ---- Per-FEATURE (column-wise) cross-sectional median imputation ----
        nan_mask = np.isnan(X)
        if nan_mask.any():
            with np.errstate(all="ignore"):
                col_median = np.nanmedian(X, axis=0)
            all_nan_cols = np.isnan(col_median)
            if all_nan_cols.any():
                logger.warning(
                    "Features %s are NaN for every symbol; imputing 0.0.",
                    [order[i] for i in np.flatnonzero(all_nan_cols)],
                )
                col_median[all_nan_cols] = 0.0
            X = np.where(nan_mask, np.broadcast_to(col_median, X.shape), X)

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
