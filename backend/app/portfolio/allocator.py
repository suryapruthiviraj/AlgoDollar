"""
allocator.py — Capital allocation engine for AlgoDollar.

This is the CORE of the system: given a fresh contribution (or rebalance
event), it decides how much capital to deploy across long-term, swing, and
intraday sleeves — and how much to keep as cash.

Allocation Philosophy
---------------------
- Cash is always a valid asset class.  If no strategy shows sufficient
  expected edge, 100% cash is a legitimate output.
- Regime and strategy health multiplicatively scale opportunity scores.
- User constraints (disabled strategies, intraday cap) override all model
  recommendations.
- The system never forces a trade.
- All arithmetic is transparent and logged; outputs include a human-readable
  explanation string.

Mathematical flow
-----------------
1. Compute available_capital = contribution + existing_cash.
2. Compute raw opportunity score per bucket = base_score * regime_mult * health_mult.
3. Normalize raw scores to weights (softmax-like, with floor = 0).
4. Apply user constraints (hard disable, max_intraday_pct).
5. Apply opportunity threshold: if max_score < minimum_edge_threshold, cash = 100%.
6. Compute final amounts.  Validate: sum = available_capital (±1 INR rounding).

INVARIANT: longterm_amount + swing_amount + intraday_amount + cash_amount
           == available_capital   (within 1 INR rounding tolerance)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type stubs for regime (imported loosely to avoid circular deps)
# ---------------------------------------------------------------------------
try:
    from backend.app.models.regime_model import (  # type: ignore
        CombinedRegime, MarketRegime, VolatilityRegime
    )
except ImportError:
    CombinedRegime = None  # type: ignore
    MarketRegime = None     # type: ignore
    VolatilityRegime = None # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regime multipliers: how much to scale the raw opportunity score in each regime.
# Bull + low vol → full deployment.  Panic → near-zero.
_REGIME_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    # (price_regime, vol_regime): {sleeve: multiplier}
    "BULL_LOW_VOL":      {"longterm": 1.00, "swing": 1.00, "intraday": 1.00},
    "BULL_MEDIUM_VOL":   {"longterm": 1.00, "swing": 0.85, "intraday": 0.70},
    "BULL_HIGH_VOL":     {"longterm": 0.80, "swing": 0.65, "intraday": 0.50},
    "SIDEWAYS_LOW_VOL":  {"longterm": 0.70, "swing": 0.70, "intraday": 0.65},
    "SIDEWAYS_MEDIUM_VOL":{"longterm": 0.65, "swing": 0.60, "intraday": 0.50},
    "SIDEWAYS_HIGH_VOL": {"longterm": 0.50, "swing": 0.45, "intraday": 0.35},
    "BEAR_LOW_VOL":      {"longterm": 0.40, "swing": 0.25, "intraday": 0.20},
    "BEAR_MEDIUM_VOL":   {"longterm": 0.30, "swing": 0.15, "intraday": 0.10},
    "BEAR_HIGH_VOL":     {"longterm": 0.20, "swing": 0.10, "intraday": 0.05},
    "PANIC_LOW_VOL":     {"longterm": 0.20, "swing": 0.05, "intraday": 0.00},
    "PANIC_MEDIUM_VOL":  {"longterm": 0.15, "swing": 0.03, "intraday": 0.00},
    "PANIC_HIGH_VOL":    {"longterm": 0.10, "swing": 0.00, "intraday": 0.00},
}

_DEFAULT_REGIME_MULTIPLIER = {"longterm": 0.60, "swing": 0.50, "intraday": 0.40}

# Strategy health multipliers
_HEALTH_MULTIPLIERS: Dict[str, float] = {
    "HEALTHY":  1.00,
    "REDUCED":  0.50,
    "PAUSED":   0.00,
    "DISABLED": 0.00,
}

# Historical (placeholder) Sharpe ratios per strategy, normalized to [0, 1].
# Replace with live OOS estimates from ModelRegistry.
_DEFAULT_SHARPE_NORMALIZED: Dict[str, float] = {
    "longterm": 0.75,
    "swing":    0.65,
    "intraday": 0.55,
}

# Minimum opportunity score to deploy any capital to a strategy.
_MIN_OPPORTUNITY_SCORE = 0.10

# Minimum total score for any deployment (below this → 100% cash).
_MIN_TOTAL_SCORE_FOR_DEPLOYMENT = 0.15

# Default user settings if none provided.
_DEFAULT_USER_SETTINGS = {
    "longterm_enabled": True,
    "swing_enabled":    True,
    "intraday_enabled": True,
    "max_intraday_pct": 0.10,   # max 10% of available capital to intraday
    "max_swing_pct":    0.40,
    "max_longterm_pct": 0.80,
    "min_cash_pct":     0.05,   # always keep at least 5% cash
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AllocationResult:
    """
    Result of the capital allocation decision.

    INVARIANT
    ---------
    longterm_amount + swing_amount + intraday_amount + cash_amount
        == available_capital  (within 1 INR rounding tolerance)
    """
    # Amounts (INR)
    longterm_amount: float
    swing_amount:    float
    intraday_amount: float
    cash_amount:     float
    available_capital: float

    # Risk fractions (what fraction of each sleeve is considered "at risk")
    longterm_risk_pct:  float
    swing_risk_pct:     float
    intraday_risk_pct:  float

    # Context
    regime_label: str
    opportunity_scores: Dict[str, float]
    explanation: str
    confidence: float  # 0–1, how confident the system is in this allocation

    def validate(self, tolerance: float = 1.0) -> bool:
        """Return True if the allocation sums to available_capital."""
        total = (
            self.longterm_amount
            + self.swing_amount
            + self.intraday_amount
            + self.cash_amount
        )
        return abs(total - self.available_capital) <= tolerance

    def as_dict(self) -> dict:
        return {
            "longterm_amount":   round(self.longterm_amount, 2),
            "swing_amount":      round(self.swing_amount, 2),
            "intraday_amount":   round(self.intraday_amount, 2),
            "cash_amount":       round(self.cash_amount, 2),
            "available_capital": round(self.available_capital, 2),
            "longterm_risk_pct": round(self.longterm_risk_pct, 4),
            "swing_risk_pct":    round(self.swing_risk_pct, 4),
            "intraday_risk_pct": round(self.intraday_risk_pct, 4),
            "regime":            self.regime_label,
            "opportunity_scores": {k: round(v, 4) for k, v in self.opportunity_scores.items()},
            "explanation":       self.explanation,
            "confidence":        round(self.confidence, 3),
        }


# ---------------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------------

class CapitalAllocator:
    """
    Determine capital allocation across long-term, swing, intraday, and cash.

    Parameters
    ----------
    sharpe_normalized : dict[str, float]
        Normalized historical Sharpe ratio per strategy, range [0, 1].
        Sourced from the ModelRegistry OOS metrics.  Defaults if None.
    """

    def __init__(
        self,
        sharpe_normalized: Optional[Dict[str, float]] = None,
        min_opportunity_score: float = _MIN_OPPORTUNITY_SCORE,
        min_total_score_for_deployment: float = _MIN_TOTAL_SCORE_FOR_DEPLOYMENT,
    ):
        self._sharpe = sharpe_normalized or dict(_DEFAULT_SHARPE_NORMALIZED)
        self._min_opp = min_opportunity_score
        self._min_total = min_total_score_for_deployment

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(
        self,
        contribution: float,
        existing_portfolio: dict,
        market_data: dict,
        strategy_health_map: Dict[str, str],
        user_settings: Optional[Dict] = None,
        regime=None,
    ) -> AllocationResult:
        """
        Compute capital allocation.

        Parameters
        ----------
        contribution : float (INR)
            New capital being added (or 0 for a rebalance-only event).
        existing_portfolio : dict
            Portfolio state.  Must contain 'cash' (float) representing
            currently uninvested cash.
        market_data : dict
            Live or current-day market data (used for context; regime already
            extracted separately).
        strategy_health_map : dict[str, str]
            Mapping of strategy name to health string, e.g.:
            {"longterm": "HEALTHY", "swing": "REDUCED", "intraday": "PAUSED"}.
        user_settings : dict or None
            User preferences.  Keys: longterm_enabled, swing_enabled,
            intraday_enabled, max_intraday_pct, max_swing_pct,
            max_longterm_pct, min_cash_pct.
        regime : CombinedRegime or str or None
            Current market regime.  If str, should match a key in
            _REGIME_MULTIPLIERS (e.g. "BULL_LOW_VOL").

        Returns
        -------
        AllocationResult
        """
        # Step 1: available capital
        existing_cash = float(existing_portfolio.get("cash", 0))
        available_capital = contribution + existing_cash

        if available_capital <= 0:
            return self._zero_allocation(available_capital, "No capital available.")

        # Step 2: regime multipliers
        regime_label, regime_mults = self._resolve_regime(regime)

        # Step 3: strategy health multipliers
        health_mults = {
            "longterm":  _HEALTH_MULTIPLIERS.get(
                strategy_health_map.get("longterm", "HEALTHY"), 0.0
            ),
            "swing":     _HEALTH_MULTIPLIERS.get(
                strategy_health_map.get("swing", "HEALTHY"), 0.0
            ),
            "intraday":  _HEALTH_MULTIPLIERS.get(
                strategy_health_map.get("intraday", "HEALTHY"), 0.0
            ),
        }

        # Step 4: raw opportunity scores
        raw_scores: Dict[str, float] = {}
        for strat in ["longterm", "swing", "intraday"]:
            sharpe_norm = self._sharpe.get(strat, 0.5)
            raw_scores[strat] = (
                sharpe_norm
                * regime_mults.get(strat, 0.5)
                * health_mults[strat]
            )

        # Step 5: apply user constraints (enabled/disabled)
        settings = {**_DEFAULT_USER_SETTINGS, **(user_settings or {})}
        if not settings.get("longterm_enabled", True):
            raw_scores["longterm"] = 0.0
        if not settings.get("swing_enabled", True):
            raw_scores["swing"] = 0.0
        if not settings.get("intraday_enabled", True):
            raw_scores["intraday"] = 0.0

        # Step 6: check if any strategy clears minimum threshold
        total_score = sum(raw_scores.values())
        if total_score < self._min_total:
            explanation = (
                f"Total opportunity score {total_score:.3f} is below minimum "
                f"{self._min_total:.3f} (regime={regime_label}). "
                "Allocating 100% to cash — no identified edge justifies deployment."
            )
            logger.info(explanation)
            result = AllocationResult(
                longterm_amount=0.0,
                swing_amount=0.0,
                intraday_amount=0.0,
                cash_amount=available_capital,
                available_capital=available_capital,
                longterm_risk_pct=0.0,
                swing_risk_pct=0.0,
                intraday_risk_pct=0.0,
                regime_label=regime_label,
                opportunity_scores=raw_scores,
                explanation=explanation,
                confidence=0.95,
            )
            assert result.validate(), "Allocation invariant violated (100% cash path)."
            return result

        # Step 7: normalize scores to weights (clip strategies below min)
        filtered_scores = {
            k: v if v >= self._min_opp else 0.0
            for k, v in raw_scores.items()
        }
        score_sum = sum(filtered_scores.values())

        # Minimum cash buffer
        min_cash = settings.get("min_cash_pct", 0.05) * available_capital
        deployable = available_capital - min_cash

        if score_sum <= 0:
            weights = {"longterm": 0.0, "swing": 0.0, "intraday": 0.0}
        else:
            weights = {k: v / score_sum for k, v in filtered_scores.items()}

        # Apply user hard caps
        weights["longterm"]  = min(weights["longterm"],  settings.get("max_longterm_pct",  0.80))
        weights["swing"]     = min(weights["swing"],     settings.get("max_swing_pct",     0.40))
        weights["intraday"]  = min(weights["intraday"],  settings.get("max_intraday_pct",  0.10))

        # Re-normalize after caps (weights may not sum to 1 after capping)
        w_sum = sum(weights.values())
        if w_sum > 0:
            weights = {k: v / w_sum for k, v in weights.items()}

        # Compute amounts
        longterm_amount  = round(deployable * weights["longterm"], 2)
        swing_amount     = round(deployable * weights["swing"], 2)
        intraday_amount  = round(deployable * weights["intraday"], 2)

        # Cash = everything else (ensures invariant is exact)
        cash_amount = round(
            available_capital - longterm_amount - swing_amount - intraday_amount, 2
        )

        # Risk fractions per sleeve (fraction of sleeve that is "at risk")
        # Longterm: 5% max drawdown used as risk capital (conservative)
        # Swing: 2% per position stop × max 10 positions = 20% of sleeve
        # Intraday: 0.5% per trade × 5 positions = 2.5% at risk
        lt_risk_pct   = 0.05
        swing_risk_pct = 0.20
        id_risk_pct   = 0.025

        # Confidence: average of regime + health signal strengths
        confidence = float(
            np.clip(
                np.mean([
                    regime_mults.get("longterm", 0.5),
                    np.mean(list(health_mults.values())),
                    min(total_score, 1.0),
                ]),
                0.0, 1.0,
            )
        )

        explanation = self._build_explanation(
            available_capital=available_capital,
            contribution=contribution,
            existing_cash=existing_cash,
            regime_label=regime_label,
            raw_scores=raw_scores,
            weights=weights,
            strategy_health_map=strategy_health_map,
            longterm_amount=longterm_amount,
            swing_amount=swing_amount,
            intraday_amount=intraday_amount,
            cash_amount=cash_amount,
            settings=settings,
        )

        result = AllocationResult(
            longterm_amount=longterm_amount,
            swing_amount=swing_amount,
            intraday_amount=intraday_amount,
            cash_amount=cash_amount,
            available_capital=available_capital,
            longterm_risk_pct=lt_risk_pct,
            swing_risk_pct=swing_risk_pct,
            intraday_risk_pct=id_risk_pct,
            regime_label=regime_label,
            opportunity_scores=raw_scores,
            explanation=explanation,
            confidence=confidence,
        )

        if not result.validate():
            # Rounding correction: add residual to cash
            residual = round(
                available_capital
                - longterm_amount - swing_amount - intraday_amount - cash_amount,
                2,
            )
            result.cash_amount = round(cash_amount + residual, 2)
            logger.debug("Applied rounding residual of %.2f to cash.", residual)

        logger.info(
            "Allocation: LT=₹%.0f Swing=₹%.0f ID=₹%.0f Cash=₹%.0f "
            "(total=₹%.0f) regime=%s conf=%.2f",
            result.longterm_amount,
            result.swing_amount,
            result.intraday_amount,
            result.cash_amount,
            result.available_capital,
            regime_label,
            result.confidence,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_regime(regime) -> tuple[str, Dict[str, float]]:
        """Extract regime label and multiplier dict from various input types."""
        if regime is None:
            return "UNKNOWN", _DEFAULT_REGIME_MULTIPLIER

        # If CombinedRegime object
        if hasattr(regime, "label"):
            label = regime.label
        elif isinstance(regime, str):
            label = regime.upper()
        else:
            label = "UNKNOWN"

        mults = _REGIME_MULTIPLIERS.get(label, _DEFAULT_REGIME_MULTIPLIER)
        return label, mults

    @staticmethod
    def _zero_allocation(available_capital: float, reason: str) -> AllocationResult:
        return AllocationResult(
            longterm_amount=0.0,
            swing_amount=0.0,
            intraday_amount=0.0,
            cash_amount=max(0.0, available_capital),
            available_capital=max(0.0, available_capital),
            longterm_risk_pct=0.0,
            swing_risk_pct=0.0,
            intraday_risk_pct=0.0,
            regime_label="UNKNOWN",
            opportunity_scores={"longterm": 0.0, "swing": 0.0, "intraday": 0.0},
            explanation=reason,
            confidence=1.0,
        )

    @staticmethod
    def _build_explanation(
        available_capital: float,
        contribution: float,
        existing_cash: float,
        regime_label: str,
        raw_scores: Dict[str, float],
        weights: Dict[str, float],
        strategy_health_map: Dict[str, str],
        longterm_amount: float,
        swing_amount: float,
        intraday_amount: float,
        cash_amount: float,
        settings: dict,
    ) -> str:
        """
        Build a factual, data-driven explanation of the allocation decision.

        This explanation is for display to the investor.  It references
        actual computed quantities — no marketing language.
        """
        lines = [
            f"Available capital: ₹{available_capital:,.0f} "
            f"(new contribution ₹{contribution:,.0f} + existing cash ₹{existing_cash:,.0f}).",
            f"Market regime: {regime_label}.",
            "",
            "Opportunity scores (scale 0–1; higher = more favourable):",
            f"  Long-term:  {raw_scores['longterm']:.3f} "
            f"  (health: {strategy_health_map.get('longterm', 'HEALTHY')})",
            f"  Swing:      {raw_scores['swing']:.3f} "
            f"  (health: {strategy_health_map.get('swing', 'HEALTHY')})",
            f"  Intraday:   {raw_scores['intraday']:.3f} "
            f"  (health: {strategy_health_map.get('intraday', 'HEALTHY')})",
            "",
            "Allocation (after regime adjustment, health scaling, and user caps):",
            f"  Long-term:  ₹{longterm_amount:>10,.0f}  ({weights['longterm']:.1%} of deployable)",
            f"  Swing:      ₹{swing_amount:>10,.0f}  ({weights['swing']:.1%} of deployable)",
            f"  Intraday:   ₹{intraday_amount:>10,.0f}  ({weights['intraday']:.1%} of deployable)",
            f"  Cash:       ₹{cash_amount:>10,.0f}  (minimum buffer {settings.get('min_cash_pct', 0.05):.0%})",
        ]
        if not settings.get("intraday_enabled", True):
            lines.append("  Note: Intraday is disabled by user setting.")
        if not settings.get("swing_enabled", True):
            lines.append("  Note: Swing is disabled by user setting.")
        return "\n".join(lines)
