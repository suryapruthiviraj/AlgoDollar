"""
allocator.py — Capital allocation engine for AlgoDollar.

This is the CORE of the system: given a fresh contribution (or rebalance
event), it decides how much capital to deploy across long-term, swing, and
intraday sleeves — and how much to keep as cash.

Allocation Philosophy
---------------------
- Cash is always a valid asset class.  If no strategy shows sufficient
  expected edge, 100% cash is a legitimate output, and *partial* deployment
  is the normal output: the deployed fraction scales continuously with the
  size of the measured edge.
- Regime and strategy health multiplicatively scale opportunity scores.
  Both are keyed on canonical enums (MarketRegime, StrategyHealth) and an
  unrecognised value RAISES.  A silent default here is how regime detection
  stops influencing allocation without anyone noticing.
- User constraints (disabled strategies, per-sleeve caps, cash floor) are
  HARD constraints on the returned rupee amounts.  They are applied last and
  are never re-normalised away.
- The system never forces a trade.
- All arithmetic is transparent and logged; outputs include a human-readable
  explanation string.

Mathematical flow
-----------------
1. available_capital = contribution + existing_cash.
2. raw_score[s] = base_score[s] * regime_mult[s] * health_mult[s], where
   base_score defaults to the normalised OOS Sharpe of the sleeve.
   Sleeves disabled by the user, or scoring below `min_opportunity_score`,
   are set to 0.
3. total_score = sum(raw_score).  If total_score < min_total_score →
   100% cash.
4. deployed_fraction = clip(total_score / full_deployment_score, 0, 1).
   This is an ABSOLUTE deployment intensity, not a simplex weight: half the
   edge deploys half the capital.
5. target_deploy = available_capital * (1 - min_cash_pct) * deployed_fraction,
   split across sleeves in proportion to their raw scores.
6. Each sleeve amount is clipped in RUPEES against its user cap
   (max_*_pct * available_capital).  Clipped remainder goes to CASH; weights
   are NEVER re-normalised after capping.
7. cash_amount = available_capital - sum(sleeve amounts)  (exact plug).
8. The result is verified: every cap holds and the sum invariant holds, or
   AllocationInvariantError is raised.  These are real raises, not `assert`
   (asserts vanish under `python -O`).

INVARIANT: longterm_amount + swing_amount + intraday_amount + cash_amount
           == available_capital   (exact; cash is the plug)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from app.risk.regime import (
    SLEEVES,
    MarketRegime,
    regime_sleeve_multipliers,
)
from app.strategies.base import StrategyHealth

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class AllocationInvariantError(RuntimeError):
    """
    A produced allocation violated a hard invariant (sum, cap, or sign).

    This is a real exception rather than an `assert` on purpose: assertions
    are stripped when the interpreter runs with -O, which is exactly the
    configuration in which a silent capital-allocation bug is most expensive.
    """


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regime multipliers live in ONE place — app.risk.regime.REGIME_SLEEVE_MULTIPLIERS,
# keyed on the MarketRegime enum.  They are looked up through
# regime_sleeve_multipliers(), which raises KeyError on an unknown regime.
# There is deliberately no local copy and no `_DEFAULT_REGIME_MULTIPLIER`.

# Strategy health multipliers, keyed on the canonical StrategyHealth enum
# from app.strategies.base.
_HEALTH_MULTIPLIERS: Dict[StrategyHealth, float] = {
    StrategyHealth.HEALTHY:  1.00,
    StrategyHealth.REDUCED:  0.50,
    StrategyHealth.PAUSED:   0.00,
    StrategyHealth.DISABLED: 0.00,
}

# Legacy synonyms emitted by app.risk.engine.StrategyHealth.status
# ("ACTIVE | REDUCED | PAUSED | STOPPED").  Mapped explicitly rather than
# silently falling through to 0.0, which used to hand a perfectly healthy
# strategy zero capital.  Anything not listed here raises.
_HEALTH_ALIASES: Dict[str, StrategyHealth] = {
    "ACTIVE":  StrategyHealth.HEALTHY,
    "STOPPED": StrategyHealth.DISABLED,
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

# Total score at which the system deploys everything it is allowed to deploy.
# Reference point: every sleeve at its full normalised Sharpe with a regime
# multiplier of 1.0 (i.e. STRONG_BULL) and full health.  Below it, deployment
# scales down linearly; there is no cliff.
_FULL_DEPLOYMENT_SCORE = sum(_DEFAULT_SHARPE_NORMALIZED.values())  # 1.95

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

_CAP_KEY = {
    "longterm": "max_longterm_pct",
    "swing":    "max_swing_pct",
    "intraday": "max_intraday_pct",
}

_ENABLED_KEY = {
    "longterm": "longterm_enabled",
    "swing":    "swing_enabled",
    "intraday": "intraday_enabled",
}


# ---------------------------------------------------------------------------
# Rupee arithmetic helpers
# ---------------------------------------------------------------------------

def _floor_paise(value: float) -> float:
    """
    Round a rupee amount DOWN to whole paise.

    Rounding down (never up) is what makes the hard caps and the cash floor
    exact rather than "exact ± half a paisa": a deployed amount can only ever
    come in at or below what the model asked for.
    """
    if value <= 0.0:
        return 0.0
    return int(value * 100.0 + 1e-6) / 100.0


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
        == available_capital  (cash is computed as the residual plug)
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

    # Fraction of deployable capital the model actually wanted to deploy,
    # before user caps.  0.0 == "no trade".  Added field, defaulted so that
    # existing keyword construction keeps working.
    deployed_fraction: float = 0.0

    @property
    def deployed_amount(self) -> float:
        return self.longterm_amount + self.swing_amount + self.intraday_amount

    def validate(self, tolerance: float = 1.0) -> bool:
        """
        Return True if the allocation sums to available_capital.

        NOTE: this reports on the amounts as they stand; it does not clamp
        anything.  A negative available_capital (a deficit) is reported
        honestly as a negative cash amount and still validates, because the
        arithmetic genuinely balances.
        """
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
            "opportunity_scores": {
                k: round(v, 4) for k, v in self.opportunity_scores.items()
            },
            "explanation":       self.explanation,
            "confidence":        round(self.confidence, 3),
            "deployed_fraction": round(self.deployed_fraction, 4),
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
    min_opportunity_score : float
        Per-sleeve score below which that sleeve gets nothing.
    min_total_score_for_deployment : float
        Total score below which the answer is 100% cash.
    full_deployment_score : float
        Total score at which the system deploys the whole deployable pot.
        Deployment scales linearly between 0 and this value — this is what
        makes "a weaker edge deploys less capital" true.
    """

    def __init__(
        self,
        sharpe_normalized: Optional[Dict[str, float]] = None,
        min_opportunity_score: float = _MIN_OPPORTUNITY_SCORE,
        min_total_score_for_deployment: float = _MIN_TOTAL_SCORE_FOR_DEPLOYMENT,
        full_deployment_score: float = _FULL_DEPLOYMENT_SCORE,
    ):
        self._sharpe = dict(sharpe_normalized or _DEFAULT_SHARPE_NORMALIZED)
        self._min_opp = float(min_opportunity_score)
        self._min_total = float(min_total_score_for_deployment)
        if not np.isfinite(full_deployment_score) or full_deployment_score <= 0:
            raise ValueError(
                f"full_deployment_score must be finite and > 0, got "
                f"{full_deployment_score!r}"
            )
        self._full_deployment_score = float(full_deployment_score)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(
        self,
        contribution: float,
        existing_portfolio: dict,
        market_data: Optional[dict],
        strategy_health_map: Mapping[str, object],
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
            currently uninvested cash.  A negative net figure is reported
            honestly as a deficit rather than clamped to zero.
        market_data : dict or None
            Live/current-day market context.  Two optional keys are read:
              - "sharpe_normalized": {sleeve: float} — live OOS Sharpe
                estimates that override the constructor defaults for this
                call.
              - "opportunity_scores": {sleeve: float} — base scores used
                directly (takes precedence over sharpe_normalized).
            Any other content is ignored.
        strategy_health_map : mapping[str, StrategyHealth | str]
            Health per sleeve, e.g. {"longterm": StrategyHealth.HEALTHY,
            "swing": "REDUCED", "intraday": "PAUSED"}.  Accepts the
            StrategyHealth enum, its string values, and the legacy
            risk-engine synonyms ACTIVE/STOPPED.  Anything else raises
            KeyError.  A missing sleeve is treated as HEALTHY (logged).
        user_settings : dict or None
            User preferences.  Keys: longterm_enabled, swing_enabled,
            intraday_enabled, max_intraday_pct, max_swing_pct,
            max_longterm_pct, min_cash_pct.  Unknown keys raise ValueError
            (a typo'd cap key must not silently fail to apply).
        regime : MarketRegime | CombinedRegime
            Current market regime.  A MarketRegime member, or any object
            exposing `.price_regime` (e.g. CombinedRegime), or the string
            value of a MarketRegime member.  Unknown values raise KeyError;
            None raises ValueError.

        Returns
        -------
        AllocationResult

        Raises
        ------
        ValueError
            Non-finite inputs, missing/invalid regime, invalid user settings.
        KeyError
            Unknown regime or unknown strategy-health value.
        AllocationInvariantError
            The computed allocation broke the sum invariant or a hard cap.
            (Should be unreachable; it is a guard, not control flow.)
        """
        # Step 1: available capital -------------------------------------
        contribution = self._as_finite(contribution, "contribution")
        existing_cash = self._as_finite(
            existing_portfolio.get("cash", 0) if existing_portfolio else 0,
            "existing_portfolio['cash']",
        )
        available_capital = contribution + existing_cash

        settings = self._resolve_settings(user_settings)

        if available_capital < 0:
            # Honest reporting: the account is short.  Do not clamp to zero —
            # that used to report available=0 and validate()==True while the
            # true figure was, say, -40,000.
            logger.warning(
                "Net capital is negative (contribution ₹%.2f + existing cash "
                "₹%.2f = ₹%.2f). Reporting the deficit; nothing to allocate.",
                contribution, existing_cash, available_capital,
            )
            return self._flat_allocation(
                available_capital,
                (
                    f"Capital deficit of ₹{-available_capital:,.2f} "
                    f"(contribution ₹{contribution:,.2f} + existing cash "
                    f"₹{existing_cash:,.2f}). No capital can be deployed; the "
                    "deficit is reported as negative cash, not hidden."
                ),
                regime_label="UNKNOWN",
            )

        if available_capital == 0:
            return self._flat_allocation(
                0.0, "No capital available.", regime_label="UNKNOWN"
            )

        # Step 2: regime multipliers (enum-keyed, raises on unknown) -----
        regime_label, regime_mults = self._resolve_regime(regime)

        # Step 3: strategy health multipliers (enum-keyed, raises) -------
        health = {s: self._resolve_health(strategy_health_map, s) for s in SLEEVES}
        health_mults = {s: _HEALTH_MULTIPLIERS[health[s]] for s in SLEEVES}

        # Step 4: raw opportunity scores ---------------------------------
        base_scores = self._resolve_base_scores(market_data)
        raw_scores: Dict[str, float] = {
            s: base_scores[s] * regime_mults[s] * health_mults[s] for s in SLEEVES
        }

        # Step 5: user enable/disable ------------------------------------
        for s in SLEEVES:
            if not settings[_ENABLED_KEY[s]]:
                raw_scores[s] = 0.0

        # Step 6: per-sleeve minimum edge --------------------------------
        filtered_scores = {
            s: (v if v >= self._min_opp else 0.0) for s, v in raw_scores.items()
        }
        total_score = sum(filtered_scores.values())

        # Step 7: ABSOLUTE deployment intensity --------------------------
        # The scores are a measure of edge, not a simplex.  Half the edge
        # deploys half the capital; below `min_total` we deploy nothing.
        if total_score < self._min_total:
            deployed_fraction = 0.0
        else:
            deployed_fraction = float(
                np.clip(total_score / self._full_deployment_score, 0.0, 1.0)
            )

        min_cash = settings["min_cash_pct"] * available_capital
        deployable_ceiling = available_capital - min_cash
        target_deploy = deployable_ceiling * deployed_fraction

        # Step 8: split the deployed pot across sleeves, then apply the
        # user's HARD caps in rupees.  No re-normalisation afterwards — the
        # clipped remainder is a decision to hold cash, not a licence to
        # push the money into a different sleeve.
        cap_amounts = {
            s: settings[_CAP_KEY[s]] * available_capital for s in SLEEVES
        }
        amounts: Dict[str, float] = {s: 0.0 for s in SLEEVES}
        capped: Dict[str, bool] = {s: False for s in SLEEVES}
        if total_score > 0 and target_deploy > 0:
            for s in SLEEVES:
                share = filtered_scores[s] / total_score
                want = share * target_deploy
                allowed = min(want, cap_amounts[s])
                capped[s] = allowed < want - 1e-9
                amounts[s] = _floor_paise(allowed)

        # Step 9: cash is the residual plug (exact invariant) ------------
        cash_amount = available_capital - sum(amounts.values())

        # Risk fractions per sleeve (fraction of sleeve that is "at risk")
        # Longterm: 5% max drawdown used as risk capital (conservative)
        # Swing: 2% per position stop × max 10 positions = 20% of sleeve
        # Intraday: 0.5% per trade × 5 positions = 2.5% at risk
        lt_risk_pct = 0.05
        swing_risk_pct = 0.20
        id_risk_pct = 0.025

        # Confidence: how strong the evidence behind this decision is.
        confidence = float(
            np.clip(
                np.mean([
                    float(np.mean(list(regime_mults.values()))),
                    float(np.mean(list(health_mults.values()))),
                    deployed_fraction,
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
            filtered_scores=filtered_scores,
            total_score=total_score,
            deployed_fraction=deployed_fraction,
            amounts=amounts,
            cap_amounts=cap_amounts,
            capped=capped,
            cash_amount=cash_amount,
            strategy_health_map=health,
            settings=settings,
        )

        result = AllocationResult(
            longterm_amount=amounts["longterm"],
            swing_amount=amounts["swing"],
            intraday_amount=amounts["intraday"],
            cash_amount=cash_amount,
            available_capital=available_capital,
            longterm_risk_pct=lt_risk_pct,
            swing_risk_pct=swing_risk_pct,
            intraday_risk_pct=id_risk_pct,
            regime_label=regime_label,
            opportunity_scores=raw_scores,
            explanation=explanation,
            confidence=confidence,
            deployed_fraction=deployed_fraction,
        )

        # Step 10: hard verification.  Explicit raises, never `assert`.
        self._verify(result, settings)

        logger.info(
            "Allocation: LT=₹%.2f Swing=₹%.2f ID=₹%.2f Cash=₹%.2f "
            "(total=₹%.2f) regime=%s deployed=%.1f%% conf=%.2f",
            result.longterm_amount,
            result.swing_amount,
            result.intraday_amount,
            result.cash_amount,
            result.available_capital,
            regime_label,
            100.0 * result.deployed_amount / available_capital,
            result.confidence,
        )
        return result

    # ------------------------------------------------------------------
    # Input resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _as_finite(value, name: str) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number, got {value!r}") from exc
        if not np.isfinite(out):
            raise ValueError(f"{name} must be finite, got {out!r}")
        return out

    @staticmethod
    def _resolve_settings(user_settings: Optional[Dict]) -> Dict:
        """
        Merge user settings over defaults and validate them.

        Unknown keys raise: a mistyped cap key that silently does nothing is
        the same class of bug as a mistyped regime label.
        """
        user_settings = dict(user_settings or {})
        unknown = set(user_settings) - set(_DEFAULT_USER_SETTINGS)
        if unknown:
            raise ValueError(
                f"Unknown user_settings key(s): {sorted(unknown)}. "
                f"Valid keys: {sorted(_DEFAULT_USER_SETTINGS)}"
            )
        settings = {**_DEFAULT_USER_SETTINGS, **user_settings}

        for key in ("max_longterm_pct", "max_swing_pct", "max_intraday_pct",
                    "min_cash_pct"):
            val = float(settings[key])
            if not np.isfinite(val) or not (0.0 <= val <= 1.0):
                raise ValueError(f"{key} must be a fraction in [0, 1], got {val!r}")
            settings[key] = val
        for key in _ENABLED_KEY.values():
            settings[key] = bool(settings[key])
        return settings

    def _resolve_base_scores(self, market_data: Optional[dict]) -> Dict[str, float]:
        """
        Base (pre-regime, pre-health) opportunity score per sleeve.

        Defaults to the normalised Sharpe priors; `market_data` may override
        them with live estimates.  This is the one place market_data is read —
        it used to be accepted and never used at all.
        """
        scores = {s: float(self._sharpe.get(s, 0.5)) for s in SLEEVES}
        if market_data:
            override = (
                market_data.get("opportunity_scores")
                or market_data.get("sharpe_normalized")
            )
            if override:
                for s in SLEEVES:
                    if s in override:
                        scores[s] = self._as_finite(
                            override[s], f"market_data score for {s}"
                        )
        for s, v in scores.items():
            if v < 0:
                raise ValueError(f"Base opportunity score for {s} is negative: {v}")
        return scores

    @staticmethod
    def _resolve_regime(regime) -> Tuple[str, Dict[str, float]]:
        """
        Resolve `regime` to (label, sleeve multipliers).

        Accepts a MarketRegime member, an object exposing `.price_regime`
        (CombinedRegime), or the string value of a MarketRegime member.
        Raises KeyError for anything else and ValueError for None — the old
        `.get(label, _DEFAULT)` meant every unrecognised label silently
        collapsed to the same default multipliers, which is precisely why
        allocations were byte-identical in every regime.
        """
        if regime is None:
            raise ValueError(
                "regime is required: pass a MarketRegime member (or a "
                "CombinedRegime).  Allocating without a regime would silently "
                "ignore market conditions."
            )

        resolved = regime
        if not isinstance(resolved, MarketRegime):
            price_regime = getattr(resolved, "price_regime", None)
            if price_regime is not None:
                resolved = price_regime
        if not isinstance(resolved, MarketRegime) and isinstance(resolved, str):
            try:
                resolved = MarketRegime(resolved.strip().upper())
            except ValueError as exc:
                raise KeyError(
                    f"Unknown market regime {regime!r}. Valid regimes: "
                    f"{[r.value for r in MarketRegime]}"
                ) from exc

        mults = regime_sleeve_multipliers(resolved)  # raises KeyError if unknown
        return resolved.value, mults

    @staticmethod
    def _resolve_health(
        strategy_health_map: Mapping[str, object], sleeve: str
    ) -> StrategyHealth:
        """
        Resolve one sleeve's health to the canonical StrategyHealth enum.

        Raises KeyError on an unrecognised value.  Previously an unrecognised
        value (e.g. the risk engine's own "ACTIVE") mapped to a 0.0 multiplier,
        so a perfectly healthy strategy was handed zero capital in silence.
        """
        if not strategy_health_map or sleeve not in strategy_health_map:
            logger.warning(
                "No health reported for sleeve '%s'; assuming %s.",
                sleeve, StrategyHealth.HEALTHY.value,
            )
            return StrategyHealth.HEALTHY

        value = strategy_health_map[sleeve]
        if isinstance(value, StrategyHealth):
            return value
        if isinstance(value, str):
            key = value.strip().upper()
            if key in _HEALTH_ALIASES:
                return _HEALTH_ALIASES[key]
            try:
                return StrategyHealth(key)
            except ValueError as exc:
                raise KeyError(
                    f"Unknown strategy health {value!r} for sleeve '{sleeve}'. "
                    f"Valid: {[h.value for h in StrategyHealth]} "
                    f"(legacy synonyms: {sorted(_HEALTH_ALIASES)})"
                ) from exc
        raise KeyError(
            f"Unknown strategy health {value!r} for sleeve '{sleeve}'. "
            f"Expected StrategyHealth or str."
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    @staticmethod
    def _verify(result: AllocationResult, settings: Dict) -> None:
        """
        Enforce the hard invariants on the produced allocation.

        Raises AllocationInvariantError.  Deliberately NOT `assert`: assert
        statements are removed by `python -O`, which would delete the only
        thing standing between a cap bug and real money.
        """
        available = result.available_capital
        amounts = {
            "longterm": result.longterm_amount,
            "swing": result.swing_amount,
            "intraday": result.intraday_amount,
        }
        eps = max(1e-6, abs(available) * 1e-12)

        total = sum(amounts.values()) + result.cash_amount
        if abs(total - available) > eps:
            raise AllocationInvariantError(
                f"Allocation does not sum to available capital: "
                f"{total!r} != {available!r} (residual {total - available!r})"
            )

        for sleeve, amount in amounts.items():
            if amount < -eps:
                raise AllocationInvariantError(
                    f"{sleeve} allocation is negative: {amount!r}"
                )
            cap_pct = settings[_CAP_KEY[sleeve]]
            cap_amount = cap_pct * available
            if amount > cap_amount + eps:
                raise AllocationInvariantError(
                    f"{sleeve} allocation ₹{amount:,.2f} exceeds the user cap "
                    f"of {cap_pct:.2%} (₹{cap_amount:,.2f}) of available "
                    f"capital ₹{available:,.2f}"
                )
            if not settings[_ENABLED_KEY[sleeve]] and amount > eps:
                raise AllocationInvariantError(
                    f"{sleeve} is disabled by the user but was allocated "
                    f"₹{amount:,.2f}"
                )

        min_cash_amount = settings["min_cash_pct"] * available
        if result.cash_amount < min_cash_amount - eps:
            raise AllocationInvariantError(
                f"Cash ₹{result.cash_amount:,.2f} is below the user's minimum "
                f"cash floor of {settings['min_cash_pct']:.2%} "
                f"(₹{min_cash_amount:,.2f})"
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flat_allocation(
        available_capital: float, reason: str, regime_label: str = "UNKNOWN"
    ) -> AllocationResult:
        """
        All-cash result.  `available_capital` is reported as-is — including a
        negative (deficit) figure — so the invariant stays true and the
        reported number stays honest.
        """
        return AllocationResult(
            longterm_amount=0.0,
            swing_amount=0.0,
            intraday_amount=0.0,
            cash_amount=available_capital,
            available_capital=available_capital,
            longterm_risk_pct=0.0,
            swing_risk_pct=0.0,
            intraday_risk_pct=0.0,
            regime_label=regime_label,
            opportunity_scores={s: 0.0 for s in SLEEVES},
            explanation=reason,
            confidence=1.0,
            deployed_fraction=0.0,
        )

    def _build_explanation(
        self,
        available_capital: float,
        contribution: float,
        existing_cash: float,
        regime_label: str,
        raw_scores: Dict[str, float],
        filtered_scores: Dict[str, float],
        total_score: float,
        deployed_fraction: float,
        amounts: Dict[str, float],
        cap_amounts: Dict[str, float],
        capped: Dict[str, bool],
        cash_amount: float,
        strategy_health_map: Dict[str, StrategyHealth],
        settings: dict,
    ) -> str:
        """
        Build a factual, data-driven explanation of the allocation decision.

        This explanation is for display to the investor.  It references
        actual computed quantities — no marketing language.
        """
        deployed = sum(amounts.values())
        lines = [
            f"Available capital: ₹{available_capital:,.0f} "
            f"(new contribution ₹{contribution:,.0f} + existing cash "
            f"₹{existing_cash:,.0f}).",
            f"Market regime: {regime_label}.",
            "",
            "Opportunity scores (higher = more favourable):",
        ]
        for s in SLEEVES:
            note = ""
            if not settings[_ENABLED_KEY[s]]:
                note = "  [disabled by user]"
            elif filtered_scores[s] == 0.0 and raw_scores[s] > 0.0:
                note = f"  [below minimum edge {self._min_opp:.2f} → no capital]"
            lines.append(
                f"  {s.capitalize():<10} {raw_scores[s]:.3f}"
                f"  (health: {strategy_health_map[s].value}){note}"
            )
        lines += [
            "",
            f"Total score {total_score:.3f} vs full-deployment reference "
            f"{self._full_deployment_score:.3f} "
            f"→ deploy {deployed_fraction:.1%} of deployable capital.",
        ]
        if deployed_fraction == 0.0:
            lines.append(
                f"  Total score is below the minimum {self._min_total:.3f} — "
                "no identified edge justifies deployment. 100% cash."
            )
        lines += [
            "",
            "Allocation (after regime adjustment, health scaling, and user caps):",
        ]
        for s in SLEEVES:
            pct = amounts[s] / available_capital if available_capital else 0.0
            cap_note = (
                f"  [capped at {settings[_CAP_KEY[s]]:.0%} "
                f"= ₹{cap_amounts[s]:,.0f}]" if capped[s] else ""
            )
            lines.append(
                f"  {s.capitalize():<10} ₹{amounts[s]:>12,.0f}  "
                f"({pct:.2%} of available){cap_note}"
            )
        lines.append(
            f"  {'Cash':<10} ₹{cash_amount:>12,.0f}  "
            f"({(cash_amount / available_capital if available_capital else 0):.2%} "
            f"of available; minimum buffer {settings['min_cash_pct']:.0%})"
        )
        if any(capped.values()):
            lines.append(
                "  Capital released by user caps is held as cash, not "
                "redirected to another sleeve."
            )
        lines.append(
            f"  Deployed ₹{deployed:,.0f} of ₹{available_capital:,.0f} "
            f"({(deployed / available_capital if available_capital else 0):.2%})."
        )
        return "\n".join(lines)
