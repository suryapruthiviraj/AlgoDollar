"""
base.py — Abstract base for all AlgoDollar trading strategies.

Design principles
-----------------
- Paper mode is DEFAULT.  Live trading requires explicit opt-in.
- No forced trades: generate_signals() can return an empty list.
- Position sizing is risk-driven, not capital-driven.
- Strategies self-monitor health and downgrade gracefully.
- All signals are timestamped and carry the feature snapshot used to
  generate them (for reproducibility and debugging).
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared cost constants
# ---------------------------------------------------------------------------
# These are calibrated against app.backtesting.costs.ZerodhaCostModel
# (2024 Zerodha rates, NSE cash) plus a slippage allowance.  They are the
# single source of truth for the strategies' pre-trade cost hurdles; the
# execution-time cost model remains authoritative for realised P&L.
#
#   ZerodhaCostModel().breakeven_return(qty, price, product=...) gives
#   round-trip STATUTORY + BROKERAGE cost only:
#       MIS (intraday) : 4.5 - 10.7 bps depending on ticket size (~8 bps typical)
#       CNC (delivery) : 22.6 - 25.5 bps depending on ticket size (~24 bps typical)
#   Slippage is NOT in that model and must be added by the caller:
#       ~5 bps per leg on liquid large caps => ~10 bps round trip.
INTRADAY_FEE_ROUND_TRIP = 0.0008   # ~8 bps  (ZerodhaCostModel, MIS)
DELIVERY_FEE_ROUND_TRIP = 0.0024   # ~24 bps (ZerodhaCostModel, CNC)
SLIPPAGE_ROUND_TRIP     = 0.0010   # ~5 bps per leg on liquid names

#: Round-trip cost hurdle for an intraday (MIS) trade, as a fraction of
#: notional.  ~18 bps.
INTRADAY_ROUND_TRIP_COST = INTRADAY_FEE_ROUND_TRIP + SLIPPAGE_ROUND_TRIP

#: Round-trip cost hurdle for a delivery (CNC) trade, as a fraction of
#: notional.  ~34 bps.
DELIVERY_ROUND_TRIP_COST = DELIVERY_FEE_ROUND_TRIP + SLIPPAGE_ROUND_TRIP


# ---------------------------------------------------------------------------
# Portfolio budget
# ---------------------------------------------------------------------------
#: A strategy's emitted signal set may never intend more than 100% of the
#: capital allocated to that sleeve.  Per-position caps alone do NOT bound
#: gross exposure (10 positions x 30% cap = 300% gross), so the whole batch
#: is normalised against this budget.
MAX_GROSS_EXPOSURE = 1.0

#: Minimum acceptable target/stop ratio for an ENTRY signal.
MIN_RISK_REWARD = 1.0


class RiskEngineError(RuntimeError):
    """
    Raised when the risk engine cannot render a verdict on a trade.

    Deliberately fatal: an unavailable or broken risk engine must BLOCK the
    trade (fail closed).  Never swallow this to "keep trading".
    """


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StrategyHealth(str, Enum):
    """
    Strategy operational health.

    HEALTHY   : Operating normally.
    REDUCED   : Operating at reduced capacity (e.g. smaller position sizes,
                fewer signals) due to recent underperformance.
    PAUSED    : Not generating new signals; existing positions managed to exit.
    DISABLED  : Completely inactive.  Requires manual intervention to re-enable.
    """
    HEALTHY  = "HEALTHY"
    REDUCED  = "REDUCED"
    PAUSED   = "PAUSED"
    DISABLED = "DISABLED"


class SignalDirection(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    EXIT  = "EXIT"


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """
    A trading signal produced by a strategy.

    Fields
    ------
    symbol : str
    direction : SignalDirection
    strategy_name : str
    timestamp : datetime
        Wall-clock time the signal was generated.  Used to detect stale signals.
    signal_date : datetime
        The market date whose data produced this signal.
    edge_score : float
        Expected net edge after estimated costs.  Signals with edge_score ≤ 0
        should not be executed.
    expected_return : float
        Model's point estimate of forward return OVER ``holding_period_days``,
        expressed as a simple fraction of notional (0.004 = 40 bps).

        UNITS CONTRACT — this is the single most abused field in the system.
        ``expected_return`` and ``edge_score`` MUST be expressed over exactly
        the horizon named by ``holding_period_days``.  A 12-month momentum
        number stored here alongside ``holding_period_days=5`` inflates every
        downstream consumer (sizing, ranking, Kelly, risk budgeting) by ~50x.
        Producers must state the horizon conversion they applied in
        ``metadata["return_units"]``.
    expected_return_std : float
        Uncertainty (1 sigma) in the expected return estimate, same horizon
        and units as ``expected_return``.  Must be >= 0.
    stop_loss_pct : float
        Stop-loss level as fraction of entry price.  E.g. 0.02 = 2%.
    target_pct : float
        Target as fraction of entry price.
    holding_period_days : int
        Expected holding horizon.  Not a hard constraint.
    feature_snapshot : dict
        Copy of the feature values at signal generation.  Used for debugging.
    metadata : dict
        Any strategy-specific extra info.
    """
    symbol: str
    direction: SignalDirection
    strategy_name: str
    timestamp: datetime
    signal_date: datetime
    edge_score: float
    expected_return: float
    expected_return_std: float
    stop_loss_pct: float
    target_pct: float
    holding_period_days: int
    feature_snapshot: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """
        Gate every signal must pass before it may be sized or executed.

        An ENTRY signal is valid only if:
          1. edge_score is finite and > 0 (positive expected NET edge — this,
             not the risk/reward ratio, is the economic test);
          2. expected_return / expected_return_std are finite and the std is
             non-negative;
          3. it carries a usable stop (stop_loss_pct > 0);
          4. its risk/reward ratio exceeds MIN_RISK_REWARD.

        (4) is the enforcement point for the documented "RR must be > 1"
        rule.  It used to be documented but unenforced: risk_reward_ratio()
        had zero call sites.  Note that RR is fixed by construction inside
        each strategy, so on its own it is a shape check, not an edge test —
        the edge test is (1).

        EXIT signals are always actionable: closing risk is never gated.
        """
        if self.direction == SignalDirection.EXIT:
            return True

        if not np.isfinite(self.edge_score) or self.edge_score <= 0.0:
            return False
        if not np.isfinite(self.expected_return):
            return False
        if not np.isfinite(self.expected_return_std) or self.expected_return_std < 0.0:
            return False
        if self.stop_loss_pct <= 0.0:
            return False

        rr = self.risk_reward_ratio()
        if not np.isfinite(rr) or rr <= MIN_RISK_REWARD:
            logger.warning(
                "Signal %s/%s rejected: risk/reward %.2f <= %.2f "
                "(target=%.4f, stop=%.4f).",
                self.strategy_name, self.symbol, rr, MIN_RISK_REWARD,
                self.target_pct, self.stop_loss_pct,
            )
            return False
        return True

    def risk_reward_ratio(self) -> float:
        """
        target_pct / stop_loss_pct.

        Descriptive only — enforced (must exceed MIN_RISK_REWARD) by
        is_valid().  A high RR is NOT evidence of edge: it is chosen by the
        strategy, not estimated from data.  Use breakeven_win_rate() to see
        what the RR actually demands of the hit rate.
        """
        if self.stop_loss_pct <= 0:
            return np.nan
        return self.target_pct / self.stop_loss_pct

    def breakeven_win_rate(self) -> float:
        """
        Hit rate required for this stop/target pair to break even, ignoring
        costs:  p* = stop / (stop + target) = 1 / (1 + RR).
        """
        rr = self.risk_reward_ratio()
        if not np.isfinite(rr) or rr <= 0:
            return np.nan
        return 1.0 / (1.0 + rr)


# ---------------------------------------------------------------------------
# Performance thresholds for health auto-update
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    """Rolling performance metrics fed to update_health()."""
    rolling_sharpe_30d: float    # 30-day rolling annualized Sharpe
    rolling_sharpe_90d: float    # 90-day rolling annualized Sharpe
    current_drawdown_pct: float  # Current drawdown from peak (positive = drawdown)
    win_rate_30d: float          # Win rate over last 30 trades
    num_trades_30d: int          # Trade count for statistical significance
    num_observations_30d: int = 0
    """
    Number of RETURN OBSERVATIONS behind rolling_sharpe_30d (e.g. 21 for a
    30-calendar-day window of daily returns).  This is what determines the
    standard error of the Sharpe estimate and therefore whether a degradation
    transition is statistically defensible.  Defaults to 0, which falls back
    to num_trades_30d and — being far below the minimum sample size — blocks
    Sharpe-driven degradation entirely rather than acting on an unknown n.
    """

    def sharpe_sample_size(self) -> int:
        """Observation count backing the 30d Sharpe (falls back to trade count)."""
        return int(self.num_observations_30d or self.num_trades_30d or 0)


# Health thresholds (conservative defaults; tune per strategy).
#
# The Sharpe levels below are the LEVELS BEING TESTED, not raw trip-wires:
# a transition fires only when the observed Sharpe sits more than
# _SHARPE_DEGRADE_Z standard errors BELOW the level (see _sharpe_significantly_below).
# Drawdown, by contrast, is an OBSERVED fact rather than a noisy estimate, so
# drawdown triggers apply unconditionally.
_HEALTH_THRESHOLDS = {
    # (from_state, to_state): condition
    "reduce":  {"sharpe_30d": 0.0,  "drawdown": 0.05},  # Sharpe < 0 OR DD > 5%
    "pause":   {"sharpe_30d": -0.5, "drawdown": 0.10},  # Sharpe < -0.5 OR DD > 10%
    "disable": {"sharpe_30d": -1.0, "drawdown": 0.20},  # Sharpe < -1.0 OR DD > 20%
    "recover": {"sharpe_90d": 0.3,  "drawdown": 0.03},  # Sustained recovery
}

# Statistical-significance controls for DEGRADATION transitions.
#
# A 30-day rolling annualized Sharpe is built on ~21 daily observations, whose
# standard error is ~sqrt(252/21) ~ 3.5.  Comparing that estimate against a raw
# -1.0 trip-wire disables a genuinely profitable strategy (true Sharpe 1.0)
# with probability ~28% PER MONTH — i.e. it is killed by sampling noise roughly
# every 3.5 months.  We therefore require (a) a minimum sample and (b) evidence
# that the true Sharpe is below the level, not merely the point estimate.
_MIN_OBS_FOR_DEGRADATION = 60   # observations required before ANY Sharpe-driven degrade
_SHARPE_DEGRADE_Z = 2.0         # observed Sharpe must be this many SEs below the level
_PERIODS_PER_YEAR = 252         # daily observations


def _annualized_sharpe_se(sharpe_annual: float, n_obs: int,
                          periods_per_year: int = _PERIODS_PER_YEAR) -> float:
    """
    Standard error of an ANNUALIZED Sharpe ratio estimated from n_obs
    observations (Lo, 2002, IID case):

        SE(SR_ann) = sqrt(periods_per_year / n) * sqrt(1 + SR_ann^2 / (2 * periods_per_year))

    Returns +inf for a sample too small to say anything.
    """
    if n_obs is None or n_obs < 2:
        return math.inf
    if not np.isfinite(sharpe_annual):
        return math.inf
    base = math.sqrt(periods_per_year / n_obs)
    correction = math.sqrt(1.0 + (sharpe_annual ** 2) / (2.0 * periods_per_year))
    return base * correction


def _sharpe_significantly_below(
    sharpe_annual: float,
    level: float,
    n_obs: int,
    z: float = _SHARPE_DEGRADE_Z,
    min_obs: int = _MIN_OBS_FOR_DEGRADATION,
) -> bool:
    """
    True only if the observed annualized Sharpe is more than `z` standard
    errors below `level` AND the sample is large enough to be meaningful.

    This is the ONLY way a Sharpe number may drive a degradation transition.
    """
    if n_obs is None or n_obs < min_obs:
        return False
    if not np.isfinite(sharpe_annual):
        return False
    se = _annualized_sharpe_se(sharpe_annual, n_obs)
    if not math.isfinite(se):
        return False
    return sharpe_annual < (level - z * se)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """
    Abstract base class for AlgoDollar trading strategies.

    Subclasses implement the specific signal logic for their time horizon.
    All common infrastructure (health management, signal validation,
    position-size guardrails) lives here.
    """

    # Subclasses must set these
    name: str = "base"
    holding_period: str = "unspecified"

    # Paper mode is default.  Set to False only in production with explicit
    # confirmation from the system's operator.
    paper_mode: bool = True

    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.health: StrategyHealth = StrategyHealth.HEALTHY
        self._health_log: List[dict] = []
        if paper_mode:
            logger.info("Strategy '%s' initialized in PAPER mode.", self.name)
        else:
            logger.warning(
                "Strategy '%s' initialized in LIVE mode. "
                "Ensure risk controls are active.",
                self.name,
            )

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signals(
        self,
        universe: List[str],
        features_df: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame],
    ) -> List[Signal]:
        """
        Produce a list of trading signals.

        Implementations MUST:
        1. Use only data available at the signal generation time (no look-ahead).
        2. Return an empty list if no edge is found (no forced trades).
        3. Return Signal objects with edge_score > 0 only.
        4. Respect self.health — PAUSED and DISABLED must return [].

        Parameters
        ----------
        universe : list[str]
            Eligible symbols (pre-filtered for liquidity, etc.).
        features_df : pd.DataFrame
            Feature panel indexed by (date, symbol) or wide format.
        market_data : dict[str, pd.DataFrame]
            Raw OHLCV data per symbol.

        Returns
        -------
        list[Signal], ranked by edge_score descending.
        """
        ...

    @abstractmethod
    def calculate_position_size(
        self,
        signal: Signal,
        available_capital: float,
        risk_engine,
    ) -> float:
        """
        Calculate position size (number of shares or rupee amount) for a signal.

        Must return 0 if:
          - available_capital <= 0
          - signal.edge_score <= 0
          - risk_engine indicates breach of risk limits

        Parameters
        ----------
        signal : Signal
        available_capital : float (INR)
        risk_engine : RiskEngine instance

        Returns
        -------
        float : rupee value of position (0 = no trade).
        """
        ...

    @abstractmethod
    def should_exit(
        self,
        position: dict,
        current_data: Dict[str, Any],
    ) -> bool:
        """
        Decide whether to exit an open position.

        Parameters
        ----------
        position : dict with at least:
            symbol, entry_price, entry_date, direction, stop_loss, target,
            signal (the original Signal object).
        current_data : dict with current OHLCV, time, features.

        Returns
        -------
        bool : True = exit now.
        """
        ...

    # ------------------------------------------------------------------
    # Health management
    # ------------------------------------------------------------------

    def update_health(
        self,
        metrics: PerformanceMetrics,
        timestamp: Optional[datetime] = None,
    ) -> StrategyHealth:
        """
        Auto-update strategy health based on rolling performance metrics.

        Degradation on the SHARPE path requires statistical evidence, not a
        point estimate crossing a line:
          - at least _MIN_OBS_FOR_DEGRADATION (60) return observations, and
          - the observed Sharpe more than _SHARPE_DEGRADE_Z (2) standard
            errors BELOW the threshold level.
        Degradation on the DRAWDOWN path is unconditional: a drawdown is an
        observed fact, not an estimate.

        Logic (transitions apply in order; first match wins):
        1. HEALTHY → REDUCED  if Sharpe_30d significantly < 0 or drawdown > 5%
        2. HEALTHY/REDUCED → PAUSED if Sharpe_30d significantly < -0.5
           or drawdown > 10%
        3. HEALTHY/REDUCED/PAUSED → DISABLED if Sharpe_30d significantly < -1.0
           or drawdown > 20%
        4. REDUCED/PAUSED → HEALTHY if Sharpe_90d > 0.3 and drawdown < 3%
           (recovery requires at least 10 trades for statistical significance)

        DISABLED is terminal: it is never auto-changed, in either direction.
        Re-enabling requires an operator to set .health explicitly.

        Parameters
        ----------
        metrics : PerformanceMetrics
        timestamp : datetime, defaults to now.

        Returns
        -------
        StrategyHealth (new state)
        """
        ts = timestamp or datetime.utcnow()
        old_health = self.health
        t = _HEALTH_THRESHOLDS

        # DISABLED is sticky and terminal — manual intervention only.  (Without
        # this, a mildly-negative Sharpe would fall through to the PAUSED branch
        # and silently *un*-disable a strategy an operator had killed.)
        if old_health == StrategyHealth.DISABLED:
            return self.health

        sharpe_30 = metrics.rolling_sharpe_30d
        sharpe_90 = metrics.rolling_sharpe_90d
        dd = metrics.current_drawdown_pct
        n_trades = metrics.num_trades_30d
        n_obs = metrics.sharpe_sample_size()

        def _sharpe_degrades(level: float) -> bool:
            return _sharpe_significantly_below(sharpe_30, level, n_obs)

        new_health = old_health

        # Disable (most severe)
        if _sharpe_degrades(t["disable"]["sharpe_30d"]) or dd > t["disable"]["drawdown"]:
            new_health = StrategyHealth.DISABLED
        # Pause
        elif _sharpe_degrades(t["pause"]["sharpe_30d"]) or dd > t["pause"]["drawdown"]:
            new_health = StrategyHealth.PAUSED
        # Reduce
        elif (
            (_sharpe_degrades(t["reduce"]["sharpe_30d"]) or dd > t["reduce"]["drawdown"])
            and old_health == StrategyHealth.HEALTHY
        ):
            new_health = StrategyHealth.REDUCED
        # Recovery (from REDUCED or PAUSED only)
        elif (
            old_health in (StrategyHealth.REDUCED, StrategyHealth.PAUSED)
            and sharpe_90 >= t["recover"]["sharpe_90d"]
            and dd <= t["recover"]["drawdown"]
            and n_trades >= 10
        ):
            new_health = StrategyHealth.HEALTHY

        self.health = new_health
        if new_health != old_health:
            logger.warning(
                "Strategy '%s' health: %s → %s at %s (sharpe30=%.2f, dd=%.1f%%)",
                self.name,
                old_health.value,
                new_health.value,
                ts,
                sharpe_30,
                dd * 100,
            )
            self._health_log.append({
                "timestamp": ts,
                "from": old_health.value,
                "to": new_health.value,
                "sharpe_30d": sharpe_30,
                "sharpe_30d_se": _annualized_sharpe_se(sharpe_30, n_obs),
                "num_observations_30d": n_obs,
                "drawdown": dd,
            })
        return self.health

    # ------------------------------------------------------------------
    # Guard helpers
    # ------------------------------------------------------------------

    def _is_operational(self) -> bool:
        """Return True if strategy can generate new signals."""
        return self.health in (StrategyHealth.HEALTHY, StrategyHealth.REDUCED)

    def _position_size_multiplier(self) -> float:
        """
        Scale factor for position sizes based on health.
        HEALTHY → 1.0, REDUCED → 0.5, PAUSED/DISABLED → 0.0.
        """
        return {
            StrategyHealth.HEALTHY:  1.0,
            StrategyHealth.REDUCED:  0.5,
            StrategyHealth.PAUSED:   0.0,
            StrategyHealth.DISABLED: 0.0,
        }[self.health]

    # ------------------------------------------------------------------
    # Risk-engine gate (fail CLOSED)
    # ------------------------------------------------------------------

    @staticmethod
    def _risk_engine_approves(risk_engine, symbol: str, size: float, sleeve: str) -> bool:
        """
        Ask the risk engine to approve a trade.  FAILS CLOSED.

        Previously this call was wrapped in ``except Exception: pass``, so a
        broken or unreachable risk engine silently PERMITTED every trade —
        the exact opposite of what a risk control is for.  Any exception now
        propagates as RiskEngineError and blocks the trade.

        ``risk_engine=None`` means "no risk engine wired in" (backtests) and
        is treated as approval; an object that *is* supplied but cannot answer
        is a malfunction and blocks.
        """
        if risk_engine is None:
            return True

        approve = getattr(risk_engine, "approve_trade", None)
        if approve is None:
            raise RiskEngineError(
                f"Risk engine {type(risk_engine).__name__} has no approve_trade(); "
                f"cannot verify {sleeve} trade in {symbol} — blocking."
            )
        try:
            verdict = approve(symbol, size, sleeve)
        except Exception as exc:  # noqa: BLE001 — deliberately re-raised
            logger.error(
                "Risk engine raised on %s (%s, size=%.2f): %s — BLOCKING trade.",
                symbol, sleeve, size, exc,
            )
            raise RiskEngineError(
                f"Risk engine failed to approve {symbol} ({sleeve}): {exc}"
            ) from exc
        return bool(verdict)

    # ------------------------------------------------------------------
    # Portfolio budget
    # ------------------------------------------------------------------

    def _stamp_target_weights(
        self,
        signals: List[Signal],
        raw_weights: List[float],
        budget: float = MAX_GROSS_EXPOSURE,
    ) -> None:
        """
        Normalise per-signal target weights across the EMITTED SIGNAL SET and
        record them in ``signal.metadata["target_weight"]``.

        Per-position caps do not bound a portfolio: 10 swing positions at the
        30% per-position cap is 300% gross.  Sizing therefore has to be a
        batch decision.  After this call,
        ``sum(s.metadata["target_weight"]) <= budget`` always holds.
        """
        total = float(sum(max(0.0, w) for w in raw_weights))
        scale = 1.0 if total <= budget or total <= 0 else budget / total
        if scale < 1.0:
            logger.info(
                "%s: gross intent %.1f%% of sleeve exceeds %.0f%% budget — "
                "scaling all %d position weights by %.3f.",
                self.name, total * 100, budget * 100, len(signals), scale,
            )
        for sig, w in zip(signals, raw_weights):
            sig.metadata["target_weight"] = float(max(0.0, w) * scale)
            sig.metadata["target_weight_raw"] = float(max(0.0, w))
            sig.metadata["portfolio_budget"] = float(budget)

    @staticmethod
    def _entry_budget(
        max_positions: int,
        num_existing: int,
        budget: float = MAX_GROSS_EXPOSURE,
    ) -> float:
        """
        Fraction of the sleeve still available to NEW signals.

        Existing positions are already consuming the sleeve, so a full book
        must not be handed another 100% of it.  Reserves pro-rata by slot.
        """
        if max_positions <= 0:
            return 0.0
        held_fraction = min(1.0, max(0, num_existing) / float(max_positions))
        return max(0.0, budget * (1.0 - held_fraction))

    def allocate_capital(
        self,
        signals: Iterable[Signal],
        available_capital: float,
        risk_engine=None,
        max_gross_exposure: float = MAX_GROSS_EXPOSURE,
    ) -> Dict[str, float]:
        """
        Size a whole signal batch under a hard portfolio budget.

        This is the API callers should prefer over calling
        calculate_position_size() signal-by-signal, because only the batch
        view can enforce a gross-exposure limit.  The returned sizes are
        guaranteed to satisfy::

            sum(sizes.values()) <= available_capital * max_gross_exposure

        Risk-engine failures propagate (RiskEngineError) — they never
        silently permit the batch.
        """
        sizes: Dict[str, float] = {}
        if available_capital <= 0:
            return sizes

        for sig in signals:
            if sig.direction == SignalDirection.EXIT:
                continue
            size = self.calculate_position_size(sig, available_capital, risk_engine)
            if size > 0:
                sizes[sig.symbol] = sizes.get(sig.symbol, 0.0) + float(size)

        budget_rs = available_capital * max_gross_exposure
        total = sum(sizes.values())
        if total > budget_rs and total > 0:
            scale = budget_rs / total
            logger.warning(
                "%s: batch sizing intended %.1f%% of sleeve capital — "
                "scaling down by %.3f to respect the %.0f%% budget.",
                self.name, total / available_capital * 100, scale,
                max_gross_exposure * 100,
            )
            sizes = {k: v * scale for k, v in sizes.items()}
        return sizes

    # ------------------------------------------------------------------
    # Common stop-loss / target checks
    # ------------------------------------------------------------------

    @staticmethod
    def _hit_stop(
        position: dict, current_price: float, atr_buffer: float = 0.0
    ) -> bool:
        """Return True if current_price has hit or passed the stop-loss level."""
        direction = position.get("direction", "LONG")
        stop = position["stop_loss"]
        if direction == "LONG":
            return current_price <= stop * (1 - atr_buffer)
        else:
            return current_price >= stop * (1 + atr_buffer)

    @staticmethod
    def _hit_target(position: dict, current_price: float) -> bool:
        """Return True if current_price has hit or passed the profit target."""
        direction = position.get("direction", "LONG")
        target = position["target"]
        if direction == "LONG":
            return current_price >= target
        else:
            return current_price <= target

    def get_health_log(self) -> List[dict]:
        """Return audit log of all health transitions."""
        return list(self._health_log)
