"""
rebalancer.py — Portfolio rebalancing logic for AlgoDollar.

Principles
----------
- Rebalance ONLY if the expected benefit (drift correction) exceeds the
  rebalancing cost plus a buffer.  Never rebalance for tiny drift.
- Net out buys and sells across the portfolio to minimize gross turnover.
- All cost estimates use the ZerodhaCostModel or a supplied cost model.
- Paper mode by default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Minimum deviation (absolute weight difference) to even consider rebalancing
_MIN_DRIFT_TO_CONSIDER = 0.01  # 1%

# Buffer on top of transaction cost (to avoid excessive churn)
_COST_BUFFER_MULTIPLIER = 1.5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RebalanceTrade:
    """
    A single rebalancing trade.

    Attributes
    ----------
    symbol : str
    action : str, 'BUY' or 'SELL'.
    current_value : float, INR.
    target_value : float, INR.
    trade_value : float, INR (target - current).  Positive = buy, negative = sell.
    shares : float, approximate number of shares (value / price).
    estimated_cost : float, INR (brokerage + STT + taxes).
    """
    symbol: str
    action: str
    current_value: float
    target_value: float
    trade_value: float
    shares: float
    current_weight: float
    target_weight: float
    estimated_cost: float


@dataclass
class RebalanceResult:
    trades: List[RebalanceTrade]
    total_turnover: float       # gross value of all trades (buys + sells)
    total_cost: float           # estimated transaction costs
    expected_drift_reduction: float  # weighted mean weight deviation before vs after
    rebalance_justified: bool   # True if benefit > cost


# ---------------------------------------------------------------------------
# Rebalancer
# ---------------------------------------------------------------------------

class Rebalancer:
    """
    Compute minimum-turnover rebalancing trades.

    Parameters
    ----------
    cost_model : optional
        Instance with a ``calculate_costs(transaction_type, qty, price, ...)``
        method returning a CostBreakdown with a ``total`` attribute.
        If None, uses a flat 20 bps estimate.
    """

    _DEFAULT_COST_RATE = 0.0020  # 20 bps round-trip estimate if no model supplied

    def __init__(self, cost_model=None):
        self._cost_model = cost_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        total_capital: float,
        threshold: float = 0.05,
        prices: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Decide whether rebalancing is worthwhile.

        Rebalance only if:
        1. At least one weight deviates by more than `threshold` from target.
        2. The expected benefit (tracking error reduction) > estimated cost × buffer.

        Parameters
        ----------
        current_weights : dict[symbol, weight]
        target_weights : dict[symbol, weight]
        total_capital : float (INR)
        threshold : float, deviation threshold (fraction).
        prices : dict[symbol, price] (optional, for cost estimation).

        Returns
        -------
        bool
        """
        all_symbols = set(current_weights) | set(target_weights)
        max_deviation = 0.0
        total_deviation = 0.0

        for sym in all_symbols:
            curr_w = current_weights.get(sym, 0.0)
            tgt_w  = target_weights.get(sym, 0.0)
            dev = abs(curr_w - tgt_w)
            max_deviation = max(max_deviation, dev)
            total_deviation += dev

        if max_deviation < _MIN_DRIFT_TO_CONSIDER:
            logger.debug("Max deviation %.3f < minimum %.3f; no rebalance needed.",
                         max_deviation, _MIN_DRIFT_TO_CONSIDER)
            return False

        if max_deviation < threshold:
            logger.debug("Max deviation %.3f < threshold %.3f; no rebalance.",
                         max_deviation, threshold)
            return False

        # Estimate cost as fraction of total capital to be traded
        total_deviation / 2.0  # each deviation is double-counted
        estimated_cost_pct = self._estimate_cost_pct(prices)
        total_cost_pct = estimated_cost_pct * _COST_BUFFER_MULTIPLIER

        # Benefit proxy: weighted deviation reduction (assume full correction)
        benefit_pct = total_deviation * 0.5  # conservative: half the drift corrected

        should = benefit_pct > total_cost_pct
        logger.info(
            "should_rebalance: max_dev=%.3f total_dev=%.3f "
            "benefit=%.4f cost=%.4f (×%.1f) → %s",
            max_deviation,
            total_deviation,
            benefit_pct,
            total_cost_pct,
            _COST_BUFFER_MULTIPLIER,
            "YES" if should else "NO",
        )
        return should

    def calculate_trades(
        self,
        current_positions: Dict[str, Dict],
        target_weights: Dict[str, float],
        total_capital: float,
        prices: Optional[Dict[str, float]] = None,
    ) -> RebalanceResult:
        """
        Compute the set of trades required to move from current to target.

        Trades are NETTED: each symbol produces at most one buy or one sell.
        The algorithm:
        1. Compute current values and current weights.
        2. Compute target values from target_weights × total_capital.
        3. Net trade for each symbol = target_value - current_value.
        4. Positive → buy, negative → sell.

        Parameters
        ----------
        current_positions : dict[symbol, dict]
            Each entry has at minimum 'value' (INR) and optionally 'price'.
        target_weights : dict[symbol, float]
            Target portfolio weights; should sum to ≤ 1.
        total_capital : float (INR)
        prices : dict[symbol, float] (optional, for share quantity estimation).

        Returns
        -------
        RebalanceResult
        """
        if not target_weights:
            return RebalanceResult(
                trades=[], total_turnover=0.0, total_cost=0.0,
                expected_drift_reduction=0.0, rebalance_justified=False,
            )

        # Normalize target weights
        w_sum = sum(target_weights.values())
        if w_sum <= 0:
            return RebalanceResult(
                trades=[], total_turnover=0.0, total_cost=0.0,
                expected_drift_reduction=0.0, rebalance_justified=False,
            )
        normalized_target = {k: v / w_sum for k, v in target_weights.items()}

        all_symbols = set(current_positions) | set(normalized_target)
        sum(
            float(pos.get("value", 0)) for pos in current_positions.values()
        )

        trades: List[RebalanceTrade] = []
        total_turnover = 0.0
        total_cost = 0.0
        drift_before = 0.0

        for sym in all_symbols:
            pos = current_positions.get(sym, {})
            current_val = float(pos.get("value", 0.0))
            current_w   = current_val / total_capital if total_capital > 0 else 0.0
            target_w    = normalized_target.get(sym, 0.0)
            target_val  = target_w * total_capital
            trade_val   = target_val - current_val
            drift_before += abs(current_w - target_w)

            # Skip tiny trades (below 0.1% of capital or ₹500)
            min_trade = max(total_capital * 0.001, 500.0)
            if abs(trade_val) < min_trade:
                continue

            action = "BUY" if trade_val > 0 else "SELL"
            price = float(pos.get("price", prices.get(sym, 1.0) if prices else 1.0))
            shares = abs(trade_val) / price if price > 0 else 0.0

            # Cost estimate
            est_cost = self._estimate_trade_cost(action, abs(trade_val), price, sym)

            trades.append(RebalanceTrade(
                symbol=sym,
                action=action,
                current_value=current_val,
                target_value=target_val,
                trade_value=trade_val,
                shares=shares,
                current_weight=current_w,
                target_weight=target_w,
                estimated_cost=est_cost,
            ))
            total_turnover += abs(trade_val)
            total_cost += est_cost

        # Drift after (ideal: 0 for all symbols)
        drift_after = 0.0  # full correction assumed
        drift_reduction = drift_before - drift_after

        justified = (drift_reduction * total_capital) > (total_cost * _COST_BUFFER_MULTIPLIER)

        logger.info(
            "Rebalancer: %d trades | turnover=₹%.0f | cost=₹%.0f | justified=%s",
            len(trades),
            total_turnover,
            total_cost,
            justified,
        )
        return RebalanceResult(
            trades=sorted(trades, key=lambda t: abs(t.trade_value), reverse=True),
            total_turnover=total_turnover,
            total_cost=total_cost,
            expected_drift_reduction=drift_reduction,
            rebalance_justified=justified,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _estimate_trade_cost(
        self,
        action: str,
        trade_value: float,
        price: float,
        symbol: str,
    ) -> float:
        """Estimate transaction cost for a single trade."""
        if self._cost_model is not None:
            try:
                qty = int(trade_value / price) if price > 0 else 0
                breakdown = self._cost_model.calculate_costs(
                    transaction_type=action,
                    qty=qty,
                    price=price,
                    exchange="NSE",
                    product="CNC",
                )
                return float(breakdown.total)
            except Exception as exc:
                logger.debug("Cost model error for %s: %s; using flat rate.", symbol, exc)

        # Flat rate fallback
        return trade_value * self._DEFAULT_COST_RATE

    def _estimate_cost_pct(self, prices: Optional[Dict[str, float]]) -> float:
        """Estimate average round-trip cost as a fraction of trade value."""
        return self._DEFAULT_COST_RATE
