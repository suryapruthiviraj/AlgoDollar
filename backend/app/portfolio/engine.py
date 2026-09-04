"""
The allocation engine: inputs in, target portfolio out, reasons attached.

ORDER OF OPERATIONS, AND WHY IT IS THIS ORDER
----------------------------------------------
    1. pre-flight gates      -> may return CASH immediately
    2. sleeve CAPITAL split  -> rupees per bucket
    3. sleeve RISK budget    -> volatility share per bucket (a separate decision)
    4. name-level sizing     -> risk-parity-ish within each sleeve
    5. hard constraints      -> position, sector, strategy, liquidity, Kelly cap
    6. delta vs the existing book
    7. turnover limit        -> applied to the DELTA, scaling it down
    8. cash floor            -> applied last, so it cannot be spent by a later step

Every step can only REDUCE exposure. There is no step that increases a position
to consume capital, which is what makes "cash is a valid allocation" structural
rather than aspirational.

Steps 5 and 7 are deliberately separated. A constraint that caps a target
position is a statement about the portfolio; a constraint that caps the trade
toward it is a statement about the cost of getting there. Merging them hides
which one bound.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from app.portfolio.allocation import (
    DEPLOYABLE,
    AllocationInputs,
    AllocationSnapshot,
    BindingConstraint,
    PositionTarget,
    RiskLimits,
    StrategyBucket,
    StrategyTarget,
    TargetPortfolio,
)

logger = logging.getLogger(__name__)

#: Quote age beyond which a price is not usable for sizing.
MAX_QUOTE_AGE_SEC = 300.0

#: Health labels that mean the sleeve may not take NEW capital.
_HEALTH_BLOCKED = {"DISABLED", "PAUSED", "STOPPED"}
#: Health labels that halve a sleeve's capital rather than blocking it.
_HEALTH_REDUCED = {"REDUCED", "DEGRADED"}

#: Baseline risk-budget shares. Not derived from the capital split — that is the
#: whole point. Intraday gets a small capital share and a disproportionately
#: LARGE risk share because intraday positions are more volatile per rupee.
_BASE_RISK_BUDGET = {
    StrategyBucket.LONG_TERM: 0.45,
    StrategyBucket.SWING: 0.35,
    StrategyBucket.INTRADAY: 0.20,
}


class PortfolioAllocationEngine:
    """
    Turns market state and signals into a target portfolio, or into cash.

    Stateless: everything it reads is on :class:`AllocationInputs`, and the same
    inputs always produce the same target.
    """

    def __init__(self, limits: Optional[RiskLimits] = None) -> None:
        self.limits = limits or RiskLimits()

    # -- entry point ------------------------------------------------------- #

    def allocate(self, inputs: AllocationInputs) -> TargetPortfolio:
        limits = inputs.limits or self.limits
        target = TargetPortfolio(
            as_of=inputs.as_of,
            input_fingerprint=inputs.fingerprint(),
            is_no_trade=True,
        )

        blocked = self._preflight(inputs, limits, target)
        if blocked:
            # Held positions are NOT force-liquidated here. A halt means "place
            # no new orders", which is not the same as "sell everything at
            # once" — an unwind is itself a large, costly, market-moving trade
            # and must be an explicit decision, never a side effect of a gate.
            target.cash_reserve = inputs.cash
            target.cash_reserve_pct = _safe_div(inputs.cash, inputs.total_capital)
            target.strategies = [self._cash_only_strategy(inputs, target.reasons)]
            return target

        usable = self._usable_signals(inputs, limits, target)
        if not usable:
            target.reasons.append(
                "No signal survived data-quality and liquidity screening, so "
                "there is nothing to allocate to. Capital stays in cash."
            )
            target.cash_reserve = inputs.cash
            target.cash_reserve_pct = _safe_div(inputs.cash, inputs.total_capital)
            target.strategies = [self._cash_only_strategy(inputs, target.reasons)]
            return target

        # ---- 2. CAPITAL per sleeve ---------------------------------------- #
        capital_by_bucket = self._allocate_capital(inputs, limits, usable, target)

        # ---- 3. RISK budget per sleeve (a separate decision) --------------- #
        risk_by_bucket = self._allocate_risk(inputs, limits, usable, target)

        # ---- 4-5. name-level targets under hard constraints ---------------- #
        positions = self._size_positions(
            inputs, limits, usable, capital_by_bucket, risk_by_bucket, target
        )

        # ---- 6-7. deltas and turnover -------------------------------------- #
        positions = self._apply_turnover_limit(inputs, limits, positions, target)

        # ---- 8. cash floor -------------------------------------------------- #
        positions = self._apply_cash_floor(inputs, limits, positions, target)

        deployed = sum(p.target_value for p in positions)
        cash = max(0.0, inputs.total_capital - deployed)

        target.positions = positions
        target.cash_reserve = cash
        target.cash_reserve_pct = _safe_div(cash, inputs.total_capital)
        target.is_no_trade = not any(p.delta_quantity != 0 for p in positions)

        target.expected_portfolio_vol = self._expected_vol(inputs, positions)
        turnover_value = sum(abs(p.delta_value) for p in positions)
        target.expected_turnover_pct = _safe_div(turnover_value, inputs.total_capital)
        target.estimated_cost = turnover_value * (inputs.cost_bps_per_side / 10_000.0)
        target.estimated_cost_pct = _safe_div(
            target.estimated_cost, inputs.total_capital
        )

        target.strategies = self._strategy_targets(
            inputs, positions, capital_by_bucket, risk_by_bucket, cash
        )

        if target.is_no_trade:
            target.reasons.append(
                "The current book already matches the target within rounding; "
                "no trade is required."
            )
        else:
            target.reasons.append(
                f"Target set from {len(usable)} usable signal(s) across "
                f"{len({p.strategy for p in positions})} sleeve(s)."
            )
        if target.expected_portfolio_vol is not None:
            target.reasons.append(
                f"Expected portfolio volatility {target.expected_portfolio_vol:.1%} "
                f"against a {limits.target_portfolio_vol:.1%} target."
            )
        return target

    # -- 1. pre-flight ----------------------------------------------------- #

    def _preflight(
        self, inputs: AllocationInputs, limits: RiskLimits, target: TargetPortfolio
    ) -> bool:
        """
        Gates that produce CASH regardless of opportunity. Fail closed.

        Returns True when allocation must stop. Each gate states its own reason,
        and ALL failing gates are recorded rather than only the first — an
        operator needs to know everything that is wrong, not the earliest thing.
        """
        blocked = False

        if inputs.kill_switch_active:
            target.reasons.append("KILL SWITCH ACTIVE: no new capital is allocated.")
            blocked = True

        if not inputs.trading_permitted:
            target.reasons.append(
                "Trading is not permitted (startup reconciliation has not "
                "succeeded). No allocation is made."
            )
            blocked = True

        if inputs.market_data_stale:
            target.reasons.append(
                "Market data is stale. Allocating on prices that no longer "
                "reflect the market would size every position wrongly."
            )
            blocked = True

        if inputs.total_capital <= 0:
            target.reasons.append(
                f"Total capital is Rs {inputs.total_capital:,.2f}. There is "
                f"nothing to allocate."
            )
            blocked = True

        dd = abs(float(inputs.current_drawdown_pct))
        if dd >= limits.max_portfolio_drawdown_pct:
            target.reasons.append(
                f"DRAWDOWN BREACH: {dd:.1%} against a "
                f"{limits.max_portfolio_drawdown_pct:.1%} limit. New capital is "
                f"withheld until the drawdown recovers."
            )
            target.binding_constraints.append(BindingConstraint(
                "max_portfolio_drawdown", "portfolio",
                limits.max_portfolio_drawdown_pct, dd, 0.0,
            ))
            blocked = True

        daily_loss = -min(0.0, float(inputs.daily_pnl_pct))
        if daily_loss >= limits.max_daily_loss_pct:
            target.reasons.append(
                f"DAILY LOSS LIMIT: down {daily_loss:.2%} against a "
                f"{limits.max_daily_loss_pct:.2%} limit. No new positions today."
            )
            target.binding_constraints.append(BindingConstraint(
                "max_daily_loss", "portfolio",
                limits.max_daily_loss_pct, daily_loss, 0.0,
            ))
            blocked = True

        return blocked

    def _cash_only_strategy(
        self, inputs: AllocationInputs, reasons: list[str]
    ) -> StrategyTarget:
        return StrategyTarget(
            bucket=StrategyBucket.CASH,
            capital_amount=inputs.total_capital,
            capital_pct=1.0,
            risk_budget_pct=0.0,
            risk_used_pct=0.0,
            n_positions=0,
            health="N/A",
            reason="; ".join(reasons) or "no deployable opportunity",
        )

    # -- signal screening --------------------------------------------------- #

    def _usable_signals(
        self, inputs: AllocationInputs, limits: RiskLimits, target: TargetPortfolio
    ) -> list:
        """
        Drop signals that cannot be sized SAFELY, and say why for each.

        A missing volatility or a missing traded value is not a small gap: it
        means the risk limit and the liquidity limit cannot be applied to that
        name. Sizing it anyway would place a position no constraint is guarding.
        """
        usable = []
        for s in inputs.signals:
            health = str(inputs.strategy_health.get(s.strategy, "HEALTHY")).upper()
            if health in _HEALTH_BLOCKED:
                target.warnings.append(
                    f"{s.symbol}: dropped — strategy '{s.strategy}' is {health}."
                )
                continue
            if s.price is None or not (s.price > 0):
                target.warnings.append(f"{s.symbol}: dropped — no usable price.")
                continue
            if s.quote_age_sec is not None and s.quote_age_sec > MAX_QUOTE_AGE_SEC:
                target.warnings.append(
                    f"{s.symbol}: dropped — quote is {s.quote_age_sec:.0f}s old "
                    f"(limit {MAX_QUOTE_AGE_SEC:.0f}s)."
                )
                continue
            if s.volatility is None or not (s.volatility > 0):
                target.warnings.append(
                    f"{s.symbol}: dropped — no volatility estimate, so it cannot "
                    f"be risk-sized and no risk limit could be applied to it."
                )
                continue
            if s.median_traded_value is None or not (s.median_traded_value > 0):
                target.warnings.append(
                    f"{s.symbol}: dropped — no traded-value estimate, so the "
                    f"liquidity limit cannot be applied."
                )
                continue
            if not (s.edge_score > 0):
                target.warnings.append(
                    f"{s.symbol}: dropped — edge score {s.edge_score:.4f} is not "
                    f"positive."
                )
                continue
            usable.append(s)
        return usable

    # -- 2. capital --------------------------------------------------------- #

    def _allocate_capital(
        self, inputs: AllocationInputs, limits: RiskLimits,
        signals: Sequence, target: TargetPortfolio,
    ) -> dict[StrategyBucket, float]:
        """
        Rupees per sleeve, driven by where the opportunities actually are.

        A sleeve with no usable signal gets NOTHING — capital is not spread
        evenly for tidiness. The deployable total is then capped by the strategy
        limit and by the cash floor.
        """
        deployable = inputs.total_capital * (1.0 - limits.min_cash_pct)
        by_bucket: dict[StrategyBucket, float] = {b: 0.0 for b in DEPLOYABLE}

        strength: dict[StrategyBucket, float] = {}
        for b in DEPLOYABLE:
            in_bucket = [s for s in signals if _bucket_of(s.strategy) is b]
            if not in_bucket:
                continue
            health = str(
                inputs.strategy_health.get(b.value, "HEALTHY")
            ).upper()
            mult = 0.5 if health in _HEALTH_REDUCED else 1.0
            strength[b] = sum(max(0.0, s.edge_score) for s in in_bucket) * mult

        total_strength = sum(strength.values())
        if total_strength <= 0:
            target.reasons.append(
                "No sleeve produced positive aggregate edge; capital stays in cash."
            )
            return by_bucket

        for b, sc in strength.items():
            share = sc / total_strength
            amount = deployable * share
            cap = inputs.total_capital * limits.max_strategy_pct
            if amount > cap:
                target.binding_constraints.append(BindingConstraint(
                    "max_strategy_pct", b.value, limits.max_strategy_pct,
                    _safe_div(amount, inputs.total_capital), limits.max_strategy_pct,
                ))
                amount = cap
            by_bucket[b] = amount
        return by_bucket

    # -- 3. risk (a SEPARATE decision) --------------------------------------- #

    def _allocate_risk(
        self, inputs: AllocationInputs, limits: RiskLimits,
        signals: Sequence, target: TargetPortfolio,
    ) -> dict[StrategyBucket, float]:
        """
        Volatility budget per sleeve. NOT derived from the capital split.

        The budget starts from fixed baseline shares reflecting how much risk
        each sleeve is *permitted* to consume, is zeroed for sleeves with no
        signal, is halved for degraded sleeves, and is then renormalised to the
        portfolio volatility target.

        Deriving this from capital share would make the two numbers identical by
        construction and destroy the only information the comparison carries.
        """
        active = {
            b for b in DEPLOYABLE
            if any(_bucket_of(s.strategy) is b for s in signals)
        }
        if not active:
            return {b: 0.0 for b in DEPLOYABLE}

        raw: dict[StrategyBucket, float] = {}
        for b in DEPLOYABLE:
            if b not in active:
                raw[b] = 0.0
                continue
            health = str(inputs.strategy_health.get(b.value, "HEALTHY")).upper()
            mult = 0.5 if health in _HEALTH_REDUCED else 1.0
            raw[b] = _BASE_RISK_BUDGET[b] * mult

        total = sum(raw.values())
        if total <= 0:
            return {b: 0.0 for b in DEPLOYABLE}

        # Scale the whole budget down when realised volatility already exceeds
        # the target: the portfolio is running hot and should take less new
        # risk, not the same amount.
        scale = 1.0
        if inputs.realised_vol and inputs.realised_vol > limits.target_portfolio_vol:
            scale = limits.target_portfolio_vol / inputs.realised_vol
            target.binding_constraints.append(BindingConstraint(
                "target_portfolio_vol", "portfolio",
                limits.target_portfolio_vol, inputs.realised_vol,
                limits.target_portfolio_vol,
            ))
            target.reasons.append(
                f"Realised volatility {inputs.realised_vol:.1%} exceeds the "
                f"{limits.target_portfolio_vol:.1%} target; the risk budget is "
                f"scaled to {scale:.2f}."
            )
        return {b: (v / total) * scale for b, v in raw.items()}

    # -- 4-5. name-level sizing --------------------------------------------- #

    def _size_positions(
        self, inputs: AllocationInputs, limits: RiskLimits, signals: Sequence,
        capital_by_bucket: Mapping[StrategyBucket, float],
        risk_by_bucket: Mapping[StrategyBucket, float],
        target: TargetPortfolio,
    ) -> list[PositionTarget]:
        held = {p.symbol: p for p in inputs.positions}
        out: list[PositionTarget] = []
        sector_value: dict[str, float] = {}

        for bucket in DEPLOYABLE:
            capital = capital_by_bucket.get(bucket, 0.0)
            if capital <= 0:
                continue
            in_bucket = [s for s in signals if _bucket_of(s.strategy) is bucket]
            if not in_bucket:
                continue

            # Inverse-volatility weights: each name contributes a similar amount
            # of risk. Equal RUPEE weights would let the most volatile name
            # dominate the portfolio's variance while looking evenly sized.
            inv_vol = np.array([1.0 / max(s.volatility, 1e-6) for s in in_bucket])
            # Edge tilts the allocation but cannot dominate it. sqrt compresses
            # the spread, so a signal with twice the score gets ~1.4x the
            # weight, not 2x — the score is an estimate, not a measurement.
            edge = np.array([math.sqrt(max(0.0, s.edge_score)) for s in in_bucket])
            raw = inv_vol * edge
            if raw.sum() <= 0:
                continue
            weights = raw / raw.sum()

            ranked = sorted(
                zip(in_bucket, weights), key=lambda t: t[1], reverse=True
            )[: limits.max_positions]

            for sig, w in ranked:
                reasons: list[str] = []
                constraints: list[str] = []
                want_value = capital * float(w)
                reasons.append(
                    f"{bucket.value}: inverse-vol weight {w:.3f} of "
                    f"Rs {capital:,.0f} sleeve capital"
                )

                # --- position cap ---
                pos_cap = inputs.total_capital * limits.max_position_pct
                if want_value > pos_cap:
                    target.binding_constraints.append(BindingConstraint(
                        "max_position_pct", sig.symbol, limits.max_position_pct,
                        _safe_div(want_value, inputs.total_capital),
                        limits.max_position_pct,
                    ))
                    constraints.append("max_position_pct")
                    want_value = pos_cap

                # --- sector cap (cumulative across everything placed so far) ---
                sec = sig.sector or "UNKNOWN"
                sec_cap = inputs.total_capital * limits.max_sector_pct
                already = sector_value.get(sec, 0.0)
                if already + want_value > sec_cap:
                    allowed = max(0.0, sec_cap - already)
                    target.binding_constraints.append(BindingConstraint(
                        "max_sector_pct", sec, limits.max_sector_pct,
                        _safe_div(already + want_value, inputs.total_capital),
                        _safe_div(already + allowed, inputs.total_capital),
                    ))
                    constraints.append("max_sector_pct")
                    want_value = allowed

                # --- liquidity ---
                liq_cap = float(sig.median_traded_value) * limits.max_liquidity_participation
                if want_value > liq_cap:
                    target.binding_constraints.append(BindingConstraint(
                        "max_liquidity_participation", sig.symbol,
                        limits.max_liquidity_participation,
                        _safe_div(want_value, max(sig.median_traded_value, 1.0)),
                        limits.max_liquidity_participation,
                    ))
                    constraints.append("max_liquidity_participation")
                    want_value = liq_cap

                # --- fractional Kelly cap, applied LAST ---
                kelly_cap = self._kelly_cap(sig, inputs.total_capital, limits)
                if kelly_cap is not None and want_value > kelly_cap:
                    target.binding_constraints.append(BindingConstraint(
                        "fractional_kelly", sig.symbol, limits.max_kelly_weight,
                        _safe_div(want_value, inputs.total_capital),
                        _safe_div(kelly_cap, inputs.total_capital),
                    ))
                    constraints.append("fractional_kelly")
                    want_value = kelly_cap

                qty = int(want_value // sig.price)
                if qty <= 0:
                    target.warnings.append(
                        f"{sig.symbol}: sized to Rs {want_value:,.0f}, below one "
                        f"share at Rs {sig.price:,.2f}. Not held."
                    )
                    continue

                final_value = qty * sig.price
                sector_value[sec] = sector_value.get(sec, 0.0) + final_value
                cur = held.get(sig.symbol)
                cur_qty = int(cur.quantity) if cur else 0

                out.append(PositionTarget(
                    symbol=sig.symbol, strategy=bucket.value, sector=sec,
                    target_weight=_safe_div(final_value, inputs.total_capital),
                    target_value=final_value, target_quantity=qty,
                    current_quantity=cur_qty, delta_quantity=qty - cur_qty,
                    price=float(sig.price), expected_vol=float(sig.volatility),
                    risk_contribution_pct=None,
                    reason="; ".join(reasons), constraints=constraints,
                ))

        # Held names with no signal are targeted to ZERO — an exit, not an
        # oversight. Leaving them out of the target entirely would silently mean
        # "hold forever", because the delta would never be computed.
        targeted = {p.symbol for p in out}
        for sym, pos in held.items():
            if sym in targeted:
                continue
            out.append(PositionTarget(
                symbol=sym, strategy=pos.strategy, sector=pos.sector,
                target_weight=0.0, target_value=0.0, target_quantity=0,
                current_quantity=int(pos.quantity),
                delta_quantity=-int(pos.quantity),
                price=float(pos.last_price),
                expected_vol=inputs.volatilities.get(sym),
                risk_contribution_pct=None,
                reason="held but no current signal supports it; target is zero",
                constraints=[],
            ))

        self._attach_risk_contributions(inputs, out)
        return out

    def _kelly_cap(
        self, sig: Any, total_capital: float, limits: RiskLimits
    ) -> Optional[float]:
        """
        Fractional-Kelly ceiling for one name. Never full Kelly.

        f* = mu / sigma^2 for a continuous approximation, multiplied by
        ``kelly_fraction`` and then hard-capped at ``max_kelly_weight``. Both
        steps are required: the fraction accounts for the edge being ESTIMATED
        rather than known, and the hard cap bounds the damage when the estimate
        is not merely noisy but wrong.
        """
        mu = float(sig.expected_return or 0.0)
        sd = float(sig.expected_return_std or 0.0)
        if mu <= 0 or sd <= 0:
            return None
        f_star = mu / (sd * sd)
        if not math.isfinite(f_star) or f_star <= 0:
            return None
        weight = min(f_star * limits.kelly_fraction, limits.max_kelly_weight)
        return total_capital * weight

    def _attach_risk_contributions(
        self, inputs: AllocationInputs, positions: list[PositionTarget]
    ) -> None:
        """
        Marginal contribution to portfolio variance, per name.

        This is the RISK view of the same book that ``target_weight`` describes
        in CAPITAL terms. Where a correlation matrix is available it is used;
        otherwise contributions assume independence, which UNDERSTATES
        concentration risk — so the assumption is recorded rather than hidden.
        """
        live = [
            p for p in positions
            if p.target_value > 0 and p.expected_vol is not None and p.expected_vol > 0
        ]
        if not live:
            return
        w = np.array([p.target_value for p in live], dtype=float)
        total = w.sum()
        if total <= 0:
            return
        w = w / total
        vol = np.array([float(p.expected_vol or 0.0) for p in live])
        corr = _correlation_for(inputs, [p.symbol for p in live])
        cov = np.outer(vol, vol) * corr
        port_var = float(w @ cov @ w)
        if port_var <= 0:
            return
        marginal = cov @ w
        contrib = w * marginal / port_var
        for p, c in zip(live, contrib):
            p.risk_contribution_pct = float(c)

    # -- 7. turnover -------------------------------------------------------- #

    def _apply_turnover_limit(
        self, inputs: AllocationInputs, limits: RiskLimits,
        positions: list[PositionTarget], target: TargetPortfolio,
    ) -> list[PositionTarget]:
        """
        Cap the TRADE, not the target.

        Applied to the delta from the existing book, which is what makes a
        monthly contribution top up toward the target instead of rebuilding it.
        When the limit binds, every delta is scaled by the same factor so the
        portfolio moves proportionally toward the target rather than completing
        a few names and ignoring the rest.
        """
        turnover_value = sum(abs(p.delta_value) for p in positions)
        cap_value = inputs.total_capital * limits.max_turnover_pct
        if turnover_value <= cap_value or turnover_value <= 0:
            return positions

        scale = cap_value / turnover_value
        target.binding_constraints.append(BindingConstraint(
            "max_turnover_pct", "portfolio", limits.max_turnover_pct,
            _safe_div(turnover_value, inputs.total_capital), limits.max_turnover_pct,
        ))
        target.reasons.append(
            f"Turnover limit bound: wanted "
            f"{_safe_div(turnover_value, inputs.total_capital):.1%}, capped at "
            f"{limits.max_turnover_pct:.1%}. Every trade is scaled by "
            f"{scale:.2f}; the portfolio moves partway toward the target and "
            f"the remainder is left for the next rebalance."
        )
        for p in positions:
            scaled = int(p.delta_quantity * scale)
            p.delta_quantity = scaled
            p.target_quantity = p.current_quantity + scaled
            p.target_value = p.target_quantity * p.price
            p.target_weight = _safe_div(p.target_value, inputs.total_capital)
            p.constraints.append("max_turnover_pct")
        return positions

    # -- 8. cash floor ------------------------------------------------------- #

    def _apply_cash_floor(
        self, inputs: AllocationInputs, limits: RiskLimits,
        positions: list[PositionTarget], target: TargetPortfolio,
    ) -> list[PositionTarget]:
        """
        Enforce the cash minimum LAST, so nothing downstream can spend it.

        Applied by scaling every BUY down. Sells are untouched: reducing a sell
        to raise cash is self-defeating.
        """
        deployed = sum(p.target_value for p in positions)
        max_deployed = inputs.total_capital * (1.0 - limits.min_cash_pct)
        if deployed <= max_deployed or deployed <= 0:
            return positions

        scale = max_deployed / deployed
        target.binding_constraints.append(BindingConstraint(
            "min_cash_pct", "portfolio", limits.min_cash_pct,
            1.0 - _safe_div(deployed, inputs.total_capital), limits.min_cash_pct,
        ))
        target.reasons.append(
            f"Cash floor bound: the target deployed "
            f"{_safe_div(deployed, inputs.total_capital):.1%}, leaving less than "
            f"the {limits.min_cash_pct:.1%} minimum. Positions scaled by "
            f"{scale:.2f}."
        )
        for p in positions:
            if p.target_quantity <= 0:
                continue
            p.target_quantity = int(p.target_quantity * scale)
            p.target_value = p.target_quantity * p.price
            p.target_weight = _safe_div(p.target_value, inputs.total_capital)
            p.delta_quantity = p.target_quantity - p.current_quantity
            p.constraints.append("min_cash_pct")
        return positions

    # -- reporting ----------------------------------------------------------- #

    def _expected_vol(
        self, inputs: AllocationInputs, positions: Sequence[PositionTarget]
    ) -> Optional[float]:
        live = [
            p for p in positions
            if p.target_value > 0 and p.expected_vol is not None and p.expected_vol > 0
        ]
        if not live:
            return 0.0
        w = np.array([p.target_value for p in live], dtype=float)
        w = w / max(inputs.total_capital, 1e-9)
        vol = np.array([float(p.expected_vol or 0.0) for p in live])
        corr = _correlation_for(inputs, [p.symbol for p in live])
        var = float(w @ (np.outer(vol, vol) * corr) @ w)
        return float(np.sqrt(max(var, 0.0)))

    def _strategy_targets(
        self, inputs: AllocationInputs, positions: Sequence[PositionTarget],
        capital_by_bucket: Mapping[StrategyBucket, float],
        risk_by_bucket: Mapping[StrategyBucket, float],
        cash: float,
    ) -> list[StrategyTarget]:
        out: list[StrategyTarget] = []
        for b in DEPLOYABLE:
            in_b = [p for p in positions if p.strategy == b.value and p.target_value > 0]
            deployed = sum(p.target_value for p in in_b)
            risk_used = sum(
                p.risk_contribution_pct or 0.0 for p in in_b
            )
            health = str(inputs.strategy_health.get(b.value, "HEALTHY")).upper()
            budget = risk_by_bucket.get(b, 0.0)
            if not in_b and deployed <= 0:
                reason = (
                    f"No capital: {'strategy is ' + health if health != 'HEALTHY' else 'no usable signal'}."
                )
            else:
                reason = (
                    f"{len(in_b)} position(s); capital "
                    f"{_safe_div(deployed, inputs.total_capital):.1%} vs risk "
                    f"budget {budget:.1%}, risk used {risk_used:.1%}. Capital and "
                    f"risk shares differ because they measure different things."
                )
            out.append(StrategyTarget(
                bucket=b, capital_amount=deployed,
                capital_pct=_safe_div(deployed, inputs.total_capital),
                risk_budget_pct=budget, risk_used_pct=risk_used,
                n_positions=len(in_b), health=health, reason=reason,
            ))
        out.append(StrategyTarget(
            bucket=StrategyBucket.CASH, capital_amount=cash,
            capital_pct=_safe_div(cash, inputs.total_capital),
            risk_budget_pct=0.0, risk_used_pct=0.0, n_positions=0,
            health="N/A",
            reason="Residual after every constraint. Cash is a position, not a leftover.",
        ))
        return out


# --------------------------------------------------------------------------- #
#  helpers                                                                      #
# --------------------------------------------------------------------------- #

def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _bucket_of(strategy: str) -> StrategyBucket:
    s = (strategy or "").lower()
    if "intraday" in s:
        return StrategyBucket.INTRADAY
    if "swing" in s:
        return StrategyBucket.SWING
    if "long" in s:
        return StrategyBucket.LONG_TERM
    return StrategyBucket.SWING


def _correlation_for(inputs: AllocationInputs, symbols: Sequence[str]) -> np.ndarray:
    """
    Correlation sub-matrix for ``symbols``, or the identity.

    The identity means "assume independence", which UNDERSTATES the risk of a
    correlated book. It is used only when no matrix was supplied, and callers
    that care should supply one.
    """
    n = len(symbols)
    if (
        inputs.correlation_matrix is None
        or not inputs.correlation_symbols
        or n == 0
    ):
        return np.eye(n)
    index = {s: i for i, s in enumerate(inputs.correlation_symbols)}
    full = np.asarray(inputs.correlation_matrix, dtype=float)
    if any(s not in index for s in symbols):
        return np.eye(n)
    idx = [index[s] for s in symbols]
    sub = full[np.ix_(idx, idx)]
    if sub.shape != (n, n) or not np.all(np.isfinite(sub)):
        return np.eye(n)
    # Symmetrise and put a unit diagonal; a supplied matrix is not trusted to
    # be well-formed just because it was supplied.
    sub = (sub + sub.T) / 2.0
    np.fill_diagonal(sub, 1.0)
    return np.clip(sub, -1.0, 1.0)


def allocate_and_snapshot(
    inputs: AllocationInputs, limits: Optional[RiskLimits] = None
) -> tuple[TargetPortfolio, AllocationSnapshot]:
    """Allocate and capture a reproducible record in one call."""
    engine = PortfolioAllocationEngine(limits)
    target = engine.allocate(inputs)
    return target, AllocationSnapshot.build(inputs, target)
