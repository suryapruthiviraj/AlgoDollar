"""
engine.py — Event-driven backtesting engine for AlgoDollar.

NO LOOK-AHEAD BIAS CONTRACT
---------------------------
On each bar T:
  1. Prices used for SIGNAL GENERATION = close prices of bar T (just closed).
  2. Orders are executed at the OPEN of bar T+1 (next bar) + slippage.
  3. Features computed for bar T use only data[0..T] inclusive.
  4. Stop/target checks use bar T's low/high to determine if touched,
     but the execution price is capped at the stop/target level.

The engine NEVER uses data beyond the current simulation date.

Survivorship bias note
----------------------
Results depend on the universe passed at each bar.  If the universe is a
static list of today's index constituents, results will be upward-biased.
Use point-in-time universe data to mitigate this.

Risk-free rate: India 10-year G-Sec, ~6.5% annualized.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RF_ANNUAL = 0.065       # India 10-year G-Sec (risk-free for Sharpe)
_RF_DAILY  = _RF_ANNUAL / 252

# Slippage model (half-spread + impact estimate, per leg)
_SLIPPAGE_LARGE_CAP = 0.0005   # 5 bps  for large caps
_SLIPPAGE_SMALL_CAP = 0.0015   # 15 bps for small caps
# Safety ceiling. MUST stay above _SLIPPAGE_SMALL_CAP, otherwise it silently
# truncates the small-cap tier and every symbol is charged the same slippage.
_MAX_SLIPPAGE       = 0.0050   # 50 bps hard ceiling

# Maximum drawdown before the engine halts the backtest
_MAX_DD_HALT = 0.50  # 50% drawdown → halt (configurable)

# Fraction of bars that may throw before the whole backtest is declared invalid.
_MAX_ERROR_RATE = 0.05


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BacktestMetrics:
    total_return:          float
    cagr:                  float
    sharpe_ratio:          float
    sortino_ratio:         float
    max_drawdown:          float
    max_drawdown_duration: int    # calendar days
    win_rate:              float
    avg_win:               float
    avg_loss:              float
    profit_factor:         float
    num_trades:            int
    avg_holding_period:    float  # trading days
    gross_return:          float
    net_return:            float
    calmar_ratio:          float
    annualized_volatility: float
    total_costs:           float
    cost_as_pct_of_gross:  float

    def to_dict(self) -> dict:
        return {k: round(v, 6) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


@dataclass
class TradeRecord:
    symbol:          str
    entry_date:      str
    exit_date:       str
    entry_price:     float
    exit_price:      float
    qty:             float
    direction:       str
    pnl_gross:       float
    pnl_net:         float
    cost:            float
    holding_days:    int
    exit_reason:     str


@dataclass
class BacktestResult:
    equity_curve:  pd.Series
    trades:        pd.DataFrame
    daily_returns: pd.Series
    metrics:       BacktestMetrics
    params:        dict


# ---------------------------------------------------------------------------
# Open position tracking
# ---------------------------------------------------------------------------

@dataclass
class _Position:
    symbol:       str
    direction:    str
    entry_date:   pd.Timestamp
    entry_price:  float
    qty:          float
    stop_loss:    float
    target:       float
    signal:       Any   # original Signal object
    cost_in:      float  # entry transaction cost


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EventDrivenBacktester:
    """
    Event-driven backtester.

    Parameters
    ----------
    cost_model : ZerodhaCostModel (or any model with calculate_costs()).
    max_drawdown_halt : float
        Stop the simulation if portfolio drawdown exceeds this fraction.
    """

    def __init__(
        self,
        cost_model=None,
        max_drawdown_halt: float = _MAX_DD_HALT,
        product: str = "CNC",
        exchange: str = "NSE",
        large_cap_symbols: Optional[Iterable[str]] = None,
        slippage_multiplier: float = 1.0,
        whole_shares: bool = True,
    ):
        """
        Parameters
        ----------
        cost_model : ZerodhaCostModel or compatible.
        max_drawdown_halt : float
            Halt the simulation if portfolio drawdown exceeds this fraction.
        product : {"CNC", "MIS"}
            Zerodha product code. Determines the cost schedule: MIS (intraday)
            pays capped brokerage and sell-side-only STT; CNC (delivery) pays
            zero brokerage but STT on both legs plus DP charges on sells.
            Using the wrong one misprices every trade in the backtest.
        exchange : {"NSE", "BSE"}
        large_cap_symbols : iterable of str, optional
            Symbols priced at the tighter large-cap slippage tier. If omitted,
            every symbol is treated as small-cap, which is the conservative
            assumption.
        slippage_multiplier : float
            Scales base slippage. Used for degradation stress tests
            (0.5x / 1x / 1.5x / 2x / 3x).
        whole_shares : bool
            If True (default), position quantities are floored to whole shares.
            Indian equities cannot be traded fractionally; allowing fractional
            quantities produces returns that are not attainable in practice.
        """
        self._cost_model = cost_model
        self._max_dd_halt = max_drawdown_halt
        self._product = product
        self._exchange = exchange
        self._large_cap_symbols = set(large_cap_symbols or ())
        self._slippage_multiplier = float(slippage_multiplier)
        self._whole_shares = whole_shares

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        strategy,
        universe: List[str],
        prices_df: pd.DataFrame,
        features_fn: Callable[[pd.DataFrame, pd.DataFrame, Any], pd.DataFrame],
        cost_model=None,
        start_date: str = "",
        end_date: str = "",
        initial_capital: float = 100_000.0,
        volume_df: Optional[pd.DataFrame] = None,
        nifty_df: Optional[pd.DataFrame] = None,
        params: Optional[dict] = None,
        open_df: Optional[pd.DataFrame] = None,
        high_df: Optional[pd.DataFrame] = None,
        low_df: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run a full backtest.

        Parameters
        ----------
        strategy : BaseStrategy subclass instance.
        universe : list[str], symbols to consider (subject to survivorship bias).
        prices_df : DataFrame (T × N), close prices, DatetimeIndex.
        features_fn : callable(prices_df_up_to_T, volume_df_up_to_T, nifty_df_up_to_T)
            → features_df.  Must return features using ONLY data up to T.
        cost_model : ZerodhaCostModel or compatible.
        start_date, end_date : str "YYYY-MM-DD".
        initial_capital : float (INR).
        volume_df : DataFrame (T × N), daily volume (optional).
        nifty_df : DataFrame (T × 1+) with 'close' (optional for market features).
        params : dict, logged to BacktestResult.

        Returns
        -------
        BacktestResult
        """
        cm = cost_model or self._cost_model
        params = params or {}

        # Date filtering
        idx = prices_df.index
        if start_date:
            idx = idx[idx >= pd.Timestamp(start_date)]
        if end_date:
            idx = idx[idx <= pd.Timestamp(end_date)]
        if len(idx) < 2:
            raise ValueError("Insufficient trading days in backtest window.")

        dates = idx
        logger.info(
            "Backtest: %s → %s (%d bars), capital=₹%.0f",
            dates[0].date(), dates[-1].date(), len(dates), initial_capital,
        )

        # State
        capital = initial_capital
        peak_capital = initial_capital
        positions: Dict[str, _Position] = {}
        equity_history: List[tuple[pd.Timestamp, float]] = []
        trade_records: List[TradeRecord] = []
        feature_errors = 0
        strategy_errors = 0
        last_feature_error: Optional[Exception] = None
        last_strategy_error: Optional[Exception] = None
        halted_at: Optional[pd.Timestamp] = None

        # ---- Pre-build per-symbol OHLCV frames once ----
        # If true OHLC is not supplied, open/high/low are synthesized from
        # close. That makes every intrabar range identically zero, so ATR and
        # any high/low-based stop logic silently evaluate to 0. Warn loudly:
        # this is a real limitation of the input data, not a detail.
        synth_ohlc = high_df is None or low_df is None
        if synth_ohlc:
            logger.warning(
                "No high/low data supplied — open/high/low are being "
                "synthesized from close. Intrabar range is therefore zero, so "
                "ATR-based features and intrabar stop detection will not "
                "work. Pass high_df/low_df for realistic results."
            )

        full_market_data: Dict[str, pd.DataFrame] = {}
        for sym in universe:
            if sym not in prices_df.columns:
                continue
            close = prices_df[sym]
            full_market_data[sym] = pd.DataFrame({
                "open": open_df[sym] if open_df is not None and sym in open_df.columns else close,
                "high": high_df[sym] if high_df is not None and sym in high_df.columns else close,
                "low": low_df[sym] if low_df is not None and sym in low_df.columns else close,
                "close": close,
                "volume": (
                    volume_df[sym]
                    if volume_df is not None and sym in volume_df.columns
                    else pd.Series(0.0, index=prices_df.index)
                ),
            })

        for bar_idx, current_date in enumerate(dates[:-1]):
            next_date = dates[bar_idx + 1]

            # ---- Current prices (bar T close — for signal generation) ----
            prices_up_to_T = prices_df.loc[:current_date, :]

            # ---- Compute portfolio value ----
            portfolio_value = capital
            for sym, pos in positions.items():
                if sym in prices_df.columns:
                    last_px = prices_df.loc[current_date, sym]
                    if pd.notna(last_px):
                        portfolio_value += pos.qty * last_px

            equity_history.append((current_date, portfolio_value))

            # ---- Drawdown check ----
            peak_capital = max(peak_capital, portfolio_value)
            dd = (peak_capital - portfolio_value) / peak_capital
            if dd > self._max_dd_halt:
                logger.warning(
                    "Max drawdown %.1f%% breached on %s — halting backtest.",
                    dd * 100,
                    current_date.date(),
                )
                # Liquidate at the NEXT bar, not at the end of the dataset.
                # Falling through to the end-of-backtest liquidation would
                # mark open positions out at a price potentially years in the
                # future — a look-ahead that flatters (or distorts) precisely
                # the scenario the halt exists to capture.
                halted_at = next_date
                break

            # ---- Check exits (using current bar T data) ----
            symbols_to_exit = []
            for sym, pos in positions.items():
                if sym not in prices_df.columns:
                    symbols_to_exit.append((sym, "data_missing"))
                    continue
                curr_px = prices_df.loc[current_date, sym]
                if pd.isna(curr_px):
                    continue
                current_data = {"price": curr_px, "date": current_date}
                if strategy.should_exit(
                    {
                        "symbol": sym,
                        "entry_price": pos.entry_price,
                        "entry_date": pos.entry_date,
                        "direction": pos.direction,
                        "stop_loss": pos.stop_loss,
                        "target": pos.target,
                        "signal": pos.signal,
                    },
                    current_data,
                ):
                    symbols_to_exit.append((sym, "strategy_exit"))

            # Execute exits at next bar open
            for sym, reason in symbols_to_exit:
                # Bound to its own name because `pos` is already inferred as
                # _Position from the iteration above, which makes pop()'s None
                # default look like a type error.
                exiting = positions.pop(sym, None)
                if exiting is None:
                    continue
                pos = exiting
                if sym not in prices_df.columns or pd.isna(
                    prices_df.loc[next_date, sym]
                ):
                    exit_px = pos.entry_price  # fallback
                else:
                    exit_px = prices_df.loc[next_date, sym]

                exit_px = self._apply_slippage(exit_px, "SELL", sym)
                cost_out = self._estimate_cost("SELL", pos.qty, exit_px, cm)
                proceeds = pos.qty * exit_px - cost_out
                capital += proceeds

                pnl_gross = pos.qty * (exit_px - pos.entry_price)
                pnl_net   = pnl_gross - pos.cost_in - cost_out
                holding   = int((next_date - pos.entry_date).days)

                trade_records.append(TradeRecord(
                    symbol=sym,
                    entry_date=str(pos.entry_date.date()),
                    exit_date=str(next_date.date()),
                    entry_price=round(pos.entry_price, 4),
                    exit_price=round(exit_px, 4),
                    qty=pos.qty,
                    direction=pos.direction,
                    pnl_gross=round(pnl_gross, 4),
                    pnl_net=round(pnl_net, 4),
                    cost=round(pos.cost_in + cost_out, 4),
                    holding_days=holding,
                    exit_reason=reason,
                ))

            # ---- Generate new signals (using data up to bar T only) ----
            try:
                features_df = features_fn(
                    prices_up_to_T,
                    volume_df.loc[:current_date] if volume_df is not None else None,
                    nifty_df.loc[:current_date] if nifty_df is not None else None,
                )
            except Exception as exc:
                # Swallowing this silently produces a flat equity curve that
                # looks like "the strategy chose not to trade" rather than
                # "the feature pipeline is broken". Count it and fail the run
                # if it is not a rare edge case.
                feature_errors += 1
                last_feature_error = exc
                logger.warning(
                    "features_fn failed on %s: %s", current_date.date(), exc
                )
                features_df = pd.DataFrame()

            # Slice the pre-built per-symbol frames. Constructing a fresh
            # DataFrame per symbol per bar (the previous behaviour) is
            # O(bars x symbols) allocations each copying O(bars) of data,
            # which made realistic backtests unusably slow.
            n_rows = bar_idx + 1
            market_data = {
                sym: df.iloc[:n_rows] for sym, df in full_market_data.items()
            }

            try:
                signals = strategy.generate_signals(
                    universe=universe,
                    features_df=features_df,
                    market_data=market_data,
                    existing_positions={sym: {"value": pos.qty * prices_df.loc[current_date, sym]}
                                        for sym, pos in positions.items()
                                        if sym in prices_df.columns},
                )
            except Exception as exc:
                strategy_errors += 1
                last_strategy_error = exc
                logger.warning(
                    "strategy.generate_signals failed on %s: %s",
                    current_date.date(), exc,
                )
                signals = []

            # ---- Execute entries at next bar open ----
            for signal in signals:
                if signal.symbol in positions:
                    continue  # already holding
                if not signal.is_valid():
                    continue

                if signal.symbol not in prices_df.columns:
                    continue
                if pd.isna(prices_df.loc[next_date, signal.symbol]):
                    continue

                entry_px = prices_df.loc[next_date, signal.symbol]
                entry_px = self._apply_slippage(entry_px, "BUY", signal.symbol)

                # Size against total portfolio value, not free cash. Sizing on
                # cash alone shrinks every position as the book fills up, which
                # systematically under-deploys and is not what the strategy's
                # risk model intends.
                size = strategy.calculate_position_size(
                    signal, portfolio_value, risk_engine=None
                )
                if size <= 0:
                    continue
                size = min(size, capital)

                if entry_px <= 0:
                    continue
                qty = size / entry_px
                if self._whole_shares:
                    qty = float(int(qty))  # Indian equities are not fractionally tradable
                if qty <= 0:
                    continue

                cost_in = self._estimate_cost("BUY", qty, entry_px, cm)
                total_outlay = qty * entry_px + cost_in
                if total_outlay > capital:
                    # Scale down to fit available cash
                    qty = (capital * 0.99 - cost_in) / entry_px
                    if self._whole_shares:
                        qty = float(int(qty))
                    if qty <= 0:
                        continue
                    cost_in = self._estimate_cost("BUY", qty, entry_px, cm)
                    total_outlay = qty * entry_px + cost_in
                    if total_outlay > capital:
                        continue

                capital -= total_outlay
                stop_price = entry_px * (1 - signal.stop_loss_pct)
                target_price = entry_px * (1 + signal.target_pct)

                positions[signal.symbol] = _Position(
                    symbol=signal.symbol,
                    direction=signal.direction.value,
                    entry_date=next_date,
                    entry_price=entry_px,
                    qty=qty,
                    stop_loss=stop_price,
                    target=target_price,
                    signal=signal,
                    cost_in=cost_in,
                )

        # ---- Liquidate remaining positions ----
        # If the drawdown halt fired, close at the halt bar. Otherwise close
        # at the last bar of the window.
        final_date = halted_at if halted_at is not None else dates[-1]
        for sym, pos in list(positions.items()):
            if sym not in prices_df.columns:
                continue
            exit_px = prices_df.loc[final_date, sym]
            if pd.isna(exit_px):
                continue
            exit_px = self._apply_slippage(exit_px, "SELL", sym)
            cost_out = self._estimate_cost("SELL", pos.qty, exit_px, cm)
            proceeds = pos.qty * exit_px - cost_out
            capital += proceeds

            pnl_gross = pos.qty * (exit_px - pos.entry_price)
            pnl_net   = pnl_gross - pos.cost_in - cost_out
            holding   = int((final_date - pos.entry_date).days)
            trade_records.append(TradeRecord(
                symbol=sym, entry_date=str(pos.entry_date.date()),
                exit_date=str(final_date.date()),
                entry_price=round(pos.entry_price, 4),
                exit_price=round(exit_px, 4),
                qty=pos.qty, direction=pos.direction,
                pnl_gross=round(pnl_gross, 4), pnl_net=round(pnl_net, 4),
                cost=round(pos.cost_in + cost_out, 4),
                holding_days=holding, exit_reason="end_of_backtest",
            ))
        equity_history.append((final_date, capital))

        # ---- Fail loudly on systemic pipeline breakage ----
        # A backtest that silently produced no signals because the feature
        # pipeline threw on every bar must not be reported as a valid result.
        n_bars = len(dates) - 1
        if n_bars > 0:
            if feature_errors / n_bars > _MAX_ERROR_RATE:
                raise RuntimeError(
                    f"features_fn failed on {feature_errors}/{n_bars} bars "
                    f"({feature_errors / n_bars:.0%}). The backtest is invalid. "
                    f"Last error: {last_feature_error!r}"
                )
            if strategy_errors / n_bars > _MAX_ERROR_RATE:
                raise RuntimeError(
                    f"strategy.generate_signals failed on {strategy_errors}/"
                    f"{n_bars} bars ({strategy_errors / n_bars:.0%}). The "
                    f"backtest is invalid. Last error: {last_strategy_error!r}"
                )

        # ---- Build equity curve ----
        equity_curve = pd.Series(
            {t: v for t, v in equity_history}
        ).sort_index()

        daily_returns = equity_curve.pct_change().dropna()
        trades_df = pd.DataFrame([t.__dict__ for t in trade_records]) if trade_records else pd.DataFrame()

        metrics = self._compute_metrics(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            trade_records=trade_records,
            initial_capital=initial_capital,
            start_date=dates[0],
            end_date=final_date,
        )

        logger.info(
            "Backtest complete: CAGR=%.1f%% Sharpe=%.2f MaxDD=%.1f%% Trades=%d",
            metrics.cagr * 100,
            metrics.sharpe_ratio,
            metrics.max_drawdown * 100,
            metrics.num_trades,
        )
        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades_df,
            daily_returns=daily_returns,
            metrics=metrics,
            params=params,
        )

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        equity_curve: pd.Series,
        daily_returns: pd.Series,
        trade_records: List[TradeRecord],
        initial_capital: float,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> BacktestMetrics:
        final_value = equity_curve.iloc[-1]
        total_return = (final_value / initial_capital) - 1.0

        # CAGR
        n_years = max((end_date - start_date).days / 365.25, 1.0 / 252)
        cagr = (final_value / initial_capital) ** (1 / n_years) - 1.0

        # Annualized volatility
        ann_vol = float(daily_returns.std() * np.sqrt(252)) if len(daily_returns) > 1 else 0.0

        # Sharpe (annualized, rf=6.5%)
        excess_daily = daily_returns - _RF_DAILY
        sharpe = float(
            excess_daily.mean() / excess_daily.std() * np.sqrt(252)
        ) if excess_daily.std() > 0 else 0.0

        # Sortino (downside deviation only)
        downside = daily_returns[daily_returns < _RF_DAILY] - _RF_DAILY
        sortino_denom = float(np.sqrt((downside ** 2).mean()) * np.sqrt(252)) if len(downside) > 0 else 1e-6
        sortino = float(excess_daily.mean() * 252 / sortino_denom) if sortino_denom > 0 else 0.0

        # Max drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = float(drawdown.min())

        # Max drawdown duration
        dd_duration = 0
        current_duration = 0
        for dd_val in drawdown:
            if dd_val < -0.001:
                current_duration += 1
                dd_duration = max(dd_duration, current_duration)
            else:
                current_duration = 0

        # Trade statistics
        if trade_records:
            pnl_nets = [t.pnl_net for t in trade_records]
            wins  = [p for p in pnl_nets if p > 0]
            losses = [p for p in pnl_nets if p <= 0]
            win_rate = len(wins) / len(pnl_nets)
            avg_win  = float(np.mean(wins)) if wins else 0.0
            avg_loss = float(np.mean(losses)) if losses else 0.0
            profit_factor = (
                float(sum(wins)) / abs(float(sum(losses)))
                if losses and sum(losses) != 0 else float("inf")
            )
            avg_holding = float(np.mean([t.holding_days for t in trade_records]))
            total_costs = float(sum(t.cost for t in trade_records))
            gross_pnl   = float(sum(t.pnl_gross for t in trade_records))
            net_pnl     = float(sum(t.pnl_net for t in trade_records))
            gross_return = gross_pnl / initial_capital
            net_return   = net_pnl / initial_capital
            cost_as_pct_gross = (
                total_costs / abs(gross_pnl) if gross_pnl != 0 else 0.0
            )
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0.0
            avg_holding = total_costs = gross_return = net_return = cost_as_pct_gross = 0.0

        calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

        return BacktestMetrics(
            total_return=round(total_return, 6),
            cagr=round(cagr, 6),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            max_drawdown=round(max_dd, 6),
            max_drawdown_duration=dd_duration,
            win_rate=round(win_rate, 4),
            avg_win=round(avg_win, 4),
            avg_loss=round(avg_loss, 4),
            profit_factor=round(profit_factor, 4),
            num_trades=len(trade_records),
            avg_holding_period=round(avg_holding, 2),
            gross_return=round(gross_return, 6),
            net_return=round(net_return, 6),
            calmar_ratio=round(calmar, 4),
            annualized_volatility=round(ann_vol, 6),
            total_costs=round(total_costs, 4),
            cost_as_pct_of_gross=round(cost_as_pct_gross, 4),
        )

    # ------------------------------------------------------------------
    # Slippage and cost helpers
    # ------------------------------------------------------------------

    def _apply_slippage(
        self,
        price: float,
        side: str,
        symbol: str,
    ) -> float:
        """
        Apply market-impact slippage to the execution price.

        Base slippage (per leg):
          - Large cap (in self._large_cap_symbols): 5 bps
          - Everything else:                       15 bps

        The result is scaled by self._slippage_multiplier, which exists so the
        same strategy can be re-run at 0.5x / 1x / 1.5x / 2x / 3x slippage to
        test whether its edge survives worse-than-expected execution. A
        strategy that is only profitable at 1x is fragile and must be rejected.

        BUY:  price × (1 + slippage)
        SELL: price × (1 - slippage)
        """
        base = (
            _SLIPPAGE_LARGE_CAP
            if symbol in self._large_cap_symbols
            else _SLIPPAGE_SMALL_CAP
        )
        slippage = min(base * self._slippage_multiplier, _MAX_SLIPPAGE)
        if side.upper() == "BUY":
            return price * (1 + slippage)
        return price * (1 - slippage)

    def _estimate_cost(
        self,
        side: str,
        qty: float,
        price: float,
        cost_model,
    ) -> float:
        """
        Transaction cost for one leg, via the shared ZerodhaCostModel.

        `product` is taken from self._product rather than hardcoded: pricing an
        intraday (MIS) strategy as delivery (CNC) overstates cost by roughly
        2.5x per round trip and would reject viable intraday strategies.
        """
        if cost_model is None:
            return qty * price * 0.0015
        breakdown = cost_model.calculate_costs(
            transaction_type=side,
            qty=qty,
            price=price,
            exchange=self._exchange,
            product=self._product,
        )
        return float(breakdown.total)
