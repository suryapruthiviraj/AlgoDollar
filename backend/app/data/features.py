"""
features.py — Timestamp-safe feature engineering for AlgoDollar.

NO LOOK-AHEAD BIAS CONTRACT
---------------------------
Every feature computed for bar T uses ONLY data available at or before the
CLOSE of bar T.  Specifically:
  - log_return_1d at T  = log(close[T] / close[T-1])          ✓
  - momentum_12_1 at T  = cumulative return from T-252 to T-21 ✓
  - RSI at T            = uses close values up to and including T ✓
  - cross-sectional ranks at T are computed from the T-bar close cross-section ✓

Intraday features (vwap_distance, vwap_slope) require intraday OHLCV bars.
They are computed on the aggregated intraday data available by the time of
signal generation (i.e., bars up to the last completed bar before signal time).

All methods accept a DataFrame indexed by date (DatetimeIndex) and return
aligned Series or DataFrames.  The caller must ensure that the input data is
itself free of future information.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ANNUALIZATION = np.sqrt(252)
_EWMA_LAMBDA = 0.94


class FeatureEngine:
    """
    Compute all alpha features for a universe of stocks.

    Usage
    -----
    engine = FeatureEngine()
    features_df = engine.compute_all_features(prices_df, volume_df, nifty_df)

    Parameters in compute_all_features
    ------------------------------------
    prices_df : DataFrame, shape (T, N), close prices, DatetimeIndex.
    volume_df : DataFrame, shape (T, N), daily volume, DatetimeIndex.
    nifty_df  : DataFrame, shape (T, 1+), columns include 'close', DatetimeIndex.
    intraday_df : optional dict {symbol: intraday_ohlcv_df} for VWAP features.
    """

    # ------------------------------------------------------------------
    # Price / Return features
    # ------------------------------------------------------------------

    @staticmethod
    def log_return(prices: pd.Series, periods: int) -> pd.Series:
        """
        Log return over `periods` days.  Strictly non-look-ahead: return at T
        uses close[T] and close[T-periods], both observable at T.

        log_return_Nd[T] = log(close[T] / close[T-N])
        """
        return np.log(prices / prices.shift(periods))

    @staticmethod
    def excess_return_vs_nifty(
        prices: pd.Series, nifty_close: pd.Series, periods: int
    ) -> pd.Series:
        """
        Excess return of stock vs NIFTY over `periods` days.
        """
        stock_ret = np.log(prices / prices.shift(periods))
        nifty_ret = np.log(nifty_close / nifty_close.shift(periods))
        return stock_ret - nifty_ret.reindex(stock_ret.index)

    @staticmethod
    def momentum_12_1(prices: pd.Series) -> pd.Series:
        """
        Fama-French 12-1 momentum: 12-month cumulative return excluding the
        most recent month.

        momentum_12_1[T] = log(close[T-21] / close[T-252])

        This skips the most recent month (short-term reversal) and uses only
        data available at T.
        """
        return np.log(prices.shift(21) / prices.shift(252))

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (Wilder's smoothing).

        RSI at T uses close[T-period..T].  No look-ahead.
        """
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder's smoothing (EMA with alpha = 1/period)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def distance_from_52w_high(prices: pd.Series) -> pd.Series:
        """
        (price[T] / rolling_max_252[T]) - 1.  ≤ 0.  Uses close[T-251..T].
        """
        rolling_max = prices.rolling(252, min_periods=126).max()
        return (prices / rolling_max) - 1

    @staticmethod
    def distance_from_52w_low(prices: pd.Series) -> pd.Series:
        """
        (price[T] / rolling_min_252[T]) - 1.  ≥ 0.  Uses close[T-251..T].
        """
        rolling_min = prices.rolling(252, min_periods=126).min()
        return (prices / rolling_min) - 1

    @staticmethod
    def price_to_sma(prices: pd.Series, window: int) -> pd.Series:
        """
        price[T] / SMA(window)[T] - 1.  SMA uses close[T-window+1..T].
        """
        sma = prices.rolling(window, min_periods=window // 2).mean()
        return (prices / sma) - 1

    # ------------------------------------------------------------------
    # Volatility features
    # ------------------------------------------------------------------

    @staticmethod
    def realized_vol(prices: pd.Series, window: int) -> pd.Series:
        """
        Annualized realized volatility over `window` days.

        Uses log returns computed on close[T-window..T].  The std of log
        returns is multiplied by sqrt(252).
        """
        log_rets = np.log(prices / prices.shift(1))
        return log_rets.rolling(window, min_periods=window // 2).std() * _ANNUALIZATION

    @staticmethod
    def ewma_vol(prices: pd.Series, lam: float = _EWMA_LAMBDA) -> pd.Series:
        """
        EWMA volatility (RiskMetrics 1994, lambda=0.94).

        sigma^2[T] = lambda * sigma^2[T-1] + (1-lambda) * r^2[T]

        Annualized: sigma[T] * sqrt(252).
        No look-ahead: all inputs are at or before T.
        """
        log_rets = np.log(prices / prices.shift(1))
        sq_rets = log_rets ** 2
        ewma_var = sq_rets.ewm(alpha=1 - lam, adjust=False).mean()
        return np.sqrt(ewma_var) * _ANNUALIZATION

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range normalized by close price.

        ATR[T] / close[T].  Uses OHLC up to and including T.
        """
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_val = tr.ewm(span=period, adjust=False).mean()
        return atr_val / close.replace(0, np.nan)

    @staticmethod
    def vol_ratio(prices: pd.Series, short_window: int = 10, long_window: int = 63) -> pd.Series:
        """
        Volatility regime ratio: short_vol / long_vol.

        Values > 1 indicate elevated recent volatility.
        """
        short_vol = FeatureEngine.realized_vol(prices, short_window)
        long_vol = FeatureEngine.realized_vol(prices, long_window)
        return short_vol / long_vol.replace(0, np.nan)

    # ------------------------------------------------------------------
    # Volume features
    # ------------------------------------------------------------------

    @staticmethod
    def volume_ratio(volume: pd.Series, window: int = 10) -> pd.Series:
        """
        volume[T] / rolling_mean_volume(window)[T-1].

        We use the T-1 rolling mean to avoid including today's volume in its
        own denominator (mild form of look-ahead, averted here).
        """
        rolling_avg = volume.shift(1).rolling(window, min_periods=window // 2).mean()
        return volume / rolling_avg.replace(0, np.nan)

    @staticmethod
    def relative_volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
        """
        Z-score of today's volume vs its rolling window.

        Uses shift(1) on rolling mean/std to exclude today from the baseline.
        """
        rolled_mean = volume.shift(1).rolling(window, min_periods=window // 2).mean()
        rolled_std = volume.shift(1).rolling(window, min_periods=window // 2).std()
        return (volume - rolled_mean) / rolled_std.replace(0, np.nan)

    @staticmethod
    def pvt(prices: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Price Volume Trend (PVT).

        PVT[T] = PVT[T-1] + volume[T] * (close[T] - close[T-1]) / close[T-1]
        """
        daily_ret = prices.pct_change()
        pvt = (volume * daily_ret).cumsum()
        return pvt

    @staticmethod
    def obv_slope(prices: pd.Series, volume: pd.Series, window: int = 10) -> pd.Series:
        """
        Slope of On-Balance Volume (OBV) over `window` days.

        OBV[T] += volume[T] if close[T] > close[T-1] else -volume[T].
        Slope is a linear regression coefficient over the last `window` bars.
        No look-ahead: computed from data available at T.
        """
        direction = np.sign(prices.diff())
        obv = (direction * volume).cumsum()
        # Rolling linear regression slope (normalized by mean OBV for scale)
        x = np.arange(window, dtype=float)
        x -= x.mean()

        def slope(y: np.ndarray) -> float:
            if np.isnan(y).any():
                return np.nan
            return float(np.polyfit(x, y, 1)[0])

        obv_slope_series = obv.rolling(window, min_periods=window).apply(slope, raw=True)
        return obv_slope_series

    # ------------------------------------------------------------------
    # VWAP features (requires intraday data)
    # ------------------------------------------------------------------

    @staticmethod
    def vwap_distance(intraday_df: pd.DataFrame) -> float:
        """
        Compute VWAP and return (last_price / VWAP) - 1 for one day.

        Parameters
        ----------
        intraday_df : DataFrame with columns open, high, low, close, volume,
            indexed by intraday timestamps (all within a single trading day).

        Returns
        -------
        float : (current_price / VWAP) - 1
        """
        if intraday_df.empty:
            return np.nan
        typical_price = (intraday_df["high"] + intraday_df["low"] + intraday_df["close"]) / 3.0
        cum_tp_vol = (typical_price * intraday_df["volume"]).cumsum()
        cum_vol = intraday_df["volume"].cumsum()
        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
        last_price = intraday_df["close"].iloc[-1]
        last_vwap = vwap.iloc[-1]
        if last_vwap == 0 or np.isnan(last_vwap):
            return np.nan
        return float(last_price / last_vwap) - 1.0

    @staticmethod
    def vwap_slope(intraday_df: pd.DataFrame, window_bars: int = 10) -> float:
        """
        Linear regression slope of VWAP over the last `window_bars` intraday bars.

        Returns slope per bar (un-normalized).
        """
        if intraday_df.empty or len(intraday_df) < window_bars:
            return np.nan
        typical_price = (intraday_df["high"] + intraday_df["low"] + intraday_df["close"]) / 3.0
        cum_tp_vol = (typical_price * intraday_df["volume"]).cumsum()
        cum_vol = intraday_df["volume"].cumsum()
        vwap = (cum_tp_vol / cum_vol.replace(0, np.nan)).iloc[-window_bars:]
        if vwap.isna().any():
            return np.nan
        x = np.arange(len(vwap), dtype=float)
        return float(np.polyfit(x, vwap.values, 1)[0])

    # ------------------------------------------------------------------
    # Cross-sectional features (universe-level)
    # ------------------------------------------------------------------

    @staticmethod
    def sector_relative_return(
        prices_df: pd.DataFrame,
        sector_map: dict[str, str],
        periods: int = 5,
    ) -> pd.DataFrame:
        """
        Stock return minus median return of its sector peers.

        Ranks and medians are computed cross-sectionally at each T using only
        data available at T (the close cross-section is fully observable at T).

        Parameters
        ----------
        prices_df : DataFrame (T × N), close prices.
        sector_map : dict {symbol: sector}.
        periods : int, return window.

        Returns
        -------
        DataFrame (T × N), excess return vs sector median.
        """
        ret = np.log(prices_df / prices_df.shift(periods))

        # For each symbol, subtract median return of its sector on same date
        sectors = {s: sector_map.get(s, "Unknown") for s in prices_df.columns}
        sector_series = pd.Series(sectors)
        unique_sectors = sector_series.unique()

        excess = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
        for sector in unique_sectors:
            cols = sector_series[sector_series == sector].index.tolist()
            cols_in_df = [c for c in cols if c in ret.columns]
            if not cols_in_df:
                continue
            sector_median = ret[cols_in_df].median(axis=1)
            for col in cols_in_df:
                excess[col] = ret[col] - sector_median

        return excess

    @staticmethod
    def cross_sectional_rank(series_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sectional percentile rank of values at each date.

        rank_df[T, sym] = percentile of sym's value among all symbols at date T.
        Uses only data at T (the cross-section of contemporaneous values).
        """
        return series_df.rank(axis=1, pct=True)

    # ------------------------------------------------------------------
    # Market (benchmark) features
    # ------------------------------------------------------------------

    @staticmethod
    def nifty_return(nifty_close: pd.Series, periods: int) -> pd.Series:
        """Log return of NIFTY over `periods` days."""
        return np.log(nifty_close / nifty_close.shift(periods))

    @staticmethod
    def nifty_vol_ratio(nifty_close: pd.Series) -> pd.Series:
        """Short/long volatility ratio for NIFTY (regime signal)."""
        return FeatureEngine.vol_ratio(nifty_close, short_window=10, long_window=63)

    @staticmethod
    def rolling_beta(
        stock_prices: pd.Series,
        nifty_prices: pd.Series,
        window: int = 30,
    ) -> pd.Series:
        """
        Rolling market beta (OLS regression of stock returns on NIFTY returns).

        beta[T] = Cov(r_stock, r_nifty)[T-window..T] / Var(r_nifty)[T-window..T]

        Uses only past returns (T-window to T), no look-ahead.
        """
        stock_ret = np.log(stock_prices / stock_prices.shift(1))
        nifty_ret = np.log(nifty_prices / nifty_prices.shift(1))

        combined = pd.concat([nifty_ret, stock_ret.reindex(nifty_ret.index)], axis=1)
        combined.columns = ["nifty", "stock"]

        # Vectorized rolling OLS slope:
        #     beta = Cov(stock, nifty) / Var(nifty)
        # computed from rolling moments. The previous implementation looped in
        # Python over every bar, re-slicing and re-intersecting indices each
        # time — O(T * window) with pandas overhead per step, which made
        # feature generation over a real universe take hours.
        min_obs = max(window // 2, 2)
        n = combined["nifty"]
        s = combined["stock"]

        roll = dict(window=window, min_periods=min_obs)
        mean_n = n.rolling(**roll).mean()
        mean_s = s.rolling(**roll).mean()
        mean_ns = (n * s).rolling(**roll).mean()
        mean_nn = (n * n).rolling(**roll).mean()

        cov = mean_ns - mean_n * mean_s
        var = mean_nn - mean_n * mean_n

        beta = cov / var.where(var > 1e-12)
        beta.name = "beta_30d"
        return beta

    # ------------------------------------------------------------------
    # Master compute
    # ------------------------------------------------------------------

    def compute_all_features(
        self,
        prices_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        nifty_df: pd.DataFrame,
        high_df: Optional[pd.DataFrame] = None,
        low_df: Optional[pd.DataFrame] = None,
        sector_map: Optional[dict[str, str]] = None,
        intraday_data: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Compute every feature for every symbol in prices_df.

        Parameters
        ----------
        prices_df : DataFrame (T × N), close prices, DatetimeIndex.
        volume_df : DataFrame (T × N), daily volume.
        nifty_df  : DataFrame (T × ≥1) with a 'close' column.
        high_df   : DataFrame (T × N), daily high prices (optional, for ATR).
        low_df    : DataFrame (T × N), daily low prices (optional, for ATR).
        sector_map : dict {symbol: sector} (optional, for sector-relative features).
        intraday_data : dict {symbol: list_of_intraday_dfs} (optional, for VWAP).

        Returns
        -------
        pd.DataFrame with MultiIndex columns (feature_name, symbol) or
        flat columns named "{feature}_{symbol}".

        The returned DataFrame has the same DatetimeIndex as prices_df.
        All features are NaN for early dates where the lookback window is
        insufficient — this is correct; do not fill with zeroes.
        """
        if "close" in nifty_df.columns:
            nifty_close = nifty_df["close"].reindex(prices_df.index)
        else:
            nifty_close = nifty_df.iloc[:, 0].reindex(prices_df.index)

        symbols = prices_df.columns.tolist()
        n_dates = len(prices_df)

        all_features: dict[str, pd.DataFrame] = {}

        # ----- Price / return features -----
        for sym in symbols:
            px = prices_df[sym]
            for periods, name in [
                (1, "log_return_1d"),
                (5, "log_return_5d"),
                (10, "log_return_10d"),
                (21, "log_return_21d"),
                (63, "log_return_63d"),
            ]:
                all_features.setdefault(name, {})[sym] = self.log_return(px, periods)

            all_features.setdefault("excess_return_vs_nifty_5d", {})[sym] = (
                self.excess_return_vs_nifty(px, nifty_close, 5)
            )
            all_features.setdefault("excess_return_vs_nifty_21d", {})[sym] = (
                self.excess_return_vs_nifty(px, nifty_close, 21)
            )
            all_features.setdefault("momentum_12_1", {})[sym] = self.momentum_12_1(px)
            all_features.setdefault("rsi_14", {})[sym] = self.rsi(px, 14)
            all_features.setdefault("distance_from_52w_high", {})[sym] = (
                self.distance_from_52w_high(px)
            )
            all_features.setdefault("distance_from_52w_low", {})[sym] = (
                self.distance_from_52w_low(px)
            )
            all_features.setdefault("price_to_sma20", {})[sym] = self.price_to_sma(px, 20)
            all_features.setdefault("price_to_sma50", {})[sym] = self.price_to_sma(px, 50)
            all_features.setdefault("price_to_sma200", {})[sym] = self.price_to_sma(px, 200)

        # ----- Volatility features -----
        for sym in symbols:
            px = prices_df[sym]
            all_features.setdefault("realized_vol_10d", {})[sym] = self.realized_vol(px, 10)
            all_features.setdefault("realized_vol_21d", {})[sym] = self.realized_vol(px, 21)
            all_features.setdefault("realized_vol_63d", {})[sym] = self.realized_vol(px, 63)
            all_features.setdefault("ewma_vol_21d", {})[sym] = self.ewma_vol(px)
            all_features.setdefault("vol_ratio", {})[sym] = self.vol_ratio(px, 10, 63)

            if high_df is not None and low_df is not None and sym in high_df.columns:
                all_features.setdefault("atr_14", {})[sym] = self.atr(
                    high_df[sym], low_df[sym], px, 14
                )
            else:
                all_features.setdefault("atr_14", {})[sym] = pd.Series(
                    np.nan, index=px.index, name=sym
                )

        # ----- Volume features -----
        for sym in symbols:
            if sym not in volume_df.columns:
                continue
            vol = volume_df[sym]
            px = prices_df[sym]
            all_features.setdefault("volume_ratio_10d", {})[sym] = self.volume_ratio(vol, 10)
            all_features.setdefault("relative_volume_zscore", {})[sym] = (
                self.relative_volume_zscore(vol, 20)
            )
            all_features.setdefault("pvt", {})[sym] = self.pvt(px, vol)
            all_features.setdefault("obv_slope_10d", {})[sym] = self.obv_slope(px, vol, 10)

        # ----- VWAP features -----
        if intraday_data is not None:
            for sym in symbols:
                sym_intraday = intraday_data.get(sym)
                if sym_intraday is None:
                    all_features.setdefault("vwap_distance", {})[sym] = pd.Series(
                        np.nan, index=prices_df.index
                    )
                    all_features.setdefault("vwap_slope", {})[sym] = pd.Series(
                        np.nan, index=prices_df.index
                    )
                else:
                    # sym_intraday is dict {date: intraday_df}
                    dist_vals, slope_vals = [], []
                    for dt in prices_df.index:
                        idf = sym_intraday.get(dt.date() if hasattr(dt, "date") else dt)
                        if idf is not None and not idf.empty:
                            dist_vals.append(self.vwap_distance(idf))
                            slope_vals.append(self.vwap_slope(idf))
                        else:
                            dist_vals.append(np.nan)
                            slope_vals.append(np.nan)
                    all_features.setdefault("vwap_distance", {})[sym] = pd.Series(
                        dist_vals, index=prices_df.index
                    )
                    all_features.setdefault("vwap_slope", {})[sym] = pd.Series(
                        slope_vals, index=prices_df.index
                    )

        # ----- Cross-sectional features -----
        sect_map = sector_map or {}
        if sect_map:
            for periods, fname in [(5, "sector_rel_return_5d")]:
                excess = self.sector_relative_return(prices_df, sect_map, periods)
                for sym in symbols:
                    if sym in excess.columns:
                        all_features.setdefault(fname, {})[sym] = excess[sym]

        # Cross-sectional momentum rank (21d returns)
        ret_21d_df = pd.DataFrame({sym: self.log_return(prices_df[sym], 21) for sym in symbols})
        cs_mom_rank = self.cross_sectional_rank(ret_21d_df)
        for sym in symbols:
            if sym in cs_mom_rank.columns:
                all_features.setdefault("cross_sectional_momentum_rank", {})[sym] = (
                    cs_mom_rank[sym]
                )

        # Cross-sectional vol rank
        vol_21d_df = pd.DataFrame({
            sym: self.realized_vol(prices_df[sym], 21) for sym in symbols
        })
        cs_vol_rank = self.cross_sectional_rank(vol_21d_df)
        for sym in symbols:
            if sym in cs_vol_rank.columns:
                all_features.setdefault("cross_sectional_vol_rank", {})[sym] = (
                    cs_vol_rank[sym]
                )

        # ----- Market features -----
        nifty_ret_5d = self.nifty_return(nifty_close, 5)
        nifty_ret_21d = self.nifty_return(nifty_close, 21)
        nifty_vr = self.nifty_vol_ratio(nifty_close)

        for sym in symbols:
            all_features.setdefault("nifty_return_5d", {})[sym] = nifty_ret_5d
            all_features.setdefault("nifty_return_21d", {})[sym] = nifty_ret_21d
            all_features.setdefault("nifty_vol_ratio", {})[sym] = nifty_vr
            all_features.setdefault("beta_30d", {})[sym] = self.rolling_beta(
                prices_df[sym], nifty_close, 30
            )

        # ----- Assemble flat DataFrame -----
        # Each entry in all_features: {feature_name: {symbol: pd.Series}}
        # Output columns: "{feature}_{symbol}"
        cols = {}
        for feat_name, sym_dict in all_features.items():
            for sym, series in sym_dict.items():
                col_name = f"{feat_name}__{sym}"
                cols[col_name] = series

        result = pd.DataFrame(cols, index=prices_df.index)
        logger.info(
            "compute_all_features: produced %d features for %d symbols over %d dates",
            len(all_features),
            len(symbols),
            n_dates,
        )
        return result

    # ------------------------------------------------------------------
    # Convenience: extract per-symbol feature panel
    # ------------------------------------------------------------------

    @staticmethod
    def get_symbol_features(
        features_df: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """
        Extract all features for a single symbol from the wide features_df.

        Returns a DataFrame where each column is one feature (without the
        symbol suffix).
        """
        suffix = f"__{symbol}"
        cols = {
            c.replace(suffix, ""): features_df[c]
            for c in features_df.columns
            if c.endswith(suffix)
        }
        return pd.DataFrame(cols, index=features_df.index)
