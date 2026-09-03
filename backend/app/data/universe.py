"""
universe.py — Stock universe construction and filtering for AlgoDollar.

SURVIVORSHIP BIAS WARNING
--------------------------
The hardcoded NIFTY_500_SYMBOLS list reflects the *current* index composition
as of the file's last update.  Historical backtests that use this static list
will suffer survivorship bias: companies that were delisted, merged, or removed
from the index (often the worst performers) are absent from the list, causing
backtested returns to be overstated.

Mitigation approaches (not yet implemented here):
  - Subscribe to a point-in-time universe service (e.g. NSE historical
    constituents file, Bloomberg, or a paid data vendor).
  - Store daily snapshots of the index in a database and join on backtest date.

Until point-in-time data is integrated, treat all backtest Sharpe ratios as
optimistic upper bounds.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded fallback universe — NIFTY 500 as of 2024-Q4.
# Replace with a database-backed call when point-in-time data is available.
# ---------------------------------------------------------------------------
_NIFTY500_FALLBACK: List[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    "HINDUNILVR", "ITC", "SBIN", "BAJFINANCE", "BHARTIARTL",
    "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND",
    "POWERGRID", "NTPC", "TATAMOTORS", "HCLTECH", "M&M",
    "BAJAJFINSV", "TECHM", "ADANIPORTS", "COALINDIA", "ONGC",
    "DRREDDY", "JSWSTEEL", "TATASTEEL", "BPCL", "CIPLA",
    "DIVISLAB", "EICHERMOT", "UPL", "APOLLOHOSP", "BRITANNIA",
    "HEROMOTOCO", "GRASIM", "HINDALCO", "SBILIFE", "HDFCLIFE",
    "INDUSINDBK", "IOC", "TATACONSUM", "SHREECEM", "PIDILITIND",
    "AMBUJACEM", "LTIM", "DMART", "NAUKRI", "BERGEPAINT",
    "ICICIGI", "MCDOWELL-N", "COLPAL", "PAGEIND", "SRF",
    "TORNTPHARM", "GODREJCP", "BIOCON", "AUROPHARMA", "LUPIN",
    "PVR", "BALKRISIND", "HAVELLS", "MUTHOOTFIN", "JUBLFOOD",
    "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB", "CANBK",
    "BOSCHLTD", "TVSMOTOR", "MOTHERSON", "BAJAJ-AUTO", "GAIL",
    "CHOLAFIN", "LICHSGFIN", "RECLTD", "PFC", "NHPC",
    "TATAPOWER", "ADANIENT", "ADANIGREEN", "ADANITRANS", "ADANIGAS",
    "SIEMENS", "ABB", "VOLTAS", "WHIRLPOOL", "BLUESTARCO",
    "MARICO", "EMAMILTD", "DABUR", "ZOMATO", "PAYTM",
    "NYKAA", "POLICYBZR", "CARTRADE", "EASEMYTRIP", "IRCTC",
    "HAPPSTMNDS", "MPHASIS", "COFORGE", "PERSISTENT", "LTTS",
    "TATAELXSI", "KPITTECH", "MASTECH", "NIITTECH", "HEXAWARE",
    "MANAPPURAM", "BAJAJHLDNG", "TATAINVEST", "HDFC", "GRUH",
    "PGHH", "3MINDIA", "HONAUT", "CRISIL", "ICRA",
    "RBLBANK", "YESBANK", "DCBBANK", "KARURVYSYA", "SOUTHBANK",
    "UJJIVANSFB", "AUBANK", "EQUITASBNK", "SURYODAY", "ESAFSFB",
    "CHOLAHLDNG", "SUNDARMFIN", "M&MFIN", "SHFL", "AAVAS",
    "HOMEFIRST", "APTUS", "REPCO", "CANFINHOME", "HUDCO",
    "IRFC", "RAILVIKAS", "RVNL", "IRCON", "RITES",
    "BEL", "HAL", "BEML", "MIDHANI", "COCHINSHIP",
    "GRINDWELL", "CUMMINSIND", "THERMAX", "BHEL", "SUZLON",
    "INOXWIND", "RPOWER", "TORNTPOWER", "CESC", "JPPOWER",
    "CENTURYPLY", "GREENPLY", "ASTRAL", "SUPREMEIND", "NOCIL",
    "AAPL", "DEEPAKNTR", "AAVAS", "ALKYLAMINE", "FINEORG",
    "ATUL", "VINATI", "NAVINFLUOR", "CLEAN", "LAXMICHEM",
    "PIIND", "RALLIS", "UPL", "BAYER", "SYNGENE",
    "LAURUSLABS", "GRANULES", "AJANTPHARM", "IPCALAB", "ALKEM",
    "ABBOTINDIA", "GLAXO", "PFIZER", "SANOFI", "NATCOPHARM",
    "FINEORG", "JUBLPHARMA", "SUVEN", "NEULANDLAB", "DIVI",
    "MCDPHARMA", "SOLARA", "SEQUENT", "ERIS", "GLAND",
    "KRSNAA", "ASTER", "RAINBOW", "METROPOLIS", "THYROCARE",
    "LALPATHLAB", "MAXHEALTH", "FORTIS", "NARAYANA", "YATHARTH",
    "INDHOTEL", "LEMONTREE", "CHALET", "LUXIND", "SAFARI",
    "VIP", "ASTRAZEN", "ISGEC", "TIMKEN", "SKF",
    "SCHAEFFLER", "ELGIEQUIP", "KIRLOSENG", "ESCORTS", "JBMA",
    "SUNDRMFAST", "ENDURANCE", "SUPRAJIT", "BFUTILITIE", "BORORENEW",
    "WELCORP", "MAHSEAMLES", "APL", "JINDALSAW", "RATNAMANI",
    "GALLANTT", "SAIL", "NMDC", "KIOCL", "GMRINFRA",
    "IRB", "KNRCON", "NCC", "AHLUCONT", "PNC",
    "HGINFRA", "GPPL", "ADANIPORTS", "CONCOR", "VRL",
    "BLUEDART", "MAHLOG", "GATI", "ALLCARGO", "TCI",
    "TTML", "GTLINFRA", "INDIAMART", "JUSTDIAL", "TRADINGO",
    "NAZARA", "ONEPOINT", "RATEGAIN", "INTELLECT", "NUCLEUS",
    "KFINTECH", "CAMS", "CDSL", "BSE", "MCX",
    "IIFL", "ANGELONE", "MOTILALOFS", "5PAISA", "GEOJITFSL",
    "ICICIPRULI", "HDFCAMC", "NIPPONLIFE", "UTIAMC", "ABSLAMC",
    "POLICYBZR", "GICRE", "NIACL", "STARHEALTH", "GODIGIT",
    "MANINFRA", "PRESTIGE", "PHOENIXLTD", "BRIGADE", "GODREJPROP",
    "MAHINDCIE", "OBEROIRLTY", "SOBHA", "KOLTEPATIL", "ARVIN",
    "CENTURYTEX", "VARDHMAN", "TRIDENT", "RAYMOND", "ARVINDLTD",
    "GRASIM", "AIAENG", "CARBORUNIV", "IFBIND", "LINDE",
    "TATA CHEM", "DEEPAKFERT", "GNFC", "GSPL", "MGL",
    "IGL", "AEGISCHEM", "GHCL", "GUJARAT GAS", "GUJALKALI",
    "DALBHARAT", "JKCEMENT", "HEIDELBERG", "BIRLACEM", "PRISM",
    "NUVOCO", "SAPPHIRE", "KCP", "KESORAMIND", "MANGLMCEM",
    "WONDERLA", "DELTACORP", "CELESTIAL", "BHAGCHEM", "TINPLATE",
    "HINDCOPPER", "NATIONALUM", "NILE", "SANDUMA", "SHYAMMET",
    "GPIL", "WOCKPHARMA", "UNICHEM", "FDC", "STRIDES",
    "CAPLIPOINT", "MEDPLUS", "TIPSFILMS", "SAREGAMA", "TIPS",
    "NETWORK18", "TV18BRDCST", "SUNTV", "ZEEL", "PVR",
    "INOXLEISUR", "BALAJITELE", "SHEMAROO", "JAGRAN", "DBCORP",
    "HT MEDIA", "HINDMOTORS", "MHRIL", "EIHOTEL", "TAJGVK",
    "ORIENTBELL", "CERAMICIN", "SOMANYCER", "CERA", "KAJARIACER",
    "ASAHI INDIA", "AEGIS", "ALICON", "AMARA RAJA", "EXIDEIND",
    "HBL POWER", "INDOASIAFM", "BATAINDIA", "RELAXO", "MIRZA",
    "LIBAS", "MANYAVAR", "VEDL", "HINDZINC", "HIND COPPER",
]

# Remove any accidental duplicates introduced above
_NIFTY500_FALLBACK = list(dict.fromkeys(_NIFTY500_FALLBACK))

# Sector mapping for filtering and cross-sectional analysis.
# This is a representative subset; extend as needed.
_SECTOR_MAP: Dict[str, str] = {
    "RELIANCE": "Energy",
    "ONGC": "Energy",
    "BPCL": "Energy",
    "IOC": "Energy",
    "GAIL": "Energy",
    "TATAPOWER": "Utilities",
    "NTPC": "Utilities",
    "POWERGRID": "Utilities",
    "NHPC": "Utilities",
    "CESC": "Utilities",
    "TCS": "IT",
    "INFY": "IT",
    "WIPRO": "IT",
    "HCLTECH": "IT",
    "TECHM": "IT",
    "LTIM": "IT",
    "MPHASIS": "IT",
    "COFORGE": "IT",
    "PERSISTENT": "IT",
    "LTTS": "IT",
    "TATAELXSI": "IT",
    "HDFCBANK": "Banking",
    "ICICIBANK": "Banking",
    "SBIN": "Banking",
    "KOTAKBANK": "Banking",
    "AXISBANK": "Banking",
    "INDUSINDBK": "Banking",
    "BANDHANBNK": "Banking",
    "FEDERALBNK": "Banking",
    "IDFCFIRSTB": "Banking",
    "PNB": "Banking",
    "CANBK": "Banking",
    "RBLBANK": "Banking",
    "YESBANK": "Banking",
    "BAJFINANCE": "NBFC",
    "BAJAJFINSV": "NBFC",
    "CHOLAFIN": "NBFC",
    "LICHSGFIN": "NBFC",
    "M&MFIN": "NBFC",
    "MUTHOOTFIN": "NBFC",
    "MANAPPURAM": "NBFC",
    "HINDUNILVR": "FMCG",
    "ITC": "FMCG",
    "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG",
    "MARICO": "FMCG",
    "DABUR": "FMCG",
    "COLPAL": "FMCG",
    "GODREJCP": "FMCG",
    "EMAMILTD": "FMCG",
    "SUNPHARMA": "Pharma",
    "DRREDDY": "Pharma",
    "CIPLA": "Pharma",
    "DIVISLAB": "Pharma",
    "LUPIN": "Pharma",
    "AUROPHARMA": "Pharma",
    "BIOCON": "Pharma",
    "LAURUSLABS": "Pharma",
    "IPCALAB": "Pharma",
    "ALKEM": "Pharma",
    "MARUTI": "Auto",
    "TATAMOTORS": "Auto",
    "M&M": "Auto",
    "BAJAJ-AUTO": "Auto",
    "HEROMOTOCO": "Auto",
    "EICHERMOT": "Auto",
    "TVSMOTOR": "Auto",
    "LT": "Capital Goods",
    "SIEMENS": "Capital Goods",
    "ABB": "Capital Goods",
    "BHEL": "Capital Goods",
    "BEL": "Capital Goods",
    "HAL": "Capital Goods",
    "JSWSTEEL": "Metals",
    "TATASTEEL": "Metals",
    "HINDALCO": "Metals",
    "SAIL": "Metals",
    "NMDC": "Metals",
    "VEDL": "Metals",
    "HINDZINC": "Metals",
    "ULTRACEMCO": "Cement",
    "GRASIM": "Cement",
    "SHREECEM": "Cement",
    "AMBUJACEM": "Cement",
    "DALBHARAT": "Cement",
    "JKCEMENT": "Cement",
    "ASIANPAINT": "Paints",
    "BERGEPAINT": "Paints",
    "PIDILITIND": "Chemicals",
    "SRF": "Chemicals",
    "ATUL": "Chemicals",
    "DEEPAKNTR": "Chemicals",
    "NAVINFLUOR": "Chemicals",
    "TITAN": "Consumer Discretionary",
    "DMART": "Consumer Discretionary",
    "JUBLFOOD": "Consumer Discretionary",
    "NAUKRI": "Technology",
    "ZOMATO": "Technology",
    "INDIAMART": "Technology",
    "APOLLOHOSP": "Healthcare",
    "MAXHEALTH": "Healthcare",
    "FORTIS": "Healthcare",
    "METROPOLIS": "Healthcare",
    "LALPATHLAB": "Healthcare",
    "BHARTIARTL": "Telecom",
    "COALINDIA": "Mining",
    "ADANIPORTS": "Infrastructure",
    "ADANIENT": "Conglomerate",
    "ADANIGREEN": "Renewables",
    "TATACHEM": "Chemicals",
    "ADANIGAS": "Utilities",
    "RECLTD": "NBFC",
    "PFC": "NBFC",
    "IRFC": "NBFC",
    "GODREJPROP": "Real Estate",
    "PRESTIGE": "Real Estate",
    "BRIGADE": "Real Estate",
    "SOBHA": "Real Estate",
    "OBEROIRLTY": "Real Estate",
    "SBILIFE": "Insurance",
    "HDFCLIFE": "Insurance",
    "ICICIGI": "Insurance",
    "STARHEALTH": "Insurance",
    "NIACL": "Insurance",
    "HDFCAMC": "Asset Management",
    "NIPPONLIFE": "Asset Management",
    "CDSL": "Financial Services",
    "BSE": "Financial Services",
    "MCX": "Financial Services",
    "ANGELONE": "Financial Services",
    "IRCTC": "Travel",
    "INDHOTEL": "Hospitality",
    "SUNTV": "Media",
    "ZEEL": "Media",
    "PVR": "Entertainment",
    "INOXLEISUR": "Entertainment",
}


class StockUniverse:
    """
    Constructs and filters the investable stock universe.

    SURVIVORSHIP BIAS WARNING
    -------------------------
    All methods that rely on `get_nifty500_symbols()` use a static snapshot of
    the index.  Historical simulations will include only stocks that are *still
    in* the index today, omitting past members that were removed (often due to
    poor performance or delistings).  This causes upward bias in backtested
    returns.  Integrate point-in-time data before drawing conclusions about
    historical edge.
    """

    # ------------------------------------------------------------------
    # Universe membership
    # ------------------------------------------------------------------

    @staticmethod
    def get_nifty500_symbols() -> List[str]:
        """
        Return the current NIFTY 500 constituent symbols.

        NOTE: This is a hardcoded snapshot, *not* a point-in-time list.
        For production use, fetch constituents from NSE's bhavdata or a paid
        vendor and filter by the backtest date to avoid survivorship bias.
        """
        logger.warning(
            "get_nifty500_symbols() returns a static snapshot. "
            "Backtests will suffer survivorship bias."
        )
        return list(_NIFTY500_FALLBACK)

    # ------------------------------------------------------------------
    # Liquidity filters — computed on historical data
    # ------------------------------------------------------------------

    @staticmethod
    def filter_liquid(
        symbols: List[str],
        prices_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        min_avg_volume: float = 500_000,
        min_avg_turnover: float = 5_000_000,
        lookback_days: int = 60,
    ) -> List[str]:
        """
        Retain symbols that satisfy minimum average daily volume AND turnover.

        Parameters
        ----------
        symbols : list of str
            Candidate symbols.
        prices_df : DataFrame, shape (dates, symbols), close prices.
        volume_df : DataFrame, shape (dates, symbols), daily shares traded.
        min_avg_volume : float
            Minimum average daily volume (shares).  Default 500 000.
        min_avg_turnover : float
            Minimum average daily value traded (INR).  Default ₹50 lakh.
        lookback_days : int
            Rolling window for the average computation.

        Returns
        -------
        list[str]
        """
        liquid = []
        for sym in symbols:
            if sym not in prices_df.columns or sym not in volume_df.columns:
                continue
            vol_series = volume_df[sym].dropna().tail(lookback_days)
            price_series = prices_df[sym].dropna().tail(lookback_days)
            if len(vol_series) < lookback_days // 2:
                continue
            avg_vol = vol_series.mean()
            # Align on common index for turnover computation
            common_idx = vol_series.index.intersection(price_series.index)
            if len(common_idx) == 0:
                continue
            avg_turnover = (
                vol_series.loc[common_idx] * price_series.loc[common_idx]
            ).mean()
            if avg_vol >= min_avg_volume and avg_turnover >= min_avg_turnover:
                liquid.append(sym)
        logger.info(
            "filter_liquid: %d/%d symbols passed (vol≥%s, turnover≥%s)",
            len(liquid),
            len(symbols),
            min_avg_volume,
            min_avg_turnover,
        )
        return liquid

    @staticmethod
    def filter_by_market_cap(
        symbols: List[str],
        market_cap_df: pd.DataFrame,
        min_cap_crore: float = 500,
    ) -> List[str]:
        """
        Filter symbols by most-recent market cap.

        Parameters
        ----------
        symbols : list[str]
        market_cap_df : DataFrame, shape (dates, symbols), market cap in INR crore.
        min_cap_crore : float
            Minimum market cap (INR crore).  Default ₹500 crore (~small cap threshold).

        Returns
        -------
        list[str]
        """
        eligible = []
        for sym in symbols:
            if sym not in market_cap_df.columns:
                continue
            latest = market_cap_df[sym].dropna()
            if latest.empty:
                continue
            if latest.iloc[-1] >= min_cap_crore:
                eligible.append(sym)
        logger.info(
            "filter_by_market_cap: %d/%d symbols passed (min_cap≥%s crore)",
            len(eligible),
            len(symbols),
            min_cap_crore,
        )
        return eligible

    # ------------------------------------------------------------------
    # Sector mapping
    # ------------------------------------------------------------------

    @staticmethod
    def get_sector_mapping() -> Dict[str, str]:
        """
        Return a symbol → sector dictionary.

        NOTE: Sector classifications can change over time (spin-offs, business
        pivots).  This mapping is a static snapshot and should be updated
        periodically or sourced from a live metadata API.
        """
        return dict(_SECTOR_MAP)

    # ------------------------------------------------------------------
    # Composite universe builder
    # ------------------------------------------------------------------

    @staticmethod
    def build_eligible_universe(
        symbols: List[str],
        prices_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        market_cap_df: pd.DataFrame | None = None,
        min_avg_volume: float = 500_000,
        min_avg_turnover: float = 5_000_000,
        min_cap_crore: float = 500,
        min_price: float = 10.0,
        min_trading_days: int = 126,
    ) -> List[str]:
        """
        Build the eligible investment universe by applying all filters.

        Pipeline
        --------
        1. Symbol must be present in prices_df and volume_df.
        2. Minimum number of trading days with non-null data (data quality).
        3. Minimum absolute price (avoid penny stocks / instrument errors).
        4. Liquidity filter (volume + turnover).
        5. Market cap filter (if market_cap_df provided).

        SURVIVORSHIP BIAS WARNING: the input `symbols` list itself is typically
        derived from today's index composition.  See class-level docstring.

        Parameters
        ----------
        symbols : list[str]
        prices_df : DataFrame (dates × symbols), close prices.
        volume_df : DataFrame (dates × symbols), daily shares traded.
        market_cap_df : DataFrame (dates × symbols) or None.
        min_avg_volume : float
        min_avg_turnover : float
        min_cap_crore : float
        min_price : float
            Reject symbols with latest price below this (INR).
        min_trading_days : int
            Minimum non-null rows in prices_df required.

        Returns
        -------
        list[str]
        """
        # Step 1: presence check
        present = [
            s for s in symbols
            if s in prices_df.columns and s in volume_df.columns
        ]

        # Step 2: minimum history
        enough_history = []
        for sym in present:
            valid_days = prices_df[sym].dropna().shape[0]
            if valid_days >= min_trading_days:
                enough_history.append(sym)

        # Step 3: minimum price
        above_min_price = []
        for sym in enough_history:
            latest_px = prices_df[sym].dropna()
            if latest_px.empty:
                continue
            if latest_px.iloc[-1] >= min_price:
                above_min_price.append(sym)

        # Step 4: liquidity
        liquid = StockUniverse.filter_liquid(
            above_min_price,
            prices_df,
            volume_df,
            min_avg_volume=min_avg_volume,
            min_avg_turnover=min_avg_turnover,
        )

        # Step 5: market cap (optional)
        if market_cap_df is not None:
            result = StockUniverse.filter_by_market_cap(
                liquid, market_cap_df, min_cap_crore=min_cap_crore
            )
        else:
            result = liquid

        logger.info(
            "build_eligible_universe: %d symbols passed all filters "
            "(started with %d)",
            len(result),
            len(symbols),
        )
        return result
