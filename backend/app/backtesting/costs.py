"""
costs.py — Zerodha transaction cost model (2024 rates).

All rate constants are in a CONFIG dict at the top.  Update rates there
without touching the logic when SEBI/exchange revises fees.

CURRENT RATES (as of 2024):
  Intraday (MIS): brokerage = min(0.03% of trade value, ₹20) per order
  Delivery (CNC): brokerage = ₹0 (Zerodha free delivery)
  STT: 0.025% sell side (intraday), 0.1% both sides (delivery)
  NSE exchange charge: 0.00322% both sides
  SEBI charge: ₹10 per crore (0.0001%)
  GST: 18% on (brokerage + exchange charge + sebi charge)
  Stamp duty: 0.003% on buy (intraday), 0.015% on buy (delivery)
  DP charges: ₹13.5 + GST = ₹15.93 per script per day (delivery sell)

Reference: https://zerodha.com/charges/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate configuration — update here when rates change
# ---------------------------------------------------------------------------

CONFIG = {
    # Brokerage
    "intraday_brokerage_pct":    0.0003,  # 0.03%
    "intraday_brokerage_max_rs": 20.0,    # ₹20 cap
    "delivery_brokerage_pct":    0.0,     # Free for delivery on Zerodha

    # STT (Securities Transaction Tax)
    "intraday_stt_sell_pct":     0.00025,  # 0.025% on sell side only
    "delivery_stt_buy_pct":      0.001,    # 0.1% on buy
    "delivery_stt_sell_pct":     0.001,    # 0.1% on sell

    # Exchange transaction charges (NSE)
    "nse_exchange_charge_pct":   0.0000322,  # 0.00322%

    # SEBI turnover fee
    "sebi_charge_pct":           0.000001,   # ₹10 per crore = 0.0001% = 1e-6

    # GST on (brokerage + exchange + SEBI)
    "gst_pct":                   0.18,

    # Stamp duty (on buy side only)
    "intraday_stamp_duty_pct":   0.00003,    # 0.003%
    "delivery_stamp_duty_pct":   0.00015,    # 0.015%

    # DP (depository participant) charges per script per debit (delivery sell)
    # Zerodha: ₹13.50 + GST = ₹15.93
    "dp_charge_rs":              15.93,
}


# ---------------------------------------------------------------------------
# Enums / types
# ---------------------------------------------------------------------------

class TransactionType(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class ProductType(str, Enum):
    MIS = "MIS"   # Intraday
    CNC = "CNC"   # Delivery
    NRML = "NRML" # F&O overnight (for future extension)


class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"


# ---------------------------------------------------------------------------
# Cost breakdown
# ---------------------------------------------------------------------------

@dataclass
class CostBreakdown:
    """
    Detailed breakdown of transaction costs for one trade leg.

    All values in INR.
    """
    brokerage:       float
    stt:             float
    exchange_charge: float
    sebi_charge:     float
    gst:             float
    stamp_duty:      float
    dp_charges:      float
    total:           float
    total_pct:       float  # total / trade_value (as a fraction, not %)

    def __repr__(self) -> str:
        return (
            f"CostBreakdown(brokerage={self.brokerage:.2f}, stt={self.stt:.2f}, "
            f"exchange={self.exchange_charge:.2f}, sebi={self.sebi_charge:.2f}, "
            f"gst={self.gst:.2f}, stamp={self.stamp_duty:.2f}, "
            f"dp={self.dp_charges:.2f}, TOTAL={self.total:.2f} "
            f"[{self.total_pct*100:.3f}% of trade value])"
        )


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

class ZerodhaCostModel:
    """
    Calculate all-in transaction costs for Zerodha (NSE cash segment).

    Usage
    -----
    model = ZerodhaCostModel()
    cost = model.calculate_costs("BUY", qty=100, price=1500, product="CNC")
    print(cost.total)         # ₹ total cost
    print(cost.total_pct)     # fraction of trade value
    """

    def __init__(self, config: dict | None = None):
        """
        Parameters
        ----------
        config : dict or None
            Override rate configuration.  Unspecified keys fall back to
            module-level CONFIG defaults.
        """
        self._cfg = {**CONFIG, **(config or {})}

    def calculate_costs(
        self,
        transaction_type: str | TransactionType,
        qty: int | float,
        price: float,
        exchange: str | Exchange = "NSE",
        product: str | ProductType = "MIS",
    ) -> CostBreakdown:
        """
        Compute all transaction costs for one order leg.

        Parameters
        ----------
        transaction_type : 'BUY' or 'SELL'
        qty : int | float, number of shares.
        price : float, execution price per share.
        exchange : 'NSE' or 'BSE' (exchange charges may differ; NSE used here).
        product : 'MIS' (intraday) or 'CNC' (delivery) or 'NRML'.

        Returns
        -------
        CostBreakdown
        """
        cfg = self._cfg
        tx = TransactionType(str(transaction_type).upper())
        prod = ProductType(str(product).upper())
        trade_value = float(qty) * float(price)

        if trade_value <= 0:
            return CostBreakdown(0, 0, 0, 0, 0, 0, 0, 0, 0)

        intraday = prod in (ProductType.MIS,)

        # ------------------------------------------------------------------
        # Brokerage
        # ------------------------------------------------------------------
        if intraday:
            brokerage = min(
                trade_value * cfg["intraday_brokerage_pct"],
                cfg["intraday_brokerage_max_rs"],
            )
        else:
            brokerage = trade_value * cfg["delivery_brokerage_pct"]  # ₹0

        # ------------------------------------------------------------------
        # STT
        # ------------------------------------------------------------------
        if intraday:
            # STT only on sell side for intraday
            stt = trade_value * cfg["intraday_stt_sell_pct"] if tx == TransactionType.SELL else 0.0
        else:
            # STT on both sides for delivery
            if tx == TransactionType.BUY:
                stt = trade_value * cfg["delivery_stt_buy_pct"]
            else:
                stt = trade_value * cfg["delivery_stt_sell_pct"]

        # ------------------------------------------------------------------
        # Exchange transaction charge (NSE)
        # ------------------------------------------------------------------
        exchange_charge = trade_value * cfg["nse_exchange_charge_pct"]

        # ------------------------------------------------------------------
        # SEBI turnover fee
        # ------------------------------------------------------------------
        sebi_charge = trade_value * cfg["sebi_charge_pct"]

        # ------------------------------------------------------------------
        # GST (18% on brokerage + exchange charge + SEBI charge)
        # ------------------------------------------------------------------
        gst_base = brokerage + exchange_charge + sebi_charge
        gst = gst_base * cfg["gst_pct"]

        # ------------------------------------------------------------------
        # Stamp duty (on buy side only)
        # ------------------------------------------------------------------
        if tx == TransactionType.BUY:
            stamp_rate = (
                cfg["intraday_stamp_duty_pct"] if intraday
                else cfg["delivery_stamp_duty_pct"]
            )
            stamp_duty = trade_value * stamp_rate
        else:
            stamp_duty = 0.0

        # ------------------------------------------------------------------
        # DP charges (delivery SELL only — per debit from demat account)
        # ------------------------------------------------------------------
        if not intraday and tx == TransactionType.SELL:
            dp_charges = cfg["dp_charge_rs"]
        else:
            dp_charges = 0.0

        # ------------------------------------------------------------------
        # Total
        # ------------------------------------------------------------------
        total = brokerage + stt + exchange_charge + sebi_charge + gst + stamp_duty + dp_charges
        total_pct = total / trade_value if trade_value > 0 else 0.0

        return CostBreakdown(
            brokerage=round(brokerage, 4),
            stt=round(stt, 4),
            exchange_charge=round(exchange_charge, 4),
            sebi_charge=round(sebi_charge, 6),
            gst=round(gst, 4),
            stamp_duty=round(stamp_duty, 4),
            dp_charges=round(dp_charges, 4),
            total=round(total, 4),
            total_pct=round(total_pct, 8),
        )

    def round_trip_cost(
        self,
        qty: int | float,
        buy_price: float,
        sell_price: float,
        product: str = "MIS",
        exchange: str = "NSE",
    ) -> CostBreakdown:
        """
        Compute the combined cost for a complete round-trip (buy + sell).

        The two CostBreakdown objects are summed field-by-field.
        total_pct is expressed relative to the BUY trade value.
        """
        buy_cost  = self.calculate_costs("BUY",  qty, buy_price,  exchange, product)
        sell_cost = self.calculate_costs("SELL", qty, sell_price, exchange, product)
        trade_value = qty * buy_price

        total = buy_cost.total + sell_cost.total
        return CostBreakdown(
            brokerage=       buy_cost.brokerage       + sell_cost.brokerage,
            stt=             buy_cost.stt             + sell_cost.stt,
            exchange_charge= buy_cost.exchange_charge + sell_cost.exchange_charge,
            sebi_charge=     buy_cost.sebi_charge     + sell_cost.sebi_charge,
            gst=             buy_cost.gst             + sell_cost.gst,
            stamp_duty=      buy_cost.stamp_duty      + sell_cost.stamp_duty,
            dp_charges=      buy_cost.dp_charges      + sell_cost.dp_charges,
            total=           round(total, 4),
            total_pct=       round(total / trade_value, 8) if trade_value > 0 else 0.0,
        )

    def breakeven_return(
        self,
        qty: int | float,
        price: float,
        product: str = "MIS",
    ) -> float:
        """
        Minimum gross return needed to cover round-trip costs.

        Returns fraction (e.g. 0.003 = 30 bps).
        """
        rt = self.round_trip_cost(qty, price, price, product=product)
        return rt.total_pct
