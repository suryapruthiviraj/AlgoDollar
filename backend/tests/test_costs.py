"""
Tests for ZerodhaCostModel.

Zerodha charges:
- Intraday brokerage: 0.03% or ₹20 per order, whichever is LOWER
- Delivery brokerage: ₹0 (zero)
- STT (Securities Transaction Tax):
    - Intraday: 0.025% on sell side only
    - Delivery: 0.1% on both buy and sell
- Exchange transaction charges: 0.00345% (NSE) approximately
- SEBI turnover fee: 0.0001%
- GST: 18% on (brokerage + exchange charges + SEBI fee)
- Stamp duty: 0.015% on buy side only (state-dependent; using standard)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pytest


# ---------------------------------------------------------------------------
# Inline cost model stub — mirrors expected ZerodhaCostModel interface
# ---------------------------------------------------------------------------

ProductType = Literal["MIS", "CNC", "NRML"]
OrderSide = Literal["BUY", "SELL"]


@dataclass
class CostConfig:
    """Configurable rates — can be updated to reflect regulatory changes."""
    intraday_brokerage_pct: float = 0.0003      # 0.03%
    intraday_brokerage_cap: float = 20.0         # ₹20 per order
    delivery_brokerage_pct: float = 0.0          # ₹0

    stt_intraday_pct: float = 0.00025            # 0.025% sell side only
    stt_delivery_pct: float = 0.001              # 0.1% both sides

    exchange_txn_charge_pct: float = 0.0000345   # ~0.00345% NSE (rough)
    sebi_fee_pct: float = 0.000001               # 0.0001%
    gst_pct: float = 0.18                        # 18% on brokerage+exchange+SEBI
    stamp_duty_pct: float = 0.00015              # 0.015% buy side only


@dataclass
class CostBreakdown:
    brokerage: float = 0.0
    stt: float = 0.0
    exchange_txn_charge: float = 0.0
    sebi_fee: float = 0.0
    gst: float = 0.0
    stamp_duty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange_txn_charge
            + self.sebi_fee
            + self.gst
            + self.stamp_duty
        )


class ZerodhaCostModel:
    """
    Zerodha equity cost model (NSE segment).

    All figures are approximate and should be validated against current
    Zerodha brokerage calculator before live deployment.
    """

    def __init__(self, config: CostConfig | None = None):
        self.config = config or CostConfig()

    def calculate(
        self,
        price: float,
        quantity: int,
        side: OrderSide,
        product: ProductType,
    ) -> CostBreakdown:
        """
        Calculate full cost breakdown for a single order leg.

        Args:
            price: Execution price per share (INR)
            quantity: Number of shares
            side: "BUY" or "SELL"
            product: "MIS" (intraday) or "CNC" (delivery)

        Returns:
            CostBreakdown with individual components and total.
        """
        c = self.config
        turnover = price * quantity

        # -- Brokerage --
        if product == "CNC":
            brokerage = 0.0
        else:  # MIS / NRML intraday
            brokerage = min(turnover * c.intraday_brokerage_pct, c.intraday_brokerage_cap)

        # -- STT --
        if product == "CNC":
            stt = turnover * c.stt_delivery_pct  # both sides
        else:
            # Intraday: STT only on sell side
            stt = turnover * c.stt_intraday_pct if side == "SELL" else 0.0

        # -- Exchange transaction charge --
        exchange_txn = turnover * c.exchange_txn_charge_pct

        # -- SEBI turnover fee --
        sebi_fee = turnover * c.sebi_fee_pct

        # -- GST (on brokerage + exchange + SEBI) --
        gst = (brokerage + exchange_txn + sebi_fee) * c.gst_pct

        # -- Stamp duty (buy side only) --
        stamp_duty = turnover * c.stamp_duty_pct if side == "BUY" else 0.0

        return CostBreakdown(
            brokerage=round(brokerage, 4),
            stt=round(stt, 4),
            exchange_txn_charge=round(exchange_txn, 4),
            sebi_fee=round(sebi_fee, 4),
            gst=round(gst, 4),
            stamp_duty=round(stamp_duty, 4),
        )

    def round_trip_cost(
        self,
        price: float,
        quantity: int,
        product: ProductType,
    ) -> float:
        """
        Total cost for a complete round trip (buy + sell at same price).

        This is an approximation for position sizing purposes.
        """
        buy_cost = self.calculate(price, quantity, "BUY", product)
        sell_cost = self.calculate(price, quantity, "SELL", product)
        return buy_cost.total + sell_cost.total


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIntradayBrokerage:
    def test_intraday_brokerage_capped_at_20(self):
        """
        For a large intraday order (>= ₹66,667 turnover), brokerage = ₹20, not 0.03%.

        0.03% of ₹66,667 = ₹20.00 — cap kicks in at exactly this point.
        For anything larger, cap applies.
        """
        model = ZerodhaCostModel()
        # 100 shares at ₹3000 = ₹3,00,000 turnover
        # 0.03% of ₹3,00,000 = ₹90 — capped at ₹20
        cost = model.calculate(price=3000.0, quantity=100, side="BUY", product="MIS")

        assert cost.brokerage == pytest.approx(20.0, abs=0.001), (
            f"Expected brokerage=₹20.00 (cap), got ₹{cost.brokerage:.4f}"
        )

    def test_intraday_brokerage_below_cap(self):
        """Small intraday trade brokerage = 0.03%, well below cap."""
        model = ZerodhaCostModel()
        # 1 share at ₹100 = ₹100 turnover; 0.03% = ₹0.03
        cost = model.calculate(price=100.0, quantity=1, side="BUY", product="MIS")

        expected = 100.0 * 0.0003  # = ₹0.03
        assert cost.brokerage == pytest.approx(expected, abs=0.0001)

    def test_intraday_brokerage_exactly_at_cap(self):
        """At the exact cap turnover (~₹66,667), brokerage ≈ ₹20."""
        model = ZerodhaCostModel()
        # Turnover = 20 / 0.0003 = 66,666.67
        cap_turnover = model.config.intraday_brokerage_cap / model.config.intraday_brokerage_pct
        price = cap_turnover  # 1 share at cap_turnover price
        cost = model.calculate(price=price, quantity=1, side="BUY", product="MIS")

        assert cost.brokerage == pytest.approx(model.config.intraday_brokerage_cap, abs=0.01)


class TestDeliveryBrokerage:
    def test_delivery_zero_brokerage(self):
        """CNC (delivery) product has zero brokerage — Zerodha's promise."""
        model = ZerodhaCostModel()
        # Large delivery order
        cost = model.calculate(price=2500.0, quantity=1000, side="BUY", product="CNC")

        assert cost.brokerage == 0.0, (
            f"Expected ₹0 brokerage for delivery, got ₹{cost.brokerage:.4f}"
        )

    def test_delivery_zero_brokerage_sell_side(self):
        """Delivery sell side also has zero brokerage."""
        model = ZerodhaCostModel()
        cost = model.calculate(price=2500.0, quantity=500, side="SELL", product="CNC")

        assert cost.brokerage == 0.0


class TestSTTIntraday:
    def test_stt_sell_side_intraday(self):
        """Intraday STT is charged only on the sell side."""
        model = ZerodhaCostModel()
        turnover = 2500.0 * 10  # ₹25,000

        buy_cost = model.calculate(price=2500.0, quantity=10, side="BUY", product="MIS")
        sell_cost = model.calculate(price=2500.0, quantity=10, side="SELL", product="MIS")

        assert buy_cost.stt == 0.0, "Intraday buy side should have zero STT"
        expected_stt = round(turnover * model.config.stt_intraday_pct, 4)
        assert sell_cost.stt == pytest.approx(expected_stt, abs=0.001), (
            f"Intraday sell STT expected ₹{expected_stt:.4f}, got ₹{sell_cost.stt:.4f}"
        )

    def test_intraday_stt_rate(self):
        """Verify intraday STT rate = 0.025% on sell."""
        model = ZerodhaCostModel()
        price, qty = 1000.0, 50
        turnover = price * qty  # ₹50,000
        cost = model.calculate(price=price, quantity=qty, side="SELL", product="MIS")

        expected = turnover * 0.00025  # 0.025%
        assert cost.stt == pytest.approx(expected, abs=0.001)


class TestSTTDelivery:
    def test_stt_both_sides_delivery(self):
        """Delivery STT is charged on BOTH buy and sell sides."""
        model = ZerodhaCostModel()
        price, qty = 1500.0, 20
        turnover = price * qty  # ₹30,000

        buy_cost = model.calculate(price=price, quantity=qty, side="BUY", product="CNC")
        sell_cost = model.calculate(price=price, quantity=qty, side="SELL", product="CNC")

        expected_stt = round(turnover * model.config.stt_delivery_pct, 4)

        assert buy_cost.stt == pytest.approx(expected_stt, abs=0.001), (
            f"Delivery buy STT expected ₹{expected_stt:.4f}, got ₹{buy_cost.stt:.4f}"
        )
        assert sell_cost.stt == pytest.approx(expected_stt, abs=0.001), (
            f"Delivery sell STT expected ₹{expected_stt:.4f}, got ₹{sell_cost.stt:.4f}"
        )

    def test_delivery_stt_rate(self):
        """Delivery STT rate = 0.1% on both sides."""
        model = ZerodhaCostModel()
        price, qty = 2000.0, 25
        turnover = price * qty  # ₹50,000
        buy_cost = model.calculate(price=price, quantity=qty, side="BUY", product="CNC")

        expected = turnover * 0.001  # 0.1%
        assert buy_cost.stt == pytest.approx(expected, abs=0.001)


class TestTotalCostsReasonable:
    def test_total_costs_reasonable_intraday(self):
        """
        For a ₹10,000 intraday trade (buy leg only), total costs < 0.5%.
        """
        model = ZerodhaCostModel()
        price, qty = 1000.0, 10
        turnover = price * qty  # ₹10,000

        buy_cost = model.calculate(price=price, quantity=qty, side="BUY", product="MIS")
        cost_pct = buy_cost.total / turnover

        assert cost_pct < 0.005, (
            f"Buy leg costs {cost_pct:.4%} exceed 0.5% of turnover ₹{turnover:.0f}"
        )

    def test_round_trip_intraday_under_one_pct(self):
        """Round-trip intraday costs should be < 1% of turnover."""
        model = ZerodhaCostModel()
        price, qty = 1000.0, 10
        turnover = price * qty

        rt_cost = model.round_trip_cost(price=price, quantity=qty, product="MIS")
        cost_pct = rt_cost / turnover

        assert cost_pct < 0.01, (
            f"Round-trip costs {cost_pct:.4%} exceed 1% of ₹{turnover:.0f} turnover"
        )

    def test_round_trip_delivery_under_one_pct(self):
        """Round-trip delivery costs should also be < 1% (mainly STT)."""
        model = ZerodhaCostModel()
        price, qty = 2500.0, 10
        turnover = price * qty

        rt_cost = model.round_trip_cost(price=price, quantity=qty, product="CNC")
        cost_pct = rt_cost / turnover

        assert cost_pct < 0.01, (
            f"Delivery round-trip costs {cost_pct:.4%} exceed 1% of ₹{turnover:.0f}"
        )

    def test_all_components_non_negative(self):
        """Every cost component must be >= 0 for any valid order."""
        model = ZerodhaCostModel()
        for product in ("MIS", "CNC"):
            for side in ("BUY", "SELL"):
                cost = model.calculate(price=500.0, quantity=20, side=side, product=product)
                assert cost.brokerage >= 0
                assert cost.stt >= 0
                assert cost.exchange_txn_charge >= 0
                assert cost.sebi_fee >= 0
                assert cost.gst >= 0
                assert cost.stamp_duty >= 0
                assert cost.total >= 0


class TestCostConfigUpdatable:
    def test_cost_config_updatable_recalculates_correctly(self):
        """
        Updating config rates should cause the model to recalculate with
        the new rates, not the old ones.
        """
        old_config = CostConfig(intraday_brokerage_pct=0.0003)
        new_config = CostConfig(intraday_brokerage_pct=0.0006)  # double the rate

        old_model = ZerodhaCostModel(config=old_config)
        new_model = ZerodhaCostModel(config=new_config)

        price, qty = 500.0, 10  # ₹5,000 turnover — below ₹20 cap at 0.03%
        turnover = price * qty

        old_cost = old_model.calculate(price=price, quantity=qty, side="BUY", product="MIS")
        new_cost = new_model.calculate(price=price, quantity=qty, side="BUY", product="MIS")

        # Old brokerage: 0.03% of ₹5000 = ₹1.50
        # New brokerage: 0.06% of ₹5000 = ₹3.00 (if below cap)
        expected_old = min(turnover * 0.0003, 20.0)
        expected_new = min(turnover * 0.0006, 20.0)

        assert old_cost.brokerage == pytest.approx(expected_old, abs=0.001)
        assert new_cost.brokerage == pytest.approx(expected_new, abs=0.001)
        assert new_cost.brokerage > old_cost.brokerage, (
            "Higher brokerage rate should produce higher brokerage charge"
        )

    def test_config_stt_rate_change(self):
        """Changing STT rate in config should reflect in calculation."""
        custom_config = CostConfig(stt_intraday_pct=0.0005)  # 0.05% instead of 0.025%
        model = ZerodhaCostModel(config=custom_config)

        price, qty = 1000.0, 20
        turnover = price * qty

        cost = model.calculate(price=price, quantity=qty, side="SELL", product="MIS")
        expected_stt = turnover * 0.0005

        assert cost.stt == pytest.approx(expected_stt, abs=0.001)

    def test_zero_gst_config(self):
        """Setting GST to 0 should produce zero GST in breakdown."""
        config = CostConfig(gst_pct=0.0)
        model = ZerodhaCostModel(config=config)

        cost = model.calculate(price=1000.0, quantity=10, side="BUY", product="MIS")
        assert cost.gst == 0.0

    def test_stamp_duty_only_on_buy(self):
        """Stamp duty must be zero on the sell side."""
        model = ZerodhaCostModel()

        buy_cost = model.calculate(price=1000.0, quantity=10, side="BUY", product="MIS")
        sell_cost = model.calculate(price=1000.0, quantity=10, side="SELL", product="MIS")

        assert sell_cost.stamp_duty == 0.0, "Stamp duty must be zero on sell"
        assert buy_cost.stamp_duty > 0.0, "Stamp duty must be positive on buy"
