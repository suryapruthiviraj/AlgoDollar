from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ── 1. User ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, server_default=func.now()
    )

    # relationships
    settings: Mapped[Optional["UserSettings"]] = relationship(
        "UserSettings", back_populates="user", uselist=False, lazy="select"
    )
    capital_allocations: Mapped[list["CapitalAllocation"]] = relationship(
        "CapitalAllocation", back_populates="user", lazy="select"
    )
    positions: Mapped[list["Position"]] = relationship(
        "Position", back_populates="user", lazy="select"
    )
    orders: Mapped[list["Order"]] = relationship(
        "Order", back_populates="user", lazy="select"
    )
    trades: Mapped[list["Trade"]] = relationship(
        "Trade", back_populates="user", lazy="select"
    )
    signals: Mapped[list["Signal"]] = relationship(
        "Signal", back_populates="user", lazy="select"
    )
    risk_events: Mapped[list["RiskEvent"]] = relationship(
        "RiskEvent", back_populates="user", lazy="select"
    )
    audit_logs: Mapped[list["AuditLog"]] = relationship(
        "AuditLog", back_populates="user", lazy="select"
    )
    notifications: Mapped[list["Notification"]] = relationship(
        "Notification", back_populates="user", lazy="select"
    )


# ── 2. UserSettings ────────────────────────────────────────────────────────────

class UserSettings(Base):
    __tablename__ = "user_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False, index=True
    )
    monthly_capital: Mapped[float] = mapped_column(Numeric(18, 2), default=0.0)
    risk_tolerance: Mapped[str] = mapped_column(
        String(10), default="medium", nullable=False
    )  # low / medium / high
    max_drawdown_pct: Mapped[float] = mapped_column(Float, default=0.15)
    intraday_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    swing_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    longterm_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    max_positions: Mapped[int] = mapped_column(Integer, default=20)
    max_sector_exposure_pct: Mapped[float] = mapped_column(Float, default=0.25)
    max_single_stock_pct: Mapped[float] = mapped_column(Float, default=0.10)
    cash_reserve_pct: Mapped[float] = mapped_column(Float, default=0.05)
    auto_execution_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    paper_trading_mode: Mapped[bool] = mapped_column(Boolean, default=True)
    kill_switch_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="settings")


# ── 3. CapitalAllocation ───────────────────────────────────────────────────────

class CapitalAllocation(Base):
    __tablename__ = "capital_allocations"
    __table_args__ = (
        UniqueConstraint("user_id", "month_year", name="uq_capital_allocation_user_month"),
        Index("ix_capital_allocations_user_month", "user_id", "month_year"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    month_year: Mapped[str] = mapped_column(String(7), nullable=False)  # YYYY-MM
    contribution_amount: Mapped[float] = mapped_column(Numeric(18, 2), default=0.0)
    longterm_amount: Mapped[float] = mapped_column(Numeric(18, 2), default=0.0)
    swing_amount: Mapped[float] = mapped_column(Numeric(18, 2), default=0.0)
    intraday_amount: Mapped[float] = mapped_column(Numeric(18, 2), default=0.0)
    cash_amount: Mapped[float] = mapped_column(Numeric(18, 2), default=0.0)
    longterm_risk_pct: Mapped[float] = mapped_column(Float, default=0.0)
    swing_risk_pct: Mapped[float] = mapped_column(Float, default=0.0)
    intraday_risk_pct: Mapped[float] = mapped_column(Float, default=0.0)
    regime: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="capital_allocations")


# ── 4. Position ────────────────────────────────────────────────────────────────

class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (
        Index("ix_positions_user_symbol", "user_id", "symbol"),
        Index("ix_positions_user_open", "user_id", "is_open"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(30), nullable=False)
    exchange: Mapped[str] = mapped_column(String(10), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    average_price: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    current_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 4), nullable=True)
    strategy: Mapped[str] = mapped_column(String(20), nullable=False)  # longterm/swing/intraday
    entry_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    stop_loss: Mapped[Optional[float]] = mapped_column(Numeric(18, 4), nullable=True)
    target_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 4), nullable=True)
    signal_strength: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sector: Mapped[Optional[str]] = mapped_column(String(60), nullable=True)
    is_open: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="positions")


# ── 5. Order ───────────────────────────────────────────────────────────────────

class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (
        Index("ix_orders_user_status", "user_id", "status"),
        Index("ix_orders_user_created", "user_id", "created_at"),
        # Enforced by the DATABASE, not by a check-then-insert in Python,
        # which races. This is the idempotency guarantee.
        UniqueConstraint("user_id", "client_order_id", name="uq_orders_user_client_order_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    order_id_broker: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    # The idempotency key. Generated by us BEFORE the broker is called, so an
    # ambiguous submission (timeout, connection reset) can always be resolved
    # by asking the broker for this tag instead of blindly retrying and
    # risking a duplicate order. UNIQUE per user: the database, not
    # application logic, is what makes a duplicate submission impossible —
    # two racing workers both inserting the same key means one gets an
    # IntegrityError rather than two orders reaching the exchange.
    client_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False)
    exchange: Mapped[str] = mapped_column(String(10), nullable=False)
    transaction_type: Mapped[str] = mapped_column(String(4), nullable=False)  # BUY / SELL
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    # Partial fills are the normal case, not an edge case: an order is not
    # binary. filled_quantity < quantity with status PARTIAL is a legitimate
    # resting state that survives a restart.
    filled_quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    average_fill_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 4), nullable=True)
    product: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)  # CNC / MIS
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    price: Mapped[Optional[float]] = mapped_column(Numeric(18, 4), nullable=True)
    order_type: Mapped[str] = mapped_column(String(10), nullable=False)  # MARKET/LIMIT/SL/SL-M
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING")
    strategy: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    signal_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("signals.id", ondelete="SET NULL"), nullable=True
    )
    slippage_actual: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="orders")
    trades: Mapped[list["Trade"]] = relationship("Trade", back_populates="order", lazy="select")
    transitions: Mapped[list["OrderStateTransition"]] = relationship(
        "OrderStateTransition", back_populates="order", lazy="select",
    )


# ── 6. Trade ───────────────────────────────────────────────────────────────────

class Trade(Base):
    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trades_user_created", "user_id", "created_at"),
        Index("ix_trades_user_strategy", "user_id", "strategy"),
        UniqueConstraint("user_id", "trade_id_broker", name="uq_trades_user_trade_id_broker"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    order_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("orders.id", ondelete="SET NULL"), nullable=True, index=True
    )
    # The broker's own identifier for THIS FILL. One row per fill, never
    # pre-aggregated — reconciliation compares fill-by-fill, and a partially
    # filled order that is then filled again must produce two rows.
    # Unique per user so replaying a broker trade book cannot double-count.
    trade_id_broker: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False)
    exchange: Mapped[str] = mapped_column(String(10), nullable=False)
    transaction_type: Mapped[str] = mapped_column(String(4), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    value: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    brokerage: Mapped[float] = mapped_column(Numeric(18, 4), default=0.0)
    stt: Mapped[float] = mapped_column(Numeric(18, 4), default=0.0)
    exchange_charges: Mapped[float] = mapped_column(Numeric(18, 4), default=0.0)
    gst: Mapped[float] = mapped_column(Numeric(18, 4), default=0.0)
    stamp_duty: Mapped[float] = mapped_column(Numeric(18, 4), default=0.0)
    sebi_charges: Mapped[float] = mapped_column(Numeric(18, 4), default=0.0)
    total_costs: Mapped[float] = mapped_column(Numeric(18, 4), default=0.0)
    net_value: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    strategy: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    # Realised P&L is booked ONLY on the closing side, against the weighted
    # average cost of the position being reduced. A BUY that opens or adds to
    # a position realises nothing and stores NULL — which is not the same as
    # 0.0 and must stay distinguishable when summing performance.
    realized_pnl: Mapped[Optional[float]] = mapped_column(Numeric(18, 4), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="trades")
    order: Mapped[Optional["Order"]] = relationship("Order", back_populates="trades")


# ── 7. Signal ──────────────────────────────────────────────────────────────────

class Signal(Base):
    __tablename__ = "signals"
    __table_args__ = (
        Index("ix_signals_user_strategy", "user_id", "strategy"),
        Index("ix_signals_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(30), nullable=False)
    strategy: Mapped[str] = mapped_column(String(20), nullable=False)
    direction: Mapped[str] = mapped_column(String(5), nullable=False)  # LONG / SHORT / FLAT
    score: Mapped[float] = mapped_column(Float, nullable=False)
    expected_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expected_volatility: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_name: Mapped[str] = mapped_column(String(60), nullable=False)
    model_version: Mapped[str] = mapped_column(String(30), nullable=False)
    features_snapshot: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    acted_upon: Mapped[bool] = mapped_column(Boolean, default=False)

    user: Mapped["User"] = relationship("User", back_populates="signals")


# ── 8. StrategyPerformance ─────────────────────────────────────────────────────

class StrategyPerformance(Base):
    __tablename__ = "strategy_performance"
    __table_args__ = (
        UniqueConstraint("strategy", "date", name="uq_strategy_performance_strategy_date"),
        Index("ix_strategy_performance_strategy_date", "strategy", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy: Mapped[str] = mapped_column(String(20), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    gross_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    net_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sharpe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sortino: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    win_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    num_trades: Mapped[int] = mapped_column(Integer, default=0)
    total_costs: Mapped[float] = mapped_column(Numeric(18, 4), default=0.0)
    status: Mapped[str] = mapped_column(
        String(20), default="healthy"
    )  # healthy / reduced / paused / disabled
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )


# ── 9. ModelVersion ────────────────────────────────────────────────────────────

class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        Index("ix_model_versions_name_active", "model_name", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(30), nullable=False)
    strategy: Mapped[str] = mapped_column(String(20), nullable=False)
    training_start: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    training_end: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    validation_sharpe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    oos_sharpe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    oos_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    parameters: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    git_commit: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )


# ── 10. RiskEvent ─────────────────────────────────────────────────────────────

class RiskEvent(Base):
    __tablename__ = "risk_events"
    __table_args__ = (
        Index("ix_risk_events_user_created", "user_id", "created_at"),
        Index("ix_risk_events_severity", "severity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(String(60), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(10), nullable=False)  # info / warning / critical
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    action_taken: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="risk_events")


# ── 11. AuditLog ──────────────────────────────────────────────────────────────

class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_user_created", "user_id", "created_at"),
        Index("ix_audit_logs_entity", "entity_type", "entity_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    action: Mapped[str] = mapped_column(String(60), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(60), nullable=False)
    entity_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    before_state: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    after_state: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="audit_logs")


# ── 12. Notification ──────────────────────────────────────────────────────────

class Notification(Base):
    __tablename__ = "notifications"
    __table_args__ = (
        Index("ix_notifications_user_read", "user_id", "is_read"),
        Index("ix_notifications_user_created", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    type: Mapped[str] = mapped_column(String(40), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="notifications")


# ── 13. OrderStateTransition ──────────────────────────────────────────────────

class OrderStateTransition(Base):
    """
    One row per state change of an order — an append-only history.

    The current `Order.status` alone cannot answer "how did this order reach
    REJECTED?", and after a crash it cannot distinguish "never submitted" from
    "submitted, outcome unknown". That distinction is the whole reason
    UNKNOWN is a real state: an order whose submission was ambiguous must
    never be retried blindly, and the evidence for that has to survive a
    restart. This table is that evidence.

    Rows are never updated or deleted.
    """

    __tablename__ = "order_state_transitions"
    __table_args__ = (
        Index("ix_ost_order_created", "order_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # NULL from_state means this is the order's first recorded state.
    from_state: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    to_state: Mapped[str] = mapped_column(String(30), nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # What caused the transition: "submission", "fill", "reconciliation",
    # "recovery", "cancel". Needed to tell an observed broker state from one
    # we asserted ourselves.
    source: Mapped[str] = mapped_column(String(30), nullable=False, default="submission")
    filled_quantity: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now(), index=True
    )

    order: Mapped["Order"] = relationship("Order", back_populates="transitions")


# ── 14. AccountCash ───────────────────────────────────────────────────────────

class AccountCash(Base):
    """
    Our record of cash, per user and trading mode.

    Split by `trading_mode` deliberately: paper and live balances must never
    be summed or mistaken for one another. A paper run must not be able to
    report a live balance, and reconciliation compares like with like.
    """

    __tablename__ = "account_cash"
    __table_args__ = (
        UniqueConstraint("user_id", "trading_mode", name="uq_account_cash_user_mode"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    trading_mode: Mapped[str] = mapped_column(String(10), nullable=False)  # paper / live
    cash: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    # Cash committed to working orders. Available = cash - reserved, and it is
    # `available` that funds a new order — otherwise two orders can each be
    # approved against the same rupee.
    reserved: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=0.0)
    total_costs: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, server_default=func.now()
    )


# ── 15. ReconciliationRun ─────────────────────────────────────────────────────

class ReconciliationRun(Base):
    """
    The outcome of one reconciliation pass, persisted.

    `state` is the authority on whether trading is permitted, and it is stored
    rather than held in memory so the reason a process refused to trade is
    still answerable after it exits. UNAVAILABLE (broker unreachable) is
    recorded as its own state and is NEVER collapsed into OK — that collapse
    is exactly the fail-open bug this whole subsystem exists to prevent.
    """

    __tablename__ = "reconciliation_runs"
    __table_args__ = (
        Index("ix_recon_user_created", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    trading_mode: Mapped[str] = mapped_column(String(10), nullable=False)
    # RECONCILIATION_OK / MISMATCH / UNAVAILABLE / ERROR
    state: Mapped[str] = mapped_column(String(30), nullable=False)
    trading_permitted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    discrepancy_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # What could not be read at all, as opposed to what disagreed. An empty
    # local list matching an empty broker list is only "OK" if both were
    # actually READ; this column is what keeps those cases apart.
    unavailable: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    discrepancies: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now(), index=True
    )
