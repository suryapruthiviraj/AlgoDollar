from __future__ import annotations

from typing import Any, Optional


class AlgoDollarError(Exception):
    """Base exception for all AlgoDollar errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


class BrokerConnectionError(AlgoDollarError):
    """Raised when the broker API is unreachable or returns an unexpected error."""


class InsufficientCapitalError(AlgoDollarError):
    """Raised when available capital is below what is required for an operation."""


class RiskLimitExceededError(AlgoDollarError):
    """Raised when a proposed trade or allocation would breach a configured risk limit."""


class KillSwitchActiveError(AlgoDollarError):
    """Raised when an operation is blocked because the user's kill switch is engaged."""


class StaleDataError(AlgoDollarError):
    """Raised when market data is too old to be used safely."""


class OrderValidationError(AlgoDollarError):
    """Raised when an order fails pre-submission validation checks."""


class ReconciliationError(AlgoDollarError):
    """Raised when broker positions and internal positions do not reconcile."""
