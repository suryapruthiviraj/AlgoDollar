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


class AmbiguousOrderStateError(AlgoDollarError):
    """
    Raised when a state-changing broker call failed with an unknown outcome.

    The distinction this encodes is the one that matters most in execution: a
    request that failed is not the same as a request that was rejected. If a
    `place_order` call times out, the order may well have reached the exchange
    — the response was simply lost.

    Retrying here places a second real trade. Assuming failure leaves an
    untracked live position. Both are worse than stopping, so this exception
    exists to force the only safe response: reconcile against the broker and
    find out what actually happened before doing anything else.

    Never handle this by retrying.
    """
