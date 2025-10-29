"""
Circuit Breaker implementation.

This module provides a simple but functional CircuitBreaker class that can be used
to wrap calls to external services or any function that may fail.  The breaker
supports the three classic states:

* CLOSED – normal operation.  Calls are forwarded to the wrapped function.
* OPEN – the breaker has detected a threshold of failures and will short‑circuit
  calls, raising :class:`CircuitOpenError`.
* HALF_OPEN – after a timeout the breaker allows a limited number of trial
  calls.  If they succeed the breaker closes again, otherwise it re‑opens.

The implementation keeps basic metrics that can be inspected for monitoring
purposes.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Callable, Any, Dict, Tuple

__all__ = ["CircuitBreaker", "CircuitOpenError"]


class CircuitOpenError(RuntimeError):
    """Raised when a call is attempted while the circuit is open."""


class CircuitBreaker:
    """A simple, thread‑unsafe circuit breaker.

    Parameters
    ----------
    failure_threshold:
        Number of consecutive failures required to open the circuit.
    recovery_timeout:
        Seconds to wait before transitioning from OPEN to HALF_OPEN.
    half_open_successes:
        Number of consecutive successes required in HALF_OPEN to close the
        circuit again.
    name:
        Optional name used in metrics.
    """

    # State constants
    _CLOSED = "closed"
    _OPEN = "open"
    _HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_successes: int = 3,
        name: str | None = None,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_successes = half_open_successes
        self.name = name or "circuit_breaker"

        self._state = self._CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_successes = 0
        self._opened_since: float | None = None

        # Metrics dictionary
        self.metrics: Dict[str, Any] = {
            "state": self._state,
            "failure_count": 0,
            "success_count": 0,
            "half_open_successes": 0,
            "opened_since": None,
        }

    # ---------------------------------------------------------------------
    # State helpers
    # ---------------------------------------------------------------------
    def _transition_to(self, new_state: str) -> None:
        self._state = new_state
        self.metrics["state"] = new_state
        if new_state == self._OPEN:
            self._opened_since = time.time()
            self.metrics["opened_since"] = self._opened_since
        else:
            self._opened_since = None
            self.metrics["opened_since"] = None

    def _reset_counts(self) -> None:
        self._failure_count = 0
        self._success_count = 0
        self._half_open_successes = 0

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Call *func* with the circuit breaker logic.

        Raises
        ------
        CircuitOpenError
            If the circuit is currently open.
        """
        # Check state and possibly transition
        if self._state == self._OPEN:
            if time.time() - self._opened_since >= self.recovery_timeout:
                # Time to try half‑open
                self._transition_to(self._HALF_OPEN)
            else:
                raise CircuitOpenError("Circuit is open")

        try:
            result = func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - exercised via tests
            self._handle_failure()
            raise
        else:
            self._handle_success()
            return result

    def _handle_failure(self) -> None:
        if self._state == self._HALF_OPEN:
            # Any failure in half‑open re‑opens the circuit
            self._transition_to(self._OPEN)
            self._reset_counts()
        else:  # CLOSED
            self._failure_count += 1
            self.metrics["failure_count"] = self._failure_count
            if self._failure_count >= self.failure_threshold:
                self._transition_to(self._OPEN)
                self._reset_counts()

    def _handle_success(self) -> None:
        if self._state == self._HALF_OPEN:
            self._half_open_successes += 1
            self.metrics["half_open_successes"] = self._half_open_successes
            if self._half_open_successes >= self.half_open_successes:
                self._transition_to(self._CLOSED)
                self._reset_counts()
        else:  # CLOSED
            self._success_count += 1
            self.metrics["success_count"] = self._success_count

    # ---------------------------------------------------------------------
    # Decorator support
    # ---------------------------------------------------------------------
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)

        return wrapper

    # ---------------------------------------------------------------------
    # Convenience methods
    # ---------------------------------------------------------------------
    def is_open(self) -> bool:
        return self._state == self._OPEN

    def is_closed(self) -> bool:
        return self._state == self._CLOSED

    def is_half_open(self) -> bool:
        return self._state == self._HALF_OPEN

    def get_metrics(self) -> Dict[str, Any]:
        """Return a copy of the current metrics dictionary."""
        return dict(self.metrics)

# -------------------------------------------------------------------------
# Example usage (uncomment to test manually)
# -------------------------------------------------------------------------
# if __name__ == "__main__":
#     import random
#
#     @CircuitBreaker(failure_threshold=3, recovery_timeout=5, half_open_successes=2)
#     def flaky():
#         if random.random() < 0.7:
#             raise ValueError("boom")
#         return "ok"
#
#     for i in range(20):
#         try:
#             print(i, flaky())
#         except Exception as e:
#             print(i, e)
#         time.sleep(0.5)

"""
End of circuit_breaker.py
"""
