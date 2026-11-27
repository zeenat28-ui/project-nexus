"""Circuit breaker for isolating repeated upstream backend failures."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class CircuitState(str, Enum):
    """Operational states of the backend circuit breaker."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass(frozen=True)
class CircuitSnapshot:
    """Point-in-time view of circuit breaker state."""

    state: CircuitState
    consecutive_failures: int
    retry_after_seconds: float


class CircuitBreaker:
    """Thread-safe circuit breaker with cooldown and single-probe half-open gating."""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 15.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be > 0")

        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_probe_in_flight = False

    def snapshot(self) -> CircuitSnapshot:
        """Inspect current state without mutating probe reservation."""
        with self._lock:
            now = self._clock()
            if self._state is CircuitState.OPEN and self._opened_at is not None:
                elapsed = now - self._opened_at
                remaining = max(0.0, self._cooldown_seconds - elapsed)
                if remaining == 0.0:
                    return CircuitSnapshot(
                        state=CircuitState.HALF_OPEN,
                        consecutive_failures=self._consecutive_failures,
                        retry_after_seconds=0.0,
                    )
                return CircuitSnapshot(
                    state=CircuitState.OPEN,
                    consecutive_failures=self._consecutive_failures,
                    retry_after_seconds=round(remaining, 3),
                )
            return CircuitSnapshot(
                state=self._state,
                consecutive_failures=self._consecutive_failures,
                retry_after_seconds=0.0,
            )

    def try_acquire(self) -> tuple[bool, bool]:
        """Check whether a request may proceed.

        Returns ``(allowed, is_half_open_probe)``.
        """
        with self._lock:
            if self._state is CircuitState.CLOSED:
                return True, False

            now = self._clock()
            if self._state is CircuitState.OPEN:
                if (
                    self._opened_at is not None
                    and (now - self._opened_at) >= self._cooldown_seconds
                ):
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_probe_in_flight = True
                    return True, True
                return False, False

            # HALF_OPEN: allow only one concurrent probe request
            if not self._half_open_probe_in_flight:
                self._half_open_probe_in_flight = True
                return True, True
            return False, False

    def record_success(self) -> CircuitState:
        """Record a successful backend call and close the circuit."""
        with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None
            self._half_open_probe_in_flight = False
            self._state = CircuitState.CLOSED
            return self._state

    def record_failure(self) -> CircuitState:
        """Record a backend failure and trip open when threshold is reached."""
        with self._lock:
            self._half_open_probe_in_flight = False
            self._consecutive_failures += 1
            if (
                self._state is CircuitState.HALF_OPEN
                or self._consecutive_failures >= self._failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
            return self._state

    def release_probe(self) -> None:
        """Release a half-open probe slot without changing failure counts (e.g., client cancel)."""
        with self._lock:
            self._half_open_probe_in_flight = False
