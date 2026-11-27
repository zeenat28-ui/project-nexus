"""Tests for the backend circuit breaker state machine."""

from __future__ import annotations

from project_nexus.resilience import CircuitBreaker, CircuitState


def test_circuit_breaker_transitions_and_half_open_gating() -> None:
    now = [100.0]
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=10.0, clock=lambda: now[0])

    assert breaker.snapshot().state is CircuitState.CLOSED
    assert breaker.try_acquire() == (True, False)

    # First failure stays CLOSED
    assert breaker.record_failure() is CircuitState.CLOSED
    assert breaker.try_acquire() == (True, False)

    # Second failure trips OPEN
    assert breaker.record_failure() is CircuitState.OPEN
    assert breaker.snapshot().state is CircuitState.OPEN
    assert breaker.try_acquire() == (False, False)

    # Advance past cooldown -> HALF_OPEN allows exactly one probe
    now[0] += 10.5
    assert breaker.snapshot().state is CircuitState.HALF_OPEN
    assert breaker.try_acquire() == (True, True)
    assert breaker.try_acquire() == (False, False)  # second concurrent request blocked

    # Probe failure trips immediately back to OPEN
    assert breaker.record_failure() is CircuitState.OPEN
    assert breaker.try_acquire() == (False, False)

    # Cooldown elapses again -> probe cancelled releases slot without tripping
    now[0] += 11.0
    assert breaker.try_acquire() == (True, True)
    breaker.release_probe()
    assert breaker.try_acquire() == (True, True)

    # Probe succeeds -> CLOSED
    assert breaker.record_success() is CircuitState.CLOSED
    assert breaker.snapshot().consecutive_failures == 0
    assert breaker.try_acquire() == (True, False)
