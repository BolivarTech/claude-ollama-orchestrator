# tests/test_circuit_breaker.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Per-model circuit breaker: opens after K failures, isolates models, resets on success."""

import threading

from circuit_breaker import CircuitBreaker


def test_opens_after_threshold_failures_for_that_model_only():
    cb = CircuitBreaker(threshold=3, cooldown=10.0)
    for _ in range(3):
        cb.record_failure("minimax-m3:cloud", now=0.0)
    assert cb._is_open("minimax-m3:cloud", now=1.0) is True
    assert cb._is_open("kimi-k2.7-code:cloud", now=1.0) is False  # other model unaffected


def test_reopens_closed_after_cooldown():
    cb = CircuitBreaker(threshold=1, cooldown=10.0)
    cb.record_failure("m", now=0.0)
    assert cb._is_open("m", now=5.0) is True
    assert cb._is_open("m", now=11.0) is False  # cooldown elapsed → half-open/closed


def test_success_resets_failure_count():
    cb = CircuitBreaker(threshold=2, cooldown=10.0)
    cb.record_failure("m", now=0.0)
    cb.record_success("m")
    cb.record_failure("m", now=0.0)
    assert cb._is_open("m", now=1.0) is False  # only 1 failure since reset


def test_half_open_allows_exactly_one_probe_after_cooldown():
    cb = CircuitBreaker(threshold=1, cooldown=10.0)
    cb.record_failure("m", now=0.0)
    assert cb._is_open("m", now=11.0) is False  # cooldown elapsed -> probe #1 let through
    assert cb._is_open("m", now=11.0) is True  # a 2nd concurrent caller sees it still open
    # (the one probe hasn't resolved yet)


def test_half_open_probe_success_closes_the_circuit():
    cb = CircuitBreaker(threshold=1, cooldown=10.0)
    cb.record_failure("m", now=0.0)
    assert cb._is_open("m", now=11.0) is False  # the probe delegation
    cb.record_success("m")  # probe succeeded
    assert cb._is_open("m", now=11.5) is False  # closed - back to normal operation
    assert cb._is_open("m", now=999.0) is False  # stays closed, no lingering half-open state


def test_half_open_probe_failure_reopens_for_a_fresh_cooldown():
    cb = CircuitBreaker(threshold=1, cooldown=10.0)
    cb.record_failure("m", now=0.0)
    assert cb._is_open("m", now=11.0) is False  # the probe delegation
    cb.record_failure("m", now=11.0)  # probe failed
    assert cb._is_open("m", now=12.0) is True  # re-opened immediately
    assert cb._is_open("m", now=20.9) is True  # still within the fresh cooldown (11+10=21)
    assert cb._is_open("m", now=21.5) is False  # fresh cooldown elapsed -> half-open again


def test_release_probe_after_cancellation_allows_a_later_probe():
    # WARNING fix (#3): if the in-flight probe delegation is cancelled/interrupted
    # (never resolves via record_success/record_failure), the slot must not stay
    # reserved forever — release_probe returns the breaker to a fresh-cooldown OPEN
    # state so a LATER probe is still admitted once that cooldown elapses.
    cb = CircuitBreaker(threshold=1, cooldown=10.0)
    cb.record_failure("m", now=0.0)
    assert cb._is_open("m", now=11.0) is False  # probe #1 let through, slot reserved
    assert cb._is_open("m", now=11.0) is True  # confirms the slot is indeed reserved

    cb.release_probe("m", now=11.2)  # probe #1 was cancelled, never resolved
    assert cb._is_open("m", now=11.2) is True  # fresh cooldown just started -> still open
    assert cb._is_open("m", now=21.3) is False  # fresh cooldown (11.2+10) elapsed -> a later
    # probe is admitted (not stuck forever)

    # release_probe is a no-op for a model that was never mid-probe (e.g. called
    # defensively/unconditionally by run_batch for every job, not just probes).
    cb2 = CircuitBreaker(threshold=1, cooldown=10.0)
    cb2.release_probe("never-failed", now=5.0)
    assert cb2._is_open("never-failed", now=5.0) is False


def test_half_open_probe_slot_race_is_guarded_under_real_thread_concurrency():
    # WARNING fix (#2): two THREADS racing right after cooldown must not both win
    # the probe slot. `threading.Barrier` maximizes the chance of a true race by
    # releasing both threads at (as close as possible to) the same instant; the
    # check-and-reserve in `is_open` is guarded by an internal lock, not just
    # accidentally atomic because of "is_open has no await" reasoning.
    cb = CircuitBreaker(threshold=1, cooldown=10.0)
    cb.record_failure("m", now=0.0)

    barrier = threading.Barrier(2)
    admitted: list[bool] = []
    lock = threading.Lock()

    def probe() -> None:
        barrier.wait()
        result = cb._is_open("m", now=11.0)
        with lock:
            admitted.append(result)

    threads = [threading.Thread(target=probe) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one thread was admitted as the probe (is_open() returned False);
    # the other must see the circuit as still open (True).
    assert admitted.count(False) == 1
    assert admitted.count(True) == 1


def test_is_definitively_open_never_reserves_the_probe_slot():
    # CRITICAL fix (gate-closing round, probe-slot leak): a READ-ONLY peek must be
    # safe to call any number of times on a half-open-eligible model WITHOUT ever
    # reserving the probe — unlike is_open/try_enter, which reserve on the first
    # call after cooldown.
    cb = CircuitBreaker(threshold=1, cooldown=10.0)
    cb.record_failure("m", now=0.0)
    assert cb.is_definitively_open("m", now=5.0) is True  # still within cooldown -> open

    # Cooldown elapsed: half-open-eligible. Calling this MANY times must NEVER
    # reserve a probe -- a later try_enter must still be admitted as the probe.
    for _ in range(5):
        assert cb.is_definitively_open("m", now=11.0) is False
    assert cb.try_enter("m", now=11.0) == "probe"  # still available to reserve


def test_is_definitively_open_observes_but_does_not_duplicate_an_in_flight_probe():
    # A probe already in flight (reserved by try_enter) IS "definitively open" for
    # anyone else asking -- this is a read of existing state, not a new reservation.
    cb = CircuitBreaker(threshold=1, cooldown=10.0)
    cb.record_failure("m", now=0.0)
    assert cb.try_enter("m", now=11.0) == "probe"  # the one real reservation
    assert cb.is_definitively_open("m", now=11.0) is True  # observes it, doesn't touch it
    assert cb.is_definitively_open("m", now=11.0) is True  # still just observing
    cb.record_success("m")  # resolve the probe
    assert cb.is_definitively_open("m", now=11.5) is False  # closed again


def test_try_enter_is_the_only_method_that_reserves_the_probe():
    # CRITICAL fix: try_enter is the sole reservation point. "closed" when healthy,
    # "open" while cooling down (nothing reserved), "probe" exactly once after
    # cooldown elapses, then "open" for anyone else until the probe resolves.
    cb = CircuitBreaker(threshold=1, cooldown=10.0)
    assert cb.try_enter("healthy", now=0.0) == "closed"  # never failed -> healthy

    cb.record_failure("m", now=0.0)
    assert cb.try_enter("m", now=5.0) == "open"  # still cooling down, nothing reserved
    assert cb.try_enter("m", now=11.0) == "probe"  # cooldown elapsed -> reserves
    assert cb.try_enter("m", now=11.0) == "open"  # a 2nd caller sees it busy, not a 2nd probe

    cb.record_failure("m", now=11.0)  # the probe failed
    assert cb.try_enter("m", now=12.0) == "open"  # fresh cooldown, nothing reserved
    assert cb.try_enter("m", now=21.5) == "probe"  # fresh cooldown elapsed -> reserves again


def test_private_is_open_is_a_boolean_alias_of_try_enter():
    # DRY: the private `_is_open` test helper is implemented purely in terms of try_enter
    # (True iff try_enter(...) == "open"), never a separately maintained copy of the
    # check-and-reserve logic -- so it reserves the probe exactly like try_enter and stays
    # in lockstep with it.
    cb = CircuitBreaker(threshold=1, cooldown=10.0)
    assert cb._is_open("healthy", now=0.0) is False
    cb.record_failure("m", now=0.0)
    assert cb._is_open("m", now=5.0) is True
    assert cb._is_open("m", now=11.0) is False  # reserves the probe, same as try_enter
    assert cb._is_open("m", now=11.0) is True  # a 2nd caller sees it busy
