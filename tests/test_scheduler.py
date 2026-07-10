# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Bounded scheduler: peak <= N, per-delegation overflow rejection, sibling isolation."""

import asyncio

from errors import DelegationError, OllamaConfigError
from scheduler import Scheduler


def test_peak_concurrency_never_exceeds_max_parallel():
    sched = Scheduler(max_parallel=2, max_queued=10)
    state = {"active": 0, "peak": 0}

    async def job():
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return "ok"

    results = asyncio.run(sched.run_all([job for _ in range(6)]))
    assert state["peak"] <= 2 and sched.peak <= 2
    assert results == ["ok"] * 6


def test_overflow_is_rejected_per_delegation_not_whole_batch():
    sched = Scheduler(max_parallel=2, max_queued=1)  # ceiling 3

    async def job():
        return "x"

    results = asyncio.run(sched.run_all([job] * 5))  # 5 > ceiling 3
    assert len(results) == 5
    assert results[:3] == ["x", "x", "x"]  # first 3 accepted, ran
    assert all(isinstance(r, DelegationError) for r in results[3:])  # 2 overflow rejected each


def test_one_failure_does_not_cancel_siblings():
    sched = Scheduler(max_parallel=3, max_queued=0)

    async def ok():
        await asyncio.sleep(0.01)
        return "ok"

    async def boom():
        raise RuntimeError("kaboom")

    results = asyncio.run(sched.run_all([ok, boom, ok]))
    assert results[0] == "ok" and results[2] == "ok"  # siblings completed
    assert isinstance(results[1], RuntimeError)  # failure reported per-delegation


def test_max_parallel_must_be_positive_and_queue_nonnegative():
    for bad in (dict(max_parallel=0, max_queued=1), dict(max_parallel=2, max_queued=-1)):
        try:
            Scheduler(**bad)
            raise AssertionError("expected OllamaConfigError")
        except OllamaConfigError:
            pass
