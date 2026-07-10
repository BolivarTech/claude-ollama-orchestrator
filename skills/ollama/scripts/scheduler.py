# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Bounded-concurrency scheduler: semaphore + hard queue cap, per-delegation rejection (R21/R21b)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from errors import DelegationError, OllamaConfigError


class Scheduler:
    """Run async jobs bounded by a semaphore, with a hard per-delegation queue cap.

    Overflow beyond ``max_parallel + max_queued`` is rejected **per delegation**
    (each excess job's result is a :class:`DelegationError`), never as a
    whole-batch raise; a single job's failure never cancels its siblings.

    The ceiling is **static**: it is computed once per :meth:`run_all` call as
    ``max_parallel + max_queued`` against the length of the ``thunks`` list
    handed to it up front. There is no API to enqueue additional thunks into an
    in-flight :meth:`run_all` call — a new batch is simply a new call with its
    own ceiling, not a dynamically growing queue.

    Attributes:
        peak: Maximum observed number of concurrently-running jobs.

    Concurrency-model note ([doc, minor] `peak` is per-instance state): `peak`
    is an ordinary instance attribute, accumulated across every `run_all` call
    made on `self` — it is not reset between calls. `run_batch` (MS5) always
    constructs a FRESH `Scheduler(...)` per call, so under that normal usage
    there is exactly one `run_all` invocation per instance and no possibility
    of two batches' concurrency accounting conflating into the same `peak`
    value. Calling `run_all` twice, overlapping, on ONE SHARED `Scheduler`
    instance is NOT a supported usage: each call's `active` counter is an
    independent local, but both would update the same shared `self.peak`, so
    the result would no longer represent either call's own peak concurrency in
    isolation. Callers that want independent peak accounting must use their
    own `Scheduler` instance per batch, exactly as `run_batch` does.
    """

    def __init__(self, max_parallel: int, max_queued: int) -> None:
        if max_parallel < 1:
            raise OllamaConfigError("max_parallel_agents must be >= 1")
        if max_queued < 0:
            raise OllamaConfigError("max_queued_agents must be >= 0")
        self._max_parallel = max_parallel
        self._max_queued = max_queued
        self.peak = 0

    async def run_all(self, thunks: list[Callable[[], Awaitable[Any]]]) -> list[Any]:
        """Run every thunk (≤ ``max_parallel`` at once); reject overflow per-delegation.

        Args:
            thunks: Zero-arg async callables to run, in submission order.

        Returns:
            One entry per thunk, in order: an accepted thunk's return value, or its
            raised exception (siblings unaffected), or a :class:`DelegationError`
            for each overflow thunk beyond ``max_parallel + max_queued``.
        """
        ceiling = self._max_parallel + self._max_queued
        sem = asyncio.Semaphore(self._max_parallel)
        active = 0
        lock = asyncio.Lock()

        async def _run(thunk: Callable[[], Awaitable[Any]]) -> Any:
            nonlocal active
            async with sem:
                async with lock:
                    active += 1
                    self.peak = max(self.peak, active)
                try:
                    return await thunk()
                finally:
                    async with lock:
                        active -= 1

        async def _reject(index: int) -> DelegationError:
            return DelegationError(
                f"queue full: delegation #{index} rejected — ceiling {ceiling} "
                f"(max_parallel={self._max_parallel} + max_queued={self._max_queued}); "
                "retry this delegation later, reduce the batch, or raise the cap"
            )

        coros = [_run(thunk) if i < ceiling else _reject(i) for i, thunk in enumerate(thunks)]
        # return_exceptions=True → an accepted job's exception becomes its result
        # entry (siblings keep running); rejected jobs already return a value.
        return list(await asyncio.gather(*coros, return_exceptions=True))
