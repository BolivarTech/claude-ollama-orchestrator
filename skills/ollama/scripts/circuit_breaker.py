# skills/ollama/scripts/circuit_breaker.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Per-model circuit breaker (in-memory, per-process, single-event-loop)."""

from __future__ import annotations

import threading


class CircuitBreaker:
    """Open a per-model circuit after *threshold* consecutive failures.

    Three states per model (R14b): **CLOSED** (healthy) -> **OPEN** (>= *threshold*
    consecutive failures; fails fast for *cooldown* seconds) -> **HALF-OPEN**
    (cooldown elapsed; exactly one probe delegation is let through) -> back to
    CLOSED (the probe succeeded) or OPEN again for a fresh cooldown (the probe
    failed, or was cancelled/interrupted — see :meth:`release_probe`). A 429
    (rate limit) is NOT a breaker failure — the caller (`run_batch`) handles it
    with the backend's own backoff and never calls :meth:`record_failure` for it.

    Assumes a **single event loop per process** — the per-process concurrency
    model this milestone targets (cross-process coordination is R7d/R21c,
    deferred to MS7).

    **Why `threading.Lock`, not `asyncio.Lock` (maintainability note — do not
    "fix" this into a bug):** `run_batch`'s per-delegation worker body
    (`_run_one_delegation`) runs on a REAL OS worker thread via
    `asyncio.to_thread` — dispatched onto the event loop's own DEFAULT
    executor, sized per loop by `_ensure_sized_default_executor` (and grown
    when a larger later batch needs more workers)
    (seventh round: a dedicated per-batch `ThreadPoolExecutor` dispatched via
    a bare `loop.run_in_executor(executor, ...)` was tried in an intervening
    round specifically to fix that sizing, but was reverted because it broke
    contextvars propagation into the worker thread — see the thread-pool-
    sizing entry in Task 5's Interfaces block for the full history; either
    design change is/was still a genuine OS thread) — not a coroutine
    cooperatively scheduled on the event loop. `CircuitBreaker` is also a
    shared, process-wide singleton (`_PROCESS_CIRCUIT_BREAKER` in
    `run_ollama.py`) that may be consulted from more than one such worker
    thread at once, and the `test_half_open_probe_slot_race_is_guarded_under_
    real_thread_concurrency` test below exercises exactly that with genuine
    `threading.Thread`/`threading.Barrier` concurrency, not just asyncio-task
    interleaving. `threading.Lock` is therefore the CORRECT primitive here —
    `asyncio.Lock` would be WRONG: an `asyncio.Lock` only provides mutual
    exclusion between coroutines cooperatively scheduled on the SAME event
    loop; it does nothing to serialize access from separate OS threads, and
    acquiring/releasing one from a thread that isn't running that event loop
    is unsafe/undefined (it isn't even guaranteed to be usable outside a
    running loop at all). `threading.Lock` provides correct mutual exclusion
    regardless of whether the caller is the event-loop thread or a worker
    thread spawned by the sized default executor — exactly the guarantee this
    class needs, and precisely why a future "simplification" to `asyncio.Lock`
    would silently reintroduce the half-open probe race this lock exists to
    close.
    """

    def __init__(self, threshold: int = 3, cooldown: float = 30.0) -> None:
        self._threshold = threshold
        self._cooldown = cooldown
        self._fails: dict[str, int] = {}
        self._open_until: dict[str, float] = {}
        self._half_open_probe: set[str] = set()
        # threading.Lock (NOT asyncio.Lock) is deliberate — see the class
        # docstring's "Why threading.Lock" note: this breaker is a shared
        # singleton that may be consulted from real worker threads of
        # whichever ThreadPoolExecutor `run_batch` is using, and asyncio.Lock
        # does not synchronize across OS threads.
        self._lock = threading.Lock()

    def record_failure(self, model: str, now: float) -> None:
        """Record a backend failure for *model*.

        If a half-open probe for *model* is in flight, this IS that probe's
        outcome: re-open for a fresh *cooldown* (the failure count is not
        double-counted towards a stale threshold). Otherwise, accumulate towards
        *threshold* as usual.
        """
        with self._lock:
            if model in self._half_open_probe:
                self._half_open_probe.discard(model)
                self._open_until[model] = now + self._cooldown
                return
            self._fails[model] = self._fails.get(model, 0) + 1
            if self._fails[model] >= self._threshold:
                self._open_until[model] = now + self._cooldown

    def record_success(self, model: str) -> None:
        """Reset *model*'s failure count and close its circuit (incl. a probe win)."""
        with self._lock:
            self._half_open_probe.discard(model)
            self._fails[model] = 0
            self._open_until.pop(model, None)

    def release_probe(self, model: str, now: float) -> None:
        """Release a half-open probe slot that never resolved (WARNING fix #3).

        Use when the probe delegation is cancelled/interrupted rather than
        succeeding or failing — e.g. Ctrl-C mid-probe, or a `BaseException`
        escaping `run_batch`'s fan-out. Without this, `model` would stay in
        `_half_open_probe` forever: `_is_open` would keep returning `True`
        indefinitely (a "probe in flight" that never resolves), so no future
        caller could ever be admitted as a new probe — the model would be
        permanently blocked.

        Conservative choice: an inconclusive probe is treated like a FAILED
        probe (a fresh *cooldown* starts from *now*), not an immediate
        readmission, since the model's health is genuinely unknown, not proven
        healthy. A no-op if *model* has no probe reservation in flight (safe to
        call unconditionally for every job, whether or not it was the probe).
        """
        with self._lock:
            if model in self._half_open_probe:
                self._half_open_probe.discard(model)
                self._open_until[model] = now + self._cooldown

    def is_definitively_open(self, model: str, now: float) -> bool:
        """Read-only peek: True iff *model* is unambiguously blocking new
        delegations RIGHT NOW — NEVER transitions state, NEVER reserves the
        half-open probe slot (CRITICAL fix, gate-closing round — closes the
        probe-slot-leak bug described below).

        Safe to call SPECULATIVELY on a delegation that may never actually
        execute — e.g. `run_ollama.py`'s pre-scheduling breaker filter
        (`_reject_if_circuit_open`), run for every job in a batch BEFORE any
        of them ever reaches the `Scheduler`. A job that filter lets through
        may still be overflow-rejected by the `Scheduler` before it ever runs
        (R21/R21b); if THIS method had reserved a probe for it (the way
        `_is_open`/`try_enter` do), that reservation would leak forever, since
        an overflow-rejected job never reaches `_execute_delegation` to
        release it. Because this method never reserves anything, calling it
        on a job that never executes is always harmless.

        Returns:
            True when a half-open probe is ALREADY in flight for *model*
            (observed, not reserved — some other delegation holds it), or
            when *model* is fully OPEN (the cooldown has not yet elapsed).
            False when CLOSED, or when the cooldown HAS elapsed (HALF-OPEN
            eligible) — in the latter case whether a probe is actually
            admitted is decided later, atomically, by :meth:`try_enter`, at
            the moment a delegation is about to run — never here.
        """
        with self._lock:
            if model in self._half_open_probe:
                return True
            until = self._open_until.get(model)
            return until is not None and now < until

    def try_enter(self, model: str, now: float) -> str:
        """Atomically decide whether a delegation for *model* may run RIGHT
        NOW — the ONLY method that ever reserves the half-open probe slot
        (CRITICAL fix, gate-closing round).

        Must be called EXCLUSIVELY from `_execute_delegation`, immediately
        before `call_worker()` actually runs — never speculatively, and never
        from a pre-scheduling filter that might still discard the delegation
        before it executes (use the read-only :meth:`is_definitively_open`
        there instead — see its docstring for why reserving speculatively is
        the exact bug this fix closes).

        Performs the OPEN -> HALF-OPEN transition: once *cooldown* has elapsed
        since the circuit opened, exactly the FIRST caller after that point is
        let through (this call returns ``"probe"`` and reserves the slot); any
        other call for the same model while that probe is unresolved still
        sees the circuit as open (``"open"``), so at most one probe is ever in
        flight. The check-and-reserve is guarded by `self._lock` (WARNING fix
        #2) so this holds under real concurrent access, not only
        single-event-loop interleaving. A caller that receives ``"probe"``
        MUST report the outcome via :meth:`record_success`,
        :meth:`record_failure`, or — if it never resolves —
        :meth:`release_probe`.

        Returns:
            ``"closed"``: the circuit is healthy — proceed, this call is NOT
            the probe. ``"open"``: fail fast — either still within the
            cooldown, or another probe is already in flight; NOTHING is
            reserved by this call. ``"probe"``: this call reserved the single
            half-open probe slot — the caller IS the probe.
        """
        with self._lock:
            if model in self._half_open_probe:
                return "open"  # a probe is already in flight -- don't admit a second one
            until = self._open_until.get(model)
            if until is None:
                return "closed"
            if now < until:
                return "open"  # still within the cooldown -- fully open
            self._half_open_probe.add(model)  # cooldown elapsed -> reserve the probe slot
            return "probe"

    def _is_open(self, model: str, now: float) -> bool:
        """PRIVATE, TEST-FACING boolean helper — True iff the circuit is OPEN, **and, as a
        SIDE EFFECT, reserves the half-open probe slot on the first call after cooldown**.

        The leading underscore is deliberate: this is **not part of the public API**. It
        MUTATES breaker state (it is literally ``try_enter(model, now) == "open"``), so a
        bare predicate name would be an attractive nuisance — a caller who assumed it was
        pure would leak the probe slot and permanently block a model (the exact bug
        :meth:`is_definitively_open` exists to prevent). Production code uses only the two
        public entry points: :meth:`is_definitively_open` (a read-only, never-reserving
        peek — for a speculative or pre-scheduling check on a delegation that might not
        run) and :meth:`try_enter` (the sole reserving call, made exactly once immediately
        before a delegation runs). This helper exists ONLY so tests can assert the combined
        boolean without restating ``try_enter(...) == "open"`` at every call site.

        A ``"probe"`` result means THIS call was just admitted as the (sole) half-open
        probe — the delegation is meant to proceed, so that case reports ``False`` (not
        blocked), same as ``"closed"``; only ``"open"`` reports ``True`` (fail-fast).
        """
        return self.try_enter(model, now) == "open"
