# skills/ollama/scripts/run_lock.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Process-liveness locking for Ollama run directories.

Each run dir carries a ``.ollama-lock`` naming the owning PID, ISO start time,
and a staleness bound. ``cleanup_old_runs`` consults :func:`is_dir_live` so a
concurrent session never prunes a run whose owner is still alive. Advisory and
self-healing: a crashed owner leaves a stale lock, reclaimed once the PID reports
dead or the bound elapses.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

LOCK_FILENAME = ".ollama-lock"
LOCK_STALE_AFTER_SECONDS = 21_600  # 6h floor for run dirs
_LOCK_SETUP_GRACE_SECONDS = 10  # TOCTOU grace: mkdtemp -> write_lock window (R15/R16)


def staleness_bound_for_timeout(timeout: int) -> int:
    """Run-dir staleness bound: ``max(2*timeout + 600, 6h)`` seconds (PID-reuse safe)."""
    return max(2 * timeout + 600, LOCK_STALE_AFTER_SECONDS)


def staleness_bound_ephemeral(timeout: int) -> int:
    """Ephemeral-lock staleness bound: ``2*timeout`` seconds, **NO 6h floor**.

    For the short-lived stdout-token / slot lockfiles MS7 writes: a crashed holder
    must free the token in minutes, not the 6h a run dir tolerates. ``is_dir_live``
    honors this persisted bound verbatim (see below), so it is not re-floored to 6h.

    Args:
        timeout: The per-delegation timeout in seconds.

    Returns:
        The short staleness bound in seconds.
    """
    return 2 * timeout


def _lock_path(run_dir: str) -> str:
    return os.path.join(run_dir, LOCK_FILENAME)


def is_pid_alive(pid: int) -> bool:
    """Return True if a process with *pid* currently exists.

    POSIX: ``os.kill(pid, 0)`` (``ProcessLookupError`` → dead; ``PermissionError``
    → alive). Windows: ``OpenProcess`` + ``WaitForSingleObject``. Any uncertainty
    is treated conservatively as alive so cleanup never prunes an unverifiable dir.
    """
    if pid <= 0:
        return False
    if pid > 4_294_967_295:
        return True
    try:
        if sys.platform == "win32":
            return _is_pid_alive_windows(pid)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True
        return True
    except Exception:  # noqa: BLE001 — conservative: unverifiable → alive.
        return True


_probe_warned = False  # one-per-process WARNING guard for unexpected liveness-probe failures


def _warn_unexpected_probe_failure(detail: str) -> None:
    """Emit at most ONE ``WARNING`` per process for an unexpected liveness-probe failure.

    R16's conservative-on-uncertainty bias means an unrecognized probe failure is treated
    as alive (never silently reclaimed), but that must stay observable rather than a silent
    no-op regression — this fires once per process so a run that repeatedly probes an
    ambiguous PID doesn't spam stderr.
    """
    global _probe_warned
    if not _probe_warned:
        _probe_warned = True
        print(
            f"WARNING: liveness probe returned an unexpected result ({detail}); "
            f"treating the PID as alive (conservative bias, R16).",
            file=sys.stderr,
        )


def _is_pid_alive_windows(pid: int) -> bool:
    """Windows liveness via ``OpenProcess`` + ``WaitForSingleObject`` (ctypes).

    ``OpenProcess`` failing is disambiguated into three outcomes (round 7 fix — previously
    only two were distinguished, and anything else silently fell through to "dead"):

    - ``ERROR_INVALID_PARAMETER`` (87): Windows confirms no such process exists → dead.
    - ``ERROR_ACCESS_DENIED`` (5): the PID exists but is inaccessible → alive.
    - anything else (unexpected/unrecognized error code): treated **conservatively as
      alive** (R16's stated bias — never wrongly reclaim a possibly-live dir) and reported
      once per process via :func:`_warn_unexpected_probe_failure` so the ambiguity stays
      observable instead of silently regressing to "dead".
    """
    try:
        import ctypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]
        k32.WaitForSingleObject.restype = ctypes.c_uint
        k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        SYNCHRONIZE, WAIT_TIMEOUT = 0x00100000, 0x00000102
        ERROR_ACCESS_DENIED, ERROR_INVALID_PARAMETER = 5, 87
        handle = k32.OpenProcess(SYNCHRONIZE, False, pid)
        if not handle:
            err = ctypes.get_last_error()
            if err == ERROR_INVALID_PARAMETER:
                return False  # confirmed: no such process
            if err == ERROR_ACCESS_DENIED:
                return True  # PID exists, just inaccessible to us
            _warn_unexpected_probe_failure(f"OpenProcess failed with error code {err}")
            return True  # conservative: unrecognized failure mode → treat as alive
        try:
            return bool(k32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT)
        finally:
            k32.CloseHandle(handle)
    except (OSError, AttributeError, ImportError):
        return True


def write_lock(run_dir: str, max_age_seconds: int | None = None) -> None:
    """Write ``<run_dir>/.ollama-lock`` atomically (PID / ISO start / bound)."""
    bound = LOCK_STALE_AFTER_SECONDS if max_age_seconds is None else int(max_age_seconds)
    payload = f"{os.getpid()}\n{datetime.now(timezone.utc).isoformat()}\n{bound}\n"
    final = _lock_path(run_dir)
    tmp = final + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, final)
    except OSError as exc:
        # Best-effort orphan cleanup (Caspar's recommended fix, WARNING): if the write
        # itself failed, `tmp` may never have been created; if `os.replace` failed
        # instead, `tmp` DOES exist (the rename never happened) and would otherwise be
        # left behind as a `.ollama-lock.tmp` orphan. Guarded by its own try/except so a
        # failure here can never mask or replace the warning below — cleanup is
        # best-effort, the diagnostic is not.
        try:
            os.remove(tmp)
        except OSError:
            pass
        # Loud, explicit: the run dir is now UNPROTECTED against concurrent pruning
        # (best-effort per R15) — the run still proceeds, but the unprotected state
        # must be observable so a lost artifact is diagnosable, not silent.
        print(
            f"WARNING: could not write run lock in {run_dir}: {exc}\n"
            f"WARNING: this run dir is UNPROTECTED against concurrent cleanup "
            f"(a parallel session may prune it); artifacts may be lost.",
            file=sys.stderr,
        )


def remove_lock(run_dir: str) -> None:
    """Remove the lock file if present (best-effort, never raises).

    On a SUCCESSFUL removal, also ages the dir's own mtime just past the
    setup-grace window. Without this, ``os.remove`` (and the earlier ``write_lock``)
    leave the dir's mtime "just now", so ``is_dir_live``'s no-lock branch would mask
    a just-torn-down dir as still-in-setup (:func:`_dir_is_within_setup_grace`) for
    up to ``_LOCK_SETUP_GRACE_SECONDS``. Teardown must be distinguishable from setup:
    once a run removes its lock the dir is done, so it is aged out of the grace window
    to report not-live immediately and to sort oldest-first under the LRU cleanup.
    Only ages when a lock was actually removed (``os.remove`` succeeded), so a spurious
    call on a mid-setup dir that never had a lock never defeats its legitimate grace.
    Best-effort throughout: any failure is swallowed.
    """
    try:
        os.remove(_lock_path(run_dir))
    except OSError:
        return
    try:
        aged = time.time() - _LOCK_SETUP_GRACE_SECONDS - 1
        os.utime(run_dir, (aged, aged))
    except OSError:
        pass


def _parse_lock(run_dir: str) -> tuple[int | None, float | None, int | None]:
    try:
        with open(_lock_path(run_dir), encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None, None, None
    pid = age = bound = None
    if lines:
        try:
            pid = int(lines[0].strip())
        except ValueError:
            pid = None
    if len(lines) > 1:
        try:
            started = datetime.fromisoformat(lines[1].strip())
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - started).total_seconds()
        except ValueError:
            age = None
    if len(lines) > 2:
        try:
            bound = int(lines[2].strip())
        except ValueError:
            bound = None
    return pid, age, bound


def _dir_is_fresh(run_dir: str) -> bool:
    try:
        return (time.time() - os.path.getmtime(run_dir)) < LOCK_STALE_AFTER_SECONDS
    except OSError:
        return True


def _dir_is_within_setup_grace(run_dir: str) -> bool:
    """True if *run_dir*'s mtime is within the TOCTOU setup-grace window.

    Used only for the no-lock-at-all branch of :func:`is_dir_live`: a dir just
    ``mkdtemp``-ed by ``create_output_dir`` but not yet ``write_lock``-ed is
    legitimately mid-setup, not abandoned. Any stat failure is treated as "not
    within grace" (conservative for THIS helper only — the outer `is_dir_live`
    total-exception guard still governs overall conservatism).
    """
    try:
        return (time.time() - os.path.getmtime(run_dir)) < _LOCK_SETUP_GRACE_SECONDS
    except OSError:
        return False


def is_dir_live(run_dir: str) -> bool:
    """Return True if *run_dir* belongs to a still-running process.

    No lock file at all, but the dir's mtime is within `_LOCK_SETUP_GRACE_SECONDS`
    (10s) → live (**TOCTOU fix**: closes the race between `create_output_dir`'s
    `mkdtemp` and `managed_run_dir`'s `write_lock`, R15/R16 — the dir is
    legitimately mid-setup, not abandoned, so a concurrent `cleanup_old_runs` must
    not prune it out from under the setup). No lock and older than the grace
    window → not live (prunable, e.g. setup crashed before ever writing a lock).
    Lock file present but unparseable PID → live iff dir is fresh (6h floor,
    unchanged). Dead PID → not live. Alive PID past the **persisted** bound → not
    live (PID-reuse mitigation). Alive within bound → live. Any unexpected error →
    conservatively live.

    The persisted bound is honored **verbatim** (never re-floored to 6h) so an
    ephemeral short bound (:func:`staleness_bound_ephemeral`) reclaims in minutes.
    """
    try:
        lock_exists = os.path.exists(_lock_path(run_dir))
        pid, age, bound = _parse_lock(run_dir)
        if pid is None:
            if lock_exists:
                return _dir_is_fresh(run_dir)
            # No lock at all: either genuinely lockless (not live) or mid-setup
            # between mkdtemp and write_lock (TOCTOU window, WARNING fix) — a very
            # freshly created dir is treated as live to close that race.
            return _dir_is_within_setup_grace(run_dir)
        if not is_pid_alive(pid):
            return False
        if age is None or age < 0:
            return _dir_is_fresh(run_dir)
        threshold = LOCK_STALE_AFTER_SECONDS if bound is None else bound
        return age < threshold
    except Exception:  # noqa: BLE001 — total: never raise into cleanup.
        return True


# --- Shared ephemeral-lock primitive (R7d stdout token, MS7 Task 3 concurrency slots) ---
#
# Reused verbatim by both the cross-process stdout token (below) and the concurrency
# slot-counter (MS7 Task 3) so every ephemeral lockfile this runtime creates shares ONE
# on-disk format, ONE parser, and ONE TOCTOU-safe acquire/reclaim algorithm (DRY) --
# tuned only by a caller-supplied SHORT staleness bound (:func:`staleness_bound_ephemeral`,
# no 6h floor), as opposed to the run-dir lock's 6h-floored
# :func:`staleness_bound_for_timeout` above.

STDOUT_TOKEN_FILENAME = ".ollama-stdout.lock"
_EPHEMERAL_RECLAIM_RETRIES = 3  # bounded reclaim attempts -> no livelock on a hot race


def _ephemeral_payload(bound: int) -> bytes:
    """Return the 3-line ephemeral lock payload (PID / ISO-8601 UTC / bound).

    Byte-for-byte the SAME shape ``write_lock`` writes for a run dir, so every lock
    this runtime creates (run dir, stdout token, concurrency slots) shares one format
    and one parser -- the ephemeral variants only differ in the SHORT ``bound`` value
    (:func:`staleness_bound_ephemeral`, no 6h floor).

    Args:
        bound: The staleness bound (seconds) to persist.

    Returns:
        The UTF-8-encoded 3-line payload.
    """
    return (f"{os.getpid()}\n{datetime.now(timezone.utc).isoformat()}\n{int(bound)}\n").encode(
        "utf-8"
    )


def _read_lock_fields(path: str) -> tuple[int | None, float | None, int | None]:
    """Parse ``(pid, age_seconds, bound)`` from a 3-line ephemeral lockfile at *path*.

    Mirrors the run-dir ``_parse_lock`` but reads an ARBITRARY lockfile path (a token or
    slot file, not ``<dir>/.ollama-lock``). Any unreadable/malformed field degrades to
    ``None`` (conservative -- the caller decides liveness from what it could parse).

    **Documented limitation -- wall-clock age, not monotonic (accepted).** ``age`` is
    computed from the persisted ISO-8601 UTC WALL-CLOCK timestamp (``datetime.now(utc)``
    at write time) compared against ``datetime.now(utc)`` at read time -- necessarily so,
    since a per-process ``time.monotonic()`` value is meaningless across process
    boundaries (each process's monotonic clock has its own arbitrary epoch); wall-clock
    is the only clock two different processes can compare at all. This makes the age
    comparison subject to WALL-CLOCK SKEW between the writer and the reader. For the
    SUPPORTED, SAME-MACHINE case (the only one this primitive targets) every process
    shares exactly one system clock, so there is no skew to speak of. Coordinating this
    lock across DIFFERENT MACHINES is already unsupported (PIDs are local-machine-only
    by construction) and would additionally require NTP-level clock synchronization to
    keep this age comparison meaningful -- that is explicitly out of scope, not a gap
    introduced here.

    Args:
        path: The ephemeral lockfile to parse.

    Returns:
        A ``(pid, age_seconds, bound)`` tuple; any field that could not be parsed is
        ``None``.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None, None, None
    pid = age = bound = None
    if lines:
        try:
            pid = int(lines[0].strip())
        except ValueError:
            pid = None
    if len(lines) > 1:
        try:
            started = datetime.fromisoformat(lines[1].strip())
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - started).total_seconds()
        except ValueError:
            age = None
    if len(lines) > 2:
        try:
            bound = int(lines[2].strip())
        except ValueError:
            bound = None
    return pid, age, bound


def _lockfile_holder_is_live(path: str) -> bool:
    """Return True if the ephemeral lockfile at *path* is held by a live process.

    **The short ephemeral bound is AUTHORITATIVE -- reclaimable if the PID is dead OR its
    persisted age has reached the bound, whichever comes first.** This is the exact same
    rule ``is_dir_live`` (above, unchanged) applies (``age < threshold`` there); the only
    difference between the two is WHICH bound gets persisted by their respective callers
    -- this primitive is always used with :func:`staleness_bound_ephemeral` (short,
    un-floored), vs. the run-dir lock's 6h-floored ``staleness_bound_for_timeout``. So the
    two locks share ONE policy, tuned by two different bound values for two different
    holder lifetimes -- not two different policies.

    Why bound-authoritative even when the PID is alive (PID-recycling safety): a
    legitimate holder of an EPHEMERAL lock (a stdout token, a concurrency slot) is a
    SINGLE delegation that cannot outlive its own timeout -- its persisted bound is
    ``2*timeout``. A lock whose age has already reached that bound is therefore
    DEFINITELY not still held by the process that wrote it: either that process died and
    released nothing (crash), or -- the case a purely liveness-based rule gets wrong --
    the OS RECYCLED its PID number to a different, unrelated, live process, which would
    make a liveness-only check report the lock "held" FOREVER, permanently wedging a
    stdout token or a concurrency slot. Treating the bound as authoritative closes that
    hole: once a lock is definitively past its own holder's maximum possible lifetime, it
    is reclaimable, regardless of whether *some* live process now happens to own that PID
    number.

    Accepted trade-off, self-healing (documented): a process genuinely SUSPENDED (e.g. a
    laptop put to sleep mid-delegation) past ``2*timeout`` could have its token/slot
    reclaimed here while it is technically still alive -- a brief over-commit at the
    endpoint (two holders of the same resource for a short window). This is judged
    acceptable because it SELF-HEALS: on resume, that delegation's own monotonic deadline
    (R25, checked before every retry/backoff) is already exceeded, so it aborts and
    releases on its own without ever completing -- it can race with, but never
    permanently coexist with, a reclaimer.

    Rules, in priority order:
      1. PID field unparseable (``pid is None``) -> INDETERMINATE: fall back to the
         age/bound heuristic -- fresh (``age < bound``) -> treated as held (not yet
         reclaimable); past the bound, or age/bound themselves unparseable -> reclaimable.
      2. PID parses but the process is dead (``is_pid_alive`` is False) -> reclaimable.
      3. PID parses and the process is alive, but ``age``/``bound`` both parse and
         ``age >= bound`` -> reclaimable (PID-recycling-safe: the bound overrides a live
         PID that cannot possibly still be the original legitimate holder).
      4. PID parses and is alive, and (``age < bound``, or either is unparseable) -> held.

    Args:
        path: The ephemeral lockfile to inspect.

    Returns:
        True if a live process holds the lock and its bound has not yet elapsed (do not
        steal it); False if it is reclaimable (dead PID, bound-expired despite a live
        PID, or an indeterminate/corrupt lock past its fallback bound).
    """
    pid, age, bound = _read_lock_fields(path)
    if pid is None:
        # Corrupt/indeterminate PID: liveness cannot be checked at all, so the bound
        # decides (same fallback role as in is_dir_live's lockless-dir branch, above).
        return age is not None and bound is not None and age < bound
    if not is_pid_alive(pid):
        return False
    if age is not None and bound is not None and age >= bound:
        return False  # bound-expired: reclaimable even though a live process owns this PID
    return True


def _acquire_ephemeral(path: str, bound: int) -> bool:
    """Atomically acquire the ephemeral lockfile at *path*, reclaiming a stale holder.

    Fresh path -> ``O_CREAT|O_EXCL`` create wins outright (we own it, no re-verify needed).
    Path held by a LIVE process -> return False. Path whose holder is DEAD/stale ->
    TOCTOU-safe bounded reclaim: for at most ``_EPHEMERAL_RECLAIM_RETRIES`` rounds,
    remove the stale file and ``O_EXCL``-recreate it, then **re-verify ownership** by
    re-reading the PID on disk -- if a competing reclaimer beat us (the PID isn't ours),
    back off (return False) rather than assume we hold it. Bounded retries avoid a
    livelock when two processes race to reclaim the same corpse.

    Args:
        path: The lockfile path to acquire.
        bound: The staleness bound (seconds) to persist (short/ephemeral).

    Returns:
        True if this process now holds the lock, False otherwise.
    """
    payload = _ephemeral_payload(bound)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(fd, payload)
        os.close(fd)
        return True
    except FileExistsError:
        pass
    for _ in range(_EPHEMERAL_RECLAIM_RETRIES):
        if _lockfile_holder_is_live(path):
            return False  # live holder -> do not steal
        try:
            os.remove(path)
        except FileNotFoundError:
            pass  # another reclaimer already removed it
        except OSError:
            return False
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(fd, payload)
            os.close(fd)
        except FileExistsError:
            continue  # lost the recreate race -> retry
        except OSError:
            return False
        pid, _age, _bound = _read_lock_fields(path)  # ownership re-verification
        return pid == os.getpid()
    return False


def _release_ephemeral(path: str) -> None:
    """Remove the ephemeral lockfile at *path* if present (best-effort, idempotent).

    Args:
        path: The ephemeral lockfile to remove.
    """
    try:
        os.remove(path)
    except OSError:
        pass


def acquire_token(path: str, timeout: int) -> bool:
    """Acquire the cross-process stdout stream token (R7d).

    At most one delegation ACROSS ALL PROCESSES streams tokens to the terminal. Uses the
    shared ephemeral primitive with a SHORT bound (:func:`staleness_bound_ephemeral`,
    ~2x``timeout``, NO 6h floor) so a crashed holder frees the token in minutes, not hours.

    **WARNING -- mid-stream holder crash (documented, bounded, not permanent).** The
    normal release path is the ``finally`` in ``run_ollama._stream_with_stdout_token``,
    which runs on any Python-level exception including ``KeyboardInterrupt``. A HARD
    crash that bypasses Python entirely (SIGKILL, OOM-kill, power loss) skips that
    ``finally``, so the token lockfile is left behind and is NOT released proactively.
    This is NOT an unbounded interleaving risk, for two independent reasons, either of
    which resolves it: (1) the crashed holder's PID dies with it, so
    ``_lockfile_holder_is_live``'s ``is_pid_alive`` check reports it dead essentially
    IMMEDIATELY on the very next `acquire_token` call from any other process -- no need
    to wait out the bound at all; (2) even in the pathological case where the OS has
    already recycled that PID number into a different, unrelated LIVE process (so the
    liveness check alone would be fooled), the SHORT ephemeral bound (~2x``timeout``) is
    authoritative and reclaims the token anyway once the bound elapses (see
    ``_lockfile_holder_is_live``). So the token is reclaimed either near-instantly (dead
    PID, the common crash case) or within one bound window (the PID-recycling edge
    case) -- no periodic liveness-polling loop is needed. Until reclaimed, a NEW
    delegation simply goes file-only (never stdout) rather than interleaving with a
    ghost -- there is no window in which two processes both believe they hold the token.

    Args:
        path: The token lockfile (``<project_run_root>/.ollama-stdout.lock``).
        timeout: Per-delegation timeout (drives the short staleness bound).

    Returns:
        True if this process now holds the token (-> stream to stdout); False if a live
        holder exists (-> route this delegation's stream to ``{cap}.stream.log`` instead).
    """
    return _acquire_ephemeral(path, staleness_bound_ephemeral(timeout))


def release_token(path: str) -> None:
    """Release the stdout token if held (best-effort, idempotent).

    Args:
        path: The token lockfile to release.
    """
    _release_ephemeral(path)
