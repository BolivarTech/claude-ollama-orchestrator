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
            # CLAMP to >= 0 (Caspar residual): a wall clock that steps BACKWARD (NTP
            # correction, manual reset) makes `now - started` negative, which would make a
            # bound-expiry check (`age >= bound`) impossible to satisfy -> a recycled-PID lock
            # could appear held indefinitely. Clamping keeps age a sane non-negative duration.
            # The clock-immune PRIMARY liveness signal is the `is_pid_alive` check in the
            # caller: a DEAD holder is reclaimed regardless of age, so a truly permanent lock
            # cannot arise from skew alone; the wall-clock bound is only the PID-RECYCLING
            # fallback for a live PID that cannot still be the original holder.
            age = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
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
        return max(0.0, time.time() - os.path.getmtime(run_dir)) < LOCK_STALE_AFTER_SECONDS
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
        return max(0.0, time.time() - os.path.getmtime(run_dir)) < _LOCK_SETUP_GRACE_SECONDS
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
# Release retry (MS7 Task 3 fix): on Windows, `os.remove` on an ephemeral lockfile can
# transiently fail (PermissionError WinError 5/32, "being used by another process") when
# several REAL OS threads contend on the same handful of tiny lockfiles at once (`run_
# batch`'s fan-out drives genuinely concurrent `asyncio.to_thread` workers) -- the same
# class of Windows filesystem contention already documented for `os.replace` elsewhere in
# this codebase. Swallowing that on the FIRST attempt (the old behavior) silently STRANDS
# the lockfile on disk for up to its own staleness bound, artificially shrinking the
# available token/slot pool for everyone else in the meantime. A short, bounded retry
# closes that gap without changing the release contract (still best-effort, still never
# raises).
_RELEASE_RETRY_ATTEMPTS = 10
_RELEASE_RETRY_DELAY_SECONDS = 0.005
# Torn-write grace (MS7 Task 3 fix): the 3-line payload is written by a SINGLE os.write
# call, but the file is briefly observable as 0 BYTES between a winning `os.open(O_CREAT
# | O_EXCL)` and that write landing -- a real, non-negligible window under genuine
# multi-threaded contention (`run_batch`'s fan-out drives several concurrent OS threads,
# via `asyncio.to_thread`, racing to acquire the same handful of slot files). A VERY
# fresh empty/corrupt lockfile is presumed mid-write (held); only one older than this
# grace is presumed a genuine abandoned corpse (reclaimable). Generous relative to a
# single small os.write's real latency, tiny relative to any legitimate holder's own
# multi-second-to-minutes lifetime.
#
# FILESYSTEM mtime GRANULARITY (Caspar residual): rule 0 compares `time.time()` against
# `os.path.getmtime()`. On a coarse-granularity filesystem (FAT/exFAT store mtime to the
# nearest 2 s) a just-created file can already report an apparent age near 2 s, which a
# 2-second grace would treat as reclaimable the instant it is written -- prematurely
# stealing a mid-write lock. The grace is therefore set ABOVE FAT's 2 s granularity (plus
# margin) so even the coarsest common filesystem keeps a freshly created empty lock "held"
# through its whole torn-write window. NTFS/ext4/APFS (sub-second) are unaffected either
# way; the wider value only lets a genuinely abandoned corpse linger a couple extra seconds
# (negligible vs. any real holder's lifetime). Defense-in-depth on top of the atomic
# `os.replace` claim in `_acquire_ephemeral`, which no longer removes blind.
_EPHEMERAL_TORN_WRITE_GRACE_SECONDS = 5


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
            # CLAMP to >= 0 (Caspar residual): a wall clock that steps BACKWARD (NTP
            # correction, manual reset) makes `now - started` negative, which would make a
            # bound-expiry check (`age >= bound`) impossible to satisfy -> a recycled-PID lock
            # could appear held indefinitely. Clamping keeps age a sane non-negative duration.
            # The clock-immune PRIMARY liveness signal is the `is_pid_alive` check in the
            # caller: a DEAD holder is reclaimed regardless of age, so a truly permanent lock
            # cannot arise from skew alone; the wall-clock bound is only the PID-RECYCLING
            # fallback for a live PID that cannot still be the original holder.
            age = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
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
      0. PID/age/bound ALL unparseable (a totally empty or unreadable file) -> a
         TORN-WRITE window (a concurrent winner's `os.open(O_CREAT|O_EXCL)` landed but
         its `os.write` has not yet -- see :data:`_EPHEMERAL_TORN_WRITE_GRACE_SECONDS`)
         is far more likely, under real thread/process contention, than a genuinely
         abandoned corpse: the file's own mtime decides -- younger than the grace
         window -> presumed mid-write (held, not yet reclaimable); older -> presumed
         abandoned (reclaimable).
      1. PID field unparseable but SOME field parsed (``pid is None``, not rule 0) ->
         INDETERMINATE: fall back to the age/bound heuristic -- fresh (``age < bound``)
         -> treated as held (not yet reclaimable); past the bound, or age/bound
         themselves unparseable -> reclaimable.
      2. PID parses but the process is dead (``is_pid_alive`` is False) -> reclaimable.
      3. PID parses and the process is alive, but ``age``/``bound`` both parse and
         ``age >= bound`` -> reclaimable (PID-recycling-safe: the bound overrides a live
         PID that cannot possibly still be the original legitimate holder).
      4. PID parses and is alive and ``age < bound`` -> held.
      5. PID parses and is alive but ``age``/``bound`` is unparseable (torn write that
         landed only the PID line, or later corruption) -> the persisted bound cannot be
         applied, so freshness falls back to the file's own mtime vs.
         :data:`_EPHEMERAL_TORN_WRITE_GRACE_SECONDS` (as in rule 0): younger -> held;
         older -> reclaimable. Without this, a recycled-PID live process would keep such a
         lock IMMORTAL, wedging a slot/token forever.

    Args:
        path: The ephemeral lockfile to inspect.

    Returns:
        True if a live process holds the lock and its bound has not yet elapsed (do not
        steal it); False if it is reclaimable (dead PID, bound-expired despite a live
        PID, a torn-write file past its grace window, or an indeterminate/corrupt lock
        past its fallback bound).
    """
    pid, age, bound = _read_lock_fields(path)
    if pid is None:
        if age is None and bound is None:
            # Rule 0: nothing at all parsed -- a 0-byte (or otherwise unreadable)
            # lockfile. Use the file's OWN mtime as the freshness signal (mirrors the
            # run-dir lock's `_dir_is_within_setup_grace` TOCTOU-grace pattern): a file
            # created moments ago is presumed to be a concurrent winner's write still in
            # flight, not an abandoned corpse.
            try:
                age_on_disk = max(0.0, time.time() - os.path.getmtime(path))
            except OSError:
                return False  # vanished mid-check -> nothing left to hold, reclaimable
            return age_on_disk < _EPHEMERAL_TORN_WRITE_GRACE_SECONDS
        # Corrupt/indeterminate PID: liveness cannot be checked at all, so the bound
        # decides (same fallback role as in is_dir_live's lockless-dir branch, above).
        return age is not None and bound is not None and age < bound
    if not is_pid_alive(pid):
        return False
    if age is None or bound is None:
        # Live PID but an INCOMPLETE/corrupt payload -- a torn write that landed the PID line
        # but not a valid age/bound, or later corruption. The persisted bound can't be
        # applied, so a purely liveness-based rule would keep this lock IMMORTAL if that PID
        # number was recycled to an unrelated live process. Fall back to the file's own mtime
        # (like rule 0): younger than the grace -> presumed a concurrent winner's write still
        # in flight (held); older -> the writer died mid-write, reclaimable. This guarantees an
        # ephemeral lock ALWAYS self-heals rather than wedging a slot/token forever.
        try:
            age_on_disk = max(0.0, time.time() - os.path.getmtime(path))
        except OSError:
            return False  # vanished mid-check -> reclaimable
        return age_on_disk < _EPHEMERAL_TORN_WRITE_GRACE_SECONDS
    if age >= bound:
        return False  # bound-expired: reclaimable even though a live process owns this PID
    return True


def _write_close_ephemeral(fd: int, payload: bytes) -> bool:
    """Write *payload* to *fd* and ALWAYS close it; return True iff the write succeeded.

    A ``try/finally`` guarantees the descriptor is closed even when ``os.write`` fails --
    without it, a transient write error after a successful ``os.open`` would leak the fd
    until GC (and repeated failures could exhaust the descriptor table). ``os.close``'s own
    (rare) error is swallowed so this helper is total/never-raise, matching the acquire
    path's fail-to-False contract.

    Args:
        fd: An open file descriptor owned by the caller (from ``os.open``).
        payload: The bytes to write.

    Returns:
        True if the write succeeded, False if it raised ``OSError``.
    """
    try:
        # os.write may perform a SHORT write (return fewer bytes than requested); loop over
        # the remaining bytes so the whole payload always lands -- a truncated write would
        # leave a torn lockfile that misparses (wrong PID/bound).
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:  # defensive: a non-positive return can't make progress
                return False
            view = view[written:]
        return True
    except OSError:
        return False
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _remove_quiet(path: str) -> None:
    """``os.remove`` *path*, swallowing any ``OSError`` (already gone / transient FS error).

    Used for best-effort cleanup of a private reclaim scratch file where a failure to remove
    is never fatal (the file is PID-scoped and self-heals on the next attempt).
    """
    try:
        os.remove(path)
    except OSError:
        pass


def _restore_stolen_lock(steal_path: str, path: str) -> None:
    """Put a stolen LIVE lock back at *path*; never delete it, never leave *path* unclaimed.

    First retries an atomic ``os.replace(steal_path, path)`` -- a same-dir rename that virtually
    always succeeds (a transient Windows ``PermissionError`` under contention clears on retry).
    If every retry fails, RE-CREATES the holder's lock at *path* from the stolen payload so the
    slot/token is never left free (which would let a new acquirer become a SECOND owner while
    the original holder still believes it holds the lock). The holder is the rightful owner, so
    re-writing ITS payload preserves mutual exclusion. Only if that re-create also fails because
    *path* was already taken by a new acquirer (``O_EXCL`` -> that acquirer is the single owner,
    no worse) is the stolen copy left to self-heal on the holder's own bound. Total: never
    raises; never deletes a live holder's lock.

    Args:
        steal_path: The private path holding the stolen live lock.
        path: The lock path to restore it to.
    """
    for attempt in range(_RELEASE_RETRY_ATTEMPTS):
        try:
            os.replace(steal_path, path)
            return  # restored in place
        except OSError:
            if attempt < _RELEASE_RETRY_ATTEMPTS - 1:
                time.sleep(_RELEASE_RETRY_DELAY_SECONDS)
    # Rename exhausted -> re-create the holder's lock at `path` from the stolen payload so the
    # resource is never left unclaimed (Caspar residual: a free `path` here -> double-owner).
    try:
        with open(steal_path, "rb") as fh:
            payload = fh.read()
    except OSError:
        return  # stolen copy vanished -> nothing to re-create (path presumably reclaimed)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError:
        return  # `path` already taken by a new acquirer (single owner) -> leave the steal copy
    if _write_close_ephemeral(fd, payload):
        _remove_quiet(steal_path)  # holder's lock safely re-created at `path`; drop the copy


def _acquire_ephemeral(path: str, bound: int) -> bool:
    """Atomically acquire the ephemeral lockfile at *path*, reclaiming a stale holder.

    Fresh path -> ``O_CREAT|O_EXCL`` create wins outright (we own it, no re-verify needed).
    Path held by a LIVE process -> return False. Path whose holder is DEAD/stale ->
    TOCTOU-safe bounded reclaim: for at most ``_EPHEMERAL_RECLAIM_RETRIES`` rounds, ATOMICALLY
    claim the stale file with ``os.replace`` into a private ``<path>.reclaim.<pid>`` (exactly
    one racer can move a given file), then inspect that STOLEN copy in isolation. If it turns
    out a competitor swapped a fresh LIVE lock in during the window, RESTORE it and back off --
    never evict a live holder (R7d/R21c mutual exclusion). If it was genuinely stale, discard
    it and ``O_EXCL``-create our own lock, then **re-verify ownership** by re-reading the PID.
    A competing reclaimer that beat us to the ``O_EXCL`` create makes us retry, and bounded
    retries avoid a livelock when two processes race the same corpse. (The restore path has a
    rare triple-race that can overwrite a third racer's fresh lock -- an accepted, bounded,
    self-healing limitation of portable lockfiles without an OS advisory lock.)

    Args:
        path: The lockfile path to acquire.
        bound: The staleness bound (seconds) to persist (short/ephemeral).

    Returns:
        True if this process now holds the lock, False otherwise.
    """
    payload = _ephemeral_payload(bound)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        fd = None  # already held -> try to reclaim below
    except OSError:
        # Any OTHER transient FS error on the atomic create (a Windows AV/indexer briefly
        # locking the new file, a read-only temp, EMFILE, ...) means we simply couldn't
        # acquire the token -- degrade to False so the caller falls back to a per-agent
        # file sink, NEVER propagate and crash the delegation (total/never-raise).
        return False
    if fd is not None:
        # Fresh create won: write the payload, ALWAYS closing the fd. On write failure the
        # (empty/torn) file is OURS -- O_EXCL guarantees no competitor created it -- so REMOVE it
        # rather than leave it to occupy `path` until its mtime grace expires (Caspar residual:
        # an empty lockfile briefly blocking the slot/token). Its own mtime grace would self-heal
        # it, but proactively removing our own failed creation is cleaner and faster.
        if _write_close_ephemeral(fd, payload):
            return True
        _remove_quiet(path)
        return False
    steal_path = f"{path}.reclaim.{os.getpid()}"
    for _ in range(_EPHEMERAL_RECLAIM_RETRIES):
        if _lockfile_holder_is_live(path):
            return False  # live holder -> do not steal
        # TOCTOU-safe reclaim (R7d/R21c mutual exclusion). A plain check-then-`os.remove`
        # could delete a FRESH LIVE lock that a competing reclaimer placed in the window
        # between the liveness check above and the remove -- evicting a live holder and
        # producing TWO owners. Instead ATOMICALLY claim whatever is at `path` by moving it
        # to a private name (`os.replace` moves exactly one file; only one racer can win a
        # given file), then inspect the STOLEN copy in isolation where no competitor can
        # mutate it:
        steal_gone = steal_perm_fail = False
        for _steal_try in range(_RELEASE_RETRY_ATTEMPTS):
            try:
                os.replace(path, steal_path)
                break  # claimed
            except FileNotFoundError:
                steal_gone = True  # another reclaimer already took it
                break
            except OSError:
                # Transient Windows PermissionError (a concurrent reader/writer briefly holding
                # `path`) clears on retry -- bounded, like the restore/release paths -- rather
                # than immediately abandoning the reclaim on a fleeting contention hiccup.
                if _steal_try == _RELEASE_RETRY_ATTEMPTS - 1:
                    steal_perm_fail = True
                else:
                    time.sleep(_RELEASE_RETRY_DELAY_SECONDS)
        if steal_perm_fail:
            return False  # persistent contention -> give up (caller falls back to a file sink)
        if steal_gone:
            continue  # path vanished -> retry the outer reclaim loop (path may be free now)
        if _lockfile_holder_is_live(steal_path):
            # A competitor swapped in a LIVE lock during our window; we must NOT evict it.
            # Restore it (best-effort) and back off. The restore can, in a rare TRIPLE-race
            # (a THIRD racer created a fresh lock at `path` between our replace-out and this
            # restore), overwrite that third racer's lock. This is the ONE residual and it is
            # an ACCEPTED, IRREDUCIBLE limitation of a portable-lockfile design: closing it
            # fully needs an OS advisory lock (`fcntl.flock`/`msvcrt.locking`), deliberately
            # rejected for the stdlib-only, cross-platform contract (NR1). It is bounded and
            # self-healing (at most a transient +1 concurrent agent, resolved on the next
            # release/reclaim), NEVER a permanent double-owner. It is also UNREACHABLE under
            # the v0.1 shipping model: the cross-process slot/token machinery (R7d/R21c) is a
            # v0.2 feature; v0.1 runs a SINGLE orchestrator, and this whole reclaim path only
            # fires on a stale lock left by a CRASHED prior process, then needs THREE
            # concurrent same-index reclaimers in overlapping microsecond windows -- a
            # pattern the single-orchestrator invariant does not produce.
            #
            # CRITICAL: never delete the stolen live lock (that evicts the live holder, the
            # exact bug this reclaim prevents) and never leave `path` unclaimed (a free `path`
            # lets a new acquirer become a SECOND owner). `_restore_stolen_lock` retries the
            # atomic rename, then re-creates the holder's lock at `path` from the stolen payload
            # if the rename can't succeed -- preserving mutual exclusion in every case a portable
            # lockfile can (the only residual, a new acquirer racing into `path` in the tiny
            # re-create window, is the accepted, bounded, self-healing triple-race below).
            _restore_stolen_lock(steal_path, path)
            return False
        # The stolen copy was genuinely stale -> discard it; `path` is now free. Create our
        # fresh lock atomically; if a competitor created one meanwhile, O_EXCL fails -> retry.
        _remove_quiet(steal_path)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue  # lost the recreate race -> retry
        except OSError:
            return False
        if not _write_close_ephemeral(fd, payload):  # ALWAYS closes the fd (no leak)
            _remove_quiet(path)  # our own O_EXCL-created empty/torn file -> remove, don't strand
            return False
        pid, _age, _bound = _read_lock_fields(path)  # ownership re-verification
        if pid == os.getpid():
            return True
        # Re-verify did NOT read back our pid. In practice this is a transient read glitch on
        # our OWN O_EXCL-created lock (a competitor cannot leave a DIFFERENT live pid at `path`:
        # the atomic steal above RESTORES a live holder's lock rather than overwriting it). When
        # the file reads unreadable/gone (pid is None), remove OUR stranded creation so it is not
        # seen as held by later acquirers until the bound expires; a different, non-None
        # (competitor) pid is LEFT ALONE -- never evict a lock we do not own.
        if pid is None:
            _remove_quiet(path)
        return False
    return False


def _release_ephemeral(path: str) -> None:
    """Remove the ephemeral lockfile at *path* if present (best-effort, idempotent).

    Retries a bounded number of times on a TRANSIENT ``OSError`` (e.g. Windows
    ``PermissionError`` WinError 5/32 from a concurrent reader/writer briefly holding
    the file under real thread/process contention) before giving up silently --
    without this, a single transient failure would strand the lockfile on disk for up
    to its own staleness bound, shrinking the available pool for every other
    contender in the meantime. ``FileNotFoundError`` (already gone -- another
    reclaimer, or a concurrent release) is the normal idempotent case and returns
    immediately, never counted as a retryable failure.

    Args:
        path: The ephemeral lockfile to remove.
    """
    for attempt in range(_RELEASE_RETRY_ATTEMPTS):
        try:
            os.remove(path)
            return
        except FileNotFoundError:
            return  # already gone -- nothing left to release (idempotent)
        except OSError:
            if attempt == _RELEASE_RETRY_ATTEMPTS - 1:
                return  # best-effort: exhausted retries, give up silently
            time.sleep(_RELEASE_RETRY_DELAY_SECONDS)


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


# --- Cross-process concurrency slot-counter (R21c, MS7 Task 3) ---
#
# Makes `max_parallel_agents` a GLOBAL cap, not merely a per-process one: the
# in-process `Scheduler` semaphore (MS5) only bounds concurrency WITHIN one
# `run_ollama.py`; two independent processes would each run up to `max_parallel_agents`
# (e.g. 3 + 3 = 6 agents against the endpoint). Reuses the SAME shared ephemeral-lock
# primitive as the stdout token above (`_acquire_ephemeral`/`_lockfile_holder_is_live`,
# O_EXCL + 3-line short-bound payload + TOCTOU-safe dead-holder reclaim) -- one on-disk
# format, one parser, one algorithm for every ephemeral lockfile this runtime creates.

SLOTS_DIRNAME = ".ollama-slots"


def _cleanup_orphaned_slots(slots_dir: str, max_parallel: int) -> None:
    """Best-effort sweep of reclaimable OUT-OF-RANGE ``slot-*.lock`` files under *slots_dir*.

    A slot file OUTSIDE the current probe range -- e.g. ``slot-5.lock`` left behind by a
    previous, larger ``max_parallel_agents`` -- is never probed again by `acquire_slot`
    (which only touches ``0..max_parallel-1``) and would otherwise accumulate under
    ``.ollama-slots/`` forever. This sweep removes each such orphan whose holder
    :func:`_lockfile_holder_is_live` reports reclaimable.

    **In-range slots (index < *max_parallel*) are deliberately LEFT ALONE** and reclaimed
    by `acquire_slot`'s own atomic ``O_EXCL`` probe/reclaim instead. Sweeping them here
    would open an R21c TOCTOU: this function's check-then-``os.remove`` is not atomic, so a
    concurrent `acquire_slot` that reclaims the same index between our liveness check and
    our remove (A reads the old dead PID, B swaps in a fresh live file, A removes B's file)
    would delete a slot that is actually held -> a freed-but-held slot -> over-subscription
    by one. Out-of-range orphans have NO concurrent acquirer under the supported model (all
    processes of a project share the same ``max_parallel`` -- an env override that differs
    per process is documented as unsupported in :func:`acquire_slot`), so removing them is
    race-free. This keeps the sweep pure hygiene, never load-bearing for the cap (enforced
    solely by `acquire_slot`'s atomic probe/reclaim). The ``.ollama-slots/`` dir lives under
    the per-project namespace (alongside the ``ollama-run-*`` dirs, R15).

    MISMATCHED ``max_parallel`` (documented, unsupported -- Caspar residual): "out of range"
    is relative to THIS caller's *max_parallel*. If two processes shared the dir with
    DIFFERENT caps (the explicitly unsupported config, see :func:`acquire_slot`), a
    smaller-cap process would treat a larger-cap holder's higher-index slot as an orphan.
    Even then a genuinely LIVE holder is safe -- the sweep removes a file ONLY when
    :func:`_lockfile_holder_is_live` reports it reclaimable, so a live holder within its bound
    is never swept; only a holder that has already crashed or blown its bound (i.e. was itself
    reclaimable) can be. Keep ``max_parallel_agents`` consistent per project to avoid it.

    Total: a listing/stat/remove failure -- or an unparseable ``slot-<x>.lock`` index -- is
    a silent no-op.

    Args:
        slots_dir: The per-project slots dir to sweep.
        max_parallel: The current global cap; only indices ``>= max_parallel`` are swept.
    """
    try:
        names = os.listdir(slots_dir)
    except OSError:
        return
    for name in names:
        if not (name.startswith("slot-") and name.endswith(".lock")):
            continue
        try:
            index = int(name[len("slot-") : -len(".lock")])
        except ValueError:
            continue  # not a `slot-<int>.lock` -- not ours to classify, leave it
        if index < max_parallel:
            continue  # in-range: owned by acquire_slot's atomic reclaim, never swept here
        path = os.path.join(slots_dir, name)
        try:
            if not _lockfile_holder_is_live(path):
                os.remove(path)
        except OSError:
            pass


def acquire_slot(slots_dir: str, max_parallel: int, timeout: int) -> int | None:
    """Acquire a cross-process concurrency slot (R21c); return its index or None.

    Bounds concurrent Ollama agents to *max_parallel* GLOBALLY across processes -- the
    in-process ``Scheduler`` semaphore (MS5) only caps one process. First opportunistically
    sweeps `.ollama-slots/` for reclaimable OUT-OF-RANGE orphans (:func:`_cleanup_orphaned_slots`
    -- indices ``>= max_parallel`` left by a previous, larger ``max_parallel_agents`` that
    would otherwise accumulate; in-range indices are left to the atomic probe/reclaim below to
    avoid a sweep-vs-acquire TOCTOU), then probes indices ``0..max_parallel-1`` and acquires
    the FIRST free one via the shared ephemeral-lock primitive (``O_EXCL`` create, 3-line
    short-bound payload, TOCTOU-safe dead-slot reclaim). All slots held by live processes ->
    returns None, which the caller treats as an R21b queue/rejection (never a silent
    over-subscription of the endpoint).

    Accepted, documented (INFO): the probe is ``O(max_parallel)`` filesystem
    round-trips per acquisition -- trivial at the default ``max_parallel_agents = 3``
    (and still negligible in the tens), not a hot loop. Correctness also assumes every
    process sharing this ``slots_dir`` agrees on the SAME ``max_parallel``: the
    per-project ``./.claude/ollama-agents.toml`` gives this for free (one config, one
    project), but an env override (``OLLAMA_AGENTS_MAX_PARALLEL``) that differs between
    two overlapping processes of the SAME project is **unsupported** -- a smaller value
    in one process only bounds the slots THAT process probes, not the ones the other
    process already holds beyond it. Keep ``max_parallel_agents`` consistent per project.

    Args:
        slots_dir: The per-project slots dir (``<project_run_root>/.ollama-slots``).
        max_parallel: The global concurrency cap (``max_parallel_agents``).
        timeout: Per-delegation timeout (drives the short staleness bound).

    Returns:
        The acquired slot index (0-based), or None if every slot is live.
    """
    try:
        os.makedirs(slots_dir, exist_ok=True)
    except OSError:
        return None
    _cleanup_orphaned_slots(slots_dir, max_parallel)
    bound = staleness_bound_ephemeral(timeout)
    for index in range(max_parallel):
        slot_path = os.path.join(slots_dir, f"slot-{index}.lock")
        if _acquire_ephemeral(slot_path, bound):
            return index
    return None


def release_slot(slots_dir: str, index: int) -> None:
    """Release the slot at *index* (best-effort, OSError-guarded, idempotent).

    Args:
        slots_dir: The per-project slots dir.
        index: The slot index to release (as returned by `acquire_slot`).
    """
    _release_ephemeral(os.path.join(slots_dir, f"slot-{index}.lock"))
