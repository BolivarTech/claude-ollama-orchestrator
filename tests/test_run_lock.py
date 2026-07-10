# tests/test_run_lock.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Process-liveness locking: write/parse, is_pid_alive, is_dir_live decision table."""

import os
import time
from datetime import datetime, timedelta, timezone

import run_lock
from run_lock import (
    SLOTS_DIRNAME,
    STDOUT_TOKEN_FILENAME,
    acquire_slot,
    acquire_token,
    is_dir_live,
    is_pid_alive,
    release_slot,
    release_token,
    remove_lock,
    staleness_bound_ephemeral,
    staleness_bound_for_timeout,
    write_lock,
)


def test_staleness_bound_floors_at_6h_and_scales_with_timeout():
    assert staleness_bound_for_timeout(10) == 21_600  # floor
    assert staleness_bound_for_timeout(20_000) == 2 * 20_000 + 600


def test_ephemeral_bound_is_short_with_no_6h_floor():
    assert staleness_bound_ephemeral(30) == 2 * 30  # ~2*timeout, no floor
    assert staleness_bound_ephemeral(30) < 21_600  # far below the run-dir floor


def test_alive_pid_past_persisted_bound_is_not_live(tmp_path):
    # PID-reuse mitigation: our own (alive) PID, but a lock whose age already
    # exceeds a short persisted bound → is_dir_live is False.
    write_lock(str(tmp_path), max_age_seconds=1)  # 1s ephemeral bound
    lock = os.path.join(str(tmp_path), ".ollama-lock")
    lines = open(lock, encoding="utf-8").read().splitlines()
    old = "2000-01-01T00:00:00+00:00"  # far in the past
    open(lock, "w", encoding="utf-8").write(f"{lines[0]}\n{old}\n{lines[2]}\n")
    assert is_dir_live(str(tmp_path)) is False  # alive PID, past bound → not live


def test_alive_pid_within_persisted_bound_is_live(tmp_path):
    write_lock(str(tmp_path), max_age_seconds=3600)  # 1h bound, fresh
    assert is_dir_live(str(tmp_path)) is True


def test_own_pid_is_alive():
    assert is_pid_alive(os.getpid()) is True


def test_pid_zero_or_negative_is_not_alive():
    assert is_pid_alive(0) is False
    assert is_pid_alive(-1) is False


def test_write_then_dir_is_live_for_own_pid(tmp_path):
    write_lock(str(tmp_path))
    assert is_dir_live(str(tmp_path)) is True


def test_write_lock_removes_orphaned_tmp_and_warns_when_replace_fails(
    tmp_path, monkeypatch, capsys
):
    # Caspar's recommended fix (WARNING): os.replace(tmp, final) can fail AFTER the
    # .tmp write already succeeded — the .tmp file must not be left behind as an
    # orphan, and write_lock must still warn to stderr rather than raise.
    def _raise(*_a, **_k):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(run_lock.os, "replace", _raise)
    write_lock(str(tmp_path))  # must not raise
    assert not os.path.exists(os.path.join(str(tmp_path), ".ollama-lock.tmp"))  # no orphan
    assert not os.path.exists(os.path.join(str(tmp_path), ".ollama-lock"))  # never renamed
    assert "WARNING" in capsys.readouterr().err


def test_freshly_created_lockless_dir_is_live_during_toctou_grace_window(tmp_path):
    # Closes the mkdtemp -> write_lock race (R15/R16, WARNING fix): a dir created
    # just now with no lock yet is legitimately mid-setup, not abandoned — treated
    # as live so a concurrent cleanup_old_runs can't prune it out from under the
    # setup between create_output_dir's mkdtemp and managed_run_dir's write_lock.
    assert is_dir_live(str(tmp_path)) is True


def test_lockless_dir_older_than_grace_window_is_not_live(tmp_path):
    old = time.time() - 3600
    os.utime(str(tmp_path), (old, old))
    assert is_dir_live(str(tmp_path)) is False


def test_dead_pid_lock_is_not_live(tmp_path, monkeypatch):
    write_lock(str(tmp_path))
    monkeypatch.setattr(run_lock, "is_pid_alive", lambda pid: False)
    assert is_dir_live(str(tmp_path)) is False


def test_remove_lock_is_idempotent(tmp_path):
    write_lock(str(tmp_path))
    remove_lock(str(tmp_path))
    remove_lock(str(tmp_path))  # no raise
    assert is_dir_live(str(tmp_path)) is False


def test_windows_unexpected_openprocess_error_is_conservatively_alive(monkeypatch, capsys):
    # WARNING fix (round 7): R16's stated bias is CONSERVATIVE-ON-UNCERTAINTY — any
    # unexpected/unrecognized OpenProcess failure must be treated as ALIVE (never silently
    # mapped to dead), same as the existing ACCESS_DENIED case, plus a one-per-process WARNING.
    # Exercised directly (not gated by sys.platform) so it runs on any host.
    import ctypes

    monkeypatch.setattr(run_lock, "_probe_warned", False)  # isolate from other tests

    class _FakeKernel32:
        def __init__(self):
            self.OpenProcess = lambda *_a, **_k: 0  # NULL handle → OpenProcess failed
            self.WaitForSingleObject = lambda *_a, **_k: 0
            self.CloseHandle = lambda *_a, **_k: None

    monkeypatch.setattr(ctypes, "WinDLL", lambda *_a, **_k: _FakeKernel32())
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 1450)  # unexpected/unknown code
    assert run_lock._is_pid_alive_windows(4321) is True  # conservative: alive
    assert "WARNING" in capsys.readouterr().err


def test_windows_invalid_parameter_error_is_dead(monkeypatch):
    # Confirms the fix did not weaken the KNOWN "no such process" case (ERROR_INVALID_PARAMETER
    # = 87): that specific code is still correctly mapped to dead, not swept into the new
    # conservative-unknown branch.
    import ctypes

    class _FakeKernel32:
        def __init__(self):
            self.OpenProcess = lambda *_a, **_k: 0
            self.WaitForSingleObject = lambda *_a, **_k: 0
            self.CloseHandle = lambda *_a, **_k: None

    monkeypatch.setattr(ctypes, "WinDLL", lambda *_a, **_k: _FakeKernel32())
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 87)  # ERROR_INVALID_PARAMETER
    assert run_lock._is_pid_alive_windows(4321) is False


def test_hostile_binary_lock_content_never_raises_and_is_treated_as_corrupt(tmp_path):
    # INFO->test (round 7): a hostile/garbage `.ollama-lock` (invalid UTF-8, non-numeric PID
    # line, missing lines, absurdly long lines) must never raise UnicodeDecodeError/ValueError/
    # any non-domain exception — it falls through to the mtime/not-live escape, same as any
    # other unparseable-PID lock in the decision table. Confirms `_parse_lock`'s
    # `errors="replace"` read is bytes-safe.
    lock = os.path.join(str(tmp_path), ".ollama-lock")
    hostile = (
        b"\xff\xfe\x00\x01not-a-pid\n"
        + b"\x80\x81\x82\xfe\xff" * 200  # invalid UTF-8, repeated
        + b"\nnotanumber\nmissing-third-line-garbage"
        + os.urandom(128)  # absurdly long trailing garbage line
    )
    with open(lock, "wb") as fh:
        fh.write(hostile)
    # Fresh dir + unparseable PID → mtime-freshness escape → live.
    assert is_dir_live(str(tmp_path)) is True
    old = time.time() - 3600 * 24
    os.utime(str(tmp_path), (old, old))
    # Aged past the 6h floor → not live. Either way: no exception ever escapes.
    assert is_dir_live(str(tmp_path)) is False


# --- Cross-process stdout token (R7d) — shared ephemeral-lock primitive ---


def _write_ephemeral(path, pid, bound, *, iso=None):
    """Fabricate a 3-line ephemeral lockfile (PID / ISO-8601 UTC / bound) for a test."""
    iso = iso or datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{pid}\n{iso}\n{bound}\n")


def test_stdout_token_is_exclusive_across_processes(tmp_path):
    tok = str(tmp_path / STDOUT_TOKEN_FILENAME)
    assert acquire_token(tok, timeout=60) is True  # holder wins → streams to stdout
    assert acquire_token(tok, timeout=60) is False  # concurrent process → file-only
    release_token(tok)
    assert acquire_token(tok, timeout=60) is True  # reclaimable once released


def test_stdout_token_uses_the_run_lock_3_line_format_and_short_bound(tmp_path):
    tok = str(tmp_path / STDOUT_TOKEN_FILENAME)
    acquire_token(tok, timeout=30)
    lines = open(tok, encoding="utf-8").read().splitlines()
    assert len(lines) == 3  # PID / ISO-8601 UTC / bound
    assert int(lines[0]) == os.getpid()
    datetime.fromisoformat(lines[1])  # line 2 parses as ISO-8601
    assert int(lines[2]) == staleness_bound_ephemeral(30) == 60  # 2*timeout, NO 6h floor


def test_stdout_token_reclaimed_from_dead_holder_with_ownership_reverify(tmp_path, monkeypatch):
    tok = str(tmp_path / STDOUT_TOKEN_FILENAME)
    _write_ephemeral(tok, pid=999_999, bound=120)  # a "dead" holder
    monkeypatch.setattr(run_lock, "is_pid_alive", lambda pid: pid == os.getpid())
    assert acquire_token(tok, timeout=60) is True  # stale token reclaimed
    assert int(open(tok, encoding="utf-8").readline()) == os.getpid()  # ownership re-verified


def test_live_holder_within_short_bound_is_never_reclaimed(tmp_path, monkeypatch):
    tok = str(tmp_path / STDOUT_TOKEN_FILENAME)
    _write_ephemeral(tok, pid=4321, bound=3600)  # fresh, within bound
    monkeypatch.setattr(run_lock, "is_pid_alive", lambda pid: True)  # holder alive
    assert acquire_token(tok, timeout=60) is False  # live PID inside bound → not stolen


def test_live_pid_past_the_ephemeral_bound_is_reclaimed_pid_recycling_safe(tmp_path, monkeypatch):
    """PID-recycling regression: for the SHORT ephemeral bound, staleness is
    AUTHORITATIVE even when the PID field currently belongs to a live process. A
    legitimate holder is a SINGLE delegation bounded by its own `2*timeout`, so a lock
    older than that can never still be that same legitimate holder — either it crashed
    (PID dead) or the OS recycled its PID into an unrelated process, which a
    liveness-only rule would report as "held" FOREVER (the bug this closes). Accepted
    trade-off (documented on `_lockfile_holder_is_live`): a genuinely SUSPENDED holder
    (e.g. laptop sleep) past its bound could have its token reclaimed here too — a brief
    over-commit — but this SELF-HEALS: on resume its own monotonic deadline (R25) is
    already exceeded, so it aborts and releases without ever completing as a second live
    holder of the same resource."""
    tok = str(tmp_path / STDOUT_TOKEN_FILENAME)
    _write_ephemeral(
        tok, pid=os.getpid(), bound=60, iso="2020-01-01T00:00:00+00:00"
    )  # ancient timestamp => age >> bound
    monkeypatch.setattr(run_lock, "is_pid_alive", lambda pid: pid == os.getpid())
    assert acquire_token(tok, timeout=30) is True  # bound-expired => reclaimed despite live PID


def test_run_dir_lock_keeps_a_distinct_longer_bound_policy_from_the_ephemeral_one(
    tmp_path, monkeypatch
):
    """Distinctness regression: `is_dir_live` (MS3, unchanged by MS7) applies the exact
    same "dead PID OR age >= bound" rule as `_lockfile_holder_is_live` above, but a RUN
    DIR's persisted bound carries the 6h floor (`staleness_bound_for_timeout`) while an
    ephemeral lock's bound is short (`staleness_bound_ephemeral`, no floor). The SAME
    5-minute age that would reclaim an ephemeral token/slot must NOT reclaim a run dir
    lock — pinning the two lock classes apart so a future edit can't accidentally
    collapse them onto the same (wrong-for-one-of-them) bound."""
    run_dir = str(tmp_path)
    write_lock(run_dir, max_age_seconds=staleness_bound_for_timeout(30))  # 6h-floored bound
    lock = os.path.join(run_dir, ".ollama-lock")
    lines = open(lock, encoding="utf-8").read().splitlines()
    backdated = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    open(lock, "w", encoding="utf-8").write(f"{lines[0]}\n{backdated}\n{lines[2]}\n")
    monkeypatch.setattr(run_lock, "is_pid_alive", lambda pid: pid == int(lines[0]))
    assert is_dir_live(run_dir) is True  # 5min age << 6h floor => still live
    # The SAME 5-minute age exceeds a typical ephemeral bound (2*30=60s here) and would be
    # reclaimed by `_lockfile_holder_is_live` instead — sanity-check the bounds differ.
    assert 300 >= staleness_bound_ephemeral(30)


def test_release_token_is_idempotent(tmp_path):
    tok = str(tmp_path / STDOUT_TOKEN_FILENAME)
    acquire_token(tok, timeout=60)
    release_token(tok)
    release_token(tok)  # no raise
    assert not os.path.exists(tok)


def test_acquire_token_degrades_to_false_on_transient_open_oserror(tmp_path, monkeypatch):
    # Total / never-raise (plan narrative + the ephemeral primitive's contract): a transient
    # OSError on the initial atomic create -- e.g. a Windows AV/indexer momentarily locking
    # the new file, or a read-only temp -- must degrade to "couldn't acquire" (return False,
    # so the caller falls back to a per-agent file sink), NEVER propagate and crash the
    # delegation. Only FileExistsError means "held by someone else, try to reclaim".
    tok = str(tmp_path / STDOUT_TOKEN_FILENAME)

    def _boom_open(*a, **k):
        raise PermissionError("transient AV lock on the new lockfile")

    monkeypatch.setattr(run_lock.os, "open", _boom_open)
    assert acquire_token(tok, timeout=60) is False


# --- Cross-process concurrency slot-counter (R21c, MS7 Task 3) ---


def test_acquire_slot_returns_first_free_index_then_none_when_full(tmp_path):
    slots = str(tmp_path / SLOTS_DIRNAME)
    assert acquire_slot(slots, max_parallel=3, timeout=60) == 0
    assert acquire_slot(slots, max_parallel=3, timeout=60) == 1
    assert acquire_slot(slots, max_parallel=3, timeout=60) == 2
    assert acquire_slot(slots, max_parallel=3, timeout=60) is None  # full → queue/reject


def test_two_processes_never_exceed_max_parallel_live_slots(tmp_path):
    slots = str(tmp_path / SLOTS_DIRNAME)
    held = [acquire_slot(slots, 2, 60) for _ in range(2)]  # two "processes"
    assert sorted(held) == [0, 1]
    assert acquire_slot(slots, 2, 60) is None  # a 3rd is refused
    live = [f for f in os.listdir(slots) if f.startswith("slot-")]
    assert len(live) == 2  # never > max_parallel


def test_release_slot_frees_it_for_reuse(tmp_path):
    slots = str(tmp_path / SLOTS_DIRNAME)
    i = acquire_slot(slots, 2, 60)
    acquire_slot(slots, 2, 60)
    assert acquire_slot(slots, 2, 60) is None  # full
    release_slot(slots, i)
    assert acquire_slot(slots, 2, 60) == i  # freed slot reusable


def test_dead_slot_is_reclaimed_not_skipped(tmp_path, monkeypatch):
    slots = str(tmp_path / SLOTS_DIRNAME)
    os.makedirs(slots)
    _write_ephemeral(os.path.join(slots, "slot-0.lock"), pid=999_999, bound=120)
    monkeypatch.setattr(run_lock, "is_pid_alive", lambda pid: pid == os.getpid())
    assert acquire_slot(slots, 2, 60) == 0  # dead slot-0 reclaimed
    assert int(open(os.path.join(slots, "slot-0.lock")).readline()) == os.getpid()


def test_orphaned_dead_slot_file_outside_probe_range_is_cleaned_up_on_acquire(
    tmp_path, monkeypatch
):
    """Hygiene regression: a slot file left behind by a PREVIOUS, larger
    `max_parallel_agents` (e.g. `slot-5.lock`) sits OUTSIDE the current probe range
    (`0..max_parallel-1`) and would otherwise never be touched again — it must still be
    swept and removed once its holder is reclaimable, not accumulate forever under
    `.ollama-slots/`."""
    slots = str(tmp_path / SLOTS_DIRNAME)
    os.makedirs(slots)
    orphan = os.path.join(slots, "slot-5.lock")
    _write_ephemeral(orphan, pid=999_999, bound=60)  # dead PID, index outside range(0, 2)
    monkeypatch.setattr(run_lock, "is_pid_alive", lambda pid: pid == os.getpid())
    acquire_slot(slots, max_parallel=2, timeout=60)  # probes only slot-0/slot-1
    assert not os.path.exists(orphan)  # swept anyway, not left behind


def test_cleanup_leaves_in_range_slots_to_atomic_reclaim_sweeps_only_out_of_range(
    tmp_path, monkeypatch
):
    """R21c TOCTOU guard: the hygiene sweep must NOT remove an IN-RANGE slot file, even a
    reclaimable-looking one. An in-range slot is owned by `acquire_slot`'s atomic O_EXCL
    probe/reclaim; removing it here can race a concurrent acquirer that just swapped in a
    fresh file at the same path (A reads the old dead PID, B reclaims, A's os.remove kills
    B's live file) -> a freed-but-held slot -> over-subscription by one. Only OUT-OF-RANGE
    orphans (index >= max_parallel, left by a previous larger cap) -- which no current
    process ever probes -- are swept here; in-range dead files are reclaimed atomically by
    the probe loop instead."""
    slots = str(tmp_path / SLOTS_DIRNAME)
    os.makedirs(slots)
    in_range = os.path.join(slots, "slot-0.lock")
    out_of_range = os.path.join(slots, "slot-9.lock")
    _write_ephemeral(in_range, pid=999_999, bound=60)  # dead PID, index inside range(0, 3)
    _write_ephemeral(out_of_range, pid=999_999, bound=60)  # dead PID, index >= 3
    monkeypatch.setattr(run_lock, "is_pid_alive", lambda pid: pid == os.getpid())
    run_lock._cleanup_orphaned_slots(slots, max_parallel=3)
    assert os.path.exists(in_range)  # in-range: NOT swept, left to atomic reclaim
    assert not os.path.exists(out_of_range)  # out-of-range orphan: swept


def test_acquire_ephemeral_does_not_evict_a_live_lock_swapped_in_during_reclaim(
    tmp_path, monkeypatch
):
    """R7d/R21c mutual exclusion: if a competing reclaimer swaps a FRESH LIVE lock into the
    path in the window between this reclaim's liveness check and its claim, _acquire_ephemeral
    must NOT evict that live holder. It must detect the live lock (by inspecting what it
    atomically claimed) and back off (return False), leaving the competitor's lock intact --
    never a plain check-then-remove that deletes the live file and produces two owners."""
    path = str(tmp_path / "ephemeral.lock")
    b_pid = 424242  # a distinct "competitor" PID we mark alive
    _write_ephemeral(path, pid=999_999, bound=60)  # stale holder this caller wants to reclaim
    monkeypatch.setattr(run_lock, "is_pid_alive", lambda pid: pid == b_pid)  # only B is alive
    real_is_live = run_lock._lockfile_holder_is_live
    calls = {"n": 0}

    def _swap_live_lock_then_report_dead(p):
        calls["n"] += 1
        if calls["n"] == 1 and p == path:
            # Simulate competitor B finishing its reclaim: a FRESH LIVE lock now sits at `path`.
            # Our caller's first check still saw the ORIGINAL stale (dead) holder.
            _write_ephemeral(path, pid=b_pid, bound=60)
            return False
        return real_is_live(p)

    monkeypatch.setattr(run_lock, "_lockfile_holder_is_live", _swap_live_lock_then_report_dead)
    result = run_lock._acquire_ephemeral(path, bound=60)
    assert result is False  # backed off; did not steal B's live lock
    pid, _age, _bound = run_lock._read_lock_fields(path)
    assert pid == b_pid  # B's live lock is intact at `path`, not evicted -> no double ownership


def test_read_lock_fields_clamps_negative_age_from_a_future_timestamp(tmp_path):
    """Clock-skew guard (Caspar residual): a lock whose ISO timestamp is in the FUTURE (the
    wall clock stepped backward after it was written) must not yield a NEGATIVE age -- which
    would make a bound-expiry check (`age >= bound`) impossible to satisfy and could hold a
    recycled-PID lock indefinitely. `_read_lock_fields` clamps age to >= 0."""
    from datetime import datetime, timedelta, timezone

    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    path = str(tmp_path / "future.lock")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{os.getpid()}\n{future}\n60\n")
    pid, age, bound = run_lock._read_lock_fields(path)
    assert pid == os.getpid()
    assert age is not None and age >= 0.0  # clamped, never negative despite the future stamp
    assert bound == 60


def test_acquire_ephemeral_writes_full_payload_even_on_a_short_os_write(tmp_path, monkeypatch):
    """os.write may perform a SHORT write (fewer bytes than requested); the lock payload must
    be written in FULL (looped) so the lockfile never lands with a truncated/torn payload that
    would misparse (wrong PID/bound) -- which could wrongly report a live holder as stale."""
    path = str(tmp_path / "eph.lock")
    real_write = run_lock.os.write
    state = {"short_done": False}

    def _short_first_write(fd, data):
        if not state["short_done"]:
            state["short_done"] = True
            return real_write(fd, data[:1])  # only 1 byte on the first call
        return real_write(fd, data)

    monkeypatch.setattr(run_lock.os, "write", _short_first_write)
    assert run_lock._acquire_ephemeral(path, bound=60) is True
    pid, _age, bound = run_lock._read_lock_fields(path)
    assert pid == os.getpid() and bound == 60  # the whole 3-line payload landed


def test_acquire_ephemeral_closes_fd_when_write_fails_no_descriptor_leak(tmp_path, monkeypatch):
    """No FD leak: if `os.write` raises after a successful `O_EXCL` open, `_acquire_ephemeral`
    must still close the descriptor (try/finally) before returning False -- otherwise the fd
    leaks until GC, and under repeated transient write failures a process could exhaust its
    descriptor table. Exercises the first (fresh-create) write path."""
    path = str(tmp_path / "ephemeral.lock")
    closed: list[int] = []
    real_close = run_lock.os.close

    def _raising_write(fd, data):
        raise OSError("simulated write failure after open")

    def _recording_close(fd):
        closed.append(fd)
        return real_close(fd)  # actually close it so the test leaks nothing either

    monkeypatch.setattr(run_lock.os, "write", _raising_write)
    monkeypatch.setattr(run_lock.os, "close", _recording_close)
    result = run_lock._acquire_ephemeral(path, bound=60)
    assert result is False  # write failed -> could not acquire the token
    assert len(closed) == 1  # the opened fd was closed exactly once, not leaked


def test_release_slot_is_idempotent(tmp_path):
    slots = str(tmp_path / SLOTS_DIRNAME)
    acquire_slot(slots, 1, 60)
    release_slot(slots, 0)
    release_slot(slots, 0)  # no raise


def test_acquire_slot_never_exceeds_max_parallel_under_real_thread_contention(tmp_path):
    # CRITICAL (R21c) under GENUINE thread concurrency: N real threads racing to acquire_slot
    # against the SAME slots dir must never hold more than max_parallel DISTINCT live slots --
    # the O_EXCL create + TOCTOU-safe reclaim guarantees at most one winner per index. (The
    # sequential "two processes" test monkeypatches the liveness probe; this exercises the
    # real primitive under real contention, mirroring run_batch's asyncio.to_thread fan-out.)
    import threading

    slots_dir = str(tmp_path / SLOTS_DIRNAME)
    max_parallel = 4
    n_threads = 16
    barrier = threading.Barrier(n_threads)
    acquired: list[int] = []
    lock = threading.Lock()

    def _worker() -> None:
        barrier.wait()  # release all threads at once to maximize contention
        idx = acquire_slot(slots_dir, max_parallel, timeout=30)
        if idx is not None:
            with lock:
                acquired.append(idx)

    threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # None released -> exactly max_parallel win, each a DISTINCT index (never two on one slot);
    # the other 12 threads get None (fail-closed), never a duplicate acquisition.
    assert len(acquired) == max_parallel
    assert len(set(acquired)) == max_parallel  # all distinct -- no double-acquire of an index
    assert all(0 <= i < max_parallel for i in acquired)


def test_live_pid_with_corrupt_bound_and_old_mtime_is_reclaimable_not_immortal(
    tmp_path, monkeypatch
):
    # An ephemeral lock whose PID line parses (and that PID is ALIVE -- e.g. the OS recycled
    # it to an unrelated live process) but whose BOUND is corrupt/missing (a torn write that
    # landed past the first line) must NEVER become IMMORTAL: the persisted bound can't be
    # applied, so a purely liveness-based rule would report it "held" forever. Fall back to
    # the file's mtime -- an OLD such lock self-heals to reclaimable.
    import os
    import time

    import run_lock
    from run_lock import _lockfile_holder_is_live

    lock = str(tmp_path / "slot-0.lock")
    with open(lock, "w", encoding="utf-8") as fh:
        fh.write(
            f"{os.getpid()}\n2026-07-10T00:00:00+00:00\nnotanumber\n"
        )  # PID+ISO ok, bound corrupt
    old = time.time() - 3600  # 1h old, far past the 2s torn-write grace
    os.utime(lock, (old, old))
    monkeypatch.setattr(run_lock, "is_pid_alive", lambda pid: True)  # PID recycled -> "alive"
    assert _lockfile_holder_is_live(lock) is False  # reclaimable, NOT immortal
