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
    STDOUT_TOKEN_FILENAME,
    acquire_token,
    is_dir_live,
    is_pid_alive,
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
