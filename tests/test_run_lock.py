# tests/test_run_lock.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Process-liveness locking: write/parse, is_pid_alive, is_dir_live decision table."""

import os
import time

import run_lock
from run_lock import (
    is_dir_live,
    is_pid_alive,
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
