# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Per-project temp namespace, unique run dirs, LRU cleanup excluding live dirs."""

import os
import tempfile

import temp_dirs
from run_lock import write_lock
from temp_dirs import OLLAMA_DIR_PREFIX, cleanup_old_runs, create_output_dir, project_run_root


def test_project_run_root_is_stable_and_hashed(tmp_path):
    a = project_run_root(str(tmp_path))
    b = project_run_root(str(tmp_path))
    assert a == b  # deterministic
    assert os.path.isdir(a)
    assert "ollama-runs" in a


def test_create_output_dir_is_unique(tmp_path):
    d1 = create_output_dir(None, str(tmp_path))
    d2 = create_output_dir(None, str(tmp_path))
    assert d1 != d2
    assert os.path.basename(d1).startswith(OLLAMA_DIR_PREFIX)


def test_cleanup_keeps_newest_n_and_removes_older(tmp_path, monkeypatch):
    monkeypatch.setattr(temp_dirs, "is_dir_live", lambda d: False)  # none live
    dirs = [create_output_dir(None, str(tmp_path)) for _ in range(4)]
    for i, d in enumerate(dirs):
        os.utime(d, (i, i))  # ascending mtime
    cleanup_old_runs(2, str(tmp_path))
    survivors = [d for d in dirs if os.path.exists(d)]
    assert set(survivors) == set(dirs[-2:])  # newest 2 kept


def test_cleanup_never_removes_a_live_dir(tmp_path, monkeypatch):
    live = create_output_dir(None, str(tmp_path))
    write_lock(live)
    monkeypatch.setattr(temp_dirs, "is_dir_live", lambda d: d == live)
    others = [create_output_dir(None, str(tmp_path)) for _ in range(3)]
    cleanup_old_runs(0, str(tmp_path))  # remove all non-live
    assert os.path.exists(live)  # live excluded
    assert not any(os.path.exists(d) for d in others)


def test_project_run_root_falls_back_to_gettempdir_on_makedirs_failure(monkeypatch, capsys):
    # Container unwritable (permissions / read-only temp) → degrade to the shared
    # gettempdir() with a warning, rather than raising (R15 best-effort fallback).
    # Task 5's `test_gettempdir_fallback_skips_cross_project_cleanup` covers the
    # consequence (cleanup skipped) once `run_ollama` observes this fallback.
    def _raise(*_a, **_k):
        raise OSError("permission denied (simulated)")

    monkeypatch.setattr(os, "makedirs", _raise)
    result = project_run_root("/some/project/root")
    assert result == tempfile.gettempdir()
    assert "WARNING" in capsys.readouterr().err


def test_cleanup_old_runs_refuses_shared_gettempdir_at_runtime(capsys):
    # Runtime enforcement of the "never call with gettempdir()" precondition (R15/R17),
    # not just a comment: even if a caller passes the shared temp root directly, cleanup
    # must no-op + warn rather than scanning/pruning OTHER projects' ollama-run-* dirs.
    cleanup_old_runs(5, tempfile.gettempdir())
    assert "WARNING" in capsys.readouterr().err
