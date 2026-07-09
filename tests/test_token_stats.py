# tests/test_token_stats.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Local token accounting per capability/model, separate from Claude usage."""

import json
import os
import threading

import pytest

from backend import DelegationResult
from errors import ValidationError
from token_stats import TokenStats


def _res(content="x", p=10, c=5, est=False, elapsed=0.5):
    return DelegationResult(content, p, c, est, elapsed)


def test_record_rejects_unknown_capability():
    ts = TokenStats()
    with pytest.raises(ValidationError):
        ts.record("bogus-capability", "m", _res())


def test_record_rejects_empty_or_blank_model():
    ts = TokenStats()
    with pytest.raises(ValidationError):
        ts.record("coder", "", _res())
    with pytest.raises(ValidationError):
        ts.record("coder", "   ", _res())


def test_record_rejects_negative_token_counts():
    ts = TokenStats()
    with pytest.raises(ValidationError):
        ts.record("coder", "m", _res(p=-1, c=5))
    with pytest.raises(ValidationError):
        ts.record("coder", "m", _res(p=5, c=-1))


def test_record_rejects_bool_token_count():
    # bool is an int subclass in Python — isinstance(True, int) is True — so a
    # bare `isinstance(x, int)` check would silently accept True/False as a
    # token count of 1/0 and corrupt the accounting. Must be rejected explicitly.
    ts = TokenStats()
    with pytest.raises(ValidationError):
        ts.record("coder", "m", _res(p=True, c=5))
    with pytest.raises(ValidationError):
        ts.record("coder", "m", _res(p=5, c=False))


def test_record_rejects_non_finite_or_negative_elapsed_s():
    # record() validates token counts but must ALSO validate elapsed_s: a
    # NaN/inf/negative elapsed_s corrupts the aggregate tok_per_s and produces
    # invalid JSON (NaN is not valid JSON), so it must be rejected fail-closed,
    # same discipline as the token-count checks above.
    ts = TokenStats()
    with pytest.raises(ValidationError):
        ts.record("coder", "m", _res(elapsed=float("nan")))
    with pytest.raises(ValidationError):
        ts.record("coder", "m", _res(elapsed=float("inf")))
    with pytest.raises(ValidationError):
        ts.record("coder", "m", _res(elapsed=-1.0))
    with pytest.raises(ValidationError):
        # bool is an int subclass — must be rejected explicitly, same
        # discipline as the token-count bool checks above.
        ts.record("coder", "m", _res(elapsed=True))


def test_record_accepts_zero_and_positive_elapsed_s():
    ts = TokenStats()
    ts.record("coder", "m", _res(elapsed=0.0))
    ts.record("coder", "m", _res(elapsed=2.5))
    assert ts.to_dict()["coder"]["m"]["delegations"] == 2


def test_record_accumulates_per_capability_model():
    ts = TokenStats()
    ts.record("coder", "kimi-k2.7-code:cloud", _res(p=10, c=20, elapsed=1.0))
    ts.record("coder", "kimi-k2.7-code:cloud", _res(p=5, c=5, elapsed=1.0))
    d = ts.to_dict()["coder"]["kimi-k2.7-code:cloud"]
    assert d["prompt_tokens"] == 15
    assert d["completion_tokens"] == 25
    assert d["http_calls"] == 2
    assert d["delegations"] == 2


def test_estimated_calls_are_flagged():
    ts = TokenStats()
    ts.record("reviewer", "glm-5.2:cloud", _res(est=True))
    assert ts.to_dict()["reviewer"]["glm-5.2:cloud"]["estimated_calls"] == 1


def test_retry_counts_two_http_calls_but_one_delegation():
    ts = TokenStats()
    ts.record("reviewer", "glm-5.2:cloud", _res(p=10, c=5, elapsed=0.5))  # attempt 1
    ts.record(
        "reviewer", "glm-5.2:cloud", _res(p=10, c=5, elapsed=0.5), counts_as_delegation=False
    )  # retry
    d = ts.to_dict()["reviewer"]["glm-5.2:cloud"]
    assert d["http_calls"] == 2 and d["delegations"] == 1
    assert d["prompt_tokens"] == 20 and d["completion_tokens"] == 10  # BOTH attempts' tokens


def test_tok_per_s_is_aggregate_completion_over_elapsed():
    ts = TokenStats()
    ts.record("coder", "m", _res(c=30, elapsed=1.5))  # 30 / 1.5 = 20 tok/s
    assert ts.to_dict()["coder"]["m"]["tok_per_s"] == 20.0


def test_record_is_thread_safe_under_concurrency():
    ts = TokenStats()

    def worker():
        for _ in range(100):
            ts.record("coder", "m", _res(p=1, c=1, elapsed=0.01))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    d = ts.to_dict()["coder"]["m"]
    assert d["http_calls"] == 800  # no lost updates under 8-way concurrency
    assert d["prompt_tokens"] == 800 and d["completion_tokens"] == 800


def test_write_creates_token_stats_json(tmp_path):
    ts = TokenStats()
    ts.record("coder", "m", _res())
    path = ts.write(str(tmp_path))
    assert path == os.path.join(str(tmp_path), "token_stats.json")
    assert json.loads(open(path, encoding="utf-8").read())["coder"]["m"]["http_calls"] == 1


def test_write_returns_none_on_unwritable_dir(tmp_path):
    ts = TokenStats()
    ts.record("coder", "m", _res())
    # parent dir does not exist → open() fails → warn + return None, never crash
    assert ts.write(os.path.join(str(tmp_path), "does", "not", "exist")) is None


def test_write_final_file_is_always_complete_valid_json(tmp_path):
    ts = TokenStats()
    ts.record("coder", "m", _res())
    ts.record("reviewer", "m2", _res(p=1, c=2))
    path = ts.write(str(tmp_path))
    # The final file only ever exists via the atomic os.replace of a FULLY-written
    # tmp file, so it must always parse as complete, well-formed JSON.
    data = json.loads(open(path, encoding="utf-8").read())
    assert set(data) == {"coder", "reviewer"}


def test_write_leaves_no_partial_final_file_when_replace_fails(tmp_path, monkeypatch):
    ts = TokenStats()
    ts.record("coder", "m", _res())

    def _boom(*_args, **_kwargs):
        raise OSError("simulated failure during os.replace")

    monkeypatch.setattr(os, "replace", _boom)
    assert ts.write(str(tmp_path)) is None
    # A crash mid-write must never leave a partial/corrupt final file behind.
    assert not os.path.exists(os.path.join(str(tmp_path), "token_stats.json"))


def test_write_calls_fsync_for_durability(tmp_path, monkeypatch):
    # Durability: os.replace alone gives atomic VISIBILITY but not durability —
    # write() must attempt an fsync of the tmp file's descriptor so the data
    # survives an OS crash shortly after the write. This asserts the BEHAVIOR
    # (fsync is attempted at least once during a successful write) rather than
    # pinning the exact internal call ORDER relative to os.replace, which is
    # an implementation detail the test shouldn't couple to.
    fsync_calls: list[int] = []
    real_fsync = os.fsync

    def _tracked_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _tracked_fsync)
    ts = TokenStats()
    ts.record("coder", "m", _res())
    path = ts.write(str(tmp_path))
    assert path is not None
    assert len(fsync_calls) >= 1


def test_write_produces_internally_consistent_snapshot_under_concurrent_record(tmp_path):
    # Regression guard for a torn read: write() must snapshot the bucket
    # fields under the SAME lock record() mutates them with, so the JSON on
    # disk never observes a partial update (some fields reflecting N recorded
    # calls, others N-1). Each record() call here bumps prompt_tokens by 1 and
    # completion_tokens by exactly 2, so completion_tokens == prompt_tokens*2
    # and http_calls == delegations must hold in EVERY written snapshot, not
    # just in the final state after all threads finish.
    ts = TokenStats()
    stop = threading.Event()

    def recorder():
        while not stop.is_set():
            ts.record("coder", "m", _res(p=1, c=2, elapsed=0.01))

    snapshots = []
    recorders = [threading.Thread(target=recorder) for _ in range(4)]
    for t in recorders:
        t.start()
    try:
        for _ in range(50):
            path = ts.write(str(tmp_path))
            if path is not None:
                with open(path, encoding="utf-8") as fh:
                    snapshots.append(json.load(fh))
    finally:
        stop.set()
        for t in recorders:
            t.join()

    assert snapshots
    for snap in snapshots:
        bucket = snap.get("coder", {}).get("m")
        if bucket is None:
            continue  # a write that raced before the very first record() is fine
        assert bucket["completion_tokens"] == bucket["prompt_tokens"] * 2
        assert bucket["http_calls"] == bucket["delegations"]


def test_concurrent_writes_do_not_collide_on_tmp_path(tmp_path):
    # Two concurrent write() calls must not share the SAME .tmp filename — a
    # shared/fixed tmp path would let one writer's in-progress file get
    # clobbered/truncated by the other before either os.replace()s it. Using
    # tempfile.mkstemp() gives each call its own unique tmp path, so all
    # concurrent writers succeed and the final file is always complete.
    ts = TokenStats()
    ts.record("coder", "m", _res())
    results: list[str | None] = []
    results_lock = threading.Lock()

    def writer():
        path = ts.write(str(tmp_path))
        with results_lock:
            results.append(path)

    writers = [threading.Thread(target=writer) for _ in range(8)]
    for t in writers:
        t.start()
    for t in writers:
        t.join()

    assert all(p is not None for p in results)
    final = os.path.join(str(tmp_path), "token_stats.json")
    data = json.loads(open(final, encoding="utf-8").read())
    assert data["coder"]["m"]["http_calls"] == 1


def test_write_returns_none_when_fsync_fails_best_effort(tmp_path, monkeypatch):
    # fsync is best-effort: a platform/filesystem that raises on fsync must
    # warn-and-return-None via the SAME OSError guard as the rest of write(),
    # never propagate, and never leave a partial final file.
    def _boom(_fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(os, "fsync", _boom)
    ts = TokenStats()
    ts.record("coder", "m", _res())
    assert ts.write(str(tmp_path)) is None
    assert not os.path.exists(os.path.join(str(tmp_path), "token_stats.json"))
