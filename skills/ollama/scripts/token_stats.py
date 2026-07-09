# skills/ollama/scripts/token_stats.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Local per-capability/model token accounting (separate from Claude/Anthropic usage)."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import threading
import time
from typing import Any

from backend import DelegationResult
from errors import ValidationError
from ollama_config import CAPABILITIES

_STATS_FILENAME = "token_stats.json"
# Windows-only: os.replace (MoveFileEx with MOVEFILE_REPLACE_EXISTING) is NOT safe under
# concurrent contention onto the SAME destination path — unlike POSIX rename(), a concurrent
# replacer can transiently fail with WinError 5 (ERROR_ACCESS_DENIED → PermissionError)
# instead of atomically succeeding. Retry a bounded number of times with a small escalating
# backoff so concurrent writers all eventually land their atomic replace. On POSIX the first
# attempt always wins, so the loop is a single no-op iteration there. (NR6: Windows is the
# documented dev floor; token_stats.json is a SHARED destination — unlike MS3's per-run-dir
# .ollama-lock, whose unique path is never contended.)
_REPLACE_MAX_RETRIES = 10
_REPLACE_BACKOFF_SECONDS = 0.003


class TokenStats:
    """Accumulate token metrics per (capability, model).

    Thread-safe: a lock guards mutation and snapshotting so the concurrent
    delegation fan-out landing in MS5 never loses an update, and ``write()``
    never observes a torn/partial bucket (see ``write()`` docstring). A
    structured retry issues two backend calls for one logical delegation — both
    calls' tokens are accounted (correct billing) via ``http_calls``, while
    ``delegations`` counts the logical delegation once.

    ``http_calls`` semantics (decided for R7a/R12 — local request/token
    accounting): it counts backend calls that **completed** and produced a
    ``DelegationResult`` — i.e., every call reaching ``record()``. A connection
    failure/timeout/5xx raises *before* any ``DelegationResult`` exists, so
    there is nothing to pass to ``record()`` and that attempt is NOT counted
    here. This is deliberate, not an oversight: tracking failed attempts
    requires a distinct signal (an exception, not token/elapsed metrics) and is
    the per-model circuit breaker's job (R14b, MS5), not this token/cost
    accumulator's. So read ``http_calls`` as "completed attempts billed", not
    "raw connection attempts made".
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def record(
        self,
        capability: str,
        model: str,
        result: DelegationResult,
        *,
        counts_as_delegation: bool = True,
    ) -> None:
        """Fold one backend call's metrics into the running totals (thread-safe).

        Args:
            capability: The capability name. Must be one of the seven known
                capabilities (``ollama_config.CAPABILITIES``).
            model: The resolved model tag. Must be a non-empty, non-blank string.
            result: The backend-call outcome carrying token metrics. Its
                ``prompt_tokens``/``completion_tokens`` must be non-negative ints,
                and NOT ``bool`` — ``bool`` is technically an ``int`` subclass in
                Python (``isinstance(True, int)`` is ``True``) but is never a valid
                token count here (Task 1's ``_resolve_usage``/``_coerce_token_count``
                already guarantee this for values coming off the wire, but
                ``record`` validates independently — it must never silently
                corrupt the accounting on a bad caller).
            counts_as_delegation: True for the first/only attempt of a delegation,
                False for a retry attempt (so ``http_calls`` counts every completed
                backend call while ``delegations`` counts the logical delegation
                once). Only ever called for a call that COMPLETED (returned a
                ``DelegationResult``) — a call that raised before producing one is
                never recorded (see the class docstring's ``http_calls`` note).

        Raises:
            ValidationError: if ``capability`` is not one of the seven known
                capabilities, ``model`` is empty/blank, either token count on
                ``result`` is a ``bool``, non-``int``, or negative, or
                ``result.elapsed_s`` is a ``bool``, non-numeric, non-finite
                (``NaN``/``inf``), or negative. Fail-closed: reject bad input
                rather than silently corrupt the accounting (NR8).
        """
        if capability not in CAPABILITIES:
            raise ValidationError(
                f"unknown capability {capability!r}; expected one of {CAPABILITIES}"
            )
        if not isinstance(model, str) or not model.strip():
            raise ValidationError(f"model must be a non-empty string, got {model!r}")
        # bool MUST be checked before the int check: bool is an int subclass, so
        # isinstance(True, int) is True and a bare int check would silently accept
        # True/False as a token count of 1/0 and corrupt the accounting.
        if (
            isinstance(result.prompt_tokens, bool)
            or not isinstance(result.prompt_tokens, int)
            or result.prompt_tokens < 0
        ):
            raise ValidationError(
                f"prompt_tokens must be a non-negative int, got {result.prompt_tokens!r}"
            )
        if (
            isinstance(result.completion_tokens, bool)
            or not isinstance(result.completion_tokens, int)
            or result.completion_tokens < 0
        ):
            raise ValidationError(
                f"completion_tokens must be a non-negative int, got {result.completion_tokens!r}"
            )
        # elapsed_s feeds the aggregate tok_per_s denominator: a NaN/inf value
        # poisons that division (and NaN is not valid JSON, so write() would
        # emit a broken token_stats.json); a negative value is nonsensical.
        # bool is (again) an int/float-compatible subclass and must be
        # rejected explicitly, same discipline as the token-count checks.
        if (
            isinstance(result.elapsed_s, bool)
            or not isinstance(result.elapsed_s, (int, float))
            or not math.isfinite(result.elapsed_s)
            or result.elapsed_s < 0
        ):
            raise ValidationError(
                f"elapsed_s must be a finite, non-negative number, got {result.elapsed_s!r}"
            )
        with self._lock:
            bucket = self._data.setdefault(capability, {}).setdefault(
                model,
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "http_calls": 0,
                    "delegations": 0,
                    "estimated_calls": 0,
                    "_elapsed": 0.0,
                },
            )
            bucket["prompt_tokens"] += result.prompt_tokens
            bucket["completion_tokens"] += result.completion_tokens
            bucket["http_calls"] += 1
            bucket["delegations"] += 1 if counts_as_delegation else 0
            bucket["estimated_calls"] += 1 if result.estimated else 0
            bucket["_elapsed"] += result.elapsed_s

    def to_dict(self) -> dict[str, Any]:
        """Return a CONSISTENT snapshot with aggregate ``tok_per_s`` per bucket.

        Thread-safe: the entire snapshot (every field of every bucket) is
        copied out while holding ``self._lock``, so a concurrent ``record()``
        can never be interleaved with a partially-built snapshot — the caller
        always gets either the state before or after any given ``record()``
        call, never a torn mix of the two. The lock is released before this
        method returns, so callers (notably ``write()``) never hold it during
        any I/O.

        ``http_calls`` counts only backend calls that completed and were
        recorded (see the class docstring) — not raw connection attempts.

        ``tok_per_s`` is an aggregate END-TO-END **delivered-tokens-per-second**
        metric = total completion tokens / total end-to-end wall-clock elapsed
        across the bucket's calls — that elapsed time includes network latency,
        429 backoff waiting, and queueing, so this is NOT a measure of raw model
        generation/decode speed (see ``DelegationResult.tok_per_s``, Task 1).
        """
        with self._lock:
            out: dict[str, Any] = {}
            for cap, models in self._data.items():
                out[cap] = {}
                for model, b in models.items():
                    tps = round(b["completion_tokens"] / b["_elapsed"], 4) if b["_elapsed"] else 0.0
                    out[cap][model] = {
                        "prompt_tokens": b["prompt_tokens"],
                        "completion_tokens": b["completion_tokens"],
                        "http_calls": b["http_calls"],
                        "delegations": b["delegations"],
                        "estimated_calls": b["estimated_calls"],
                        "tok_per_s": tps,
                    }
            return out

    def write(self, output_dir: str) -> str | None:
        """Atomically write ``token_stats.json`` into *output_dir*; return its path.

        Sequence: (1) take a CONSISTENT SNAPSHOT via ``to_dict()`` — this
        acquires ``self._lock`` just long enough to copy out every bucket's
        fields, then releases it, BEFORE any file is opened — so a concurrent
        ``record()`` can never race the serialization/I/O below, and this
        method never observes (or writes out) a torn/partial bucket; (2)
        serialize the snapshot to a **per-call unique** tmp file created with
        ``tempfile.mkstemp(dir=output_dir, ...)`` — never a fixed/shared tmp
        name, so two concurrent ``write()`` calls never clobber each other's
        in-progress tmp file; (3) `fsync` the tmp file's descriptor to push it
        out of the OS page cache and onto durable storage; (4) ``os.replace``
        it into place — a filesystem-atomic rename (same pattern MS3 uses for
        ``.ollama-lock``). On Windows this rename is retried with a bounded
        backoff on ``PermissionError`` (WinError 5), because — unlike POSIX
        ``rename()`` — MoveFileEx onto a SHARED destination is not safe under
        concurrent contention and one racing replacer can transiently lose. A reader can
        therefore never observe a partial/corrupt file: either the final file
        is complete valid JSON, or (on any failure) the previous final file —
        or none at all — is left untouched. The `fsync` closes the gap where
        `os.replace` alone gives atomic *visibility* but not *durability* — an
        OS crash/power-loss between a bare `replace` and the page cache
        flushing to disk could still lose the write; `fsync`-ing the tmp file
        before the rename means the data is already on disk by the time the
        rename is visible.

        Best-effort (R12/NR8): an unwritable dir, a failed `mkstemp`, a failed
        `fsync`, or a failed ``os.replace``, warns to stderr and returns
        ``None`` instead of crashing the run — `fsync` is wrapped in the SAME
        `OSError` guard as the rest of this method (some filesystems/platforms
        may not support it) so a best-effort durability step never turns into a
        hard failure; the leftover unique ``.tmp`` file (if any) is best-effort
        removed. MS3 migrates the default location from cwd into the managed
        run directory.

        Args:
            output_dir: The directory to write ``token_stats.json`` into.

        Returns:
            The full path to the written file, or ``None`` on any I/O failure.
        """
        path = os.path.join(output_dir, _STATS_FILENAME)
        # Snapshot FIRST, outside any tmp-file/I/O machinery: to_dict() holds
        # self._lock only for the duration of the copy, so the lock is never
        # held across file creation, fsync, or the rename below.
        snapshot = self.to_dict()
        tmp_path: str | None = None
        try:
            # A unique tmp path per call (random suffix, via mkstemp) — never
            # a fixed shared name — so concurrent write() calls cannot
            # collide on the same in-progress tmp file.
            fd, tmp_path = tempfile.mkstemp(
                dir=output_dir, prefix=f".{_STATS_FILENAME}.", suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(snapshot, fh, indent=2)
                fh.flush()
                # Best-effort durability: push the tmp file's content out of the
                # OS page cache and onto disk BEFORE the atomic rename, so a
                # crash right after `os.replace` cannot lose the write. Guarded
                # by the same `except OSError` below — never raises on its own.
                os.fsync(fh.fileno())
            # Windows-safe atomic replace: retry on PermissionError (WinError 5) under
            # concurrent contention onto the shared destination; a non-Permission OSError
            # (e.g. disk full) is NOT retried — it falls through to the best-effort
            # `except OSError` below → None. On POSIX the first attempt always succeeds.
            for attempt in range(_REPLACE_MAX_RETRIES):
                try:
                    os.replace(tmp_path, path)
                    break
                except PermissionError:
                    if attempt == _REPLACE_MAX_RETRIES - 1:
                        raise
                    time.sleep(_REPLACE_BACKOFF_SECONDS * (attempt + 1))
        except (OSError, ValueError) as exc:
            # ValueError (not an OSError subclass) is raised by tempfile.mkstemp/os.open when
            # the path contains an embedded NUL byte; catch it too so write() honors its
            # never-raise, best-effort contract instead of letting it escape.
            print(f"WARNING: could not write {path}: {exc}", file=sys.stderr)
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            return None
        return path
