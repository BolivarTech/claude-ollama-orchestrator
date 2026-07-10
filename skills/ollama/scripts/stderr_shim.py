# skills/ollama/scripts/stderr_shim.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Capture stderr for persistence (R18), live or buffered depending on the display.

Per-delegation capture (MS5, R7c) is implemented via ``contextvars`` rather than a
process-global ``sys.stderr`` swap: a single dispatching proxy (``_DispatchingStderr``)
is installed once and routes each write to the CURRENT asyncio task's own capture
buffer. Because ``asyncio`` copies the context for every task, concurrent delegations
each see their own buffer/tee state with no shared mutable state to race on.
"""

from __future__ import annotations

import contextlib
import contextvars
import io
import sys
from collections.abc import Iterable, Iterator
from typing import Any, TextIO


_current_capture: contextvars.ContextVar[io.StringIO | None] = contextvars.ContextVar(
    "_ollama_stderr_capture", default=None
)
_current_tee: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_ollama_stderr_tee", default=False
)


class _DispatchingStderr:
    """A ``sys.stderr`` proxy: routes each write to the current task's capture buffer (a
    ``ContextVar``) if one is active, else to the real stderr. When the active capture also
    has ``tee`` set, every write reaches BOTH the buffer and the real stderr (reproduces
    MS3's ``active=False`` live-tee mode; buffer-only reproduces MS3's ``active=True`` mode).

    Because asyncio tasks copy the context, concurrent delegations each see their own
    capture buffer via the ``ContextVar`` — no shared global mutation, no cross-contamination,
    even though this proxy object itself is a single process-global ``sys.stderr``.

    Only ``write``/``writelines``/``flush`` are overridden; every OTHER attribute (``isatty``,
    ``encoding``, ``errors``, ``fileno``, ``reconfigure``, ``buffer``, ...) is forwarded to the
    real stderr via ``__getattr__`` (CRITICAL fix) — without it, :class:`StatusDisplay`'s
    TTY/encoding probes (R20) and the Windows console hardening's
    ``sys.stderr.reconfigure(...)`` (R26) would raise ``AttributeError`` the instant this
    proxy is installed.
    """

    def __init__(self, real: TextIO) -> None:
        self._real = real

    def write(self, s: str) -> int:
        buf = _current_capture.get()
        if buf is None:
            return self._real.write(s)
        if _current_tee.get():
            self._real.write(s)
        return buf.write(s)

    def writelines(self, lines: Iterable[str]) -> None:
        """Route each line through :meth:`write` — never bypass the tee/capture path."""
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        buf = _current_capture.get()
        if buf is not None:
            buf.flush()
        if buf is None or _current_tee.get():
            self._real.flush()

    @property
    def real(self) -> TextIO:
        """The underlying real stderr (used by the status display, R20/MS5 INFO fix #7)."""
        return self._real

    def __getattr__(self, name: str) -> Any:
        # Proxy every non-overridden attribute to the real stderr so the stream contract
        # (fileno/buffer/encoding/isatty/reconfigure/...) is honored regardless of whether a
        # capture is currently active. Any is idiomatic for a dynamic passthrough.
        return getattr(self._real, name)


def _ensure_dispatching_stderr_installed() -> None:
    """Lazily install the dispatching stderr proxy, idempotently, with NO restore.

    Private, one-way helper backing the legacy single-delegation path
    (``capture_stderr_for_delegation`` calls this internally, so MS3's call site,
    ``run_delegation`` -> ``buffered_stderr_while``, which never wraps itself in
    :func:`install_dispatching_stderr`, keeps working unchanged). Appropriate ONLY because a
    long-lived CLI process never "uninstalling" a proxy that transparently forwards to the
    real stream when no capture ``ContextVar`` is set is harmless. Batch fan-out (Task 5)
    does NOT rely on this — it uses the guaranteed install/uninstall
    :func:`install_dispatching_stderr` context manager instead.
    """
    if not isinstance(sys.stderr, _DispatchingStderr):
        sys.stderr = _DispatchingStderr(sys.stderr)


@contextlib.contextmanager
def install_dispatching_stderr() -> Iterator[None]:
    """Install the dispatching stderr proxy for this block, ALWAYS restoring the prior
    ``sys.stderr`` on exit — success, exception, OR cancellation (WARNING fix, MS5 gate).

    A process-global ``sys.stderr`` proxy installed with no guaranteed uninstall would leak
    across runs/tests/multi-plugin environments. Callers that need the whole-batch invariant
    (``run_batch``, Task 5) wrap their ENTIRE fan-out in this context manager ONCE, at the top
    of the batch — never per-delegation: tearing the proxy down around each individual
    delegation would pull it out from under sibling delegations still relying on it mid-flight.

    Nests correctly: if ``sys.stderr`` is already a :class:`_DispatchingStderr` on entry (e.g.
    a nested call, or a prior lazy install via :func:`_ensure_dispatching_stderr_installed`),
    this wraps that proxy's REAL stream rather than double-wrapping, and restores the
    immediately-prior ``sys.stderr`` (proxy or not) on exit — ordinary context-manager
    stacking.
    """
    original = sys.stderr
    real = original.real if isinstance(original, _DispatchingStderr) else original
    sys.stderr = _DispatchingStderr(real)
    try:
        yield
    finally:
        sys.stderr = original


@contextlib.contextmanager
def capture_stderr_for_delegation(*, tee: bool = False) -> Iterator[io.StringIO]:
    """Route the current task's stderr to its own buffer for the block's duration.

    Args:
        tee: If True, every write ALSO reaches the real stderr immediately (live
            visibility), in addition to the buffer — used to reproduce MS3's
            ``active=False`` mode. If False (default), writes land ONLY in the buffer.

    Yields:
        The capture buffer so the caller can persist it to ``{cap}.stderr.log``. Isolated
        per asyncio task via ``ContextVar``\\ s: safe under concurrent fan-out (each task's
        ``tee``/buffer pair is independent, even though the proxy itself is a single
        process-global object).
    """
    _ensure_dispatching_stderr_installed()
    buf = io.StringIO()
    cap_token = _current_capture.set(buf)
    tee_token = _current_tee.set(tee)
    try:
        yield buf
    finally:
        _current_capture.reset(cap_token)
        _current_tee.reset(tee_token)


@contextlib.contextmanager
def buffered_stderr_while(active: bool) -> Iterator[io.StringIO]:
    """Capture ``sys.stderr`` for the duration, always yielding the capture buffer.

    R18 requires ``{cap}.stderr.log`` to be written whether or not the live status display
    (R20) is active, so this always captures. Only the delivery to the *real* stderr differs:

    - ``active=True`` (a live :class:`StatusDisplay` owns the terminal): stderr is buffered
      ONLY — writes never touch the real stream during the block, protecting the display's
      in-place ANSI redraws — then flushed to the real stderr once, on exit.
    - ``active=False`` (``--no-status``): stderr is tee'd — every write is visible on the
      real stderr immediately AND captured, so diagnostics are live *and* persisted.

    MS5 re-expresses this on top of :func:`capture_stderr_for_delegation` (per-delegation,
    ``contextvars``-based capture) instead of a bespoke global ``sys.stderr`` swap, so it
    stays safe if a caller ever runs it from within a concurrent fan-out — but its two-mode
    contract for the single-delegation caller (``run_delegation``) is unchanged.

    Args:
        active: Whether a live display owns the terminal (buffer) or not (tee).

    Yields:
        The capture buffer (an ``io.StringIO``), never ``None``.

    Note:
        The exit-path flush (``active=True``) is **best-effort**: if the real stderr is
        broken/closed at that point, the flush is silently skipped rather than raising —
        an exception escaping here could otherwise be mistaken for the delegation itself
        having failed, even though the delegation already completed successfully.
    """
    # WARNING fix (#5): resolve the REAL stderr target BEFORE entering the capture context,
    # not after. Under concurrent fan-out, another task can install/touch `sys.stderr` while
    # THIS block runs; reading it only after the `with` exits risks resolving against
    # whatever `sys.stderr` happens to be at that LATER moment (a different task's proxy
    # state) instead of the target that was live when this delegation started — a cross-task
    # flush landing in the wrong place. Pinning it up front makes the final flush target
    # independent of anything another task does to `sys.stderr` while this one is in flight.
    real = sys.stderr.real if isinstance(sys.stderr, _DispatchingStderr) else sys.stderr
    with capture_stderr_for_delegation(tee=not active) as buf:
        yield buf
    if active:
        # Buffer-only mode: nothing reached the real stream during the block — flush the
        # captured text to it exactly once, now that the block exited. Best-effort (INFO fix,
        # round 7): a broken/closed real stderr here must never mask a successful delegation
        # as failed — the captured text is only lost from this final echo-to-real-stderr,
        # callers persist `buf.getvalue()` to `{cap}.stderr.log` independently (R18).
        try:
            real.write(buf.getvalue())
            real.flush()
        except (OSError, ValueError):
            pass
