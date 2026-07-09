# skills/ollama/scripts/stderr_shim.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Capture stderr for persistence (R18), live or buffered depending on the display."""

from __future__ import annotations

import contextlib
import io
import sys
from collections.abc import Iterable, Iterator
from typing import Any, TextIO


class _TeeStderr:
    """Write-through proxy: forwards every write to the real stream AND a capture buffer.

    Used when no live status display owns the terminal (``--no-status``): the user still
    sees diagnostics as they happen, while the same bytes land in the capture buffer so the
    caller can persist them to ``{cap}.stderr.log`` (R18) after the run.

    Both single-string writes (``write``) and batched writes (``writelines``, e.g.
    ``sys.stderr.writelines([...])``) are routed through the same tee/capture path — a bare
    ``write()`` override alone would let a ``writelines`` call bypass capture entirely.
    """

    def __init__(self, real: TextIO, buffer: io.StringIO) -> None:
        self._real = real
        self._buffer = buffer

    def write(self, s: str) -> int:
        self._real.write(s)
        self._buffer.write(s)
        return len(s)

    def writelines(self, lines: Iterable[str]) -> None:
        """Route each line through :meth:`write` — never bypass the tee/capture path."""
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._real.flush()

    def __getattr__(self, name: str) -> Any:
        # Dynamic proxy to the wrapped real stderr: the attribute's type is whatever the
        # underlying stream exposes, so Any is the idiomatic annotation (mypy-strict needs
        # an explicit return type on every def).
        return getattr(self._real, name)


class _BufferOnlyStderr:
    """Capture-only stderr proxy: writes go ONLY to the buffer, never the real stream.

    Used when a live status display owns the terminal (``active=True``): stderr writes are
    withheld from the real stream (so the display's in-place ANSI redraw is never corrupted)
    and captured for later persistence, then flushed to the real stream once on block exit.

    Unlike a bare ``io.StringIO``, this proxies every OTHER attribute (``fileno``,
    ``buffer``, ``encoding``, ``isatty``, ...) to the real stderr via ``__getattr__``, so
    code that probes ``sys.stderr`` during the block still sees a faithful stream interface.
    A raw ``StringIO`` has no ``.buffer``/``.encoding`` and its ``fileno()`` raises, which
    violates the ``sys.stderr`` contract callers may rely on.
    """

    def __init__(self, real: TextIO, buffer: io.StringIO) -> None:
        self._real = real
        self._buffer = buffer

    def write(self, s: str) -> int:
        return self._buffer.write(s)

    def writelines(self, lines: Iterable[str]) -> None:
        """Route each line to the capture buffer ONLY (never the real stream)."""
        for line in lines:
            self._buffer.write(line)

    def flush(self) -> None:
        # Buffer-only: nothing reaches the real stream until the on-exit flush, so a
        # mid-block flush is a no-op (matching the raw io.StringIO this replaces).
        pass

    def __getattr__(self, name: str) -> Any:
        # Proxy every non-overridden attribute to the real stderr so the stream contract
        # (fileno/buffer/encoding/isatty/...) is honored during the block. Any is idiomatic.
        return getattr(self._real, name)


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

    Args:
        active: Whether a live display owns the terminal (buffer) or not (tee).

    Yields:
        The capture buffer (an ``io.StringIO``), never ``None``.

    Note:
        The exit-path flush (``active=True``) is **best-effort**: if the real stderr is
        broken/closed at that point, the flush is silently skipped rather than raising —
        an exception escaping here could otherwise be mistaken for the delegation itself
        having failed, even though the delegation already completed successfully. The
        restore of ``sys.stderr`` to the real stream happens unconditionally, before the
        guarded flush is attempted.
    """
    real = sys.stderr
    buffer = io.StringIO()
    # active: capture-only but proxy the real stream's attributes (contract-faithful);
    # inactive (--no-status): tee live to the real stream AND capture.
    sys.stderr = _BufferOnlyStderr(real, buffer) if active else _TeeStderr(real, buffer)
    try:
        yield buffer
    finally:
        sys.stderr = real
        if active:
            try:
                real.write(buffer.getvalue())
                real.flush()
            except (OSError, ValueError):
                # Best-effort (INFO fix, round 7): a broken/closed real stderr here must
                # never mask a successful delegation as failed. The captured text is only
                # lost from this final echo-to-real-stderr — callers persist `buf.getvalue()`
                # to `{cap}.stderr.log` independently (R18), so no artifact is lost.
                pass
