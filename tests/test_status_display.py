# tests/test_status_display.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Live status tree: plain line on non-TTY, ANSI redraw on TTY, ASCII fallback."""

import io

import pytest

from status_display import VALID_STATES, StatusDisplay


class _FakeStream:
    """A write-capturing stream with configurable ``encoding`` and ``isatty``."""

    def __init__(self, encoding="utf-8", tty=False):
        self._buf = io.StringIO()
        self.encoding = encoding
        self._tty = tty

    def write(self, s):
        return self._buf.write(s)

    def flush(self):
        pass

    def isatty(self):
        return self._tty

    def getvalue(self):
        return self._buf.getvalue()


def test_non_tty_emits_one_plain_line_per_update():
    out = _FakeStream(tty=False)
    disp = StatusDisplay(["coder", "reviewer"], stream=out)
    disp.update("coder", "running", tok_per_s=42.0)
    disp.update("coder", "success")
    text = out.getvalue()
    assert text.count("\n") == 2  # one line per update
    assert "coder" in text and "success" in text
    assert "\x1b[" not in text  # no ANSI on a non-TTY


def test_tty_redraw_emits_ansi_cursor_controls():
    out = _FakeStream(encoding="utf-8", tty=True)
    disp = StatusDisplay(["coder", "reviewer"], stream=out)
    disp.update("coder", "running", tok_per_s=10.0)
    disp.update("reviewer", "success")
    text = out.getvalue()
    assert "\x1b[2K" in text  # clear-line control
    assert "\x1b[2A" in text  # cursor-up over the 2-row frame


def test_utf8_stream_uses_unicode_glyphs():
    out = _FakeStream(encoding="utf-8", tty=False)
    StatusDisplay(["coder"], stream=out).update("coder", "success")
    assert "✓" in out.getvalue()  # ✓


def test_cp1252_stream_falls_back_to_ascii_glyphs():
    out = _FakeStream(encoding="cp1252", tty=False)
    StatusDisplay(["coder"], stream=out).update("coder", "success")
    text = out.getvalue()
    assert "+" in text  # ASCII glyph
    assert "✓" not in text  # no UTF-8 ✓ on cp1252


def test_invalid_state_rejected():
    disp = StatusDisplay(["coder"], stream=_FakeStream())
    with pytest.raises(ValueError):
        disp.update("coder", "not-a-state")


def test_valid_states_cover_the_lifecycle():
    assert {"pending", "running", "success", "failed", "timeout"} <= VALID_STATES


def test_empty_agents_list_is_safe_to_construct_and_stop():
    # No agents to track (e.g. a delegation batch of zero) must not crash construction,
    # a TTY-mode redraw path, or teardown — there is simply nothing to render.
    out = _FakeStream(tty=True)
    disp = StatusDisplay([], stream=out)
    disp.stop()  # no-op teardown, never raises
    assert out.getvalue() == ""  # nothing drawn without a single update()


def test_status_display_unwraps_dispatching_stderr_proxy_to_the_real_stream():
    # INFO fix (#7): constructing StatusDisplay AFTER the per-delegation dispatching
    # proxy is installed must resolve `self._stream` to the REAL stream, not the
    # proxy object itself.
    import sys

    from stderr_shim import _DispatchingStderr
    from status_display import StatusDisplay

    class _FakeReal:
        def isatty(self) -> bool:
            return False

        def write(self, s: str) -> int:
            return len(s)

        def flush(self) -> None:
            pass

    real = _FakeReal()
    old_stderr = sys.stderr
    sys.stderr = _DispatchingStderr(real)
    try:
        display = StatusDisplay(["coder"])
        assert display._stream is real  # unwrapped to the REAL stream, not the proxy
    finally:
        sys.stderr = old_stderr
