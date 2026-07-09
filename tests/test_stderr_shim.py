# tests/test_stderr_shim.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Buffer real stderr while the status display renders; tee it live otherwise."""

import sys

from stderr_shim import buffered_stderr_while


def test_display_active_buffers_then_flushes_once(capsys):
    with buffered_stderr_while(active=True) as buf:
        print("during", file=sys.stderr)
        captured_mid = capsys.readouterr().err
        assert "during" not in captured_mid  # buffered, not yet on real stderr
        assert "during" in buf.getvalue()  # but already in the capture buffer
    assert "during" in capsys.readouterr().err  # flushed to real stderr on exit


def test_display_off_tees_live_and_still_captures(capsys):
    with buffered_stderr_while(active=False) as buf:
        print("live", file=sys.stderr)
        assert "live" in capsys.readouterr().err  # visible immediately (tee, --no-status)
        assert "live" in buf.getvalue()  # AND captured for {cap}.stderr.log
    assert "live" not in capsys.readouterr().err  # nothing further flushed on exit (already live)


def test_active_yields_the_capture_buffer(capsys):
    with buffered_stderr_while(active=True) as buf:
        print("diag", file=sys.stderr)
        assert "diag" in buf.getvalue()  # readable for persistence
    assert "diag" in capsys.readouterr().err  # still flushed on exit


def test_inactive_also_yields_a_buffer(capsys):
    # Both modes must yield a buffer — R18 persists {cap}.stderr.log either way.
    with buffered_stderr_while(active=False) as buf:
        print("x", file=sys.stderr)
    assert buf is not None
    capsys.readouterr()


def test_tee_writelines_routes_through_write_live_and_captured(capsys):
    # _TeeStderr.writelines must not bypass the tee/capture path — a bare write()
    # override alone misses sys.stderr.writelines([...]) call sites (WARNING fix).
    with buffered_stderr_while(active=False) as buf:
        sys.stderr.writelines(["a\n", "b\n"])
        assert capsys.readouterr().err == "a\nb\n"  # live, tee mode
    assert buf.getvalue() == "a\nb\n"  # AND captured


def test_buffer_writelines_is_captured_then_flushed_once(capsys):
    with buffered_stderr_while(active=True) as buf:
        sys.stderr.writelines(["x\n", "y\n"])
        assert capsys.readouterr().err == ""  # buffered only, nothing live yet
        assert buf.getvalue() == "x\ny\n"
    assert capsys.readouterr().err == "x\ny\n"  # flushed once on exit


def test_exit_flush_failure_is_best_effort_and_never_raises(monkeypatch):
    # INFO fix (round 7): a broken/closed real stderr at exit-time flush must never
    # propagate — that could otherwise be misread as the delegation itself having failed,
    # even though the delegation (whatever ran inside the `with` block) succeeded.
    class _BrokenStream:
        def write(self, _s):
            raise OSError("broken pipe (simulated)")

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stderr", _BrokenStream())
    # Must not raise: the exit-path flush/restore is best-effort.
    with buffered_stderr_while(active=True) as buf:
        buf.write("captured but never successfully flushed")
    # Reaching this line (context manager exited cleanly) is the assertion.
