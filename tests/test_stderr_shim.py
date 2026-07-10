# tests/test_stderr_shim.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Buffer real stderr while the status display renders; tee it live otherwise."""

import asyncio
import sys

import pytest

from stderr_shim import (
    buffered_stderr_while,
    capture_stderr_for_delegation,
    install_dispatching_stderr,
)


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


def test_active_shim_proxies_real_stderr_stream_attributes(monkeypatch):
    # Balthasar WARNING: in active mode the shimmed sys.stderr must honor the stream
    # contract — fileno/encoding/isatty proxy to the real stream. A bare io.StringIO has no
    # `.encoding` and its `fileno()` raises, breaking code that probes sys.stderr while the
    # display owns the terminal. Writes still go to the capture buffer ONLY (never the real
    # stream mid-block) so the display's ANSI redraw is protected.
    class _FakeReal:
        encoding = "utf-8"

        def __init__(self) -> None:
            self.written: list[str] = []

        def fileno(self) -> int:
            return 2

        def isatty(self) -> bool:
            return False

        def write(self, s: str) -> int:
            self.written.append(s)
            return len(s)

        def flush(self) -> None:
            pass

    fake = _FakeReal()
    monkeypatch.setattr(sys, "stderr", fake)
    with buffered_stderr_while(active=True) as buf:
        assert sys.stderr.encoding == "utf-8"  # proxied (raw StringIO has no .encoding)
        assert sys.stderr.fileno() == 2  # proxied (raw StringIO.fileno() raises)
        assert sys.stderr.isatty() is False  # proxied
        sys.stderr.write("captured\n")
        assert fake.written == []  # active mode: NOTHING reaches the real stream mid-block
    assert buf.getvalue() == "captured\n"
    assert "".join(fake.written) == "captured\n"  # delivered to the real stream once, on exit


def test_concurrent_delegations_capture_to_distinct_buffers(capsys):
    with install_dispatching_stderr():

        async def deleg(tag):
            with capture_stderr_for_delegation() as buf:
                print(f"from-{tag}", file=sys.stderr)
                await asyncio.sleep(0.01)  # yield -> interleave with the other task
                print(f"more-{tag}", file=sys.stderr)
                return buf.getvalue()

        async def main():
            return await asyncio.gather(deleg("A"), deleg("B"))

        a, b = asyncio.run(main())
    assert a == "from-A\nmore-A\n"  # A's buffer holds ONLY A's lines
    assert b == "from-B\nmore-B\n"  # no cross-contamination despite interleave
    assert "from-A" not in capsys.readouterr().err  # captured, not leaked to real stderr


def test_stderr_outside_a_delegation_goes_to_real_stderr(capsys):
    with install_dispatching_stderr():
        print("orchestrator diag", file=sys.stderr)  # no capture context active
    assert "orchestrator diag" in capsys.readouterr().err


def test_install_dispatching_stderr_restores_original_on_normal_exit():
    # WARNING fix: the guaranteed uninstall -- after the `with` block exits
    # normally, `sys.stderr` must be the EXACT original object again, not left
    # as (or wrapping) the proxy.
    original = sys.stderr
    with install_dispatching_stderr():
        assert sys.stderr is not original
        from stderr_shim import _DispatchingStderr

        assert isinstance(sys.stderr, _DispatchingStderr)
    assert sys.stderr is original


def test_install_dispatching_stderr_restores_original_on_exception():
    # WARNING fix: the guaranteed uninstall must ALSO hold when the block
    # raises -- a proxy installed but never restored would leak into every
    # later test/run in the same process.
    original = sys.stderr
    with pytest.raises(RuntimeError):
        with install_dispatching_stderr():
            raise RuntimeError("boom mid-block")
    assert sys.stderr is original


def test_install_dispatching_stderr_restores_original_on_cancellation():
    # [WARNING] The guaranteed uninstall must ALSO hold when the block is
    # CANCELLED, not just when it raises an ordinary Exception --
    # `asyncio.CancelledError` is a BaseException (Python 3.8+), a genuinely
    # different propagation path than `test_..._on_exception` above (which
    # only exercises a plain `RuntimeError`). `run_batch` (Task 5) wraps its
    # entire fan-out in `install_dispatching_stderr()`, and a task driving
    # that fan-out can be cancelled by its own caller (e.g. the CLI's own
    # Ctrl-C handling) -- the proxy must never leak in that case either.
    original = sys.stderr

    async def _install_then_get_cancelled():
        with install_dispatching_stderr():
            assert sys.stderr is not original
            await asyncio.sleep(10)  # never completes -- cancelled below

    async def _drive():
        task = asyncio.ensure_future(_install_then_get_cancelled())
        await asyncio.sleep(0)  # let it start and enter the with-block
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_drive())
    assert sys.stderr is original


def test_dispatching_stderr_delegates_isatty_and_encoding_to_real_stderr():
    # CRITICAL fix: StatusDisplay calls isatty()/encoding, and Windows console
    # hardening calls reconfigure(...) -- none of those are overridden explicitly
    # on the proxy, so they must reach the REAL stderr via __getattr__.
    from stderr_shim import _DispatchingStderr

    class _FakeReal:
        def isatty(self) -> bool:
            return True

        encoding = "utf-8"
        errors = "backslashreplace"

        def fileno(self) -> int:
            return 2

        def reconfigure(self, **kwargs) -> None:
            self.reconfigured_with = kwargs

        def write(self, s: str) -> int:
            return len(s)

        def flush(self) -> None:
            pass

    real = _FakeReal()
    proxy = _DispatchingStderr(real)
    assert proxy.isatty() is True
    assert proxy.encoding == "utf-8"
    assert proxy.errors == "backslashreplace"
    assert proxy.fileno() == 2
    proxy.reconfigure(encoding="utf-8", errors="backslashreplace")
    assert real.reconfigured_with == {"encoding": "utf-8", "errors": "backslashreplace"}


def test_buffered_stderr_while_active_true_preserves_ms3_buffer_then_flush_once(capsys):
    # MS3 parity (WARNING fix): active=True buffers only, flushed once on exit --
    # verbatim the same assertions as MS3.md's test_display_active_buffers_then_flushes_once.
    with buffered_stderr_while(active=True) as buf:
        print("during", file=sys.stderr)
        assert "during" not in capsys.readouterr().err  # buffered, not yet on real stderr
        assert "during" in buf.getvalue()
    assert "during" in capsys.readouterr().err  # flushed to real stderr on exit


def test_buffered_stderr_while_active_false_preserves_ms3_live_tee(capsys):
    # MS3 parity (WARNING fix): active=False tees live AND still captures -- verbatim
    # the same assertions as MS3.md's test_display_off_tees_live_and_still_captures.
    # This is the mode `run_delegation` uses under --no-status, so it is NOT dead code.
    with buffered_stderr_while(active=False) as buf:
        print("live", file=sys.stderr)
        assert "live" in capsys.readouterr().err  # visible immediately (tee)
        assert "live" in buf.getvalue()  # AND captured
    assert "live" not in capsys.readouterr().err  # nothing further flushed on exit


def test_buffered_stderr_while_flushes_to_the_real_target_pinned_at_entry():
    # WARNING fix (#5): buffered_stderr_while must resolve the REAL stderr target
    # BEFORE entering the capture block, so a later reassignment of the global
    # sys.stderr (e.g. by another concurrently-running task/delegation) during the
    # block can't redirect THIS call's final flush to the wrong target.
    from stderr_shim import _DispatchingStderr

    class _Fake:
        def __init__(self):
            self.chunks: list[str] = []

        def write(self, s: str) -> int:
            self.chunks.append(s)
            return len(s)

        def flush(self) -> None:
            pass

        def isatty(self) -> bool:
            return False

    real_at_entry = _Fake()
    sys.stderr = _DispatchingStderr(real_at_entry)
    try:
        with buffered_stderr_while(active=True):
            print("during", file=sys.stderr, end="")
            # Simulate a concurrent task reassigning the global proxy mid-flight --
            # this must NOT affect where THIS call's flush lands.
            impostor = _Fake()
            sys.stderr = _DispatchingStderr(impostor)

        assert "during" in "".join(real_at_entry.chunks)  # flushed to the ORIGINAL real target
        assert impostor.chunks == []  # the impostor never saw the flush
    finally:
        sys.stderr = sys.__stderr__
