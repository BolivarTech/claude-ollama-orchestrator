# tests/test_ollama_stream.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Hardened SSE reader: deltas→sink, [DONE], tolerance, idle timeout, bounds, HTTP errors."""

import dataclasses
import io
import socket
import urllib.error

import pytest

import ollama_stream
from errors import OllamaBackendError, SinkError  # domain exceptions come from errors
from ollama_config import resolve_config
from ollama_stream import MAX_SSE_LINE_BYTES, stream_run


def _cfg(**overrides):
    """Canonical MS4 test config factory (used everywhere instead of ad-hoc
    ``object.__setattr__`` mutation of a resolved config — see Task 4's `_cfg`):
    resolve the REAL config, then apply field overrides via ``dataclasses.replace``."""
    base = resolve_config(global_path=None, repo_path=None, env={})
    return dataclasses.replace(base, **overrides) if overrides else base


def _sse(*lines):
    body = b"".join(ln.encode("utf-8") if isinstance(ln, str) else ln for ln in lines)

    def _open(req, timeout=None):
        return io.BytesIO(body)  # supports .read(n) like a real urllib response

    return _open


def test_stream_accumulates_deltas_and_calls_sink_in_order():
    got: list[str] = []
    urlopen = _sse(
        'data: {"choices":[{"delta":{"content":"he"}}]}\n',
        ": keep-alive\n",
        'data: {"choices":[{"delta":{"content":"llo"}}]}\n',
        'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n',
        "data: [DONE]\n",
    )
    res = stream_run(_cfg(), "sys", "p", "m", 60, sink=got.append, urlopen=urlopen)
    assert res.content == "hello" and got == ["he", "llo"]
    assert (res.prompt_tokens, res.completion_tokens, res.estimated) == (5, 2, False)
    assert res.truncated is False and res.parsed is None


def test_stream_captures_usage_from_a_mid_stream_chunk():
    """Some servers emit `usage` on a NON-final chunk (not only adjacent to [DONE]);
    the reader must not require usage to arrive on the last chunk."""
    urlopen = _sse(
        'data: {"choices":[{"delta":{"content":"he"}}],'
        '"usage":{"prompt_tokens":5,"completion_tokens":1}}\n',
        'data: {"choices":[{"delta":{"content":"llo"}}]}\n',
        'data: {"choices":[{"delta":{}}]}\n',
        "data: [DONE]\n",
    )
    res = stream_run(_cfg(), "sys", "p", "m", 60, sink=lambda _s: None, urlopen=urlopen)
    assert res.content == "hello"
    assert (res.prompt_tokens, res.completion_tokens, res.estimated) == (5, 1, False)


def test_stream_skips_malformed_and_non_string_deltas():
    urlopen = _sse(
        'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
        "data: {not json}\n",  # malformed → skip
        'data: {"choices":[{"delta":{"content":null}}]}\n',  # non-string → skip, no TypeError
        'data: {"choices":[{"delta":{"content":{"x":1}}}]}\n',  # dict content → skip
        "data: [DONE]\n",
    )
    res = stream_run(_cfg(), "s", "p", "m", 60, sink=lambda _s: None, urlopen=urlopen)
    assert res.content == "ok"


def test_stream_skips_delta_causing_recursion_error_in_json_loads(monkeypatch):
    """A pathologically deeply-nested JSON delta can raise `RecursionError` from
    `json.loads` (a theoretical DoS) — it must be treated as a malformed delta
    (skipped) and NEVER propagate, same tolerant-parser discipline as MS1's
    `parse_output` (which maps RecursionError -> a handled JSONDecodeError)."""
    real_loads = ollama_stream.json.loads

    def _loads(data, *a, **kw):
        if data == '{"boom": true}':
            raise RecursionError("maximum recursion depth exceeded")
        return real_loads(data, *a, **kw)

    monkeypatch.setattr(ollama_stream.json, "loads", _loads)
    urlopen = _sse(
        'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
        'data: {"boom": true}\n',  # triggers the patched RecursionError
        "data: [DONE]\n",
    )
    res = stream_run(_cfg(), "s", "p", "m", 60, sink=lambda _s: None, urlopen=urlopen)
    assert res.content == "ok"


def test_stream_estimates_usage_when_absent():
    urlopen = _sse('data: {"choices":[{"delta":{"content":"abcd"}}]}\n', "data: [DONE]\n")
    res = stream_run(_cfg(), "sys", "user", "m", 60, sink=lambda _s: None, urlopen=urlopen)
    assert res.estimated is True and res.completion_tokens == 1


def test_stream_usage_with_non_numeric_token_counts_falls_back_to_estimate():
    """A malformed `usage` object (string/float/null counts) from an untrusted server
    must NOT raise via a bare `int(...)` — it falls back to the local estimate for
    BOTH counts (via the shared `_resolve_usage`/`_coerce_token_count`, same as MS2's
    transactional core), never a non-domain ValueError/TypeError."""
    urlopen = _sse(
        'data: {"choices":[{"delta":{"content":"abcd"}}],'
        '"usage":{"prompt_tokens":"five","completion_tokens":null}}\n',
        "data: [DONE]\n",
    )
    res = stream_run(_cfg(), "sys", "user", "m", 60, sink=lambda _s: None, urlopen=urlopen)
    assert res.estimated is True
    assert res.completion_tokens == 1  # estimate_tokens("abcd") == 1


def test_stream_output_cap_truncates_a_single_oversized_delta_at_the_boundary():
    """A SINGLE delta larger than the whole cap must be truncated AT the byte
    boundary — the cap must NOT overshoot by an entire delta's worth of bytes just
    because it arrived in one SSE chunk. Uses CJK content (3 bytes/char in UTF-8)
    so the truncation must also land on a valid UTF-8 character boundary (a naive
    byte-slice could split a code point and corrupt the trailing character)."""
    big = "你" * 100  # 100 chars, 300 UTF-8 bytes
    payload = '{"choices":[{"delta":{"content":"' + big + '"}}]}'
    urlopen = _sse(f"data: {payload}\n", "data: [DONE]\n")
    res = stream_run(
        _cfg(), "s", "p", "m", 60, sink=lambda _s: None, urlopen=urlopen, max_output_bytes=10
    )
    assert res.truncated is True
    encoded = res.content.encode("utf-8")
    assert len(encoded) <= 10  # never overshoots the cap
    assert len(encoded) % 3 == 0  # cut on a whole-character boundary


def test_stream_output_cap_truncates_by_utf8_bytes():
    """max_output_bytes is a UTF-8 BYTE budget, not a code-point/char count. CJK
    ideographs are 3 bytes/char in UTF-8: "你好" is 2 code points but 6 bytes — a
    code-point-based cap would let this (and the next delta) through untruncated; the
    byte-based cap must stop right after the first delta once 6 UTF-8 bytes are hit."""
    urlopen = _sse(
        'data: {"choices":[{"delta":{"content":"你好"}}]}\n',  # 2 chars, 6 bytes
        'data: {"choices":[{"delta":{"content":"世界"}}]}\n',  # would add 6 more bytes
        "data: [DONE]\n",
    )
    res = stream_run(
        _cfg(), "s", "p", "m", 60, sink=lambda _s: None, urlopen=urlopen, max_output_bytes=6
    )
    assert res.truncated is True
    assert res.content == "你好"
    assert len(res.content.encode("utf-8")) == 6


def test_stream_bounded_line_buffer_rejects_giant_line():
    giant = "data: " + "x" * (MAX_SSE_LINE_BYTES + 10) + "\n"
    with pytest.raises(OllamaBackendError):
        stream_run(_cfg(), "s", "p", "m", 60, sink=lambda _s: None, urlopen=_sse(giant))


def test_stream_idle_timeout_on_stall():
    class _Stall:
        def read(self, n):
            raise socket.timeout("stalled")  # socket timeout == idle timeout

    with pytest.raises(TimeoutError):
        stream_run(
            _cfg(),
            "s",
            "p",
            "m",
            60,
            sink=lambda _s: None,
            urlopen=lambda req, timeout=None: _Stall(),
            idle_timeout=1,
        )


def test_stream_closes_response_on_error_path():
    """The urlopen response handle must be closed even when `_consume` raises
    mid-stream (idle timeout here) — not just on the success path."""
    closed = {"n": 0}

    class _Stall:
        def read(self, n):
            raise socket.timeout("stalled")

        def close(self):
            closed["n"] += 1

    with pytest.raises(TimeoutError):
        stream_run(
            _cfg(),
            "s",
            "p",
            "m",
            60,
            sink=lambda _s: None,
            urlopen=lambda req, timeout=None: _Stall(),
            idle_timeout=1,
        )
    assert closed["n"] == 1


def test_stream_connection_reset_mid_stream():
    """An `OSError` subclass raised mid-stream by `resp.read()` (e.g. a peer resetting
    the TCP connection) must NOT leak as a raw `ConnectionResetError` — it is mapped to
    the domain `OllamaBackendError` (redacted), distinct from the `socket.timeout`/
    `TimeoutError` idle-timeout branch handled separately above."""

    class _Reset:
        def read(self, n):
            raise ConnectionResetError("connection reset by peer")

    with pytest.raises(OllamaBackendError) as exc:
        stream_run(
            _cfg(),
            "s",
            "p",
            "m",
            60,
            sink=lambda _s: None,
            urlopen=lambda req, timeout=None: _Reset(),
        )
    assert not isinstance(exc.value, ConnectionResetError)


def test_stream_socket_timeout_floors_at_one_second_near_deadline(monkeypatch):
    """When the remaining deadline is smaller than 1.0s, the per-attempt socket
    timeout handed to `urlopen` must be CLAMPED to the 1.0s floor — never a
    smaller/zero/negative timeout (impractical: excessive wakeups, OS/driver
    granularity). This is the SAME accepted 1s floor as MS1's transactional core."""
    seen_timeouts: list[float] = []
    monkeypatch.setattr(ollama_stream.time, "monotonic", lambda: 0.0)

    def _open(req, timeout=None):
        seen_timeouts.append(timeout)
        return io.BytesIO(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n' + b"data: [DONE]\n")

    # timeout=0.1 -> deadline = start(0.0) + 0.1 = 0.1; deadline - now(0.0) = 0.1,
    # well under the 1.0s floor -> eff_timeout must clamp UP to 1.0, not use 0.1.
    stream_run(_cfg(), "s", "p", "m", 0.1, sink=lambda _s: None, urlopen=_open, idle_timeout=60)
    assert seen_timeouts == [1.0]


def test_stream_429_exhausts_backoffs_then_raises_backend_error():
    """All attempts return 429 (backoffs exhausted, not just one retry) -> raises
    OllamaBackendError. Distinct from `test_stream_429_backs_off_then_succeeds`
    (one 429 then success) — this covers the EXHAUSTION path."""

    def _open(req, timeout=None):
        raise urllib.error.HTTPError("u", 429, "Too Many", {"Retry-After": "0"}, io.BytesIO(b""))

    with pytest.raises(OllamaBackendError):
        stream_run(
            _cfg(),
            "s",
            "p",
            "m",
            60,
            sink=lambda _s: None,
            urlopen=_open,
            sleep=lambda _d: None,
            max_backoffs=2,
        )


def test_stream_deadline_exceeded_mid_stream(monkeypatch):
    # Slow-drip: data trickles (each read returns keep-alive, never [DONE]); the
    # per-delegation deadline must still kill it. Advance the clock deterministically.
    clock = {"t": 1000.0}

    def _mono():
        clock["t"] += 0.6
        return clock["t"]

    monkeypatch.setattr(ollama_stream.time, "monotonic", _mono)

    class _Drip:
        def read(self, n):
            return b": keep-alive\n"  # always data, never [DONE]/EOF

    with pytest.raises(TimeoutError):
        stream_run(
            _cfg(),
            "s",
            "p",
            "m",
            2,
            sink=lambda _s: None,  # deadline ≈ now + 2
            urlopen=lambda req, timeout=None: _Drip(),
            idle_timeout=60,
        )


def test_stream_redacts_api_key_in_error_messages():
    cfg = resolve_config(
        global_path=None, repo_path=None, env={"OLLAMA_AGENTS_API_KEY": "sk-secret"}
    )

    def _open(req, timeout=None):
        raise urllib.error.HTTPError("u", 500, "Err sk-secret", {}, io.BytesIO(b"boom sk-secret"))

    with pytest.raises(OllamaBackendError) as exc:
        stream_run(cfg, "s", "p", "m", 60, sink=lambda _s: None, urlopen=_open)
    assert "sk-secret" not in str(exc.value)  # api_key redacted, never leaked


def test_stream_sink_error_is_distinct_from_transport_error():
    # SinkError is imported once, at module top, from `errors` (not from `ollama_stream`).
    def _bad_sink(_s):
        raise BrokenPipeError("stdout closed")

    with pytest.raises(SinkError):  # NOT mapped as an HTTP/backend fault
        stream_run(
            _cfg(),
            "s",
            "p",
            "m",
            60,
            sink=_bad_sink,
            urlopen=_sse('data: {"choices":[{"delta":{"content":"x"}}]}\n', "data: [DONE]\n"),
        )


def test_stream_sink_error_wraps_closed_stream_value_error():
    """Writing to a CLOSED stream raises `ValueError` ("I/O operation on closed
    file"), which is NOT an `OSError` subclass — the sink-write guard must catch it
    too and wrap it as the domain SinkError, never leak a raw ValueError."""

    def _closed_sink(_s):
        raise ValueError("I/O operation on closed file")

    with pytest.raises(SinkError):
        stream_run(
            _cfg(),
            "s",
            "p",
            "m",
            60,
            sink=_closed_sink,
            urlopen=_sse('data: {"choices":[{"delta":{"content":"x"}}]}\n', "data: [DONE]\n"),
        )


def test_stream_5xx_raises_backend_error():
    def _open(req, timeout=None):
        raise urllib.error.HTTPError("u", 503, "Down", {}, io.BytesIO(b"boom"))

    with pytest.raises(OllamaBackendError):
        stream_run(_cfg(), "s", "p", "m", 60, sink=lambda _s: None, urlopen=_open)


def test_stream_429_backs_off_then_succeeds():
    calls = {"n": 0}
    slept: list[float] = []

    def _open(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                "u", 429, "Too Many", {"Retry-After": "0"}, io.BytesIO(b"")
            )
        return io.BytesIO(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n' + b"data: [DONE]\n")

    res = stream_run(
        _cfg(), "s", "p", "m", 60, sink=lambda _s: None, urlopen=_open, sleep=slept.append
    )
    assert res.content == "ok" and slept == [0.0]


def test_stream_400_downgrades_and_drops_response_format():
    seen: list[bool] = []

    def _open(req, timeout=None):
        import json as _j

        seen.append("response_format" in _j.loads(req.data))
        if seen[-1]:
            raise urllib.error.HTTPError(
                "u", 400, "Bad", {}, io.BytesIO(b"invalid response_format")
            )
        return io.BytesIO(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n' + b"data: [DONE]\n")

    res = stream_run(
        _cfg(),
        "s",
        "p",
        "m",
        60,
        sink=lambda _s: None,
        urlopen=_open,
        response_format={"type": "json_object"},
    )
    assert res.content == "ok" and seen == [True, False]  # sent with, then without


def test_stream_eof_without_done_sentinel_returns_accumulated_content():
    """A clean EOF without a `[DONE]` sentinel (the connection just closes after the
    last useful chunk) is TOLERATED as a best-effort completion — not an error. The
    `_sse` mock naturally produces this: once its bytes are exhausted, `.read()`
    returns b"" (EOF) with no further sentinel line."""
    urlopen = _sse(
        'data: {"choices":[{"delta":{"content":"partial"}}]}\n',
        # no [DONE] line — the mocked connection simply ends here (EOF)
    )
    res = stream_run(_cfg(), "s", "p", "m", 60, sink=lambda _s: None, urlopen=urlopen)
    assert res.content == "partial"
    assert res.truncated is False


def _scripted_monotonic(*values: float):
    """Return a fake ``time.monotonic`` yielding *values* in order, then REPEATING
    the last value for any extra calls beyond the script. This makes elapsed-time
    assertions deterministic (exact scripted timestamps) WITHOUT the test needing
    to hard-code — or break on — the exact number of internal `time.monotonic()`
    calls the implementation happens to make (an implementation detail): any call
    past the end of the script simply repeats the final scripted value instead of
    raising `StopIteration`."""
    it = iter(values)
    state = {"last": values[0]}

    def _mono() -> float:
        try:
            state["last"] = next(it)
        except StopIteration:
            pass
        return state["last"]

    return _mono


def test_stream_elapsed_s_spans_the_full_400_downgrade_retry(monkeypatch):
    """elapsed_s must measure from the START of the FIRST attempt — including the
    rejected 400 attempt — not restart its clock at the successful downgraded retry.

    Uses a SCRIPTED fake clock (exact, injected return values) instead of a
    tick-counting/wall-clock approximation, so the assertion is fully deterministic:
    the clock starts at 0.0, the failed-400 attempt happens at t=1.0, and the
    retry finishes consuming the stream at t=2.5 -> elapsed_s must equal EXACTLY
    2.5, regardless of how many internal `time.monotonic()` calls occur in between
    (see `_scripted_monotonic`)."""
    monkeypatch.setattr(
        ollama_stream.time, "monotonic", _scripted_monotonic(0.0, 1.0, 1.0, 2.0, 2.5)
    )

    def _open(req, timeout=None):
        import json as _j

        if "response_format" in _j.loads(req.data):
            raise urllib.error.HTTPError(
                "u", 400, "Bad", {}, io.BytesIO(b"invalid response_format")
            )
        return io.BytesIO(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n' + b"data: [DONE]\n")

    res = stream_run(
        _cfg(),
        "s",
        "p",
        "m",
        60,
        sink=lambda _s: None,
        urlopen=_open,
        response_format={"type": "json_object"},
    )
    assert res.content == "ok"
    assert res.elapsed_s == 2.5
