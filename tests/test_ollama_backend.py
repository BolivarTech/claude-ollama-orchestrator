# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Transactional OllamaBackend: extraction, auth, error mapping, downgrade-on-400."""

import io
import json
import urllib.error
from dataclasses import replace

import pytest

from backend import MAX_ERROR_BODY_BYTES, MAX_RESPONSE_BYTES, OllamaBackend
from errors import OllamaBackendError
from ollama_config import resolve_config


def _cfg(api_key=None):
    return replace(resolve_config(global_path=None, repo_path=None, env={}), api_key=api_key)


class _Recorder:
    """Callable urlopen replacement recording the last Request + returning a body."""

    def __init__(self, content=None, error=None, error_once=None):
        self.content, self.error, self.error_once = content, error, error_once
        self.requests: list = []
        self._raised_once = False
        # A persistent `error` (raised on EVERY call — e.g. 429-exhaustion / multi-attempt
        # tests) must behave like a real server: every attempt gets its OWN response
        # object. Production code now closes an HTTPError's body after reading it, so
        # replaying the exact same object/fp on a second call would hit "I/O operation on
        # closed file" — snapshot the body ONCE here and rebuild a fresh, independently
        # readable+closable HTTPError on every raise instead of reusing the same instance.
        self._error_snapshot: tuple[str, int, str, object, bytes] | None = None
        if isinstance(error, urllib.error.HTTPError) and error.fp is not None:
            body = error.fp.read()
            error.fp.seek(0)
            self._error_snapshot = (error.filename, error.code, error.msg, error.hdrs, body)

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        if self.error_once is not None and not self._raised_once:
            self._raised_once = True
            raise self.error_once
        if self.error is not None:
            if self._error_snapshot is not None:
                url, code, msg, hdrs, body = self._error_snapshot
                raise urllib.error.HTTPError(url, code, msg, hdrs, io.BytesIO(body))
            raise self.error
        body = {"choices": [{"message": {"content": self.content}}]}
        return io.BytesIO(json.dumps(body).encode("utf-8"))


def test_run_extracts_content():
    rec = _Recorder(content="def f(): pass")
    out = OllamaBackend(_cfg(), urlopen=rec).run("coder", "sys", "write f", "m", 60)
    assert out == "def f(): pass"


def test_dict_content_is_serialized_as_json_not_str():
    rec = _Recorder(content={"agent": "reviewer", "findings": []})
    out = OllamaBackend(_cfg(), urlopen=rec).run("reviewer", "sys", "p", "m", 60)
    assert json.loads(out) == {"agent": "reviewer", "findings": []}  # valid JSON, not str(dict)


def test_null_content_raises_backend_error_not_string_none():
    # A model returning a null `content` (JSON `"content": null`) must surface as a domain
    # error — NOT str(None) == "None", which would silently masquerade as real output for
    # Claude to review/apply.
    rec = _Recorder(content=None)
    with pytest.raises(OllamaBackendError) as exc:
        OllamaBackend(_cfg(), urlopen=rec).run("coder", "s", "p", "m", 60)
    assert "null content" in str(exc.value)


def test_auth_header_only_when_api_key_present():
    rec = _Recorder(content="x")
    OllamaBackend(_cfg(api_key="sk-secret"), urlopen=rec).run("coder", "s", "p", "m", 60)
    assert rec.requests[-1].get_header("Authorization") == "Bearer sk-secret"
    rec2 = _Recorder(content="x")
    OllamaBackend(_cfg(), urlopen=rec2).run("coder", "s", "p", "m", 60)
    assert rec2.requests[-1].get_header("Authorization") is None


@pytest.mark.parametrize(
    "raw_body",
    [
        pytest.param(b'{"unexpected": true}', id="missing-choices-key"),
        pytest.param(b"[]", id="top-level-list"),
        pytest.param(b"null", id="top-level-null"),
        pytest.param(b'{"choices": []}', id="empty-choices"),
        pytest.param(b'{"choices": [{"message": null}]}', id="null-message"),
    ],
)
def test_unexpected_shape_raises_backend_error(raw_body):
    class _Bad:
        def __init__(self, body):
            self._body = body

        def __call__(self, req, timeout=None):
            return io.BytesIO(self._body)

    with pytest.raises(OllamaBackendError):
        OllamaBackend(_cfg(), urlopen=_Bad(raw_body)).run("coder", "s", "p", "m", 60)


def test_timeout_maps_to_timeout_error():
    import socket

    rec = _Recorder(error=socket.timeout("slow"))
    with pytest.raises(TimeoutError):
        OllamaBackend(_cfg(), urlopen=rec).run("coder", "s", "p", "m", 1)


def test_http_error_redacts_api_key():
    err = urllib.error.HTTPError("u", 500, "Server Error", {}, io.BytesIO(b""))
    rec = _Recorder(error=err)
    with pytest.raises(OllamaBackendError) as exc:
        OllamaBackend(_cfg(api_key="sk-secret"), urlopen=rec).run("coder", "s", "p", "m", 60)
    assert "sk-secret" not in str(exc.value)


def test_downgrade_on_400_response_format_retries_without_it():
    err = urllib.error.HTTPError(
        "u", 400, "Bad Request", {}, io.BytesIO(b"response_format not supported")
    )
    rec = _Recorder(content="ok", error_once=err)  # first call 400, second succeeds
    out = OllamaBackend(_cfg(), urlopen=rec).run(
        "reviewer", "s", "p", "m", 60, response_format={"type": "json_object"}
    )
    assert out == "ok"
    assert len(rec.requests) == 2  # downgrade retry happened


def test_429_backs_off_respecting_retry_after_then_succeeds():
    err = urllib.error.HTTPError("u", 429, "Too Many", {"Retry-After": "0"}, io.BytesIO(b""))
    slept: list[float] = []
    rec = _Recorder(content="ok", error_once=err)  # first call 429, second succeeds
    out = OllamaBackend(_cfg(), urlopen=rec, sleep=slept.append).run("coder", "s", "p", "m", 60)
    assert out == "ok"
    assert slept == [0.0]  # honored Retry-After: 0 (distinct from the parse retry)


def test_429_exhausted_raises_backend_error_with_redaction():
    err = urllib.error.HTTPError("u", 429, "Too Many", {}, io.BytesIO(b""))
    rec = _Recorder(error=err)  # always 429
    be = OllamaBackend(
        _cfg(api_key="sk-secret"),
        urlopen=rec,
        sleep=lambda _s: None,
        rng=lambda: 0.0,
        max_backoffs=2,
    )
    with pytest.raises(OllamaBackendError) as exc:
        be.run("coder", "s", "p", "m", 60)
    assert "sk-secret" not in str(exc.value)


def test_non_json_server_response_maps_to_backend_error():
    # A proxy/gateway can return HTTP 200 with a non-JSON body; the backend must map the
    # resulting JSONDecodeError to a domain error, not crash the process.
    def _html(req, timeout=None):
        return io.BytesIO(b"<html>502 Bad Gateway</html>")

    with pytest.raises(OllamaBackendError):
        OllamaBackend(_cfg(), urlopen=_html).run("coder", "s", "p", "m", 60)


def test_invalid_utf8_response_bytes_do_not_crash():
    # The chat-completions response body is untrusted server output. Malformed UTF-8
    # bytes must decode via errors="replace" (U+FFFD substitution) — NEVER raise a raw
    # UnicodeDecodeError (a non-domain exception). The substituted text is not valid
    # JSON, so this surfaces as the existing domain OllamaBackendError, exactly like any
    # other non-JSON body — not a crash.
    def _bad_utf8(req, timeout=None):
        return io.BytesIO(b"\xff\xfe garbage \x80\x81")

    with pytest.raises(OllamaBackendError):
        OllamaBackend(_cfg(), urlopen=_bad_utf8).run("coder", "s", "p", "m", 60)


def test_429_backoff_is_bounded_by_the_per_delegation_deadline():
    # R25: a 429 with a Retry-After beyond the delegation's time budget must raise
    # deadline-exceeded rather than sleep the impossible delay (no unbounded backoff).
    err = urllib.error.HTTPError("u", 429, "Too Many", {"Retry-After": "999999"}, io.BytesIO(b""))
    slept: list[float] = []
    rec = _Recorder(error=err)
    be = OllamaBackend(_cfg(), urlopen=rec, sleep=slept.append)
    with pytest.raises(OllamaBackendError):
        be.run("coder", "s", "p", "m", 1)  # 1s budget « 999999s Retry-After → deadline
    assert slept == []  # never slept past the deadline


def test_run_honors_a_caller_supplied_deadline_over_timeout():
    # R25 propagation: when the caller (dispatch) passes an ALREADY-ELAPSED deadline,
    # run must NOT derive a fresh timeout budget — the first 429 backoff is refused
    # immediately, so a large per-call `timeout` cannot re-expand the shared budget.
    import time as _t

    err = urllib.error.HTTPError("u", 429, "Too Many", {"Retry-After": "5"}, io.BytesIO(b""))
    slept: list[float] = []
    rec = _Recorder(error=err)
    be = OllamaBackend(_cfg(), urlopen=rec, sleep=slept.append)
    past = _t.monotonic() - 1.0  # deadline already in the past
    with pytest.raises(OllamaBackendError):
        be.run("coder", "s", "p", "m", 9999, deadline=past)  # huge timeout ignored
    assert slept == []  # deadline (not timeout) governed → no sleep


def test_per_call_socket_timeout_clamped_to_remaining_deadline():
    # R25: each urlopen's socket timeout is the REMAINING budget, never the nominal
    # `timeout`, so one call plus a retry can never together exceed the deadline (no 2×).
    import time as _t

    seen: list[float] = []

    def _urlopen(req, timeout=None):
        seen.append(timeout)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())

    be = OllamaBackend(_cfg(), urlopen=_urlopen)
    be.run("coder", "s", "p", "m", 9999, deadline=_t.monotonic() + 5)  # 5s left « 9999
    assert seen and 1.0 <= seen[0] <= 5.0  # clamped to remaining budget, not 9999


def test_429_retry_after_http_date_is_parsed():
    # Retry-After may be an HTTP-date (RFC 7231), not only integer seconds.
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=2))
    err = urllib.error.HTTPError("u", 429, "Too Many", {"Retry-After": when}, io.BytesIO(b""))
    slept: list[float] = []
    rec = _Recorder(content="ok", error_once=err)  # 429 (date) once, then success
    be = OllamaBackend(_cfg(), urlopen=rec, sleep=slept.append)
    be.run("coder", "s", "p", "m", 60)
    assert slept and 0.0 <= slept[0] <= 3.0  # ~2s derived from the date, not jitter


def test_oversized_response_body_raises_backend_error():
    # Always-on DoS backstop: a runaway/hostile server response over MAX_RESPONSE_BYTES
    # must be rejected — never fully decoded/JSON-parsed, never loaded unbounded.
    class _Oversized:
        def __call__(self, req, timeout=None):
            return io.BytesIO(b"x" * (MAX_RESPONSE_BYTES + 1))

    with pytest.raises(OllamaBackendError) as exc:
        OllamaBackend(_cfg(), urlopen=_Oversized()).run("coder", "s", "p", "m", 60)
    assert "MAX_RESPONSE_BYTES" in str(exc.value)


def test_response_read_is_called_with_a_bound_never_unbounded():
    # The core backstop only holds if the read itself is bounded (`resp.read(N)`), not a
    # bare `resp.read()`/`resp.read(-1)` that would defeat the whole point of the cap.
    seen: dict[str, int] = {}

    class _Resp:
        def read(self, n=-1):
            seen["n"] = n
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")

    class _Rec:
        def __call__(self, req, timeout=None):
            return _Resp()

    OllamaBackend(_cfg(), urlopen=_Rec()).run("coder", "s", "p", "m", 60)
    assert seen["n"] == MAX_RESPONSE_BYTES + 1


def test_double_400_response_format_rejection_raises_backend_error_not_internal_signal():
    # If even the DOWNGRADED (no response_format) retry is STILL rejected as a
    # response_format/json_schema 400 (e.g. a proxy whose error body generically mentions
    # those terms regardless of what was actually sent), the internal _ResponseFormatRejected
    # signal must never escape run() — it must surface as a domain OllamaBackendError.
    # Each attempt gets its OWN fresh HTTPError (a real server would never hand back an
    # already-closed response on a second attempt).
    calls: list = []

    def _urlopen(req, timeout=None):
        calls.append(req)
        raise urllib.error.HTTPError(
            "u", 400, "Bad Request", {}, io.BytesIO(b"response_format not supported")
        )

    with pytest.raises(OllamaBackendError):
        OllamaBackend(_cfg(), urlopen=_urlopen).run(
            "reviewer", "s", "p", "m", 60, response_format={"type": "json_object"}
        )
    assert len(calls) == 2  # original + downgraded, then gave up (no 3rd attempt)


def test_http_error_body_read_is_called_with_a_bound_never_unbounded():
    # The error-body backstop only holds if the read itself is bounded (`exc.read(N)`),
    # never a bare `exc.read()`/`exc.read(-1)` that would defeat the whole point of the cap.
    seen: dict[str, int] = {}

    class _ErrFP:
        def read(self, n=-1):
            seen["n"] = n
            return b"Internal Server booboo"

        def close(self):
            pass

    def _urlopen(req, timeout=None):
        raise urllib.error.HTTPError("u", 500, "Server Error", {}, _ErrFP())

    with pytest.raises(OllamaBackendError):
        OllamaBackend(_cfg(), urlopen=_urlopen).run("coder", "s", "p", "m", 60)
    assert seen["n"] == MAX_ERROR_BODY_BYTES + 1


def test_success_response_body_is_closed_after_reading():
    # No leaked descriptors on the success path: the HTTPResponse-like object must be
    # closed once its body has been read, not left open for the lifetime of the process.
    body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")
    resp = io.BytesIO(body)

    def _urlopen(req, timeout=None):
        return resp

    OllamaBackend(_cfg(), urlopen=_urlopen).run("coder", "s", "p", "m", 60)
    assert resp.closed


def test_http_error_response_body_is_closed_after_reading():
    # No leaked descriptors on the error path either: the HTTPError's body (itself a
    # response-like object) must be closed once its detail has been read. Raised directly
    # (not via `_Recorder`, which snapshots+rebuilds HTTPErrors for repeat-attempt
    # realism) so we can assert on the SAME fp instance that was closed.
    fp = io.BytesIO(b"error detail")
    err = urllib.error.HTTPError("u", 500, "Server Error", {}, fp)

    def _urlopen(req, timeout=None):
        raise err

    with pytest.raises(OllamaBackendError):
        OllamaBackend(_cfg(), urlopen=_urlopen).run("coder", "s", "p", "m", 60)
    assert fp.closed


def test_connection_reset_during_read_maps_to_backend_error_not_raw_oserror():
    # A mid-response connection drop (ConnectionResetError is an OSError subclass, common
    # on a `:cloud` endpoint) must map to a domain error, never propagate as a raw
    # non-domain exception across the module boundary.
    class _Resp:
        def read(self, n=-1):
            raise ConnectionResetError("connection reset by peer")

        def close(self):
            pass

    def _urlopen(req, timeout=None):
        return _Resp()

    with pytest.raises(OllamaBackendError) as exc:
        OllamaBackend(_cfg(), urlopen=_urlopen).run("coder", "s", "p", "m", 60)
    assert "connection" in str(exc.value).lower()


def test_deeply_nested_json_body_maps_to_backend_error_not_raw_recursion_error():
    # A malicious/hostile server can return a deeply-nested JSON body that trips
    # Python's recursion limit inside json.loads. RecursionError is a RuntimeError, NOT
    # a json.JSONDecodeError, so it is NOT caught by `except json.JSONDecodeError`
    # alone — it must still map to the domain OllamaBackendError, never escape as a raw
    # RecursionError past this module's boundary.
    nested = ("[" * 20000) + ("]" * 20000)

    def _deep(req, timeout=None):
        return io.BytesIO(nested.encode("utf-8"))

    with pytest.raises(OllamaBackendError) as exc:
        OllamaBackend(_cfg(), urlopen=_deep).run("coder", "s", "p", "m", 60)
    assert "not valid JSON" in str(exc.value)


def test_incomplete_read_during_read_maps_to_backend_error_not_raw_exception():
    # http.client.IncompleteRead (a truncated read on a mid-response drop) is NOT an
    # OSError subclass — it must be mapped explicitly, not silently propagate raw.
    import http.client

    class _Resp:
        def read(self, n=-1):
            raise http.client.IncompleteRead(b"partial")

        def close(self):
            pass

    def _urlopen(req, timeout=None):
        return _Resp()

    with pytest.raises(OllamaBackendError):
        OllamaBackend(_cfg(), urlopen=_urlopen).run("coder", "s", "p", "m", 60)
