# skills/ollama/scripts/ollama_stream.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Hardened SSE streaming reader for the visible-streaming layer (R7b/R7c).

Reuses backend's shared request builder / error mapper / backoff (DRY), so it
matches the transactional core's HTTP semantics (429 backoff, 5xx/timeout→domain,
400→downgrade) while adding streaming-only guards: an idle timeout, a bounded
per-line buffer, and an anti-runaway output cap.

Scope note (divergence from R25): the HTTP-400 ``response_format`` downgrade handled
in this module (drop the schema, retry once — see ``stream_run``'s downgrade loop) is
a TRANSPORT-level concern, and is SEPARATE from the dispatch-layer R25 parse/schema
retry (the ``---RETRY-FEEDBACK---`` reinjection lives entirely in
``run_ollama.dispatch`` and operates on the returned ``content`` AFTER a stream has
already completed successfully). This module never reinjects retry-feedback — that
responsibility stays in ``dispatch``, same as for the transactional core.
"""

from __future__ import annotations

import http.client
import json
import random
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

from backend import (
    DelegationResult,
    ResponseFormatRejected,
    _resolve_usage,
    _safe_close,
    build_chat_request,
    make_redactor,
    map_http_error,
    retry_after_delay,
)
from errors import (  # domain exceptions from errors, never defined locally in this
    OllamaBackendError,  # module (SinkError itself lives in errors.py — see Task 2's
    SinkError,  # "Modify: errors.py" step)
)
from ollama_config import OllamaAgentsConfig

_SSE_PREFIX = "data:"
MAX_SSE_LINE_BYTES = 1_048_576  # bound a single SSE line (anti-DoS)
DEFAULT_IDLE_TIMEOUT = 60  # seconds without data → hung stream
DEFAULT_MAX_OUTPUT_BYTES = 2_000_000  # public: reused as ollama_vision.stream_vision's default
_DEFAULT_MAX_OUTPUT_BYTES = (
    DEFAULT_MAX_OUTPUT_BYTES  # back-compat internal alias (this module's own default param)
)
_READ_CHUNK_BYTES = 65_536  # bounded read granularity for SSE reassembly


def stream_run(
    config: OllamaAgentsConfig,
    system_prompt: str,
    prompt: str,
    model: str,
    timeout: int,
    *,
    sink: Callable[[str], None],
    response_format: dict[str, Any] | None = None,
    idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
    max_backoffs: int = 3,
) -> DelegationResult:
    """Stream one delegation, emit each delta to *sink*, return the DelegationResult.

    The per-delegation wall-clock budget is ``deadline = time.monotonic() + timeout``
    EXACTLY (``--timeout`` is the hard end-to-end budget, retry + 429 backoff included —
    same as MS1's transactional core; NO ``2*timeout`` doubling). A REAL redactor built
    from ``config.api_key`` is applied to every error message so the key never leaks.
    ``start`` is captured ONCE, before the 400-downgrade retry loop, and threaded through
    ``_stream_once``/``_consume`` so the returned ``elapsed_s`` spans the FULL delegation —
    including a rejected ``response_format`` attempt — not just the final successful one.

    Args:
        config: Resolved config.
        system_prompt: Capability system prompt.
        prompt: Sanitized user prompt.
        model: Resolved model tag.
        timeout: Per-delegation wall-clock budget (the deadline) and read clamp.
        sink: Receiver for each content delta (stdout or file writer).
        response_format: Structured-output shape (dropped on a 400 downgrade).
        idle_timeout: Max seconds BETWEEN chunks before aborting a hung stream.
        max_output_bytes: Anti-runaway cap in UTF-8 BYTES (not code points/chars);
            exceeding it truncates the result and sets ``truncated=True``.
        urlopen: Injectable urlopen.
        sleep: Injectable sleep (429 backoff).
        rng: Injectable RNG (jitter).
        max_backoffs: Max 429 backoffs before failing.

    Returns:
        The accumulated :class:`DelegationResult` (``parsed`` is None; structured
        parsing happens in ``dispatch``).

    Raises:
        TimeoutError: idle stall or deadline exceeded mid-stream.
        OllamaBackendError: 5xx/unreachable/oversized line/backoff exhausted.
        SinkError: the output sink (stdout/file) failed to write a delta.
    """
    redact = make_redactor(config.api_key)  # REAL redaction — api_key never leaks
    start = time.monotonic()  # spans ALL attempts (incl. a failed 400
    # before a downgrade retry) — elapsed_s
    # must reflect the FULL delegation, not
    # just the final successful attempt
    deadline = start + timeout  # hard end-to-end budget (== MS1)
    rf = response_format
    for _downgrade in range(2):
        try:
            return _stream_once(
                config,
                system_prompt,
                prompt,
                model,
                sink,
                rf,
                idle_timeout,
                max_output_bytes,
                urlopen,
                sleep,
                rng,
                max_backoffs,
                deadline,
                redact,
                start,
            )
        except ResponseFormatRejected:
            rf = None  # R11 downgrade: retry once without response_format
    raise OllamaBackendError("response_format downgrade exhausted")


def _stream_once(
    config: OllamaAgentsConfig,
    system_prompt: str,
    prompt: str,
    model: str,
    sink: Callable[[str], None],
    response_format: dict[str, Any] | None,
    idle_timeout: int,
    max_output_bytes: int,
    urlopen: Callable[..., Any],
    sleep: Callable[[float], None],
    rng: Callable[[], float],
    max_backoffs: int,
    deadline: float,
    redact: Callable[[str], str],
    start: float,
) -> DelegationResult:
    """One stream attempt with 429 backoff (bounded by *deadline*).

    The response is consumed OUTSIDE the transport ``try/except`` so a ``TimeoutError``
    (idle/deadline) or ``SinkError`` raised WHILE reading the stream is not re-mapped
    as a transport error by ``map_http_error``. The response handle is closed in a
    ``finally`` around the ``_consume`` call — including on these mid-stream
    exception paths, not just on success (guarded: `resp` may not support ``.close()``).

    NOTE: this loop has no ``for/else`` — on the LAST attempt (``attempt ==
    max_backoffs``), the ``exc.code == 429 and attempt < max_backoffs`` guard is
    False, so a 429 falls straight through to ``map_http_error(exc, redact=redact)``,
    which is annotated ``NoReturn`` (it ALWAYS raises). Every path through this loop
    therefore ends in ``break``, ``continue``, or a raise — the loop can never fall
    off the end normally, so a trailing ``for/else: raise ...`` would be dead code
    (unreachable) and is intentionally omitted; the exhaustion case is exercised by
    `test_stream_429_exhausts_backoffs_then_raises_backend_error`.
    """
    req = build_chat_request(config, system_prompt, prompt, model, response_format, stream=True)
    resp: Any = None
    for attempt in range(max_backoffs + 1):
        # Socket timeout = the smaller of the idle timeout (per-read stall detection) and
        # the remaining deadline, CLAMPED to a 1.0s floor (`max(remaining, 1.0)`): a socket
        # timeout below 1s is impractical (excessive wakeups, OS/driver granularity), so the
        # per-attempt call can deliberately overshoot the deadline by up to ~1s in the worst
        # case (e.g. remaining == 0.1s → eff_timeout == 1.0s). This is the SAME accepted 1s
        # floor as MS1's transactional core — the mid-stream `deadline` check inside
        # `_consume`'s read loop is what actually bounds the overall stream duration
        # regardless of this per-attempt floor, so the ~1s overshoot never compounds.
        eff_timeout = max(min(idle_timeout, deadline - time.monotonic()), 1.0)
        try:
            resp = urlopen(req, timeout=eff_timeout)
            break  # got the response; consume below
        except urllib.error.HTTPError as exc:
            # Close the HTTPError's response body (`exc.fp`) on EVERY path out of this
            # handler — the 429-backoff `continue`, the deadline-exceeded raise, AND the
            # terminal `map_http_error` — or the error body leaks a file descriptor on every
            # HTTP error (`map_http_error` reads it but does not own its lifetime). This
            # mirrors the transactional core's finally-guarded close. `_safe_close` is
            # best-effort and never masks the exception in flight.
            try:
                if exc.code == 429 and attempt < max_backoffs:
                    delay = retry_after_delay(exc, attempt, rng)
                    if time.monotonic() + delay > deadline:
                        raise OllamaBackendError(redact("stream 429: deadline exceeded")) from None
                    sleep(delay)
                    continue
                map_http_error(exc, redact=redact)  # NoReturn: 400→ResponseFormatRejected,
                # 429-exhausted/5xx→OllamaBackendError
            finally:
                _safe_close(exc)
        except (TimeoutError, OSError, http.client.IncompleteRead) as exc:
            # socket timeout / URLError / (defensively) a truncated read at connect time →
            # domain error. IncompleteRead is not an OSError subclass, so it is named here
            # too, mirroring the read-loop and the transactional core.
            map_http_error(exc, redact=redact)  # NoReturn
    try:
        return _consume(
            resp, sink, max_output_bytes, deadline, system_prompt, prompt, redact, start
        )
    finally:
        # The response handle must be closed even when `_consume` raises mid-stream
        # (TimeoutError from idle/deadline, OllamaBackendError from an oversized line,
        # SinkError from a broken sink) — not just on the success path. Uses backend's
        # shared `_safe_close`: it guards BOTH a missing `.close` (a bare test double) AND a
        # `.close()` that RAISES — an unguarded close in this `finally` would otherwise
        # REPLACE the real exception in flight (the exact `_safe_close` policy the
        # transactional core already applies to response/error bodies).
        _safe_close(resp)


def _truncate_utf8_bytes(data: bytes, max_bytes: int) -> bytes:
    """Truncate *data* to at most *max_bytes* bytes WITHOUT splitting a multi-byte
    UTF-8 character: back off from the cut point over any trailing continuation
    bytes (``0b10xxxxxx``) until it lands on an ASCII byte or a lead byte, so the
    result always ``.decode("utf-8")``-s cleanly.

    Args:
        data: The UTF-8 encoded bytes to truncate.
        max_bytes: The maximum length of the result, in bytes.

    Returns:
        ``data`` unchanged if it already fits; otherwise a prefix of ``data`` of at
        most ``max_bytes`` bytes, cut on a whole-character boundary.
    """
    if len(data) <= max_bytes:
        return data
    cut = max_bytes
    while cut > 0 and (data[cut] & 0xC0) == 0x80:  # 0x80 = continuation-byte marker
        cut -= 1
    return data[:cut]


def _consume(
    resp: Any,
    sink: Callable[[str], None],
    max_output_bytes: int,
    deadline: float,
    system_prompt: str,
    prompt: str,
    redact: Callable[[str], str],
    start: float,
) -> DelegationResult:
    """Read the SSE body with a BOUNDED reassembly buffer.

    Reads in ``_READ_CHUNK_BYTES`` chunks (never ``for line in resp``, which would
    materialize an unbounded line) and enforces ``MAX_SSE_LINE_BYTES`` on the buffer
    BEFORE a newline appears (anti-DoS). A stalled read raises ``socket.timeout`` (the
    socket timeout was set to the idle timeout in ``_stream_once``) → re-raised as a
    ``TimeoutError`` (idle). The per-delegation ``deadline`` is re-checked on every read
    (anti slow-drip). A sink write failure — an ``OSError`` (e.g. a broken pipe) OR a
    ``ValueError`` (e.g. "I/O operation on closed file", raised by a CLOSED stream,
    which is NOT an ``OSError`` subclass) — raises :class:`SinkError`, kept distinct
    from transport errors.

    ``start`` is the monotonic timestamp of the FIRST attempt of the delegation (passed
    in from ``stream_run``, NOT re-taken here) so ``elapsed_s`` spans a prior rejected
    ``response_format`` attempt too, not just this (possibly retried) consume call.

    Anti-runaway cap: ``max_output_bytes`` is measured in UTF-8 BYTES, not code points —
    a running ``byte_count`` accumulates the (possibly truncated) delta's UTF-8 byte
    length per delta (never re-encodes the whole buffer), so multibyte content (e.g.
    CJK, ~3 bytes/char) is capped against the real byte budget instead of overshooting
    it ~3-4x. A SINGLE delta that by itself exceeds the remaining budget is truncated
    AT the cap boundary (via `_truncate_utf8_bytes`, which never splits a multi-byte
    UTF-8 character) BEFORE it is appended/sunk — a single oversized delta can never
    overshoot the cap by a whole delta's worth of bytes just because it arrived in one
    SSE chunk.

    Graceful EOF without ``[DONE]``: if the connection closes cleanly before a
    ``[DONE]`` sentinel arrives, ``_iter_lines`` simply stops yielding (see its EOF
    branch below) and this loop ends normally — the accumulated content is returned
    as a best-effort completion, NOT an error. A missing terminator at EOF is
    tolerated by design (some servers/proxies close the socket without an explicit
    ``[DONE]`` after the last useful chunk).
    """
    parts: list[str] = []
    byte_count = 0  # UTF-8 BYTES (not code points/chars)
    usage: dict[str, Any] | None = None
    truncated = False
    buf = bytearray()

    def _iter_lines() -> Iterator[bytes]:
        """Yield newline-delimited SSE lines from `resp`, reassembling across
        `_READ_CHUNK_BYTES`-sized reads with a bounded buffer (`MAX_SSE_LINE_BYTES`).

        Buffer lifecycle: bytes accumulate in `buf` until a `\\n` is found, at which
        point everything up to (and excluding) it is yielded and drained from the
        front of the buffer. On a clean EOF (`resp.read()` returns `b""`), any
        TRAILING partial buffer (bytes accumulated with no terminating newline yet —
        the connection closed mid-line) is yielded ONCE more as a final best-effort
        "line" and then cleared, rather than silently discarded — this is what lets
        a `[DONE]`-less final delta still be recovered (see
        `test_stream_eof_without_done_sentinel_returns_accumulated_content`). If the
        trailing buffer is empty at EOF there is nothing left to yield. Either way
        the generator then returns (`StopIteration`), ending the `for raw in
        _iter_lines()` loop in `_consume` normally — a missing `[DONE]` sentinel is
        by design NOT an error (see the "Graceful EOF" note in `_consume`'s docstring).
        """
        while True:
            if time.monotonic() > deadline:  # anti slow-drip: deadline mid-stream
                raise TimeoutError("stream deadline exceeded")
            try:
                chunk = resp.read(_READ_CHUNK_BYTES)  # bounded read; socket timeout == idle timeout
            except (socket.timeout, TimeoutError):
                raise TimeoutError("stream idle timeout") from None
            except (OSError, http.client.IncompleteRead) as exc:
                # A NON-timeout OSError subclass (e.g. ConnectionResetError,
                # ConnectionAbortedError) OR an http.client.IncompleteRead — a truncated
                # read on a mid-stream connection drop, which is NOT an OSError subclass and
                # so must be named explicitly, exactly as the transactional core's _call_once
                # already does — reaching mid-stream must not leak as a raw transport
                # exception; map it to the domain OllamaBackendError, kept distinct from the
                # idle-timeout branch above.
                raise OllamaBackendError(redact(f"stream connection failed: {exc}")) from None
            if not chunk:  # EOF — NO [DONE] required here: a clean
                if buf:  # close without a sentinel is tolerated as
                    yield bytes(buf)  # a best-effort completion (see docstring).
                    del buf[:]
                return
            buf.extend(chunk)
            nl = buf.find(b"\n")
            while nl != -1:
                # Bound the LINE ITSELF (not just a still-incomplete trailing buffer):
                # a line whose terminating "\n" completes within/at a chunk boundary
                # would otherwise be yielded whole regardless of size — checking only
                # the leftover (no-newline-yet) buffer below misses exactly this case.
                if nl > MAX_SSE_LINE_BYTES:
                    raise OllamaBackendError("SSE line exceeds bound")
                yield bytes(buf[:nl])
                del buf[: nl + 1]
                nl = buf.find(b"\n")
            if len(buf) > MAX_SSE_LINE_BYTES:  # bound BEFORE a newline is ever seen
                raise OllamaBackendError("SSE line exceeds bound")

    for raw in _iter_lines():
        line = raw.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":") or not line.startswith(_SSE_PREFIX):
            continue
        data = line[len(_SSE_PREFIX) :].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except (json.JSONDecodeError, RecursionError):
            # A pathologically deeply-nested delta can raise RecursionError (a
            # theoretical DoS) — treat it exactly like malformed JSON and skip this
            # delta, same tolerant-parser discipline as MS1's `parse_output` (which
            # maps RecursionError -> a handled JSONDecodeError); never let it
            # propagate and crash the run.
            continue
        if not isinstance(chunk, dict):
            # A `data:` line that is valid JSON but not a JSON object (e.g. an array or a
            # bare scalar) is malformed for our envelope: skip it rather than crash on
            # `chunk.get("usage")` below (a non-dict has no `.get`) — R7b: a malformed
            # delta must never crash the run, only be tolerated/skipped.
            continue
        try:
            delta = chunk["choices"][0]["delta"].get("content")
        except (KeyError, IndexError, TypeError, AttributeError):
            delta = None
        if isinstance(delta, str) and delta:  # non-string content → skip, no TypeError
            encoded = delta.encode("utf-8")
            remaining = max_output_bytes - byte_count
            if remaining <= 0:  # already at/over the cap: drop it whole
                truncated = True
                break
            if len(encoded) > remaining:  # a SINGLE delta overshoots the cap —
                encoded = _truncate_utf8_bytes(encoded, remaining)  # truncate AT the boundary
                delta = encoded.decode("utf-8")  # safe: cut only on a char boundary
                truncated = True
            parts.append(delta)
            try:
                sink(delta)  # a sink failure is NOT a transport error
            except (OSError, ValueError) as exc:
                # A CLOSED stream raises `ValueError` ("I/O operation on closed file"),
                # which is NOT an `OSError` subclass — both are wrapped as the same
                # domain SinkError so a closed sink never leaks a raw ValueError.
                raise SinkError(redact(f"output sink write failed: {exc}")) from None
            byte_count += len(encoded)  # UTF-8 BYTES, not code points (CJK-safe)
            if truncated or byte_count >= max_output_bytes:
                truncated = True
                break
        if isinstance(chunk.get("usage"), dict):  # captured whether mid-stream or final
            usage = chunk["usage"]
    elapsed = time.monotonic() - start  # spans the FULL delegation (incl. 400s)
    content = "".join(parts)
    # Reuse MS2's fail-soft coercion (`_resolve_usage`/`_coerce_token_count`, imported
    # from `backend`) instead of a bare `int(...)` on the untrusted SSE `usage` object:
    # a non-numeric/missing/bool/inf/nan token count falls back to the local estimate
    # for BOTH counts together (never a partial real+estimated mix, never a non-domain
    # ValueError/TypeError crash on a malformed value from the stream).
    prompt_tokens, completion_tokens, estimated = _resolve_usage(
        usage, len(system_prompt) + len(prompt), content
    )
    return DelegationResult(
        content, prompt_tokens, completion_tokens, estimated, elapsed, truncated=truncated
    )
