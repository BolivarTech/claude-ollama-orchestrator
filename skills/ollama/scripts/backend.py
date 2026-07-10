# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Backend strategy contract + transactional OpenAI-compatible Ollama backend."""

from __future__ import annotations

import http.client
import json
import math
import random
import socket
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, NoReturn

from errors import OllamaBackendError, RateLimitError
from ollama_config import OllamaAgentsConfig
from validate import _truncate_utf8_bytes

_REDACTED = "***"
_BACKOFF_CAP_SECONDS = 30
# Absolute, always-on DoS backstop on the transactional response body (MS1 core): no
# config, no toggle — a runaway/hostile server must never be able to force an unbounded
# in-memory read. 64 MiB is generous enough that no legitimate chat/completions envelope
# ever approaches it, yet small enough to bound worst-case memory. MS6 LAYERS a tighter,
# config-driven `MAX_TRANSACTIONAL_BODY_BYTES` (derived from `max_output_bytes`, R24c) ON
# TOP of this bound — the two are complementary (MS6's can only tighten, never loosen,
# this floor), never conflicting.
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
# Error bodies (4xx/5xx diagnostic text/JSON) are typically small; cap the read
# independently and much tighter than MAX_RESPONSE_BYTES so a malicious/broken endpoint
# cannot force a large in-memory read via its *error* path either.
MAX_ERROR_BODY_BYTES = 64 * 1024

# R24c anti-runaway output cap (MS6): the content-level byte budget for a transactional
# delegation's extracted `content`. Matches `ollama_stream.DEFAULT_MAX_OUTPUT_BYTES` (MS4)
# so both the transactional and streaming paths share the SAME default budget unless
# overridden, and is the same default the layered config resolver falls back to when
# `max_output_bytes` is not overridden (Task 8).
DEFAULT_MAX_OUTPUT_BYTES = 2_000_000

# R24c: named, CALIBRATED headroom for the JSON envelope wrapping `content` — everything
# in the raw body OTHER than `content` itself: object/array punctuation, `id`/`model`/
# `object` string fields, `finish_reason`, an optional `usage` block, a `system_fingerprint`,
# an optional `logprobs` block, and (defensively) more than one `choices` entry if a server
# ignores this project's implicit `n=1`. This is NOT a second content-size cap — it only
# widens the RAW BODY bound so a well-formed, within-cap response is never falsely aborted
# just because the envelope around it costs a few extra bytes. 256 KiB is deliberately
# generous relative to a typical envelope's real overhead (usually well under a few KiB)
# while staying a small, fixed fraction of DEFAULT_MAX_OUTPUT_BYTES — not unbounded, and
# NOT itself user-configurable (only the content-level max_output_bytes is, Task 8).
_ENVELOPE_MARGIN_BYTES = 256 * 1024  # 256 KiB (recalibrated from an uncalibrated 64 KiB)


def _safe_close(closable: Any) -> None:
    """Best-effort ``close()`` on a response/error body.

    Args:
        closable: Any object that may expose a ``close`` method (an
            ``http.client.HTTPResponse``, ``urllib.error.HTTPError``, or a test double).

    A missing OR non-callable ``close`` attribute (some test doubles are plain objects,
    and a data attribute happening to be named ``close`` is not something to invoke) or a
    failure while closing must never mask the real result/exception already in flight —
    this is resource cleanup, not a correctness path. ``callable(close)`` covers both the
    missing (``None``) and non-callable cases, so a non-callable ``close`` never raises a
    ``TypeError`` out of this cleanup helper.
    """
    close = getattr(closable, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:  # noqa: BLE001
        # Resource cleanup must NEVER mask the real result/exception in flight. `close()`
        # most commonly raises `OSError`, but a stream can also raise a non-OSError (e.g.
        # a `ValueError` "I/O operation on closed file", or a driver-specific error), which
        # a narrow `except OSError` would let escape from a `finally` and REPLACE the real
        # exception. Swallow ANY close failure — this is best-effort teardown, not a
        # correctness path.
        pass


def estimate_tokens(text: str) -> int:
    """Estimate token count as ``len(text) // 4`` (stdlib heuristic; never raises).

    Public (MS4 Task 1): shared with ``ollama_stream.py`` so the transactional core
    and the streaming reader use the SAME estimator (DRY) — the local fail-soft
    fallback used whenever the server omits ``usage`` (R7a).
    """
    return len(text) // 4


def estimate_tokens_from_len(n: int) -> int:
    """Estimate tokens from a character count as ``n // 4`` — no string allocation.

    Public (MS4 Task 1): shared with ``ollama_stream.py`` (see ``estimate_tokens``).

    Args:
        n: A non-negative character length.

    Returns:
        The estimated token count.
    """
    return n // 4


def _coerce_token_count(value: Any) -> int | None:
    """Safely coerce an untrusted ``usage`` token count to a non-negative int.

    The ``usage`` object comes from an untrusted remote server and may be missing,
    a bool, a float, a string, or null — a bare ``int(...)`` on it can crash (on a
    string/None) or silently produce a wrong value. This coercion is fail-soft: it
    returns ``None`` (never raises) for anything it cannot safely represent,
    signaling the caller to fall back to the local estimate.

    Args:
        value: The raw value at ``usage["prompt_tokens"]``/``["completion_tokens"]``.

    Returns:
        A non-negative ``int`` if *value* is an ``int`` (excluding ``bool``, an
        ``int`` subclass that is never a valid token count here) or a ``float``
        that is both non-negative and ``is_integer()`` (this rejects ``inf``/``nan``,
        neither of which is ever integral). ``None`` for anything else.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer():
        return int(value) if value >= 0 else None
    return None


def _resolve_usage(
    usage: dict[str, Any] | None, prompt_len: int, content: str
) -> tuple[int, int, bool]:
    """Resolve ``(prompt_tokens, completion_tokens, estimated)`` fail-soft.

    Trusts the server's ``usage`` object only if BOTH ``prompt_tokens`` and
    ``completion_tokens`` coerce safely (see ``_coerce_token_count``); otherwise
    falls back to the local ``chars/4`` estimate for BOTH counts together — never
    a partial mix of one real + one estimated count, and never a crash on
    malformed/missing ``usage``.

    Args:
        usage: The parsed ``usage`` object from the response envelope, or ``None``.
        prompt_len: ``len(system_prompt) + len(prompt)``, pre-computed by the caller
            so no throwaway concatenated/padded string is allocated here.
        content: The extracted model content, for the completion-token estimate.

    Returns:
        A ``(prompt_tokens, completion_tokens, estimated)`` tuple.
    """
    if isinstance(usage, dict):
        prompt_tokens = _coerce_token_count(usage.get("prompt_tokens"))
        completion_tokens = _coerce_token_count(usage.get("completion_tokens"))
        if prompt_tokens is not None and completion_tokens is not None:
            return prompt_tokens, completion_tokens, False
    return estimate_tokens_from_len(prompt_len), estimate_tokens(content), True


@dataclass(frozen=True)
class DelegationResult:
    """The outcome of one delegation: content plus token metrics.

    ``tok_per_s`` is an END-TO-END **delivered-tokens-per-second** metric
    (``completion_tokens / elapsed_s``), where ``elapsed_s`` is the call's full
    wall-clock duration — network latency, any 429 backoff/retry waiting, and
    (transactional) the entire round-trip until the response is fully received are
    ALL included. It is NOT the model's raw generation/decode speed.

    Attributes:
        content: The extracted model content.
        prompt_tokens: Prompt tokens (from ``usage`` or estimated).
        completion_tokens: Completion tokens (from ``usage`` or estimated).
        estimated: True if the metrics were estimated (server omitted ``usage``).
        elapsed_s: End-to-end wall-clock seconds for the call.
        parsed: For a structured capability, the validated output dict; ``None`` for
            free-text. Lets ``dispatch`` return a single type carrying both the
            content and the parsed object.
        truncated: True when the output was cut short by an anti-runaway output-size
            cap. FIRST INTRODUCED in MS4 (Task 1): the streaming reader (Task 2) sets
            this when it hits its bounded-buffer/output cap; the transactional core
            never sets it here (MS6's R24c wires the transactional output cap onto
            this SAME field rather than introducing a new one). Defaults to ``False``
            so every MS1/MS2 construction site is unaffected.
    """

    content: str
    prompt_tokens: int
    completion_tokens: int
    estimated: bool
    elapsed_s: float
    parsed: dict[str, Any] | None = None
    truncated: bool = False

    @property
    def tok_per_s(self) -> float:
        """End-to-end delivered tokens/sec = completion_tokens / elapsed_s.

        Guarded: if ``elapsed_s`` is zero, (defensively) negative, or non-finite
        (``NaN``/``inf``), returns ``0.0`` instead of dividing — never raises
        ``ZeroDivisionError``, never returns a negative/absurd rate, and never
        propagates ``NaN``/``inf`` out of this public property (``NaN <= 0`` is
        ``False``, so an explicit ``isfinite`` check is required). NOT raw model
        generation speed.
        """
        if not math.isfinite(self.elapsed_s) or self.elapsed_s <= 0:
            return 0.0
        return round(self.completion_tokens / self.elapsed_s, 4)


class ResponseFormatRejected(Exception):
    """Signal: HTTP 400 rejected ``response_format`` → retry once without it (R11).

    An internal HTTP-flow control signal — deliberately NOT part of the domain
    exception hierarchy in ``errors.py`` (see MS4's import-consistency note): it
    never surfaces past ``OllamaBackend.run`` / the streaming reader's downgrade
    retry. Public (MS4 Task 1) so ``ollama_stream.py`` shares this ONE signal with
    the transactional core instead of duplicating it.
    """


class _RateLimited(Exception):
    """Internal signal: HTTP 429. Carries the raw ``HTTPError`` so the caller can
    compute the backoff delay via the shared ``retry_after_delay`` (DRY)."""

    def __init__(self, exc: urllib.error.HTTPError) -> None:
        super().__init__("rate limited")
        self.exc = exc


def build_chat_request(
    config: OllamaAgentsConfig,
    system_prompt: str,
    prompt: str,
    model: str,
    response_format: dict[str, Any] | None,
    *,
    stream: bool,
    content_parts: list[dict[str, Any]] | None = None,
) -> urllib.request.Request:
    """Build the shared ``/chat/completions`` request (transactional or streaming).

    Public (MS4 Task 1): both ``OllamaBackend`` (transactional, ``stream=False``)
    and ``ollama_stream.py`` (streaming, ``stream=True``) build their request
    through this ONE function, so auth/shape/response_format handling can never
    drift between the two paths (DRY).

    Args:
        config: Resolved config (``base_url``/``api_key``).
        system_prompt: Capability system prompt.
        prompt: Sanitized user prompt (used as the plain-string user content UNLESS
            *content_parts* is given).
        model: Resolved model tag.
        response_format: Structured-output shape, or ``None`` for free text
            (omitted from the body entirely when ``None``, never sent as JSON
            ``null``).
        stream: ``True`` adds ``"stream": true`` plus
            ``stream_options: {"include_usage": true}`` so the final SSE chunk
            still carries ``usage`` (R7a); ``False`` (MS1-MS6 default) is the
            transactional shape, byte-identical to before this helper existed.
        content_parts: Optional OpenAI-compatible multimodal content-parts array
            for the user message (e.g. ``[{"type": "text", "text": ...},
            {"type": "image_url", "image_url": {...}}]``). When provided, the user
            message's ``content`` is this list INSTEAD OF the plain *prompt*
            string. When ``None`` (the MS1-MS6 default), behavior is
            byte-identical to before this parameter existed. This is the
            **forward-compatible seam for MS7**: MS7's vision image transport
            fills ``content_parts`` for the ``image_url`` data-URI part WITHOUT
            any refactor of this shared helper.

    Returns:
        The prepared POST request (``Authorization`` header only if an api_key
        exists — R9).
    """
    user_content: str | list[dict[str, Any]] = (
        content_parts if content_parts is not None else prompt
    )
    body: dict[str, Any] = {
        "model": model,
        "stream": stream,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    if response_format is not None:
        body["response_format"] = response_format
    req = urllib.request.Request(
        f"{config.base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if config.api_key:
        req.add_header("Authorization", f"Bearer {config.api_key}")
    return req


def map_http_error(exc: BaseException, *, redact: Callable[[str], str]) -> NoReturn:
    """Map a transport error to a domain exception. Always raises (``NoReturn``) —
    callers never need a fall-through/re-raise after calling this.

    Public (MS4 Task 1), shared by ``OllamaBackend`` and ``ollama_stream.py``.

    HTTP 429 is deliberately NOT handled here: the caller must check
    ``exc.code == 429`` and branch to its own backoff loop (via
    ``retry_after_delay``) BEFORE calling this function — a 429 is not a terminal
    failure, so it must never reach this always-raises mapper.

    Args:
        exc: The transport exception to map (``HTTPError``, ``URLError``,
            ``socket.timeout``/``TimeoutError``, or any other transport failure).
        redact: A ``str -> str`` redactor (see ``make_redactor``) applied to every
            message so a configured ``api_key`` can never leak (NR3).

    Raises:
        ResponseFormatRejected: HTTP 400 rejecting ``response_format``/
            ``json_schema`` (→ downgrade retry).
        TimeoutError: socket timeout.
        OllamaBackendError: any other 4xx/5xx, unreachable host, or unexpected
            transport error (message redacted).
    """
    if isinstance(exc, urllib.error.HTTPError):
        detail = ""
        if exc.fp is not None:
            try:
                # Bounded read (MAX_ERROR_BODY_BYTES, MS1's DoS backstop, preserved
                # verbatim here): a malicious/broken endpoint must not be able to
                # force an unbounded in-memory read via its *error* body. Best-effort:
                # a read failure (e.g. the peer resets the socket mid-read of the
                # error page) degrades to an empty/unavailable detail rather than
                # letting this SECONDARY, non-domain failure mask the error already
                # being mapped below.
                raw_detail = exc.read(MAX_ERROR_BODY_BYTES + 1)
            except OSError:
                raw_detail = b""
            detail = raw_detail.decode("utf-8", errors="replace").lower()
        if exc.code == 400 and ("response_format" in detail or "json_schema" in detail):
            raise ResponseFormatRejected() from None
        raise OllamaBackendError(
            redact(f"Ollama HTTP {exc.code}: {exc.reason} {detail}".strip())
        ) from None
    if isinstance(exc, (socket.timeout, TimeoutError)):
        raise TimeoutError(redact(f"Delegation timed out: {exc}")) from None
    if isinstance(exc, urllib.error.URLError):
        raise OllamaBackendError(redact(f"Cannot reach Ollama: {exc.reason}")) from None
    raise OllamaBackendError(redact(f"Unexpected transport error: {exc}")) from None


def make_redactor(api_key: str | None) -> Callable[[str], str]:
    """Return a redactor that removes *api_key* from any message (NR3).

    Public (MS4 Task 1): both the transactional core (``OllamaBackend``) and the
    streaming reader build their redactor from this ONE source, so the api_key can
    never leak through an error message via either path. An empty/``None`` key
    yields an identity redactor (nothing secret to hide); a present key is always
    replaced with ``***``.

    Args:
        api_key: The resolved api_key, or ``None``.

    Returns:
        A ``str -> str`` redactor.
    """
    if not api_key:
        return lambda msg: msg
    secret = api_key
    return lambda msg: msg.replace(secret, _REDACTED)


def retry_after_delay(exc: urllib.error.HTTPError, attempt: int, rng: Callable[[], float]) -> float:
    """Backoff delay for a 429: honor ``Retry-After`` (numeric seconds or an
    RFC-7231 HTTP-date), else a bounded exponential backoff with jitter (R8).

    Public (MS4 Task 1), shared by ``OllamaBackend`` and ``ollama_stream.py`` so
    the two paths compute IDENTICAL 429 backoff delays.

    Args:
        exc: The 429 ``HTTPError`` (its ``Retry-After`` header, if present, wins).
        attempt: The zero-based backoff attempt number (used for the exponential
            fallback only).
        rng: Injectable random source in ``[0, 1)`` for deterministic jitter tests.

    Returns:
        The delay in seconds to sleep before the next attempt.
    """
    header = exc.headers.get("Retry-After") if exc.headers else None
    if header:
        header = header.strip()
        if header.isdigit():
            return float(header)
        try:  # best-effort HTTP-date (RFC 7231); else fall back to jitter below.
            when = parsedate_to_datetime(header)
            if when is not None:
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                return max((when - datetime.now(timezone.utc)).total_seconds(), 0.0)
        except (TypeError, ValueError, OverflowError):
            pass
    # `2**attempt` alone is typed `Any` by mypy (int.__pow__'s overload can't decide
    # int-vs-float for a non-literal exponent) — pin it to `int` explicitly so the
    # `Any` never silently leaks into this function's declared `-> float` return.
    capped_backoff: int = min(2**attempt, _BACKOFF_CAP_SECONDS)
    return capped_backoff * (0.5 + rng())


class AgentBackend(ABC):
    """Strategy interface: turn a delegation into extracted content text."""

    @abstractmethod
    def run(
        self,
        capability: str,
        system_prompt: str,
        prompt: str,
        model: str,
        timeout: int,
        *,
        response_format: dict[str, Any] | None = None,
        deadline: float | None = None,
    ) -> DelegationResult:
        """Run one delegation and return its result plus token metrics.

        Args:
            capability: The capability name (for logging/telemetry).
            system_prompt: The capability's system prompt.
            prompt: The user prompt (already sanitized upstream).
            model: The resolved model tag.
            timeout: Per-delegation socket timeout (seconds).
            response_format: Optional structured-output request shape.
            deadline: Optional shared monotonic deadline (R25). When the caller
                (``dispatch``) supplies its own deadline, the SAME instant bounds both
                the caller's parse-retry loop and this call's 429-backoff loop, so a
                delegation's total retry+backoff time can never exceed one budget. When
                ``None``, the backend derives ``time.monotonic() + timeout``.

        Returns:
            The ``DelegationResult`` (content plus token metrics).
        """
        raise NotImplementedError


class OllamaBackend(AgentBackend):
    """Transactional (``stream: False``) OpenAI-compatible backend over stdlib urllib."""

    def __init__(
        self,
        config: OllamaAgentsConfig,
        *,
        urlopen: Callable[..., Any] = urllib.request.urlopen,
        sleep: Callable[[float], None] | None = None,
        rng: Callable[[], float] | None = None,
        max_backoffs: int = 3,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        """Initialize the backend.

        Args:
            config: Resolved layered config (base_url, api_key, models, ...).
            urlopen: Injectable HTTP transport (defaults to ``urllib.request.urlopen``);
                tests replace this with a fake/mock at the ``urlopen`` edge, never real
                network.
            sleep: Injectable sleep function for deterministic 429-backoff tests.
            rng: Injectable random source (``[0, 1)``) for deterministic jitter tests.
            max_backoffs: Maximum number of 429 backoff attempts before giving up (R8).
            max_output_bytes: R24c anti-runaway output cap (MS6) — the content-level UTF-8
                byte budget for the extracted ``content``. Derives
                ``self._max_transactional_body_bytes``, the bound applied to the RAW body
                read (the JSON envelope wrapping ``content``, hence the added envelope
                margin) — clamped to never EXCEED the absolute, always-on
                ``MAX_RESPONSE_BYTES`` backstop, so a huge/misconfigured override can only
                tighten this bound, never loosen the floor.
        """
        self._config = config
        self._urlopen = urlopen
        self._sleep = sleep or time.sleep
        self._rng = rng or random.random
        self._max_backoffs = max_backoffs
        self._max_output_bytes = max_output_bytes
        self._max_transactional_body_bytes = min(
            MAX_RESPONSE_BYTES, max_output_bytes + _ENVELOPE_MARGIN_BYTES
        )
        # DRY (MS4 Task 1): built from the SAME `make_redactor` the streaming reader
        # uses, so the api_key can never leak through either path's error messages.
        self._redact: Callable[[str], str] = make_redactor(config.api_key)

    def _build_request(
        self, system_prompt: str, prompt: str, model: str, response_format: dict[str, Any] | None
    ) -> urllib.request.Request:
        """Build the OpenAI-compatible ``/chat/completions`` request (transactional).

        Thin wrapper over the shared ``build_chat_request`` (``stream=False``),
        kept as a method so the rest of this class's call sites are unchanged.

        Args:
            system_prompt: The capability's system prompt.
            prompt: The user prompt.
            model: The resolved model tag.
            response_format: Optional structured-output request shape (omitted from the
                body entirely when ``None``, not sent as JSON ``null``).

        Returns:
            A prepared, not-yet-sent :class:`urllib.request.Request`. The
            ``Authorization`` header is added only when ``api_key`` is present (R9).
        """
        return build_chat_request(
            self._config, system_prompt, prompt, model, response_format, stream=False
        )

    def _call(
        self, req: urllib.request.Request, deadline: float
    ) -> tuple[str, dict[str, Any] | None, bool]:
        """Execute *req*, backing off on HTTP 429 up to ``max_backoffs`` (R8), bounded by
        the per-delegation monotonic *deadline* (R25).

        429 is distinct from the parse-retry (R25): it respects ``Retry-After`` (or a
        bounded exponential jitter if the header is absent). A backoff that would sleep
        PAST *deadline* raises ``OllamaBackendError`` immediately instead of sleeping (so a
        429 storm with large ``Retry-After`` values can never exceed the delegation's time
        budget); once backoffs are exhausted it likewise raises.

        Args:
            req: The prepared request.
            deadline: ``time.monotonic()`` instant past which no further backoff is allowed;
                it also caps each attempt's socket timeout to the remaining budget, so the
                dead per-call ``timeout`` parameter is gone (the deadline is the single
                source of truth for the time budget).

        Returns:
            A ``(content, usage, truncated)`` tuple: the extracted ``content`` string
            (possibly truncated by the R24c anti-runaway output cap, MS6), the raw
            ``usage`` dict from the response envelope (``None`` if the server omitted
            it), and whether the content was truncated.

        Raises:
            OllamaBackendError: on HTTP/connection/shape failure, or on 429 exhaustion
                or deadline-exceeded (key redacted).
            TimeoutError: on socket timeout.
        """
        # `while True` (never `break`) is deliberate: mypy treats the code after a
        # break-less `while True` as unreachable, so this needs exactly ONE terminal
        # raise (the exhausted-backoffs branch below) instead of a `for` loop's trailing
        # "just in case" raise, which is dead code — every iteration either `return`s
        # (success) or `raise`s (exhausted backoffs on the final attempt, or
        # deadline-exceeded); it can never fall through.
        attempt = 0
        while True:
            # Clamp the per-call socket timeout to the REMAINING budget so a call plus a
            # retry can never together exceed the delegation deadline (closes the 2×timeout
            # gap; R25). Never below 1s so a nearly-elapsed deadline still attempts once.
            eff_timeout = max(deadline - time.monotonic(), 1.0)
            try:
                return self._call_once(req, eff_timeout)
            except _RateLimited as rl:
                if attempt >= self._max_backoffs:
                    # MS5 Task 3: raise the RateLimitError SUBTYPE (not the plain base
                    # class) so a caller (`run_batch` + the per-model circuit breaker,
                    # Task 5/Task 2) can `except RateLimitError` — an isinstance check,
                    # not a string match — ahead of the generic `OllamaBackendError`
                    # branch, and exclude a healthy-but-throttled model from tripping
                    # its breaker. It IS-A OllamaBackendError, so this is additive and
                    # backward-compatible: every existing `except OllamaBackendError`
                    # still catches it unchanged.
                    raise RateLimitError(
                        self._redact(
                            f"Ollama rate-limited (429) after {self._max_backoffs} backoffs; "
                            "the model is healthy but throttled — retry later or reduce "
                            "concurrency."
                        )
                    ) from None
                # DRY (MS4 Task 1): the SAME `retry_after_delay` helper the streaming
                # reader uses computes this delay (Retry-After header, numeric or
                # HTTP-date, else bounded exponential + jitter) — identical to the
                # inline logic this replaces.
                delay = retry_after_delay(rl.exc, attempt, self._rng)
                if time.monotonic() + delay > deadline:
                    # A deadline hit DURING a 429 backoff is still throttling (R14b), not a
                    # dead model — raise the RateLimitError SUBTYPE (same as the exhausted-
                    # backoffs branch above) so `_execute_delegation`'s `except RateLimitError`
                    # arm EXCLUDES it from the per-model breaker. A plain OllamaBackendError
                    # here would trip the breaker for a healthy-but-throttled model whose
                    # backoff merely outran the delegation's time budget. RateLimitError
                    # IS-A OllamaBackendError, so every `except OllamaBackendError` still
                    # catches it unchanged (additive, backward-compatible).
                    raise RateLimitError(
                        self._redact(
                            "Ollama 429 rate limit: retry deadline exceeded before backoff."
                        )
                    ) from None
                self._sleep(delay)
                attempt += 1

    def _call_once(
        self, req: urllib.request.Request, timeout: float
    ) -> tuple[str, dict[str, Any] | None, bool]:
        """Execute *req* once and extract ``(content, usage, truncated)``, or raise.

        Args:
            req: The prepared request.
            timeout: Socket timeout for this single attempt (already clamped to the
                remaining deadline budget by the caller).

        Returns:
            A ``(content, usage, truncated)`` tuple: the extracted ``content`` string
            (possibly truncated by the R24c anti-runaway output cap, MS6), the raw
            ``usage`` dict from the SAME already bounded-read/decoded/parsed ``payload``
            (``None`` if the server omitted it) — no additional I/O or decoding — and
            whether ``content`` was truncated.

        Raises:
            ResponseFormatRejected: on a 400 that mentions ``response_format``/
                ``json_schema`` (caller downgrades and retries once).
            _RateLimited: on a 429 (caller backs off).
            OllamaBackendError: on any other HTTP error, connection failure, oversized
                response (either the absolute ``MAX_RESPONSE_BYTES`` backstop or the
                R24c ``max_output_bytes``-derived bound), non-JSON body, unexpected
                envelope shape, or null content (all messages redacted).
            TimeoutError: on socket timeout.
        """
        resp: Any = None
        try:
            resp = self._urlopen(req, timeout=timeout)
            # R24c (MS6): bound the RAW read itself — never read-then-truncate. The
            # transactional body is a JSON ENVELOPE (`content` lives inside
            # choices[0].message.content), so a byte-truncated envelope is not reliably
            # parseable; a runaway response must therefore be ABORTED, not truncated
            # after the fact. `self._max_transactional_body_bytes` (derived from
            # `max_output_bytes` + `_ENVELOPE_MARGIN_BYTES`, clamped to never exceed the
            # absolute `MAX_RESPONSE_BYTES` backstop) is the bound; `+1` is exactly
            # enough to distinguish "at the bound" from "over the bound" without reading
            # a single byte further (mirrors `binary_input.load_binary`'s bounded read) —
            # a runaway response is NEVER fully read into memory.
            raw = resp.read(self._max_transactional_body_bytes + 1)
            if len(raw) > self._max_transactional_body_bytes:
                # The transactional body is a JSON envelope — a byte-truncated envelope
                # is not reliably parseable, so this ABORTS the transaction (R24c:
                # "aborta la transacción") rather than truncating raw bytes and trying
                # to parse a corrupted JSON fragment.
                print(
                    f"WARNING: transactional response exceeded "
                    f"{self._max_transactional_body_bytes} bytes; aborting "
                    "(R24c anti-runaway output cap) — use streaming for large outputs.",
                    file=sys.stderr,
                )
                raise OllamaBackendError(
                    self._redact(
                        "transactional response exceeded max body size — "
                        "use streaming for large outputs"
                    )
                )
            # The chat-completions response body is untrusted server output: decode
            # with errors="replace" (U+FFFD substitution) so malformed UTF-8 bytes
            # never raise a raw UnicodeDecodeError (a non-domain exception) — a
            # still-invalid-JSON result after substitution surfaces as the existing
            # OllamaBackendError below.
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            try:
                # 429 is handled HERE, BEFORE the shared `map_http_error` (DRY, MS4
                # Task 1): it is not a terminal failure — the caller (`_call`) backs
                # off and retries, computing the delay via the shared
                # `retry_after_delay` (identical logic to the streaming reader's).
                if exc.code == 429:
                    raise _RateLimited(exc) from None
                # Every other HTTPError (400 downgrade-signal or any other 4xx/5xx)
                # goes through the ONE shared mapper both the transactional core and
                # the streaming reader use — same bounded error-body read
                # (MAX_ERROR_BODY_BYTES), same redaction, same downgrade detection.
                map_http_error(exc, redact=self._redact)
            finally:
                _safe_close(exc)
        except (socket.timeout, TimeoutError) as exc:
            map_http_error(exc, redact=self._redact)
        except urllib.error.URLError as exc:
            map_http_error(exc, redact=self._redact)
        except (OSError, http.client.IncompleteRead) as exc:
            # Any other connect/read failure not already covered above (e.g. a
            # ConnectionResetError or a truncated read on a mid-response connection
            # drop, realistic for `:cloud` endpoints) must map to a domain error, never
            # propagate as a raw non-domain exception. NOTE: urllib.error.URLError (and
            # its HTTPError subclass) are themselves OSError subclasses, so both are
            # caught above with their more actionable messages BEFORE this catch-all.
            raise OllamaBackendError(self._redact(f"Ollama connection error: {exc}")) from None
        except (json.JSONDecodeError, RecursionError) as exc:
            # A proxy/gateway can return 200 with a non-JSON body (e.g. b'<html>502
            # Bad Gateway</html>'), OR a malicious/hostile server can return a deeply
            # nested JSON body that trips Python's recursion limit inside json.loads
            # (RecursionError is a RuntimeError, NOT a JSONDecodeError, so it is NOT
            # caught by `except json.JSONDecodeError` alone) — map BOTH to a domain
            # error instead of letting either crash the process with a raw exception.
            raise OllamaBackendError(self._redact("Unexpected response: not valid JSON")) from exc
        finally:
            _safe_close(resp)
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OllamaBackendError(
                self._redact("Unexpected OpenAI-compatible response shape")
            ) from exc
        if content is None:
            # A model that returns a null `content` (e.g. an empty/failed completion) must
            # surface as a domain error, not str(None) == "None" — a literal "None" string
            # would silently masquerade as real output for Claude to review/apply.
            raise OllamaBackendError(
                self._redact("Ollama returned null content (model produced no output)")
            )
        content_str = json.dumps(content) if isinstance(content, dict) else str(content)
        # R24c (MS6) secondary guard: the raw-body bound above already ensures the
        # ENVELOPE was never unboundedly read; this caps the EXTRACTED content itself to
        # `max_output_bytes` UTF-8 bytes (never code points — see `_truncate_utf8_bytes`,
        # shared with `ollama_stream.py`, Task 3), so a within-bound envelope carrying a
        # content string right at the edge of the budget is still capped precisely.
        content_str, truncated = _truncate_utf8_bytes(content_str, self._max_output_bytes)
        if truncated:
            print(
                f"WARNING: response truncated at {self._max_output_bytes} bytes "
                "(R24c anti-runaway output cap).",
                file=sys.stderr,
            )
        # NEW in MS2: read `usage` from the SAME already bounded-read/decoded/parsed
        # `payload` — zero additional I/O, zero additional decoding.
        usage = payload.get("usage")
        return content_str, usage if isinstance(usage, dict) else None, truncated

    def run(
        self,
        capability: str,
        system_prompt: str,
        prompt: str,
        model: str,
        timeout: int,
        *,
        response_format: dict[str, Any] | None = None,
        deadline: float | None = None,
    ) -> DelegationResult:
        """Run one delegation transactionally; downgrade once on a 400 that rejects
        ``response_format``.

        Args:
            capability: The capability name (for logging/telemetry).
            system_prompt: The capability's system prompt.
            prompt: The user prompt (already sanitized upstream).
            model: The resolved model tag.
            timeout: Per-delegation socket timeout (seconds).
            response_format: Optional structured-output request shape.
            deadline: Optional shared monotonic deadline (R25). When ``dispatch`` passes
                its own deadline, that SAME instant bounds this call's 429-backoff loop,
                so the delegation's parse-retry (dispatch) + 429-backoff (here) share ONE
                time budget instead of each getting a fresh ``timeout``. ``None`` → derive
                ``time.monotonic() + timeout``.

        Returns:
            A ``DelegationResult`` with real ``usage`` counts when the server provides
            them safely (both coerce; see ``_coerce_token_count``), else a fail-soft
            local estimate for BOTH counts (``estimated=True``). Never raises on
            malformed/missing ``usage``.

        Raises:
            OllamaBackendError: on HTTP/connection/shape failure (key redacted).
            TimeoutError: on socket timeout.
        """
        # Reuse the caller's deadline so 429-backoff can't extend beyond the delegation's
        # single R25 budget; only derive one when called directly (deadline is None).
        if deadline is None:
            deadline = time.monotonic() + timeout
        # `capability` is intentionally unused in this transactional core; it is threaded
        # through the ``AgentBackend`` contract for MS2 telemetry (token accounting by
        # capability/model), not dead code.
        req = self._build_request(system_prompt, prompt, model, response_format)
        prompt_len = len(system_prompt) + len(prompt)
        start = time.monotonic()
        try:
            content, usage, truncated = self._call(req, deadline)
        except ResponseFormatRejected:
            downgraded = self._build_request(system_prompt, prompt, model, None)
            try:
                content, usage, truncated = self._call(downgraded, deadline)
            except ResponseFormatRejected:
                # The DOWNGRADED (no response_format) request was STILL rejected as a
                # response_format/json_schema 400 — e.g. a proxy whose error body
                # generically mentions those terms regardless of what was actually sent.
                # No further downgrade is possible; surface a domain error instead of
                # letting this internal signal escape the module boundary.
                raise OllamaBackendError(
                    self._redact(
                        "Ollama rejected response_format on both the original and the "
                        "downgraded (no response_format) request; no further downgrade "
                        "is possible."
                    )
                ) from None
        elapsed = time.monotonic() - start
        prompt_tokens, completion_tokens, estimated = _resolve_usage(usage, prompt_len, content)
        return DelegationResult(
            content, prompt_tokens, completion_tokens, estimated, elapsed, truncated=truncated
        )
