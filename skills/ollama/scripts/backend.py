# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Backend strategy contract + transactional OpenAI-compatible Ollama backend."""

from __future__ import annotations

import http.client
import json
import random
import socket
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable

from errors import OllamaBackendError
from ollama_config import OllamaAgentsConfig

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


def _safe_close(closable: Any) -> None:
    """Best-effort ``close()`` on a response/error body.

    Args:
        closable: Any object that may expose a ``close`` method (an
            ``http.client.HTTPResponse``, ``urllib.error.HTTPError``, or a test double).

    A missing ``close`` attribute (some test doubles are plain objects) or a failure
    while closing must never mask the real result/exception already in flight — this is
    resource cleanup, not a correctness path.
    """
    close = getattr(closable, "close", None)
    if close is None:
        return
    try:
        close()
    except OSError:
        pass


class _ResponseFormatRejected(Exception):
    """Internal signal: the server rejected ``response_format`` (400) → downgrade."""


class _RateLimited(Exception):
    """Internal signal: HTTP 429. Carries the parsed ``Retry-After`` (or None)."""

    def __init__(self, retry_after: float | None) -> None:
        super().__init__("rate limited")
        self.retry_after = retry_after


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
    ) -> str:
        """Run one delegation and return the extracted ``content`` string.

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
            The extracted ``content`` string.
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
        """
        self._config = config
        self._urlopen = urlopen
        self._sleep = sleep or time.sleep
        self._rng = rng or random.random
        self._max_backoffs = max_backoffs

    def _redact(self, message: str) -> str:
        """Redact the configured ``api_key`` (if any) from an error message (NR3)."""
        key = self._config.api_key
        return message.replace(key, _REDACTED) if key else message

    def _build_request(
        self, system_prompt: str, prompt: str, model: str, response_format: dict[str, Any] | None
    ) -> urllib.request.Request:
        """Build the OpenAI-compatible ``/chat/completions`` request.

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
        body: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if response_format is not None:
            body["response_format"] = response_format
        req = urllib.request.Request(
            f"{self._config.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self._config.api_key:
            req.add_header("Authorization", f"Bearer {self._config.api_key}")
        return req

    def _call(self, req: urllib.request.Request, deadline: float) -> str:
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
            The extracted ``content`` string.

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
                    raise OllamaBackendError(
                        self._redact(
                            "Ollama 429 rate limit: backoffs exhausted; raise your plan "
                            "or lower max_parallel_agents."
                        )
                    ) from None
                delay = (
                    rl.retry_after
                    if rl.retry_after is not None
                    else min(2**attempt, _BACKOFF_CAP_SECONDS) * (0.5 + self._rng())
                )
                if time.monotonic() + delay > deadline:
                    raise OllamaBackendError(
                        self._redact(
                            "Ollama 429 rate limit: retry deadline exceeded before backoff."
                        )
                    ) from None
                self._sleep(delay)
                attempt += 1

    def _call_once(self, req: urllib.request.Request, timeout: float) -> str:
        """Execute *req* once and extract the ``content`` string, or raise.

        Args:
            req: The prepared request.
            timeout: Socket timeout for this single attempt (already clamped to the
                remaining deadline budget by the caller).

        Returns:
            The extracted ``content`` string.

        Raises:
            _ResponseFormatRejected: on a 400 that mentions ``response_format``/
                ``json_schema`` (caller downgrades and retries once).
            _RateLimited: on a 429 (caller backs off).
            OllamaBackendError: on any other HTTP error, connection failure, oversized
                response, non-JSON body, unexpected envelope shape, or null content
                (all messages redacted).
            TimeoutError: on socket timeout.
        """
        resp: Any = None
        try:
            resp = self._urlopen(req, timeout=timeout)
            # Absolute DoS backstop (MAX_RESPONSE_BYTES, always-on): read at most ONE
            # byte past the bound so a runaway/hostile server can never force an
            # unbounded in-memory load — never a bare resp.read(). MS6 layers a tighter,
            # config-driven MAX_TRANSACTIONAL_BODY_BYTES (derived from max_output_bytes)
            # on top of this floor; the two bounds are complementary, not conflicting.
            raw = resp.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise OllamaBackendError(
                    self._redact(
                        f"response body exceeded MAX_RESPONSE_BYTES ({MAX_RESPONSE_BYTES} "
                        "bytes) — server sent an oversized response"
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
                detail = ""
                if exc.fp is not None:
                    # Bounded read, mirroring the success-path backstop above: a
                    # malicious/broken endpoint must not be able to force an unbounded
                    # in-memory read via its *error* body either. Best-effort: a read
                    # failure degrades to an empty detail rather than masking exc.
                    try:
                        raw_detail = exc.read(MAX_ERROR_BODY_BYTES + 1)
                    except OSError:
                        raw_detail = b""
                    detail = raw_detail.decode("utf-8", errors="replace").lower()
                if exc.code == 400 and ("response_format" in detail or "json_schema" in detail):
                    raise _ResponseFormatRejected() from None
                if exc.code == 429:
                    header = exc.headers.get("Retry-After") if exc.headers else None
                    retry_after: float | None = None
                    if header:
                        header = header.strip()
                        if header.isdigit():
                            retry_after = float(header)
                        else:  # best-effort HTTP-date (RFC 7231); else fall back to jitter.
                            try:
                                when = parsedate_to_datetime(header)
                                if when is not None:
                                    if when.tzinfo is None:
                                        when = when.replace(tzinfo=timezone.utc)
                                    retry_after = max(
                                        (when - datetime.now(timezone.utc)).total_seconds(),
                                        0.0,
                                    )
                            except (TypeError, ValueError, OverflowError):
                                retry_after = None
                    raise _RateLimited(retry_after) from None
                raise OllamaBackendError(
                    self._redact(f"Ollama HTTP {exc.code}: {exc.reason} {detail}".strip())
                ) from None
            finally:
                _safe_close(exc)
        except (socket.timeout, TimeoutError) as exc:
            raise TimeoutError(self._redact(f"Delegation timed out: {exc}")) from None
        except urllib.error.URLError as exc:
            raise OllamaBackendError(
                self._redact(f"Cannot reach Ollama at {self._config.base_url}: {exc.reason}")
            ) from None
        except (OSError, http.client.IncompleteRead) as exc:
            # Any other connect/read failure not already covered above (e.g. a
            # ConnectionResetError or a truncated read on a mid-response connection
            # drop, realistic for `:cloud` endpoints) must map to a domain error, never
            # propagate as a raw non-domain exception. NOTE: urllib.error.URLError (and
            # its HTTPError subclass) are themselves OSError subclasses, so both are
            # caught above with their more actionable messages BEFORE this catch-all.
            raise OllamaBackendError(self._redact(f"Ollama connection error: {exc}")) from None
        except json.JSONDecodeError as exc:
            # A proxy/gateway can return 200 with a non-JSON body (e.g. b'<html>502
            # Bad Gateway</html>'); map it to a domain error instead of crashing.
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
        return json.dumps(content) if isinstance(content, dict) else str(content)

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
    ) -> str:
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
            The extracted ``content`` string.

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
        try:
            return self._call(req, deadline)
        except _ResponseFormatRejected:
            downgraded = self._build_request(system_prompt, prompt, model, None)
            try:
                return self._call(downgraded, deadline)
            except _ResponseFormatRejected:
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
