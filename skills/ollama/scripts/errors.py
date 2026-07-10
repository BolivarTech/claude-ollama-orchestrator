# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Domain exceptions for the Ollama delegation runtime.

Hierarchy (deliberate): config/preflight failures are ``ValidationError``
subclasses so callers can catch the domain family; ``InvalidInputError`` is a
*sibling* of ``ValidationError`` (both inherit ``Exception``) so a fail-closed
security event is never swallowed by ``except (ValidationError, ...)`` in the
retry path.
"""

from __future__ import annotations


class ValidationError(Exception):
    """Base domain error for config/preflight validation failures."""


class OllamaConfigError(ValidationError):
    """Raised when config resolution fails (malformed TOML, invalid value)."""


class OllamaPreflightError(ValidationError):
    """Raised when preflight fails (unreachable host, missing model, auth)."""


class OllamaBackendError(Exception):
    """Raised on transport/backend failures (HTTP, timeout, bad response)."""


class RateLimitError(OllamaBackendError):
    """Raised when a 429 (rate limit) exhausts its backoff budget (R8).

    A subclass of :class:`OllamaBackendError` — deliberately additive and
    backward-compatible: every MS1 ``except OllamaBackendError`` still catches
    it unchanged. It exists so callers that need to distinguish "the model is
    unreachable/erroring" from "the model is healthy but throttling" (R14b: the
    per-model circuit breaker must NOT trip on 429) can do a clean
    ``isinstance``/``except RateLimitError`` check instead of a string match on
    the error message.
    """


class SinkError(OllamaBackendError):
    """A failure WRITING a delta to a streaming output sink (stdout/file), raised by
    `ollama_stream._consume` (MS4). A `BrokenPipeError`/`OSError` from the sink is
    wrapped as this distinct type so it is never misreported as a transport/HTTP
    fault; it is still an `OllamaBackendError`, so callers that only catch that
    base type (e.g. `run_ollama.main`) still report it actionably, while callers
    that need to distinguish "the model/transport failed" from "writing the output
    failed" (e.g. `run_ollama.dispatch`) can catch `SinkError` specifically.
    """


class DelegationError(Exception):
    """Raised on orchestration failures (queue full, deadline exceeded)."""


class InvalidInputError(Exception):
    """Fail-closed security event (e.g. nonce collision). NOT a ValidationError."""
