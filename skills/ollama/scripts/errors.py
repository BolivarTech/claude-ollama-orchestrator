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


class DelegationError(Exception):
    """Raised on orchestration failures (queue full, deadline exceeded)."""


class InvalidInputError(Exception):
    """Fail-closed security event (e.g. nonce collision). NOT a ValidationError."""
