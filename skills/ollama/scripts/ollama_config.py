# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Layered, per-key config resolver for the Ollama delegation runtime."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from errors import OllamaConfigError

DEFAULT_BASE_URL = "http://localhost:11434/v1"


def normalize_base_url(raw: str) -> str:
    """Normalize a base URL idempotently.

    Strip a trailing ``/``; prepend ``http://`` if no scheme; append ``/v1``
    **only** when the authority has no path (a value already carrying a path is
    used verbatim, never ``/v1/v1``).

    Args:
        raw: Raw base URL from config/env. Callers going through
            ``resolve_config`` are guaranteed a ``str`` here — every TOML/env
            source of ``base_url`` is type-checked via ``_require_str`` (Task 3)
            before it ever reaches this function, so this function itself does
            not re-check the type, only the empty/whitespace content below.

    Returns:
        The normalized OpenAI-compatible base URL.

    Raises:
        OllamaConfigError: if *raw* is empty/whitespace-only (or collapses to
            empty after stripping trailing slashes) — an explicit empty value is
            a misconfiguration, distinct from an *unset* base_url (which
            resolve_config's per-key precedence already falls back to
            ``DEFAULT_BASE_URL`` for, never reaching this function empty).
    """
    value = raw.strip().rstrip("/")
    if not value:
        raise OllamaConfigError(
            "base_url must not be empty; leave it unset to use the default "
            f"({DEFAULT_BASE_URL}) or provide a valid host/URL"
        )
    if "://" not in value:
        value = "http://" + value
    parts = urlsplit(value)
    if not parts.path:
        parts = parts._replace(path="/v1")
    return urlunsplit(parts)
