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
        OllamaConfigError: if *raw* has a scheme but no authority/host (e.g.
            ``"http://"``) — stripping trailing slashes from a bare scheme
            would otherwise collapse it to ``"http:"``, which then looks
            scheme-less and gets a bogus ``"http://"`` re-prepended
            (``"http://http:"``). An empty authority is always invalid.
    """
    stripped = raw.strip()
    if not stripped:
        raise OllamaConfigError(
            "base_url must not be empty; leave it unset to use the default "
            f"({DEFAULT_BASE_URL}) or provide a valid host/URL"
        )
    # Decide scheme-presence on the merely-stripped value, *before* trailing
    # slashes are removed — for a bare scheme like "http://" every trailing
    # char is "/", so rstrip below would erase the "://" marker itself and
    # make the value look scheme-less to a check performed afterwards.
    has_scheme = "://" in stripped
    value = stripped.rstrip("/")
    if not value:
        raise OllamaConfigError(
            "base_url must not be empty; leave it unset to use the default "
            f"({DEFAULT_BASE_URL}) or provide a valid host/URL"
        )
    if not has_scheme:
        value = "http://" + value
    parts = urlsplit(value)
    if not parts.netloc:
        raise OllamaConfigError(
            f"base_url {raw!r} has no host (empty authority after the "
            "scheme); provide a value like 'http://localhost:11434'"
        )
    if not parts.path:
        parts = parts._replace(path="/v1")
    return urlunsplit(parts)
