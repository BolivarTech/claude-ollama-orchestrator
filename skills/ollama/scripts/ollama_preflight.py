# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Fail-fast preflight for the Ollama endpoint."""

from __future__ import annotations

import http.client
import json
import socket
import sys
import urllib.error
import urllib.request
from typing import Any, Callable

from errors import OllamaPreflightError
from ollama_config import OllamaAgentsConfig

PREFLIGHT_TIMEOUT = 10
# Bounded read for the /models body (mirrors backend.MAX_RESPONSE_BYTES, Task 6): a
# smaller, still-generous bound since a model listing is far lighter than a chat
# completion. No legitimate /models response ever approaches this; it exists purely as
# an always-on DoS backstop against a runaway/hostile endpoint.
PREFLIGHT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_REDACTED = "***"


def _redact(message: str, api_key: str | None) -> str:
    """Replace *api_key* with ``***`` in *message* (never leak the key).

    Args:
        message: The error message to sanitize.
        api_key: The configured API key, or ``None`` if unset.

    Returns:
        *message* with every occurrence of *api_key* replaced by ``***``; the
        message unchanged if *api_key* is ``None``/empty.
    """
    return message.replace(api_key, _REDACTED) if api_key else message


def _safe_close(closable: Any) -> None:
    """Best-effort ``close()`` on a response/error body (mirrors ``backend._safe_close``).

    Args:
        closable: Any object that may expose a ``close`` method (an
            ``http.client.HTTPResponse``, ``urllib.error.HTTPError``, ``None``, or a
            test double).

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


def preflight(
    config: OllamaAgentsConfig,
    *,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
    capability: str | None = None,
    effective_model: str | None = None,
) -> None:
    """Verify the endpoint is reachable and all configured models exist.

    Performs a ``GET {base_url}/models`` with a short timeout, before any
    delegation is attempted (R10). An unreachable host or a missing configured
    model aborts fail-fast with an actionable message; a 401/403 aborts as an
    auth failure; a 404/501 (endpoint doesn't support listing models) warns
    once to stderr and proceeds without the model-existence check, since a
    missing model then only surfaces later as a chat-time 404 (TOCTOU,
    accepted per spec).

    Args:
        config: The resolved configuration.
        urlopen: Injectable ``urlopen`` (the test suite mocks HTTP here).
        capability: When paired with *effective_model*, the capability whose
            configured model should be REPLACED by *effective_model* in the
            model-existence check below — i.e. validate what will ACTUALLY be
            delegated (a ``--model`` override), not the config default that
            would otherwise be checked in its place. ``None`` leaves every
            configured model checked unchanged (the prior, still-default
            behavior).
        effective_model: The model tag that will actually be used for
            *capability* (``ns.model or cfg.models[capability]`` — R28).
            Ignored unless *capability* is also given.

    Raises:
        OllamaPreflightError: on unreachable host, auth failure (401/403), a
            missing configured (or effective-override) model, an oversized
            response body (``PREFLIGHT_MAX_RESPONSE_BYTES``), a non-JSON
            response body (e.g. an HTML error page from a misconfigured proxy,
            or malformed UTF-8 bytes decoded via ``errors="replace"``), a
            mid-read connection drop (``http.client.IncompleteRead`` or any
            other ``OSError`` such as ``ConnectionResetError``), or a
            valid-JSON body of the wrong shape (not an object, or its ``data``
            key not a list) on an otherwise-200 response. 404/501 warn-and-proceed
            instead of raising.
    """
    url = f"{config.base_url}/models"
    req = urllib.request.Request(url, method="GET")
    if config.api_key:
        req.add_header("Authorization", f"Bearer {config.api_key}")
    resp: Any = None
    try:
        resp = urlopen(req, timeout=PREFLIGHT_TIMEOUT)
        # Bounded read (mirrors backend.MAX_RESPONSE_BYTES): read at most ONE byte past
        # the bound so a runaway/hostile endpoint can never force an unbounded in-memory
        # load — never a bare resp.read().
        raw = resp.read(PREFLIGHT_MAX_RESPONSE_BYTES + 1)
        if len(raw) > PREFLIGHT_MAX_RESPONSE_BYTES:
            raise OllamaPreflightError(
                _redact(
                    f"/models response exceeded PREFLIGHT_MAX_RESPONSE_BYTES "
                    f"({PREFLIGHT_MAX_RESPONSE_BYTES} bytes) — endpoint sent an oversized "
                    "response",
                    config.api_key,
                )
            )
        # The /models response body is untrusted server output: decode with
        # errors="replace" (U+FFFD substitution) so malformed UTF-8 bytes never raise a
        # raw UnicodeDecodeError (a non-domain exception) — a still-invalid-JSON result
        # after substitution surfaces as the existing OllamaPreflightError below.
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        try:
            if exc.code in (401, 403):
                raise OllamaPreflightError(
                    _redact(
                        f"Preflight auth failed ({exc.code}): check api_key / `ollama signin`.",
                        config.api_key,
                    )
                ) from None
            if exc.code in (404, 501):
                print(
                    "WARNING: endpoint does not support listing models; skipping model check.",
                    file=sys.stderr,
                )
                return
            raise OllamaPreflightError(
                _redact(f"Preflight HTTP {exc.code} at {url}.", config.api_key)
            ) from None
        finally:
            # The HTTPError itself carries the (unread here) error body; close it on
            # every path — auth-abort, warn-and-proceed, or generic-abort — mirroring
            # backend.OllamaBackend's own cleanup of the HTTPError body (no fd leak).
            _safe_close(exc)
    except (socket.timeout, TimeoutError, urllib.error.URLError, http.client.IncompleteRead) as exc:
        raise OllamaPreflightError(
            _redact(
                f"Cannot reach Ollama at {config.base_url}: {exc}. "
                "Is it running? Try `ollama signin` for cloud.",
                config.api_key,
            )
        ) from None
    except (json.JSONDecodeError, RecursionError) as exc:
        # A reverse proxy / misconfigured endpoint can return HTTP 200 with a non-JSON
        # body (e.g. an HTML error page), OR a malicious/hostile endpoint can return a
        # deeply nested JSON body that trips Python's recursion limit inside json.loads
        # (RecursionError is a RuntimeError, NOT a JSONDecodeError, so it is NOT caught
        # by `except json.JSONDecodeError` alone). Map BOTH to a domain error instead of
        # letting either crash the process with a raw traceback.
        raise OllamaPreflightError(
            _redact(
                f"Preflight failed: {url} returned a non-JSON response ({exc}).", config.api_key
            )
        ) from None
    except OSError as exc:
        # Any other connect/read failure not already covered above (e.g. a
        # ConnectionResetError on a mid-read connection drop) must map to a domain
        # error, never propagate as a raw non-domain exception. NOTE: urllib.error.URLError
        # (and its HTTPError subclass) are themselves OSError subclasses, so both are
        # caught above with their more actionable messages BEFORE this catch-all.
        raise OllamaPreflightError(
            _redact(f"Preflight connection error: {exc}", config.api_key)
        ) from None
    finally:
        # Close the /models response on EVERY path — success, oversized-body abort, or a
        # shape/JSON error raised further below — never leave the socket/fd open
        # (mirrors backend.OllamaBackend's `finally: _safe_close(resp)`). `resp` is
        # `None` here whenever `urlopen` itself raised (e.g. the HTTPError/URLError
        # arms above), and `_safe_close(None)` is a no-op.
        _safe_close(resp)

    # The /models body is untrusted server output: it may be valid JSON of the WRONG
    # shape (a bare array/string/number/null, or {"data": null}). Guard the shape BEFORE
    # iterating so a malformed-but-valid-JSON body maps to the domain error this module
    # exists to guarantee, instead of a raw AttributeError (payload not a dict) or
    # TypeError (payload["data"] not a list — note `.get("data", [])`'s default only
    # applies when the key is ABSENT, not when it is present with value `null`).
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise OllamaPreflightError(
            _redact(
                f"Preflight failed: {url} returned an unexpected /models response shape "
                "(expected a JSON object with a `data` list).",
                config.api_key,
            )
        )
    available = {m["id"] for m in data if isinstance(m, dict) and "id" in m}
    # R28/R10: validate the EFFECTIVE model — a caller-supplied --model override for
    # *capability* takes the place of that capability's configured model in the check,
    # so an override to a non-existent model aborts HERE (actionable) instead of
    # surfacing as a later chat-time 404 (TOCTOU-acceptable-but-late). Every other
    # capability's configured model is still checked unchanged.
    check_models = dict(config.models)
    if capability is not None and effective_model is not None:
        check_models[capability] = effective_model
    missing = sorted(set(check_models.values()) - available)
    if missing:
        raise OllamaPreflightError(
            _redact(
                f"Missing models: {', '.join(missing)}. "
                "Run `ollama pull <model>` / `ollama signin` (cloud) / edit the TOML / "
                "`--ollama-init`.",
                config.api_key,
            )
        )
