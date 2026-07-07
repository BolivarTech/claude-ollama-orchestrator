# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Fail-fast preflight for the Ollama endpoint."""

from __future__ import annotations

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
            response body (``PREFLIGHT_MAX_RESPONSE_BYTES``), or a non-JSON
            response body (e.g. an HTML error page from a misconfigured proxy,
            or malformed UTF-8 bytes decoded via ``errors="replace"``) on an
            otherwise-200 response. 404/501 warn-and-proceed instead of
            raising.
    """
    url = f"{config.base_url}/models"
    req = urllib.request.Request(url, method="GET")
    if config.api_key:
        req.add_header("Authorization", f"Bearer {config.api_key}")
    try:
        resp = urlopen(req, timeout=PREFLIGHT_TIMEOUT)
        # Bounded read (mirrors backend.MAX_RESPONSE_BYTES): read at most ONE byte past
        # the bound so a runaway/hostile endpoint can never force an unbounded in-memory
        # load — never a bare resp.read().
        raw = resp.read(PREFLIGHT_MAX_RESPONSE_BYTES + 1)
        if len(raw) > PREFLIGHT_MAX_RESPONSE_BYTES:
            raise OllamaPreflightError(
                f"/models response exceeded PREFLIGHT_MAX_RESPONSE_BYTES "
                f"({PREFLIGHT_MAX_RESPONSE_BYTES} bytes) — endpoint sent an oversized "
                "response"
            )
        # The /models response body is untrusted server output: decode with
        # errors="replace" (U+FFFD substitution) so malformed UTF-8 bytes never raise a
        # raw UnicodeDecodeError (a non-domain exception) — a still-invalid-JSON result
        # after substitution surfaces as the existing OllamaPreflightError below.
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
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
    except (socket.timeout, TimeoutError, urllib.error.URLError) as exc:
        raise OllamaPreflightError(
            f"Cannot reach Ollama at {config.base_url}: {exc}. "
            "Is it running? Try `ollama signin` for cloud."
        ) from None
    except json.JSONDecodeError as exc:
        # A reverse proxy / misconfigured endpoint can return HTTP 200 with a non-JSON
        # body (e.g. an HTML error page). Map it to a domain error instead of letting an
        # uncaught json.JSONDecodeError crash the process with a raw traceback.
        raise OllamaPreflightError(
            _redact(
                f"Preflight failed: {url} returned a non-JSON response ({exc}).", config.api_key
            )
        ) from None

    available = {m["id"] for m in payload.get("data", []) if isinstance(m, dict) and "id" in m}
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
            f"Missing models: {', '.join(missing)}. "
            "Run `ollama pull <model>` / `ollama signin` (cloud) / edit the TOML / "
            "`--ollama-init`."
        )
