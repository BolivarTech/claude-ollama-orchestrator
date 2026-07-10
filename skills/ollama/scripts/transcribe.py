# skills/ollama/scripts/transcribe.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Experimental, gated audio transcription (R2 transcribe).

Two transports, resolved by capability probe: a dedicated OpenAI-style
``POST /audio/transcriptions`` (multipart upload), or — for an audio-multimodal chat
model — the audio as a chat content-part. If the endpoint supports NEITHER, transcribe
is deferred with an actionable error (never a crash), and the other six capabilities are
unaffected. Stdlib-only (multipart body built by hand — no `requests`).
"""

from __future__ import annotations

import base64
import functools
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from typing import Any

from backend import DelegationResult, estimate_tokens, make_redactor
from binary_input import audio_mime_from_bytes, load_binary
from errors import OllamaBackendError
from ollama_config import OllamaAgentsConfig
from ollama_stream import stream_run

_MULTIMODAL_AUDIO_HINTS = ("gemma", "qwen2-audio", "whisper", "audio")

# Upper bound (seconds) on the capability-probe's OWN GET (INFO fix): the probe must never
# be allowed to block for the full delegation `timeout` — see `transcribe`'s `min(timeout,
# PROBE_TIMEOUT_SECONDS)` wiring below.
PROBE_TIMEOUT_SECONDS = 5


def _default_probe(
    url: str,
    timeout: int = PROBE_TIMEOUT_SECONDS,
    *,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> bool:
    """Best-effort: True if *url* (the ``/audio/transcriptions`` endpoint) looks available.

    A ``GET`` that returns anything OTHER than 404/501 (typically 405 Method Not Allowed
    or 400 for a GET on a POST-only endpoint) means the route EXISTS. 404/501 → absent.
    Any transport failure → treated as absent (fall through to the chat/gated path).

    Args:
        url: The ``/audio/transcriptions`` URL to probe.
        timeout: Upper bound (seconds) for this GET. INFO fix: previously hardcoded to
            ``5`` with no link at all to the delegation's own timeout. ``transcribe`` now
            derives this as ``min(timeout, PROBE_TIMEOUT_SECONDS)`` so a long delegation
            timeout can never make the probe itself hang longer than intended.
        urlopen: Injectable urlopen.
    """
    try:
        urlopen(urllib.request.Request(url, method="GET"), timeout=timeout)
        return True
    except urllib.error.HTTPError as exc:
        return exc.code not in (404, 501)
    except OSError:
        return False


def _default_multimodal_audio(model: str) -> bool:
    """Best-effort heuristic: True if *model*'s tag looks audio-capable (multimodal-audio).

    A plain substring match against :data:`_MULTIMODAL_AUDIO_HINTS` — it can reject a
    non-standard multimodal model name that doesn't happen to contain one of these
    substrings (accepted, INFO: false negatives fall through to the gated error, never a
    silent misroute). Two ways to override for a non-standard deployment, from cheapest to
    most involved: (1) extend `_MULTIMODAL_AUDIO_HINTS` with the deployment's model-name
    substring; (2) pass a custom `multimodal_audio` predicate to :func:`transcribe`
    (already the injection point every test above uses) — e.g. an explicit allow-set
    (`{"my-audio-model:latest"}.__contains__`) instead of a substring heuristic.
    """
    m = model.lower()
    return any(hint in m for hint in _MULTIMODAL_AUDIO_HINTS)


def transcribe(
    config: OllamaAgentsConfig,
    audio_path: str,
    model: str,
    timeout: int,
    *,
    transport: str | None = None,
    probe: Callable[[str], bool] | None = None,
    multimodal_audio: Callable[[str], bool] = _default_multimodal_audio,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
    stream_fn: Callable[..., DelegationResult] = stream_run,
    sink: Callable[[str], None] | None = None,
    system_prompt: str = "Transcribe the provided audio verbatim.",
) -> DelegationResult:
    """Transcribe *audio_path*, gated on endpoint capability (R2 transcribe).

    Args:
        config: Resolved config (``config.transcribe_transport`` is the fallback for
            *transport* when it is left ``None``).
        audio_path: Path to the audio input (validated by ``load_binary``).
        model: The transcribe model tag.
        timeout: Per-delegation timeout (forwarded to BOTH transports — the
            ``/audio/transcriptions`` POST no longer hardcodes its own timeout).
        transport: ``None`` (default) resolves to ``config.transcribe_transport``
            (``"auto"|"endpoint"|"chat"``). ``"auto"`` runs *probe* then falls back to
            the audio-chat transport, then the gated error (the original R2 behavior).
            ``"endpoint"``/``"chat"`` force that transport and SKIP *probe* entirely —
            for a caller that already knows the endpoint's capability.
        probe: Returns True if ``{base_url}/audio/transcriptions`` is available. Not
            called when *transport* resolves to ``"endpoint"`` or ``"chat"``. ``None``
            (default) resolves to :func:`_default_probe` bound via
            ``functools.partial(timeout=min(timeout, PROBE_TIMEOUT_SECONDS), urlopen=urlopen)``
            (INFO fix) — the probe's own GET is capped at ``PROBE_TIMEOUT_SECONDS`` and
            never independent of the delegation's *timeout*/*urlopen*, so it can't hang
            longer than intended nor bypass the injected transport in tests.
        multimodal_audio: Returns True if *model* accepts audio in chat. Only consulted
            under ``"auto"``, after a failed probe.
        urlopen: Injectable urlopen (dedicated-endpoint transport / probe).
        stream_fn: Injectable streamer (audio-chat transport).
        sink: Delta receiver for the audio-chat transport (defaults to a no-op).
        system_prompt: System prompt for the audio-chat transport.

    Returns:
        The transcript in a :class:`DelegationResult`.

    Raises:
        OllamaBackendError: (actionable, gated) under ``"auto"``, if neither transport is
            available — ``transcribe`` is deferred for this build, never crashing. Under a
            FORCED transport that turns out to be unavailable, the underlying transport's
            own error propagates instead (the caller asked to skip the probe/heuristic, so
            a hard failure is the honest outcome, not a silent fallback).
        ValidationError: the audio file is unreadable / oversize / wrong magic bytes.
    """
    data = load_binary(audio_path, kind="audio")  # R24b guard on ALL transports
    resolved_transport = transport if transport is not None else config.transcribe_transport
    endpoint = f"{config.base_url}/audio/transcriptions"
    if probe is None:
        # INFO fix: bound the probe's own GET independently of `timeout` (previously
        # `_default_probe` hardcoded `timeout=5` with no link to the delegation timeout at
        # all); also forward the injected `urlopen` so the default probe stays testable.
        probe = functools.partial(
            _default_probe, timeout=min(timeout, PROBE_TIMEOUT_SECONDS), urlopen=urlopen
        )
    if resolved_transport == "endpoint":
        return _via_audio_endpoint(config, endpoint, data, audio_path, model, urlopen, timeout)
    if resolved_transport == "chat":
        return _via_audio_chat(
            config, system_prompt, data, audio_path, model, timeout, stream_fn, sink
        )
    # "auto" (default): probe first, then the audio-multimodal chat fallback, then the gate.
    if probe(endpoint):
        return _via_audio_endpoint(config, endpoint, data, audio_path, model, urlopen, timeout)
    if multimodal_audio(model):
        return _via_audio_chat(
            config, system_prompt, data, audio_path, model, timeout, stream_fn, sink
        )
    raise OllamaBackendError(
        "transcribe experimental: the endpoint does not support audio "
        "(/audio/transcriptions absent and the model is not audio-multimodal); "
        "this capability is deferred for this build"
    )


def _strip_header_controls(value: str) -> str:
    """Remove every C0 control char, DEL, and the Unicode line/paragraph separators from
    *value* so no control character can reach an interpolated MIME header (CR/LF split
    headers; the rest is stripped as cheap defense-in-depth). Printable content -- including
    non-ASCII -- is preserved verbatim. Unlike :func:`_escape_multipart_filename` this does
    NOT escape the quote/backslash: use it for header TOKENS (a ``Content-Type`` mime) and
    form-field part bodies, where a backslash-escape would corrupt the value, not protect it.
    """
    return "".join(
        c for c in value if ord(c) >= 0x20 and c != "\x7f" and c not in "\u0085\u2028\u2029"
    )


def _escape_multipart_filename(filename: str) -> str:
    """Escape *filename* for safe interpolation into a ``Content-Disposition`` header.

    **[CRITICAL/SECURITY] header-injection fix.** ``_multipart_body`` interpolates
    *filename* directly into ``filename="<name>"`` inside a
    ``Content-Disposition: form-data; ...`` header line. Without escaping, a filename
    containing a double quote or backslash can break out of the quoted value, and a
    filename containing CR/LF can smuggle an entirely new header/part into the
    multipart body (classic MIME header injection). This function makes that
    unrepresentable:

    1. Backslash-escape backslashes and double quotes (RFC 2183/6266-style quoted-string
       escaping: ``\\`` → ``\\\\``, ``"`` → ``\\"``) so a quote/backslash in the name can
       never terminate the quoted value early.
    2. Strip ``\r`` and ``\n`` entirely — a raw newline is what would let an attacker
       inject a new header line or multipart boundary; there is no legitimate reason a
       filename needs one, so it is removed rather than escaped.

    The caller (`_via_audio_endpoint`) additionally passes ``os.path.basename(audio_path)``
    rather than the raw path, so path separators never reach this function in the first
    place (defense-in-depth: this function does not depend on that alone, since a
    filesystem can still permit ``"``/``\\``/newlines in a basename on some platforms).

    Args:
        filename: The raw (untrusted) filename to interpolate.

    Returns:
        A filename string safe to interpolate inside a double-quoted header value.
    """
    escaped = filename.replace("\\", "\\\\").replace('"', '\\"')
    # Reuse the shared control-strip; the `\`/`"` escape above yields printable `\`/`\"`,
    # so it survives this pass. CR/LF are the only exploitable chars, the rest defense-in-depth.
    return _strip_header_controls(escaped)


def _multipart_body(
    fields: dict[str, str], *, file_field: str, filename: str, file_bytes: bytes, mime: str
) -> tuple[bytes, str]:
    """Build a ``multipart/form-data`` body by hand (stdlib-only, no `requests`).

    Returns ``(body, content_type)`` where *content_type* carries the boundary.

    Security: EVERY string interpolated into a header is neutralized, not only *filename*.
    A ``"``/``\\``/CR/LF in any of the field ``name``, the ``file_field``, the ``filename``,
    or the ``mime`` could otherwise break out of its ``Content-Disposition``/``Content-Type``
    header and inject an arbitrary header or extra multipart part (header injection). The
    quoted-string values (field name, file_field, filename) go through
    :func:`_escape_multipart_filename` (escape ``"``/``\\`` + strip controls); the
    ``Content-Type`` token (mime) and the form-field part bodies go through
    :func:`_strip_header_controls` (strip controls only -- a backslash-escape would corrupt a
    token or a body value). The current caller passes only trusted constants, so this is
    defense-in-depth against any future untrusted field data. The multipart boundary is an
    unguessable ``uuid4``, so a part body can never accidentally contain it.
    """
    boundary = f"----ollama{uuid.uuid4().hex}"
    crlf = b"\r\n"
    safe_filename = _escape_multipart_filename(filename)
    parts: list[bytes] = []
    for name, value in fields.items():
        safe_name = _escape_multipart_filename(name)  # quoted `name="..."` header value
        safe_value = _strip_header_controls(value)  # part body: strip CR/LF, keep quotes
        parts += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="{safe_name}"'.encode(),
            b"",
            safe_value.encode("utf-8"),
        ]
    safe_file_field = _escape_multipart_filename(file_field)  # quoted header value
    safe_mime = _strip_header_controls(mime)  # Content-Type token
    parts += [
        f"--{boundary}".encode(),
        (
            f'Content-Disposition: form-data; name="{safe_file_field}"; filename="{safe_filename}"'
        ).encode(),
        f"Content-Type: {safe_mime}".encode(),
        b"",
        file_bytes,
        f"--{boundary}--".encode(),
        b"",
    ]
    return crlf.join(parts), f"multipart/form-data; boundary={boundary}"


def _via_audio_endpoint(
    config: OllamaAgentsConfig,
    endpoint: str,
    data: bytes,
    audio_path: str,
    model: str,
    urlopen: Callable[..., Any],
    timeout: int,
) -> DelegationResult:
    """POST the audio to ``/audio/transcriptions`` (OpenAI-style) and return the transcript.

    Args:
        config: Resolved config (for the optional ``Authorization`` header).
        endpoint: The full ``{base_url}/audio/transcriptions`` URL.
        data: The validated raw audio bytes (from ``load_binary``).
        audio_path: Original path, used only for its basename/extension (filename + MIME).
        model: The transcribe model tag, sent as a form field.
        urlopen: Injectable urlopen.
        timeout: The DELEGATION's timeout, forwarded to the socket — previously this was
            a hardcoded ``60`` regardless of the caller's timeout (bug, fixed here).

    Returns:
        The transcript in a :class:`DelegationResult`.

    Raises:
        OllamaBackendError: (NR4 fix) any transport failure — an HTTP error status, an
            unreachable/reset connection, a socket timeout, or a non-JSON response body —
            is caught HERE and re-raised as this domain exception, redacted via the SAME
            :func:`backend.make_redactor` every other HTTP transport in this plugin uses.
            A raw `urllib.error.HTTPError`/`URLError`/`json.JSONDecodeError` never escapes
            to the orchestrator (violating NR4's "domain exceptions only").
    """
    start = time.monotonic()
    redact = make_redactor(config.api_key)
    body, content_type = _multipart_body(
        {"model": model},
        file_field="file",
        filename=os.path.basename(audio_path),
        file_bytes=data,
        mime=audio_mime_from_bytes(data),
    )
    req = urllib.request.Request(
        endpoint, data=body, method="POST", headers={"Content-Type": content_type}
    )
    if config.api_key:
        req.add_header("Authorization", f"Bearer {config.api_key}")
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — best-effort detail extraction only.
            detail = ""
        raise OllamaBackendError(
            redact(f"transcribe endpoint HTTP {exc.code}: {exc.reason} {detail}".strip())
        ) from None
    except urllib.error.URLError as exc:
        raise OllamaBackendError(redact(f"transcribe endpoint unreachable: {exc.reason}")) from None
    except OSError as exc:  # socket timeout and other transport-level failures
        raise OllamaBackendError(redact(f"transcribe endpoint transport error: {exc}")) from None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OllamaBackendError(
            redact(f"transcribe endpoint returned a non-JSON response: {exc}")
        ) from None
    text = payload.get("text", "") if isinstance(payload, dict) else ""
    elapsed = time.monotonic() - start
    return DelegationResult(text, 0, estimate_tokens(text), True, elapsed)


def _via_audio_chat(
    config: OllamaAgentsConfig,
    system_prompt: str,
    data: bytes,
    audio_path: str,
    model: str,
    timeout: int,
    stream_fn: Callable[..., DelegationResult],
    sink: Callable[[str], None] | None,
) -> DelegationResult:
    """Send audio as an ``input_audio`` chat content-part (audio-multimodal models)."""
    b64 = base64.b64encode(data).decode("ascii")
    fmt = audio_mime_from_bytes(data).split("/", 1)[1]
    content_parts = [
        {"type": "text", "text": "Transcribe this audio verbatim."},
        {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
    ]
    return stream_fn(
        config,
        system_prompt,
        "Transcribe this audio verbatim.",
        model,
        timeout,
        sink=(sink or (lambda _s: None)),
        content_parts=content_parts,
    )
