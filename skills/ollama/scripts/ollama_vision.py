# skills/ollama/scripts/ollama_vision.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Vision delegation transport: multimodal image_url content-part (R2 vision, MS7)."""

from __future__ import annotations

import random
import time
import urllib.request
from collections.abc import Callable
from typing import Any

from backend import DelegationResult
from binary_input import load_binary, to_data_uri
from errors import DelegationError
from ollama_config import OllamaAgentsConfig
from ollama_stream import DEFAULT_IDLE_TIMEOUT, DEFAULT_MAX_OUTPUT_BYTES, stream_run

# MIME derived from the image's VALIDATED magic bytes (INFO fix — NOT the file
# extension, which is caller-controlled and can lie or be absent/misleading). Mirrors
# `binary_input`'s own "image" magic-byte allow-list (PNG / JPEG / RIFF-WEBP) — kept in
# sync with it. `_DEFAULT_IMAGE_MIME` is a defensive backstop only; in practice one of
# the three branches always matches, since this is only ever called on bytes that
# already passed `load_binary(..., kind="image")`'s own magic-byte check.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_RIFF_WEBP_FORM_TYPE = b"WEBP"
_DEFAULT_IMAGE_MIME = "image/png"
# Substrings of model tags known to accept image content-parts (multimodal). Overridable
# via the injectable `is_multimodal` predicate for configs/tests.
_MULTIMODAL_VISION_HINTS = ("minimax", "qwen", "gemma", "llava", "vision", "gpt-4o", "pixtral")


def _default_is_multimodal(model: str) -> bool:
    """Best-effort heuristic: True if *model*'s tag looks image-capable (multimodal).

    A plain substring match against :data:`_MULTIMODAL_VISION_HINTS`. This is a DELIBERATE
    v0.1 default whose BOTH failure modes degrade gracefully (never a crash, never a silent
    misroute): a false NEGATIVE (a non-standard multimodal name missed) surfaces as the
    actionable `DelegationError` below; a false POSITIVE (a text-only model matched) sends the
    image content-part, and the server rejects it as a normal backend error routed through the
    circuit breaker. An exact allow-list (Caspar residual) is a v0.2 refinement, already
    reachable WITHOUT a code change via the injectable predicate. Two ways to override,
    cheapest first: (1) extend `_MULTIMODAL_VISION_HINTS` with the deployment's model-name
    substring; (2) pass a custom `is_multimodal` predicate to :func:`stream_vision` (already
    the injection point the tests below use) — e.g. an explicit allow-set
    (`{"my-vision-model:latest"}.__contains__`) instead of the substring heuristic.
    """
    m = model.lower()
    return any(hint in m for hint in _MULTIMODAL_VISION_HINTS)


def _image_mime_from_bytes(data: bytes) -> str:
    """Return the image MIME type detected from *data*'s VALIDATED magic bytes.

    INFO fix: this replaces a prior file-EXTENSION-based lookup, which mislabeled the
    data-URI whenever the extension didn't match the real content (e.g. a PNG saved as
    ``photo.txt``, or a path with no extension at all) — the extension is caller-supplied
    and untrusted, while the magic bytes are exactly what `load_binary` already validated.
    Called only on bytes that already passed `load_binary(..., kind="image")`'s magic-byte
    allow-list, so one of the first three branches is guaranteed to match in practice;
    `_DEFAULT_IMAGE_MIME` is a defensive backstop only, never actually reached given that
    precondition.

    Args:
        data: The validated image bytes (post ``load_binary``).

    Returns:
        ``"image/png"``, ``"image/jpeg"``, ``"image/webp"``, or the PNG default.
    """
    if data.startswith(_PNG_MAGIC):
        return "image/png"
    if data.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == _RIFF_WEBP_FORM_TYPE:
        return "image/webp"
    return _DEFAULT_IMAGE_MIME


def build_vision_content_parts(image_path: str, prompt: str) -> list[dict[str, Any]]:
    """Build the OpenAI-compatible multimodal content-parts for a vision request.

    Loads + validates the image (magic-bytes + size cap; load-on-dispatch per R24b) and
    encodes it as a base64 ``image_url`` data-URI part alongside the text prompt. The
    data-URI's MIME is derived from the same VALIDATED magic bytes
    (:func:`_image_mime_from_bytes`), never from *image_path*'s file extension — a
    mismatched or extension-less path still gets the correct MIME.

    Memory note (CORRECTED, INFO — document near `to_data_uri`/`load_binary`, MS6): a base64
    data-URI is ~4/3 the size of the raw bytes it encodes (3 raw bytes → 4 base64 chars), but
    the PEAK is not that 4/3 buffer alone — during encoding, `load_binary`'s raw ``bytes``
    object AND `to_data_uri`'s base64-encoded ``str`` are BOTH resident in memory at the same
    time (the raw buffer is not freed before the encoded one exists), so the peak per
    delegation is ``raw_bytes × (1 + 4/3)`` = ``raw_bytes × 7/3``. The PEAK memory bound for
    concurrent vision delegations is therefore ``max_parallel_agents × max_input_bytes × 7/3``
    — not ``× max_input_bytes`` as R24b's memory-under-concurrency note states for the
    raw-bytes-only case, and not the encoded-buffer-only ``× 4/3`` an earlier draft of this
    note used. `load_binary`'s load-on-dispatch (R24b) already keeps this bounded to the
    RUNNING set (the queue holds the path, not the buffer), so the 7/3 factor only applies to
    that running-set bound.

    Args:
        image_path: Path to the image input.
        prompt: The text instruction accompanying the image.

    Returns:
        ``[{"type":"text",...}, {"type":"image_url","image_url":{"url":<data-uri>}}]``.

    Raises:
        ValidationError: oversize image or magic-byte mismatch (from ``load_binary``).
    """
    data = load_binary(image_path, kind="image")  # R24b guard, load-on-dispatch
    data_uri = to_data_uri(data, _image_mime_from_bytes(data))  # MIME from magic bytes
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]


def stream_vision(
    config: OllamaAgentsConfig,
    system_prompt: str,
    prompt: str,
    model: str,
    timeout: int,
    *,
    sink: Callable[[str], None],
    image_path: str | None = None,
    is_multimodal: Callable[[str], bool] = _default_is_multimodal,
    response_format: dict[str, Any] | None = None,
    idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
    max_backoffs: int = 3,
) -> DelegationResult:
    """Stream a vision delegation (R2 vision). MS7 fills the MS4 image seam.

    With *image_path*: guard that *model* is multimodal (else actionable
    ``DelegationError`` — a text-only model cannot accept an image), then stream a
    multimodal ``image_url`` request. Without it: a plain text stream (same as any other
    capability). EVERY one of ``stream_run``'s injection/robustness params forwards
    unchanged, so vision streaming keeps identical robustness to text streaming.

    Args:
        config: Resolved config.
        system_prompt: The ``ollama-vision`` system prompt.
        prompt: The text instruction.
        model: Resolved vision model tag.
        timeout: Per-delegation timeout.
        sink: Delta receiver (stdout or file writer).
        image_path: Path to the image (validated by ``load_binary``); None → text-only.
        is_multimodal: Predicate deciding if *model* accepts images (injectable).
        response_format: Optional structured shape.
        idle_timeout: Forwarded to ``stream_run`` unchanged.
        max_output_bytes: Forwarded to ``stream_run`` unchanged.
        urlopen: Injectable urlopen, forwarded unchanged.
        sleep: Injectable sleep (429 backoff), forwarded unchanged.
        rng: Injectable RNG (jitter), forwarded unchanged.
        max_backoffs: Forwarded to ``stream_run`` unchanged.

    Returns:
        The accumulated :class:`DelegationResult`.

    Raises:
        DelegationError: *image_path* given but *model* is not multimodal.
        ValidationError: oversize/bad-magic image (from ``load_binary``).
    """
    content_parts = None
    if image_path is not None:
        if not is_multimodal(model):
            raise DelegationError(
                f"vision model {model!r} is not multimodal (cannot accept an image); "
                "set a multimodal model (e.g. minimax-m3:cloud) in [models].vision"
            )
        content_parts = build_vision_content_parts(image_path, prompt)
    return stream_run(
        config,
        system_prompt,
        prompt,
        model,
        timeout,
        sink=sink,
        response_format=response_format,
        idle_timeout=idle_timeout,
        max_output_bytes=max_output_bytes,
        urlopen=urlopen,
        sleep=sleep,
        rng=rng,
        max_backoffs=max_backoffs,
        content_parts=content_parts,
    )
