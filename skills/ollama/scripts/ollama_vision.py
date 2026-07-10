# skills/ollama/scripts/ollama_vision.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Vision streaming wrapper. MS4: text pass-through + reserved image interface; MS7 fills it."""

from __future__ import annotations

import random
import time
import urllib.request
from collections.abc import Callable
from typing import Any

from backend import DelegationResult
from ollama_config import OllamaAgentsConfig
from ollama_stream import DEFAULT_IDLE_TIMEOUT, DEFAULT_MAX_OUTPUT_BYTES, stream_run


def stream_vision(
    config: OllamaAgentsConfig,
    system_prompt: str,
    prompt: str,
    model: str,
    timeout: int,
    *,
    sink: Callable[[str], None],
    image: bytes | None = None,
    response_format: dict[str, Any] | None = None,
    idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
    max_backoffs: int = 3,
) -> DelegationResult:
    """Stream a vision delegation. MS4 streams the text response.

    A thin wrapper over `stream_run` that forwards EVERY one of its injection/
    robustness parameters unchanged — idle timeout, output cap, urlopen, sleep, rng,
    max_backoffs — so vision streaming has IDENTICAL robustness to text streaming
    (429 backoff, idle timeout, bounded output, deterministic test injection), never
    a silently-stripped-down subset of it.

    Args:
        config: Resolved config.
        system_prompt: Vision system prompt.
        prompt: The user prompt.
        model: Resolved vision model tag.
        timeout: Per-delegation timeout.
        sink: Delta receiver.
        image: Reserved for the MS7 ``image_url`` base64 content-part; must be None in MS4.
        response_format: Optional structured shape.
        idle_timeout: Forwarded to `stream_run` unchanged.
        max_output_bytes: Forwarded to `stream_run` unchanged.
        urlopen: Injectable urlopen, forwarded to `stream_run` unchanged.
        sleep: Injectable sleep (429 backoff), forwarded to `stream_run` unchanged.
        rng: Injectable RNG (jitter), forwarded to `stream_run` unchanged.
        max_backoffs: Forwarded to `stream_run` unchanged.

    Returns:
        The accumulated :class:`DelegationResult`.

    Raises:
        NotImplementedError: if *image* is provided (MS7 transport not yet wired).
    """
    if image is not None:
        raise NotImplementedError("vision image transport lands in MS7")
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
    )
