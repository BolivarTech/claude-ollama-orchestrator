# tests/test_ollama_vision.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Vision streaming wrapper (text path in MS4; image_url transport lands in MS7)."""

import dataclasses
import io

import pytest

from backend import DelegationResult
from ollama_config import resolve_config
from ollama_vision import stream_vision


def _cfg(**overrides):
    """Canonical MS4 test config factory (same pattern as `test_ollama_stream.py` /
    `test_run_ollama.py`): resolve the REAL config, then apply overrides via
    `dataclasses.replace` — never ad-hoc `object.__setattr__` mutation."""
    base = resolve_config(global_path=None, repo_path=None, env={})
    return dataclasses.replace(base, **overrides) if overrides else base


def _sse(*lines):
    body = b"".join(ln.encode("utf-8") if isinstance(ln, str) else ln for ln in lines)

    def _open(req, timeout=None):
        return io.BytesIO(body)  # supports .read(n) like a real urllib response

    return _open


def test_stream_vision_streams_text_deltas():
    got: list[str] = []
    urlopen = _sse('data: {"choices":[{"delta":{"content":"a UI"}}]}\n', "data: [DONE]\n")
    res = stream_vision(
        _cfg(), "sys", "describe", "minimax-m3:cloud", 60, sink=got.append, urlopen=urlopen
    )
    assert res.content == "a UI" and got == ["a UI"]


def test_stream_vision_image_arg_is_reserved_for_ms7():
    with pytest.raises(NotImplementedError):
        stream_vision(
            _cfg(),
            "s",
            "p",
            "minimax-m3:cloud",
            60,
            sink=lambda _s: None,
            image=b"\x89PNG",
            urlopen=_sse(),
        )


def test_stream_vision_forwards_all_stream_run_injection_params(monkeypatch):
    """`stream_vision` must forward EVERY one of `stream_run`'s injection/robustness
    params unchanged (idle_timeout, max_output_bytes, urlopen, sleep, rng,
    max_backoffs) — not a stripped-down subset — so vision streaming gets identical
    robustness to text streaming."""
    import ollama_vision

    captured: dict = {}

    def _fake_stream_run(
        config,
        system_prompt,
        prompt,
        model,
        timeout,
        *,
        sink,
        response_format=None,
        idle_timeout=None,
        max_output_bytes=None,
        urlopen=None,
        sleep=None,
        rng=None,
        max_backoffs=None,
    ):
        captured.update(
            idle_timeout=idle_timeout,
            max_output_bytes=max_output_bytes,
            urlopen=urlopen,
            sleep=sleep,
            rng=rng,
            max_backoffs=max_backoffs,
        )
        return DelegationResult("ok", 1, 1, True, 0.1)

    monkeypatch.setattr(ollama_vision, "stream_run", _fake_stream_run)
    sentinel_urlopen, sentinel_sleep, sentinel_rng = object(), object(), object()
    res = stream_vision(
        _cfg(),
        "sys",
        "describe",
        "minimax-m3:cloud",
        60,
        sink=lambda _s: None,
        idle_timeout=5,
        max_output_bytes=42,
        urlopen=sentinel_urlopen,
        sleep=sentinel_sleep,
        rng=sentinel_rng,
        max_backoffs=7,
    )
    assert res.content == "ok"
    assert captured == {
        "idle_timeout": 5,
        "max_output_bytes": 42,
        "urlopen": sentinel_urlopen,
        "sleep": sentinel_sleep,
        "rng": sentinel_rng,
        "max_backoffs": 7,
    }
