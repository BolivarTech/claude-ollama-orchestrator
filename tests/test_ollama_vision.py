# tests/test_ollama_vision.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Vision streaming wrapper (text path in MS4; image_url transport lands in MS7)."""

import base64
import dataclasses
import io
import json

import pytest

from backend import DelegationResult
from errors import DelegationError, ValidationError
from ollama_config import resolve_config
from ollama_vision import stream_vision

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32  # valid PNG magic + padding


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


def test_stream_vision_sends_image_url_data_uri_content_part(tmp_path):
    img = tmp_path / "ui.png"
    img.write_bytes(_PNG)
    seen = {}

    def _open(req, timeout=None):
        seen["content"] = json.loads(req.data)["messages"][1]["content"]
        return io.BytesIO(
            b'data: {"choices":[{"delta":{"content":"a login form"}}]}\ndata: [DONE]\n'
        )

    res = stream_vision(
        _cfg(),
        "sys",
        "describe",
        "minimax-m3:cloud",
        60,
        sink=lambda _s: None,
        image_path=str(img),
        urlopen=_open,
    )
    assert res.content == "a login form"
    text_part, image_part = seen["content"]
    assert text_part == {"type": "text", "text": "describe"}
    assert image_part["type"] == "image_url"
    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == _PNG  # the real image bytes, base64'd


def test_stream_vision_derives_mime_from_magic_bytes_not_the_file_extension(tmp_path):
    # A real PNG saved under a misleading `.txt` name/extension must still produce an
    # `image/png` data-URI — the MIME comes from the VALIDATED magic bytes (`load_binary`
    # already confirmed them), never from the untrustworthy file extension.
    img = tmp_path / "not-really-a-text-file.txt"
    img.write_bytes(_PNG)
    seen = {}

    def _open(req, timeout=None):
        seen["content"] = json.loads(req.data)["messages"][1]["content"]
        return io.BytesIO(b'data: {"choices":[{"delta":{"content":"ok"}}]}\ndata: [DONE]\n')

    stream_vision(
        _cfg(),
        "sys",
        "describe",
        "minimax-m3:cloud",
        60,
        sink=lambda _s: None,
        image_path=str(img),
        urlopen=_open,
    )
    url = seen["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")  # driven by magic bytes, not ".txt"


def test_stream_vision_rejects_a_non_multimodal_model(tmp_path):
    img = tmp_path / "ui.png"
    img.write_bytes(_PNG)
    with pytest.raises(DelegationError):
        stream_vision(
            _cfg(),
            "s",
            "p",
            "kimi-k2.7-code:cloud",  # a text-only coder model
            60,
            sink=lambda _s: None,
            image_path=str(img),
            is_multimodal=lambda _m: False,
            urlopen=_sse(),
        )


def test_stream_vision_oversize_or_bad_magic_image_raises_validation_error(tmp_path):
    bad = tmp_path / "not-an-image.png"
    bad.write_bytes(b"this is plain text, not a PNG")
    with pytest.raises(ValidationError):
        stream_vision(
            _cfg(),
            "s",
            "p",
            "minimax-m3:cloud",
            60,
            sink=lambda _s: None,
            image_path=str(bad),
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
        content_parts=None,
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
