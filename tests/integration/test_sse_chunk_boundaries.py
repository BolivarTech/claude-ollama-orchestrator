# tests/integration/test_sse_chunk_boundaries.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Enforceable integration gate (R7b): drives `stream_run` against a local, in-process
fake SSE response that deliberately splits the byte stream at AWKWARD boundaries a
single-`io.BytesIO(body)` mock never produces — mid `data:` line, and a multi-byte UTF-8
character split across two separate `.read()` calls. Marked `@pytest.mark.integration`
so `make verify` (via `python -m pytest tests/ -v`, which recurses into `tests/integration/`)
enforces it automatically — this is a checked-in gate, not a one-off manual run.
"""

import pytest

from ollama_config import resolve_config
from ollama_stream import stream_run


def _cfg():
    return resolve_config(global_path=None, repo_path=None, env={})


class _ChunkedResponse:
    """A fake `urlopen` response whose `.read(n)` yields PRE-SLICED byte chunks in
    order, ignoring the requested `n` — this is what forces the reader to reassemble
    lines/characters split at the awkward boundaries below, unlike the single-
    `io.BytesIO(body)` double used by `test_ollama_stream.py` (whose `.read(n)`
    returns the whole body in one call and never exercises split reassembly)."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    def read(self, n: int) -> bytes:  # noqa: ARG002 — n intentionally ignored
        return self._chunks.pop(0) if self._chunks else b""

    def close(self) -> None:
        pass


def _split_mid_data_line() -> list[bytes]:
    """Split a single `data: {...}` line's bytes in the MIDDLE — the reader must
    buffer the first half and only decode/parse once the second half (with the
    trailing newline) arrives."""
    line = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n'
    mid = len(line) // 2
    return [line[:mid], line[mid:], b"data: [DONE]\n"]


def _split_multibyte_char_across_reads() -> list[bytes]:
    """Split a multi-byte UTF-8 character (the euro sign, 3 bytes: 0xE2 0x82 0xAC)
    across TWO separate `.read()` returns — the reader must not raise/mangle it; a
    partial UTF-8 sequence must not be decoded until the line (and thus the full
    character) is complete."""
    payload = 'data: {"choices":[{"delta":{"content":"€5"}}]}\n'.encode("utf-8")
    split_point = payload.index("€".encode("utf-8")) + 1  # cut inside the euro's 3 bytes
    return [payload[:split_point], payload[split_point:], b"data: [DONE]\n"]


@pytest.mark.integration
def test_sse_reassembles_a_data_line_split_mid_line():
    got: list[str] = []
    resp = _ChunkedResponse(_split_mid_data_line())
    res = stream_run(
        _cfg(), "sys", "p", "m", 60, sink=got.append, urlopen=lambda req, timeout=None: resp
    )
    assert res.content == "hello"
    assert got == ["hello"]


@pytest.mark.integration
def test_sse_reassembles_a_multibyte_char_split_across_reads():
    got: list[str] = []
    resp = _ChunkedResponse(_split_multibyte_char_across_reads())
    res = stream_run(
        _cfg(), "sys", "p", "m", 60, sink=got.append, urlopen=lambda req, timeout=None: resp
    )
    assert res.content == "€5"
    assert "".join(got) == "€5"
