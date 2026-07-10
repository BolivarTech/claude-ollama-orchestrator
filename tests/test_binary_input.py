# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Binary input guard: bounded-read size cap (never reads past the cap), magic-byte
allow-list (incl. RIFF form-type disambiguation and the MP3 ID3/frame-sync-mask
variants), base64 data-uri, domain errors."""

from typing import Literal

import pytest

from binary_input import MAX_INPUT_BYTES, load_binary, to_data_uri
from errors import ValidationError

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
_JPEG = b"\xff\xd8\xff" + b"\x00" * 16


def _riff(form_type: bytes) -> bytes:
    return b"RIFF" + b"\x00\x00\x00\x00" + form_type + b"\x00" * 8


def test_valid_png_loads(tmp_path):
    p = tmp_path / "img.png"
    p.write_bytes(_PNG)
    assert load_binary(str(p), kind="image") == _PNG


def test_valid_jpeg_loads(tmp_path):
    p = tmp_path / "img.jpg"
    p.write_bytes(_JPEG)
    assert load_binary(str(p), kind="image") == _JPEG


def test_unrecognized_magic_raises_validation_error(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"NOTPNG" + b"\x00" * 16)
    with pytest.raises(ValidationError):
        load_binary(str(p), kind="image")


def test_oversize_raises_validation_error_checked_on_bytes_actually_read(tmp_path, monkeypatch):
    # The size guard is authoritative on len(data) AFTER the read, never a pre-read
    # os.path.getsize — this test caps the budget below the real file size and
    # confirms rejection, with no pre-read stat call in the path to race against.
    import binary_input

    monkeypatch.setattr(binary_input, "MAX_INPUT_BYTES", 4)
    p = tmp_path / "img.png"
    p.write_bytes(_PNG)
    with pytest.raises(ValidationError):
        load_binary(str(p), kind="image")


def test_oversize_read_is_bounded_and_never_reads_the_whole_file(tmp_path, monkeypatch):
    # INFO fix (Caspar residual): reading the ENTIRE file before checking len(data)
    # would fully read a multi-GB file into memory before rejecting it. `load_binary`
    # must bound the read itself to `MAX_INPUT_BYTES + 1` — enough to detect "over
    # the cap" without ever reading further. This test proves the READ CALL is
    # bounded (never an unbounded `.read()`/`.read(-1)`), not just that the result is
    # eventually rejected.
    import binary_input

    monkeypatch.setattr(binary_input, "MAX_INPUT_BYTES", 4)
    read_sizes: list[int] = []

    class _FakeFile:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            assert size != -1, "load_binary must never perform an unbounded read()"
            return self._data[:size]

        def __enter__(self) -> "_FakeFile":
            return self

        def __exit__(self, *exc: object) -> Literal[False]:
            return False

    oversized = _PNG + b"\x00" * 1000  # far larger than the 4-byte cap in effect
    monkeypatch.setattr(
        binary_input, "open", lambda path, mode: _FakeFile(oversized), raising=False
    )
    with pytest.raises(ValidationError):
        load_binary("fake-path.png", kind="image")
    assert read_sizes == [5]  # MAX_INPUT_BYTES + 1 == 5, and read() called exactly once


def test_missing_file_raises_validation_error_not_oserror():
    with pytest.raises(ValidationError):
        load_binary("definitely-not-a-real-file-xyz.bin", kind="image")


def test_id3_tagged_mp3_is_recognized_as_audio(tmp_path):
    # INFO fix (Caspar residual): a common real-world MP3 layout — an ID3v2 tag
    # header, distinct from the raw MPEG frame-sync case below.
    id3_mp3 = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 16
    p = tmp_path / "song.mp3"
    p.write_bytes(id3_mp3)
    assert load_binary(str(p), kind="audio") == id3_mp3


def test_mpeg2_frame_sync_mp3_is_recognized_as_audio(tmp_path):
    # INFO fix (Caspar residual): broadened MP3 detection. The previous allow-list
    # matched ONLY the exact MPEG-1 Layer III frame-sync byte pair (b"\xff\xfb"); an
    # MP3 with no ID3 tag encoded as MPEG-2 (a different, equally valid frame-sync
    # second byte, still matching the 0xE0 mask) was incorrectly rejected before this
    # fix. b"\xff\xf3" below is such a variant (not the previously-hardcoded 0xfb).
    mpeg2_mp3 = b"\xff\xf3" + b"\x00" * 16
    p = tmp_path / "song.mp3"
    p.write_bytes(mpeg2_mp3)
    assert load_binary(str(p), kind="audio") == mpeg2_mp3


def test_wav_riff_is_recognized_as_audio(tmp_path):
    wav = _riff(b"WAVE")
    p = tmp_path / "x.wav"
    p.write_bytes(wav)
    assert load_binary(str(p), kind="audio") == wav


def test_webp_riff_is_recognized_as_image(tmp_path):
    webp = _riff(b"WEBP")
    p = tmp_path / "x.webp"
    p.write_bytes(webp)
    assert load_binary(str(p), kind="image") == webp


def test_wav_riff_is_rejected_as_image_form_type_mismatch(tmp_path):
    # RIFF alone is ambiguous (WAV and WebP share the same 4-byte prefix) — the
    # form-type field at offset 8 disambiguates; a WAVE form-type must NOT pass as
    # an "image".
    wav = _riff(b"WAVE")
    p = tmp_path / "x.wav"
    p.write_bytes(wav)
    with pytest.raises(ValidationError):
        load_binary(str(p), kind="image")


def test_webp_riff_is_rejected_as_audio_form_type_mismatch(tmp_path):
    webp = _riff(b"WEBP")
    p = tmp_path / "x.webp"
    p.write_bytes(webp)
    with pytest.raises(ValidationError):
        load_binary(str(p), kind="audio")


def test_to_data_uri_is_base64():
    uri = to_data_uri(b"hi", "image/png")
    assert uri.startswith("data:image/png;base64,")


def test_max_input_bytes_is_20mb():
    assert MAX_INPUT_BYTES == 20 * 1024 * 1024
