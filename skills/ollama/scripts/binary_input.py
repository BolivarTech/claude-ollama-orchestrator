# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Bounded, magic-byte-checked binary input loader for vision/transcribe (R24b).

Load-on-call: I/O and validation happen ONLY when `load_binary` is invoked, never
eagerly — see the docstring below for the queue-by-reference memory bound and the MS7
wiring seam.
"""

from __future__ import annotations

import base64

from errors import ValidationError

MAX_INPUT_BYTES = 20 * 1024 * 1024

# Signatures that need no further disambiguation.
_MAGIC: dict[str, tuple[bytes, ...]] = {
    "image": (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff"),  # PNG, JPEG (WebP via RIFF, below)
    "audio": (b"ID3", b"fLaC", b"OggS"),  # MP3 w/ ID3 tag, FLAC, OGG (MP3 frame-sync
    # without an ID3 tag is a MASK check, below —
    # NOT a fixed-prefix member of this tuple)
}
# RIFF-based containers (WAV, WebP) share the same 4-byte "RIFF" prefix, so checking
# ONLY that prefix is ambiguous — a WAV would incorrectly pass as an "image" and vice
# versa. The disambiguating field is the 4-byte form-type at offset 8 (RIFF layout:
# b"RIFF" + 4-byte little-endian size + 4-byte form-type).
#
# Both allow-lists (_MAGIC and _RIFF_FORM_TYPE) are DELIBERATELY EXPLICIT rather than a
# generic sniffing library: accepting a new format (e.g. HEIC, Opus) is a one-line
# addition here, reviewed alongside its own test. This is an ACCEPTED, intentional
# maintenance cost (INFO, not a defect) — an explicit allow-list is exactly what a
# security-sensitive magic-byte check should be (fail-closed by construction: an
# unlisted format is rejected, never silently guessed), trading a small amount of
# future editing for never having a format silently slip through unvalidated.
_RIFF_FORM_TYPE: dict[str, bytes] = {"audio": b"WAVE", "image": b"WEBP"}
_RIFF_HEADER_LEN = 12

# MP3 frame sync (INFO fix, Caspar residual — broadened MP3 detection): an MP3 file
# WITHOUT a leading ID3 tag starts directly with an MPEG audio frame header, whose
# first 11 bits are always set (the "frame sync"). The exact second byte varies by
# MPEG version (1/2/2.5) and layer — `_MAGIC`'s previous single fixed prefix
# (`b"\xff\xfb"`, MPEG-1 Layer III only) rejected the equally-common MPEG-2/2.5
# variants. Checked as a MASK (`byte0 == 0xFF and (byte1 & 0xE0) == 0xE0`), covering
# every MPEG version/layer/protection-bit combination, rather than enumerating every
# valid second-byte value as separate fixed-prefix tuple members.
_MP3_FRAME_SYNC_BYTE0 = 0xFF
_MP3_FRAME_SYNC_BYTE1_MASK = 0xE0


def _is_mp3_frame_sync(data: bytes) -> bool:
    """Return True if *data* starts with an MPEG audio frame-sync (any version/layer)."""
    return (
        len(data) >= 2
        and data[0] == _MP3_FRAME_SYNC_BYTE0
        and (data[1] & _MP3_FRAME_SYNC_BYTE1_MASK) == _MP3_FRAME_SYNC_BYTE1_MASK
    )


def _matches_kind(data: bytes, kind: str) -> bool:
    """Return True if *data*'s magic bytes are on the allow-list for *kind*."""
    if any(data.startswith(sig) for sig in _MAGIC.get(kind, ())):
        return True
    if kind == "audio" and _is_mp3_frame_sync(data):
        return True
    if data.startswith(b"RIFF") and len(data) >= _RIFF_HEADER_LEN:
        return data[8:12] == _RIFF_FORM_TYPE.get(kind)
    return False


# Prefix -> MIME for the audio signatures load_binary admits (MP3-via-frame-sync and WAV are
# handled below since they are a mask check / a RIFF form-type, not a plain prefix).
_AUDIO_MIME_BY_PREFIX: tuple[tuple[bytes, str], ...] = (
    (b"fLaC", "audio/flac"),
    (b"OggS", "audio/ogg"),
    (b"ID3", "audio/mpeg"),
)
_DEFAULT_AUDIO_MIME = "audio/wav"


def audio_mime_from_bytes(data: bytes) -> str:
    """Return the audio ``Content-Type`` from *data*'s MAGIC BYTES, never a filename extension.

    Mirrors the image side's magic-byte MIME detection so a mislabeled extension (a FLAC named
    ``.mp3``) can never smuggle a wrong ``Content-Type`` past the transcribe upload to a server
    that trusts the declared type over sniffing. Recognizes the SAME signatures
    :func:`load_binary` admits for ``kind="audio"`` -- FLAC (``fLaC``), OGG (``OggS``), MP3
    (an ``ID3`` tag or a raw MPEG frame-sync), and WAV (``RIFF``/``WAVE``). Anything else falls
    back to :data:`_DEFAULT_AUDIO_MIME` (unreachable after ``load_binary``'s magic-byte gate,
    kept defensive/total).

    Args:
        data: The raw audio bytes (already validated by ``load_binary``).

    Returns:
        The magic-byte-derived MIME string (e.g. ``"audio/flac"``).
    """
    for sig, mime in _AUDIO_MIME_BY_PREFIX:
        if data.startswith(sig):
            return mime
    if _is_mp3_frame_sync(data):
        return "audio/mpeg"
    if data.startswith(b"RIFF") and len(data) >= _RIFF_HEADER_LEN and data[8:12] == b"WAVE":
        return "audio/wav"
    return _DEFAULT_AUDIO_MIME


def load_binary(path: str, *, kind: str) -> bytes:
    """Read *path*, enforcing the size cap and a magic-byte allow-list for *kind*.

    Load-on-call semantics (R24b): this performs I/O + validation ONLY when called —
    never eagerly. A caller that QUEUES a binary delegation beyond
    `max_parallel_agents` (R21b) should hold the *path* by reference, not pre-loaded
    bytes, and call this at slot-acquisition/dispatch time. That bounds the memory of
    a parallel+queued fan-out to `max_parallel_agents x MAX_INPUT_BYTES` (the running
    set only, never the queued set), and a TOCTOU at dispatch — the referenced file
    vanished or changed size after it was queued — surfaces HERE as a `ValidationError`
    for that one delegation, never an unhandled crash of the whole batch.

    Wiring this loader into the actual vision/transcribe fan-out (the batch runner
    holding paths across MS5's semaphore/queue, materializing bytes only on slot
    acquisition) is an MS7 seam: MS1's `dispatch` currently rejects both capabilities
    outright (`_MS1_UNSUPPORTED_CAPS`) since their multimodal transport lands in MS7.
    MS6 provides the loader/guard in isolation; MS7 is the consumer.

    The size check is AUTHORITATIVE on the bytes actually read (`len(data)`), never a
    pre-read `os.path.getsize` — a TOCTOU where the file grows between a hypothetical
    size check and the read (or a symlink swapped underneath) can never slip an
    oversized payload through undetected.

    Bounded read (INFO fix, Caspar residual): the read itself is capped at
    `MAX_INPUT_BYTES + 1` bytes (`f.read(MAX_INPUT_BYTES + 1)`), never a bare `f.read()`
    of the whole file. `+1` is exactly enough to distinguish "at the cap" from "over the
    cap" (`len(data) > MAX_INPUT_BYTES`) without reading a single byte further, so a
    multi-GB file is rejected after reading only `MAX_INPUT_BYTES + 1` bytes into
    memory — never the entire file. This keeps the post-read, TOCTOU-free size check
    above AND bounds worst-case memory to `MAX_INPUT_BYTES + 1` regardless of the
    on-disk file size.

    Args:
        path: Path to the binary input.
        kind: ``"image"`` or ``"audio"``.

    Returns:
        The file bytes (at most `MAX_INPUT_BYTES` of them, on success).

    Raises:
        ValidationError: if the file cannot be read, the bytes actually read exceed
            `MAX_INPUT_BYTES`, or the magic bytes don't match `kind`'s allow-list
            (including the RIFF form-type disambiguation for WAV vs. WebP).
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read(MAX_INPUT_BYTES + 1)
    except OSError as exc:
        raise ValidationError(f"cannot read {kind} input {path!r}: {exc}") from exc
    if len(data) > MAX_INPUT_BYTES:
        raise ValidationError(
            f"{kind} input too large: exceeds {MAX_INPUT_BYTES} bytes "
            "(read bounded to the cap + 1 byte; rejected without reading the rest "
            "of the file)"
        )
    if not _matches_kind(data, kind):
        raise ValidationError(f"unrecognized {kind} format (magic-byte check failed)")
    return data


def to_data_uri(data: bytes, mime: str) -> str:
    """Return a base64 ``data:`` URI for *data* with *mime* type."""
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
