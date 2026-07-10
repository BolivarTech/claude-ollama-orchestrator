# tests/test_transcribe.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Transcribe is experimental/gated: success via /audio/transcriptions or audio-chat;
an actionable error (never a crash) when the endpoint supports neither."""

import io
import json
import urllib.error

import pytest

from errors import OllamaBackendError
from ollama_config import resolve_config
from transcribe import transcribe

# A minimal valid WAV: RIFF + 4-byte size + WAVE form-type (MS6 magic-byte allow-list).
_WAV = b"RIFF" + b"\x24\x00\x00\x00" + b"WAVE" + b"\x00" * 8


def _cfg(**env):
    return resolve_config(global_path=None, repo_path=None, env=env)


def test_transcribe_errors_actionably_when_endpoint_has_no_audio(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(_WAV)
    with pytest.raises(OllamaBackendError) as exc:
        transcribe(
            _cfg(),
            str(audio),
            "gemma4:cloud",
            60,
            probe=lambda url: False,  # no /audio/transcriptions
            multimodal_audio=lambda model: False,
        )  # model not audio-multimodal
    assert "experimental" in str(exc.value).lower()
    assert "audio" in str(exc.value).lower()


def test_transcribe_via_audio_endpoint_returns_the_transcript(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(_WAV)
    posted = {}

    def _fake_urlopen(req, timeout=None):
        posted["url"] = req.full_url
        posted["ctype"] = req.get_header("Content-type")
        posted["timeout"] = timeout
        return io.BytesIO(json.dumps({"text": "hello world"}).encode("utf-8"))

    res = transcribe(
        _cfg(), str(audio), "whisper-1", 60, probe=lambda url: True, urlopen=_fake_urlopen
    )
    assert res.content == "hello world"
    assert posted["url"].endswith("/audio/transcriptions")
    assert posted["ctype"].startswith("multipart/form-data")  # multipart upload
    assert posted["timeout"] == 60  # the DELEGATION timeout, not a hardcoded 60


def test_transcribe_via_audio_endpoint_forwards_a_non_default_timeout(tmp_path):
    # Regression: `_via_audio_endpoint` used to hardcode `timeout=60` regardless of the
    # caller's timeout. Use a value that would fail this assertion under the old bug.
    audio = tmp_path / "a.wav"
    audio.write_bytes(_WAV)
    posted = {}

    def _fake_urlopen(req, timeout=None):
        posted["timeout"] = timeout
        return io.BytesIO(json.dumps({"text": "ok"}).encode("utf-8"))

    transcribe(_cfg(), str(audio), "whisper-1", 137, probe=lambda url: True, urlopen=_fake_urlopen)
    assert posted["timeout"] == 137


def test_via_audio_endpoint_wraps_http_error_as_a_domain_exception(tmp_path):
    # NR4 regression: a raw `urllib.error.HTTPError` from the `/audio/transcriptions`
    # transport must never escape to the orchestrator — wrapped into the SAME domain
    # `OllamaBackendError` every other HTTP transport in this plugin raises, redacted.
    audio = tmp_path / "a.wav"
    audio.write_bytes(_WAV)

    def _fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 500, "Internal Server Error", hdrs=None, fp=io.BytesIO(b"boom")
        )

    with pytest.raises(OllamaBackendError):
        transcribe(
            _cfg(), str(audio), "whisper-1", 60, probe=lambda url: True, urlopen=_fake_urlopen
        )


def test_via_audio_endpoint_wraps_url_error_as_a_domain_exception(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(_WAV)

    def _fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    with pytest.raises(OllamaBackendError):
        transcribe(
            _cfg(), str(audio), "whisper-1", 60, probe=lambda url: True, urlopen=_fake_urlopen
        )


def test_via_audio_endpoint_wraps_malformed_json_as_a_domain_exception(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(_WAV)

    def _fake_urlopen(req, timeout=None):
        return io.BytesIO(b"this is not json {")

    with pytest.raises(OllamaBackendError):
        transcribe(
            _cfg(), str(audio), "whisper-1", 60, probe=lambda url: True, urlopen=_fake_urlopen
        )


def test_default_probe_receives_a_timeout_bounded_by_probe_timeout_seconds(tmp_path):
    # INFO fix: `_default_probe` used to hardcode `timeout=5` with no link at all to the
    # delegation's own `timeout`. `transcribe` now derives a bounded probe timeout
    # (`min(timeout, PROBE_TIMEOUT_SECONDS)`) and forwards it — plus the injected
    # `urlopen` — to the default probe, so a slow/hanging probe can never outlast this
    # bound regardless of how large the delegation `timeout` is.
    #
    # DEVIATION from the plan's literal test body (documented, see task report): the
    # plan's version recorded the timeout in a single `seen["timeout"]` dict key. With no
    # `probe=` override AND the default probe succeeding (no exception raised), `transcribe`
    # makes a SECOND `urlopen` call for the real `/audio/transcriptions` POST (using the
    # full delegation timeout, 600 — required/asserted by
    # `test_transcribe_via_audio_endpoint_returns_the_transcript` et al.) — that second
    # call would overwrite a single shared dict key, making the original assertion
    # (`== PROBE_TIMEOUT_SECONDS`) unsatisfiable by ANY correct implementation. Recording
    # every call's timeout in a list and asserting on the FIRST entry (the probe's own
    # GET) preserves the test's actual intent — the probe is bounded independently of the
    # delegation timeout — without depending on whether a second call happens afterward.
    from transcribe import PROBE_TIMEOUT_SECONDS

    audio = tmp_path / "a.wav"
    audio.write_bytes(_WAV)
    seen_timeouts: list[int | None] = []

    def _fake_urlopen(req, timeout=None):
        seen_timeouts.append(timeout)
        return io.BytesIO(json.dumps({"text": "ok"}).encode("utf-8"))

    # No `probe=` override: exercises the real `_default_probe` default-resolution path.
    transcribe(_cfg(), str(audio), "whisper-1", 600, urlopen=_fake_urlopen)
    assert seen_timeouts[0] == PROBE_TIMEOUT_SECONDS == min(600, PROBE_TIMEOUT_SECONDS)


def test_transcribe_transport_endpoint_forces_the_endpoint_transport_and_skips_the_probe(
    tmp_path,
):
    audio = tmp_path / "a.wav"
    audio.write_bytes(_WAV)
    probed = {"called": False}

    def _probe(url):
        probed["called"] = True
        return True

    def _fake_urlopen(req, timeout=None):
        return io.BytesIO(json.dumps({"text": "forced endpoint"}).encode("utf-8"))

    res = transcribe(
        _cfg(),
        str(audio),
        "whisper-1",
        60,
        transport="endpoint",
        probe=_probe,
        urlopen=_fake_urlopen,
    )
    assert res.content == "forced endpoint"
    assert probed["called"] is False  # "endpoint" bypasses the probe entirely


def test_transcribe_transport_chat_forces_the_chat_transport_and_skips_the_probe(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(_WAV)
    probed = {"called": False}

    def _probe(url):
        probed["called"] = True
        return False  # would otherwise fall through to the gate

    def _fake_stream_run(
        config, system_prompt, prompt, model, timeout, *, sink, content_parts=None, **_kw
    ):
        assert content_parts is not None
        from backend import DelegationResult

        return DelegationResult("forced chat", 3, 2, False, 0.1)

    res = transcribe(
        _cfg(),
        str(audio),
        "gemma4:cloud",
        60,
        transport="chat",
        probe=_probe,
        stream_fn=_fake_stream_run,
        sink=lambda _s: None,
    )
    assert res.content == "forced chat"
    assert probed["called"] is False  # "chat" bypasses the probe entirely


def test_transcribe_transport_defaults_to_the_resolved_config_value(tmp_path):
    # No explicit `transport=` kwarg: falls back to `config.transcribe_transport`
    # ("auto" by default), which runs the probe exactly like the pre-existing behavior.
    audio = tmp_path / "a.wav"
    audio.write_bytes(_WAV)
    probed = {"called": False}

    def _probe(url):
        probed["called"] = True
        return True

    def _fake_urlopen(req, timeout=None):
        return io.BytesIO(json.dumps({"text": "auto"}).encode("utf-8"))

    res = transcribe(_cfg(), str(audio), "whisper-1", 60, probe=_probe, urlopen=_fake_urlopen)
    assert res.content == "auto" and probed["called"] is True


def test_transcribe_via_audio_multimodal_chat_returns_the_transcript(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(_WAV)

    def _fake_stream_run(
        config, system_prompt, prompt, model, timeout, *, sink, content_parts=None, **_kw
    ):
        assert content_parts is not None  # audio sent as a content-part
        from backend import DelegationResult

        return DelegationResult("spoken text", 3, 2, False, 0.2)

    res = transcribe(
        _cfg(),
        str(audio),
        "gemma4:cloud",
        60,
        probe=lambda url: False,  # no dedicated endpoint
        multimodal_audio=lambda model: True,  # but the model accepts audio
        stream_fn=_fake_stream_run,
        sink=lambda _s: None,
    )
    assert res.content == "spoken text"


def test_transcribe_via_audio_chat_forwards_the_caller_system_prompt(tmp_path):
    # WARNING fix (regression): `_via_audio_chat` must forward the CALLER-supplied
    # `system_prompt` to `stream_fn` as the system message, never a hardcoded string.
    # Use a non-default value so the assertion fails under the old hardcoded-string bug.
    audio = tmp_path / "a.wav"
    audio.write_bytes(_WAV)
    seen = {}

    def _fake_stream_run(
        config, system_prompt, prompt, model, timeout, *, sink, content_parts=None, **_kw
    ):
        seen["system_prompt"] = system_prompt
        from backend import DelegationResult

        return DelegationResult("ok", 1, 1, False, 0.1)

    custom_prompt = "Custom domain-specific transcription instructions."
    transcribe(
        _cfg(),
        str(audio),
        "gemma4:cloud",
        60,
        probe=lambda url: False,
        multimodal_audio=lambda model: True,
        stream_fn=_fake_stream_run,
        sink=lambda _s: None,
        system_prompt=custom_prompt,
    )
    assert seen["system_prompt"] == custom_prompt


def test_transcribe_rejects_a_non_audio_file(tmp_path):
    from errors import ValidationError

    bad = tmp_path / "a.wav"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(ValidationError):
        transcribe(_cfg(), str(bad), "whisper-1", 60, probe=lambda url: True)


def test_multipart_body_round_trips_through_a_stdlib_multipart_parser():
    """The hand-rolled `_multipart_body` encoder alone can't catch a boundary/CRLF bug
    that still happens to produce something `urlopen` swallows without complaint — round
    -trip the encoded body through a REAL stdlib multipart parser (`email`) and assert the
    field + file bytes survive byte-for-byte. Catches encoding bugs the encoder-only tests
    above cannot (they only assert on what the encoder claims to have written)."""
    import email
    from email import policy

    from transcribe import _multipart_body

    body, content_type = _multipart_body(
        {"model": "whisper-1"},
        file_field="file",
        filename="a.wav",
        file_bytes=_WAV,
        mime="audio/wav",
    )
    # email.message_from_bytes needs a Content-Type header to locate the boundary.
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii")
    msg = email.message_from_bytes(header + body, policy=policy.compat32)
    assert msg.is_multipart()
    fields = {p.get_param("name", header="Content-Disposition"): p for p in msg.get_payload()}
    assert fields["model"].get_payload(decode=True) == b"whisper-1"
    file_part = fields["file"]
    assert file_part.get_filename() == "a.wav"
    assert file_part.get_content_type() == "audio/wav"
    assert file_part.get_payload(decode=True) == _WAV  # exact bytes survive round-trip


def test_multipart_body_escapes_a_filename_with_quotes_backslashes_and_a_newline():
    """[CRITICAL/SECURITY] a filename containing `"`, `\\`, and CR/LF must not be able to
    break out of the `Content-Disposition` header's quoted `filename="..."` value or
    inject a new header/part (MIME header injection). Round-trip through the SAME real
    stdlib multipart parser used above so the assertion is about the actual produced
    bytes, not just what the encoder claims to have written."""
    import email
    from email import policy

    from transcribe import _multipart_body

    hostile_filename = 'evil"\\name\r\nX-Injected: pwned.wav'
    body, content_type = _multipart_body(
        {"model": "whisper-1"},
        file_field="file",
        filename=hostile_filename,
        file_bytes=_WAV,
        mime="audio/wav",
    )

    # DEVIATION from the plan's literal `assert b"X-Injected" not in body` (documented,
    # see task report): per `_escape_multipart_filename`'s own docstring, CR/LF are
    # REMOVED (not the whole tail truncated), so the harmless LITERAL text
    # "X-Injected: pwned.wav" still appears, concatenated onto the escaped filename value
    # — it is no longer preceded by a CRLF, so it can never start a new header line. The
    # actual security invariant is that it can NEVER be interpreted as a real header/part
    # (checked below via the real stdlib parser), not that the substring is textually
    # absent. No raw CR/LF survives anywhere in the body — that is exactly what would let
    # the filename smuggle a new header line or multipart boundary.
    assert b"\r\n\r\nX-Injected" not in body
    assert b"X-Injected: pwned.wav\r\n" not in body

    # The body still parses as a well-formed multipart message with exactly one file part
    # (a broken/injected header would either fail to parse or produce extra parts).
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii")
    msg = email.message_from_bytes(header + body, policy=policy.compat32)
    assert msg.is_multipart()
    payload = msg.get_payload()
    assert len(payload) == 2  # exactly "model" + "file", no extras
    fields = {p.get_param("name", header="Content-Disposition"): p for p in payload}
    file_part = fields["file"]
    assert file_part.get_payload(decode=True) == _WAV  # file content is untouched
    assert file_part.get("X-Injected") is None  # never became a real MIME header
    # The quote/backslash were escaped (not dropped) and the newline was stripped — the
    # parsed filename reflects that escaping, never the raw hostile string verbatim.
    recovered = file_part.get_filename()
    assert "\r" not in recovered and "\n" not in recovered
    assert recovered != hostile_filename  # never passed through unescaped


def test_multipart_body_regenerates_a_boundary_that_collides_with_the_file_content():
    """[SECURITY/defense-in-depth] The random uuid4 boundary makes a content collision
    cryptographically negligible, but if the boundary somehow occurred in file_bytes a crafted
    file could smuggle a premature part terminator. The builder scans and regenerates until the
    boundary is absent from the content."""
    from transcribe import _multipart_body

    colliding = "----ollamaCOLLIDES"
    clean = "----ollamaSAFEBOUNDARY"
    # File content that CONTAINS the first (colliding) boundary's delimiter form `--<boundary>`.
    file_bytes = b"audio...--" + colliding.encode() + b"...more"
    boundaries = iter([colliding, clean])

    body, content_type = _multipart_body(
        {"model": "whisper-1"},
        file_field="file",
        filename="a.wav",
        file_bytes=file_bytes,
        mime="audio/wav",
        _boundary_factory=lambda: next(boundaries),
    )
    assert clean in content_type  # regenerated to the non-colliding boundary
    assert colliding not in content_type  # never used the colliding one as the delimiter


def test_escape_multipart_filename_strips_every_control_char_not_only_crlf():
    """[SECURITY/defense-in-depth] CR/LF are the only *exploitable* chars (MIME headers
    split solely on them), but the escape strips EVERY C0 control char, DEL, and the
    Unicode line/paragraph separators (VT, FF, NUL, TAB, DEL, NEL, LS, PS) as well, so no
    control character rides into the quoted header value at all. Printable content — and
    the escaping of `"`/`\\` into `\\"`/`\\\\` — is preserved."""
    from transcribe import _escape_multipart_filename

    control = "a\x00\t\x0b\x0c\x7f  b"
    assert _escape_multipart_filename(control) == "ab"
    # quote and backslash are escaped (kept as printable), not stripped
    assert _escape_multipart_filename('q"x\\y') == 'q\\"x\\\\y'
    # a legitimate non-ASCII filename is untouched
    assert _escape_multipart_filename("café_日本.wav") == "café_日本.wav"


def test_multipart_body_neutralizes_every_header_interpolated_field_not_only_filename():
    """[SECURITY] Every string interpolated into a multipart HEADER -- the field `name`, the
    `file_field`, and the `mime` -- must be neutralized, not only the filename. A CR/LF in any
    of them could otherwise break out of a `Content-Disposition`/`Content-Type` header and
    inject an arbitrary header or extra part. Round-trip through the real stdlib multipart
    parser and assert no injected header survives on any part, and no raw injected header line
    appears in the header region of the body."""
    import email
    from email import policy

    from transcribe import _multipart_body

    body, content_type = _multipart_body(
        {"model\r\nX-Injected: pwned": "v\r\nY-Injected: pwned"},
        file_field='fi\r\nZ-Injected: pwned"le',
        filename="ok.wav",
        file_bytes=_WAV,
        mime="audio/wav\r\nW-Injected: pwned",
    )
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii")
    msg = email.message_from_bytes(header + body, policy=policy.compat32)
    assert msg.is_multipart()
    for part in msg.get_payload():
        for injected in ("X-Injected", "Y-Injected", "Z-Injected", "W-Injected"):
            assert part.get(injected) is None  # none became a real MIME header
    # No raw CR/LF-delimited injected header line survives anywhere in the body.
    assert b"\r\nX-Injected: pwned" not in body
    assert b"\r\nZ-Injected: pwned" not in body
    assert b"\r\nW-Injected: pwned" not in body
