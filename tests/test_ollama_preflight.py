# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Fail-fast preflight: reachability, model existence, auth, warn-and-proceed."""

import http.client
import io
import json
import urllib.error
from dataclasses import replace
from types import MappingProxyType

import pytest

from errors import OllamaPreflightError
from ollama_config import resolve_config
from ollama_preflight import PREFLIGHT_MAX_RESPONSE_BYTES, preflight


def _cfg(models=None, api_key=None):
    cfg = resolve_config(global_path=None, repo_path=None, env={})
    return replace(
        cfg, api_key=api_key, models=MappingProxyType(models or {"coder": "m-a", "reviewer": "m-b"})
    )


def _fake_urlopen(payload=None, error=None):
    def _open(req, timeout=None):
        if error is not None:
            raise error
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    return _open


def test_preflight_passes_when_all_models_present():
    payload = {"data": [{"id": "m-a"}, {"id": "m-b"}, {"id": "other"}]}
    preflight(_cfg(), urlopen=_fake_urlopen(payload))


def test_preflight_aborts_on_missing_model_with_actionable_message():
    with pytest.raises(OllamaPreflightError) as exc:
        preflight(_cfg(), urlopen=_fake_urlopen({"data": [{"id": "m-a"}]}))
    assert "m-b" in str(exc.value)
    assert "ollama pull" in str(exc.value) or "signin" in str(exc.value)


def test_preflight_aborts_on_unreachable_host():
    with pytest.raises(OllamaPreflightError):
        preflight(_cfg(), urlopen=_fake_urlopen(error=urllib.error.URLError("refused")))


def test_preflight_401_aborts_and_redacts_api_key():
    err = urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b""))
    with pytest.raises(OllamaPreflightError) as exc:
        preflight(_cfg(api_key="sk-secret"), urlopen=_fake_urlopen(error=err))
    assert "sk-secret" not in str(exc.value)


def test_preflight_404_warns_and_proceeds(capsys):
    err = urllib.error.HTTPError("u", 404, "Not Found", {}, io.BytesIO(b""))
    preflight(_cfg(), urlopen=_fake_urlopen(error=err))  # no raise: proceeds normally
    captured = capsys.readouterr()
    assert captured.out == ""
    warnings = [line for line in captured.err.splitlines() if line.strip()]
    assert len(warnings) == 1
    assert "does not support listing models" in warnings[0]


def test_preflight_aborts_on_non_json_response_with_actionable_message():
    # A proxy/misconfigured endpoint can return HTTP 200 with a non-JSON body (e.g. an
    # HTML error page from a reverse proxy); this must map to a domain
    # OllamaPreflightError, never an uncaught json.JSONDecodeError/raw traceback.
    def _html(req, timeout=None):
        return io.BytesIO(b"<html>502 Bad Gateway</html>")

    with pytest.raises(OllamaPreflightError) as exc:
        preflight(_cfg(), urlopen=_html)
    assert "non-JSON" in str(exc.value)


def test_preflight_invalid_utf8_response_bytes_do_not_crash():
    # The /models response body is untrusted server output. Malformed UTF-8 bytes must
    # decode via errors="replace" (U+FFFD substitution) — NEVER raise a raw
    # UnicodeDecodeError (a non-domain exception). The substituted text is not valid
    # JSON, so this surfaces as the existing domain OllamaPreflightError, exactly like
    # any other non-JSON body — not a crash.
    def _bad_utf8(req, timeout=None):
        return io.BytesIO(b"\xff\xfe not valid utf-8 \x80\x81")

    with pytest.raises(OllamaPreflightError):
        preflight(_cfg(), urlopen=_bad_utf8)


def test_preflight_oversized_models_response_raises_actionable_error():
    # Mirrors backend.MAX_RESPONSE_BYTES: the /models read must be bounded too, never a
    # bare resp.read() — a runaway/hostile response is rejected before any decode.
    def _oversized(req, timeout=None):
        return io.BytesIO(b"x" * (PREFLIGHT_MAX_RESPONSE_BYTES + 1))

    with pytest.raises(OllamaPreflightError) as exc:
        preflight(_cfg(), urlopen=_oversized)
    assert "PREFLIGHT_MAX_RESPONSE_BYTES" in str(exc.value)


def test_preflight_checks_the_effective_model_override_not_the_config_default():
    # R10 + R28: a `--model` override must be validated by preflight under the SAME
    # actionable-abort path as any other configured model — never silently skipped so a
    # bad override only surfaces as a later chat-time 404. Config says coder="m-a" (which
    # IS present), but the effective override "does-not-exist:cloud" is NOT present.
    payload = {"data": [{"id": "m-a"}, {"id": "m-b"}]}
    with pytest.raises(OllamaPreflightError) as exc:
        preflight(
            _cfg(),
            urlopen=_fake_urlopen(payload),
            capability="coder",
            effective_model="does-not-exist:cloud",
        )
    assert "does-not-exist:cloud" in str(exc.value)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        None,
        "x",
        {"data": None},
    ],
    ids=["bare-array", "bare-null", "bare-string", "data-null"],
)
def test_preflight_unexpected_models_shape_raises_domain_error_not_raw_exception(payload):
    # Untrusted /models output that is valid JSON but the WRONG shape must map to the
    # domain OllamaPreflightError this module exists to guarantee — never a raw
    # AttributeError (payload not a dict) or TypeError (payload["data"] not iterable).
    # {"data": None} is the key regression case: dict.get("data", []) only applies its
    # default when the key is ABSENT, not when present with value null.
    with pytest.raises(OllamaPreflightError) as exc:
        preflight(_cfg(), urlopen=_fake_urlopen(payload))
    assert "shape" in str(exc.value)


class _RaisingReadResponse:
    """Fake response whose ``.read()`` raises *exc* (simulates a mid-read connection drop)."""

    def __init__(self, exc):
        self._exc = exc

    def read(self, *args, **kwargs):
        raise self._exc


def test_preflight_incomplete_read_maps_to_domain_error_not_raw_traceback():
    # A truncated /models response (connection dropped mid-body) raises
    # http.client.IncompleteRead from resp.read() -- an HTTPException, NOT an
    # OSError/URLError subclass, so it must be explicitly guarded (mirrors
    # backend.OllamaBackend's own (OSError, http.client.IncompleteRead) catch-all) rather
    # than escape preflight() as a raw, uncaught exception.
    def _open(req, timeout=None):
        return _RaisingReadResponse(http.client.IncompleteRead(b"partial"))

    with pytest.raises(OllamaPreflightError):
        preflight(_cfg(), urlopen=_open)


def test_preflight_connection_reset_during_read_maps_to_domain_error():
    # Any other read-time OSError not already covered by the more specific
    # (socket.timeout, TimeoutError, URLError, IncompleteRead) arm -- e.g. a
    # ConnectionResetError -- must also map to the domain OllamaPreflightError via the
    # residual `except OSError` catch-all, never propagate as a raw exception.
    def _open(req, timeout=None):
        return _RaisingReadResponse(ConnectionResetError("connection reset by peer"))

    with pytest.raises(OllamaPreflightError):
        preflight(_cfg(), urlopen=_open)


def test_preflight_effective_model_present_passes_and_other_models_still_checked():
    # The override substitutes ONLY the given capability's slot; every other configured
    # capability's model is still validated against /models unchanged.
    payload = {"data": [{"id": "override-model:cloud"}, {"id": "m-b"}]}
    preflight(
        _cfg(),
        urlopen=_fake_urlopen(payload),
        capability="coder",
        effective_model="override-model:cloud",
    )  # no raise
    with pytest.raises(OllamaPreflightError) as exc:
        preflight(
            _cfg(),
            urlopen=_fake_urlopen({"data": [{"id": "override-model:cloud"}]}),
            capability="coder",
            effective_model="override-model:cloud",
        )
    assert "m-b" in str(exc.value)  # reviewer's configured model is still enforced
