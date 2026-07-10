# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Pre-dispatch hardening, tested in isolation from run_ollama.py's CLI surface
(R26 console IO, NR3 config permissions, NR9 remote transport, R24 oversize input)."""

import os

import pytest

from startup_hardening import (
    enable_utf8_console_io,
    warn_if_config_world_readable,
    warn_if_oversize,
    warn_if_remote_http_with_api_key,
)


def test_windows_console_io_hardening_reconfigures_stdout_and_stderr(monkeypatch):
    import startup_hardening

    class _Stream:
        def __init__(self):
            self.calls = []

        def reconfigure(self, **kw):
            self.calls.append(kw)

    out, err = _Stream(), _Stream()
    monkeypatch.setattr(startup_hardening.sys, "platform", "win32")
    monkeypatch.setattr(startup_hardening.sys, "stdout", out)
    monkeypatch.setattr(startup_hardening.sys, "stderr", err)
    enable_utf8_console_io()
    assert out.calls == [{"encoding": "utf-8", "errors": "backslashreplace"}]
    assert err.calls == [{"encoding": "utf-8", "errors": "backslashreplace"}]


def test_windows_console_io_hardening_is_a_noop_off_windows_and_never_raises(monkeypatch):
    import startup_hardening

    monkeypatch.setattr(startup_hardening.sys, "platform", "linux")
    enable_utf8_console_io()  # must not raise, even without .reconfigure


def test_world_readable_config_emits_chmod_warning(tmp_path, capsys):
    if os.name != "posix":
        pytest.skip("POSIX-only permission semantics (NR3)")
    p = tmp_path / "ollama-agents.toml"
    p.write_text('base_url = "http://localhost:11434/v1"\n', encoding="utf-8")
    os.chmod(p, 0o644)
    warn_if_config_world_readable(str(p))
    assert "chmod 600" in capsys.readouterr().err.lower()


def test_private_config_emits_no_warning(tmp_path, capsys):
    if os.name != "posix":
        pytest.skip("POSIX-only permission semantics (NR3)")
    p = tmp_path / "ollama-agents.toml"
    p.write_text('base_url = "http://localhost:11434/v1"\n', encoding="utf-8")
    os.chmod(p, 0o600)
    warn_if_config_world_readable(str(p))
    assert capsys.readouterr().err == ""


def test_missing_config_path_emits_no_warning(tmp_path, capsys):
    # Best-effort (NR3): a missing config path is silently skipped, never raises.
    warn_if_config_world_readable(str(tmp_path / "does-not-exist.toml"))
    assert capsys.readouterr().err == ""


def test_remote_http_with_api_key_warns_without_echoing_the_key(capsys):
    warn_if_remote_http_with_api_key("http://192.168.0.30:11434/v1", "sk-secret")
    err = capsys.readouterr().err.lower()
    assert "https" in err
    assert "sk-secret" not in err


def test_localhost_http_with_api_key_does_not_warn(capsys):
    warn_if_remote_http_with_api_key("http://localhost:11434/v1", "sk-secret")
    assert capsys.readouterr().err == ""


def test_ipv6_loopback_bracketed_http_with_api_key_does_not_warn(capsys):
    # INFO fix: a bracketed IPv6 loopback URL host (`urlsplit(...).hostname` returns
    # the unbracketed, lower-cased `::1`) must be treated as local, same as
    # `localhost`/`127.0.0.1` above.
    warn_if_remote_http_with_api_key("http://[::1]:11434/v1", "sk-secret")
    assert capsys.readouterr().err == ""


def test_ipv6_loopback_expanded_form_does_not_warn(capsys):
    # INFO fix: the fully-expanded form of the SAME loopback address
    # (`0:0:0:0:0:0:0:1`) must also be treated as local, not just its compressed
    # `::1` shorthand.
    warn_if_remote_http_with_api_key("http://[0:0:0:0:0:0:0:1]:11434/v1", "sk-secret")
    assert capsys.readouterr().err == ""


def test_full_ipv4_loopback_range_does_not_warn(capsys):
    # [WARNING → fix, this revision] The previous allow-list only recognized the exact
    # literal `127.0.0.1` — `127.0.0.2` (a valid loopback alias, RFC 5735/6890, sometimes
    # used to disambiguate multiple local services on one host) was incorrectly treated
    # as remote. `ipaddress.ip_address("127.0.0.2").is_loopback` is True for the ENTIRE
    # 127.0.0.0/8 block, not just the one address.
    warn_if_remote_http_with_api_key("http://127.0.0.2:11434/v1", "sk-secret")
    assert capsys.readouterr().err == ""


def test_real_lan_ip_with_api_key_still_warns(capsys):
    # Guards against the broadened check over-widening into "everything is local": a
    # genuine LAN address (outside 127.0.0.0/8, not a loopback) must still warn.
    warn_if_remote_http_with_api_key("http://192.168.1.50:11434/v1", "sk-secret")
    err = capsys.readouterr().err.lower()
    assert "https" in err
    assert "sk-secret" not in err


def test_remote_https_with_api_key_does_not_warn(capsys):
    warn_if_remote_http_with_api_key("https://api.cloud/v1", "sk-secret")
    assert capsys.readouterr().err == ""


def test_no_api_key_never_warns_regardless_of_scheme(capsys):
    warn_if_remote_http_with_api_key("http://192.168.0.30:11434/v1", None)
    assert capsys.readouterr().err == ""


def test_warn_if_oversize_emits_warning_over_threshold(capsys):
    warn_if_oversize("a" * 4000, threshold=1)
    assert "large" in capsys.readouterr().err.lower()


def test_warn_if_oversize_is_silent_under_threshold(capsys):
    warn_if_oversize("short", threshold=1_000_000)
    assert capsys.readouterr().err == ""
