# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Config resolver + base_url normalization (idempotent, per-key precedence)."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from errors import OllamaConfigError
from ollama_config import DEFAULT_BASE_URL, normalize_base_url


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("localhost:11434", "http://localhost:11434/v1"),
        ("http://localhost:11434", "http://localhost:11434/v1"),
        ("http://localhost:11434/", "http://localhost:11434/v1"),
        ("http://localhost:11434/v1", "http://localhost:11434/v1"),
        ("http://localhost:11434/v1/", "http://localhost:11434/v1"),
        ("http://proxy/ollama/v1", "http://proxy/ollama/v1"),
        ("https://api.cloud/v1", "https://api.cloud/v1"),
        ("https://api.cloud", "https://api.cloud/v1"),  # bare https host (no path) → /v1
        ("https://api.cloud/", "https://api.cloud/v1"),
    ],
)
def test_normalize_base_url_is_idempotent_and_never_doubles_v1(raw, expected):
    assert normalize_base_url(raw) == expected
    assert normalize_base_url(normalize_base_url(raw)) == expected


def test_default_base_url_constant():
    assert DEFAULT_BASE_URL == "http://localhost:11434/v1"


def test_empty_or_whitespace_base_url_raises_config_error():
    # An explicitly-empty/whitespace-only base_url is a misconfiguration, not "unset" —
    # resolve_config's `or`-chain already skips empty values when falling back to the
    # next layer/default (presence-semantics), so this guard only fires when
    # normalize_base_url is called directly with a value that has no real content.
    with pytest.raises(OllamaConfigError):
        normalize_base_url("")
    with pytest.raises(OllamaConfigError):
        normalize_base_url("   ")
    with pytest.raises(OllamaConfigError):
        normalize_base_url("///")  # all-slashes collapses to empty after stripping


@given(st.text(alphabet="abcdefghijklmnop.:/-", min_size=1, max_size=40))
def test_normalize_base_url_never_produces_double_v1(s):
    out = normalize_base_url(s if "://" in s else "http://" + s)
    assert "/v1/v1" not in out
