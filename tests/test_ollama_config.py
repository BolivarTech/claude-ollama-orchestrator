# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Config resolver + base_url normalization (idempotent, per-key precedence)."""

import string

import pytest
from hypothesis import assume, given
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


@pytest.mark.parametrize("raw", ["http://", "https://"])
def test_bare_scheme_with_empty_authority_raises_config_error(raw):
    # A scheme with nothing after it has no host: stripping trailing slashes
    # would otherwise collapse "http://" to "http:", which then looks
    # scheme-less and gets a bogus "http://" re-prepended ("http://http:").
    with pytest.raises(OllamaConfigError):
        normalize_base_url(raw)


_HOST_ALPHABET = string.ascii_letters + string.digits + ".-"
_PATH_ALPHABET = string.ascii_letters + string.digits + ".-/"

# Candidates built from a random host plus a segment *deliberately* drawn from
# values that contain "v1" (with noise variants like casing/duplication), so a
# large share of generated examples genuinely exercise the "don't double an
# already-present /v1" branch — not left to the astronomically low odds of a
# plain random-character strategy happening to spell out "v1" on its own.
_v1_bearing_candidates = st.builds(
    lambda host, use_https, v1_segment, extra: (
        ("https://" if use_https else "http://")
        + host
        + (f"/{v1_segment}" if v1_segment else "")
        + (f"/{extra}" if extra else "")
    ),
    host=st.text(alphabet=_HOST_ALPHABET, min_size=1, max_size=15),
    use_https=st.booleans(),
    v1_segment=st.sampled_from(["", "v1", "v1/", "v10", "v1v1", "V1"]),
    extra=st.text(alphabet=_PATH_ALPHABET, min_size=0, max_size=15),
)

# General-purpose noise: fully random text over an alphabet that *can* spell
# "v1" too, so the property isn't solely dependent on the constructed strategy.
_free_form_candidates = st.text(
    alphabet=string.ascii_letters + string.digits + ".:/-",
    min_size=1,
    max_size=40,
)


@given(st.one_of(_free_form_candidates, _v1_bearing_candidates))
def test_normalize_base_url_never_doubles_v1_and_stays_idempotent(s):
    candidate = s if "://" in s else "http://" + s
    try:
        out = normalize_base_url(candidate)
    except OllamaConfigError:
        # Some generated candidates are legitimately invalid (e.g. collapse to
        # a bare scheme with empty authority) — not in scope for this property.
        assume(False)
        return
    assert "/v1/v1" not in out
    # The real invariant the test name claims: normalizing an already
    # normalized value must be a no-op.
    assert normalize_base_url(out) == out
