# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Config resolver + base_url normalization (idempotent, per-key precedence)."""

import string
import textwrap

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from errors import OllamaConfigError
from ollama_config import (
    CAPABILITIES,
    DEFAULT_BASE_URL,
    DEFAULT_MODELS,
    OllamaAgentsConfig,
    normalize_base_url,
    resolve_config,
)


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


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(p)


def test_defaults_when_no_files_no_env():
    cfg = resolve_config(global_path=None, repo_path=None, env={})
    assert isinstance(cfg, OllamaAgentsConfig)
    assert cfg.base_url == "http://localhost:11434/v1"
    assert cfg.api_key is None
    assert cfg.models["coder"] == "kimi-k2.7-code:cloud"
    assert cfg.max_parallel_agents == 3
    assert cfg.max_queued_agents == 32
    assert cfg.structured["reviewer"] == "schema"
    assert cfg.structured["coder"] == "off"
    assert cfg.stream["coder"] is True
    assert cfg.stream["reviewer"] is False
    assert tuple(cfg.models) == CAPABILITIES


def test_per_key_merge_repo_base_url_global_models(tmp_path):
    g = _write(tmp_path, "global.toml", '[models]\ncoder = "global-coder:cloud"\n')
    r = _write(tmp_path, "repo.toml", 'base_url = "http://repo:9999"\n')
    cfg = resolve_config(global_path=g, repo_path=r, env={})
    assert cfg.base_url == "http://repo:9999/v1"
    assert cfg.models["coder"] == "global-coder:cloud"


def test_env_overrides_files_for_model(tmp_path):
    r = _write(tmp_path, "repo.toml", '[models]\ncoder = "repo-coder:cloud"\n')
    cfg = resolve_config(
        global_path=None, repo_path=r, env={"OLLAMA_AGENTS_MODEL_CODER": "env-coder:cloud"}
    )
    assert cfg.models["coder"] == "env-coder:cloud"


def test_api_key_presence_semantics_empty_env_means_none(tmp_path):
    r = _write(tmp_path, "repo.toml", 'api_key = "sk-from-file"\n')
    cfg = resolve_config(global_path=None, repo_path=r, env={"OLLAMA_AGENTS_API_KEY": ""})
    assert cfg.api_key is None


def test_generic_ollama_host_env_is_base_url_fallback():
    # R6: with no OLLAMA_AGENTS_HOST / repo / global base_url, the generic OLLAMA_HOST env
    # is used and normalized idempotently (bare host:port → append /v1).
    cfg = resolve_config(
        global_path=None, repo_path=None, env={"OLLAMA_HOST": "http://192.168.0.30:11434"}
    )
    assert cfg.base_url == "http://192.168.0.30:11434/v1"


def test_empty_string_model_override_is_not_an_override():
    # Presence-semantics (R6): a present-but-empty override is DISTINCT from absent —
    # it does not set an empty model tag; it falls through to the built-in default.
    cfg = resolve_config(global_path=None, repo_path=None, env={"OLLAMA_AGENTS_MODEL_CODER": ""})
    assert cfg.models["coder"] == "kimi-k2.7-code:cloud"


def test_malformed_toml_raises_domain_error(tmp_path):
    bad = _write(tmp_path, "bad.toml", "this is = = not toml")
    with pytest.raises(OllamaConfigError):
        resolve_config(global_path=None, repo_path=bad, env={})


def test_invalid_max_parallel_raises():
    with pytest.raises(OllamaConfigError):
        resolve_config(global_path=None, repo_path=None, env={"OLLAMA_AGENTS_MAX_PARALLEL": "0"})


def test_stream_string_false_coerces_to_false_not_true(tmp_path):
    # The bool('false') is True trap: a STRING "false" in TOML must coerce to False.
    r = _write(tmp_path, "repo.toml", '[stream]\ncoder = "false"\n')
    cfg = resolve_config(global_path=None, repo_path=r, env={})
    assert cfg.stream["coder"] is False


def test_stream_env_and_native_bool_coerce_correctly(tmp_path):
    r = _write(tmp_path, "repo.toml", "[stream]\ncoder = false\n")  # native TOML bool
    cfg = resolve_config(
        global_path=None, repo_path=r, env={"OLLAMA_AGENTS_STREAM_REVIEWER": "true"}
    )
    assert cfg.stream["coder"] is False and cfg.stream["reviewer"] is True


def test_stream_invalid_value_raises(tmp_path):
    r = _write(tmp_path, "repo.toml", '[stream]\ncoder = "maybe"\n')
    with pytest.raises(OllamaConfigError):
        resolve_config(global_path=None, repo_path=r, env={})


def test_config_is_frozen():
    cfg = resolve_config(global_path=None, repo_path=None, env={})
    with pytest.raises(Exception):
        cfg.base_url = "mutated"  # type: ignore[misc]


def test_non_string_base_url_raises_config_error(tmp_path):
    # A TOML `base_url = 123` (int) must never reach normalize_base_url()'s `raw.strip()`
    # or a later f"{base_url}/chat/completions" as a non-string — caught here as an
    # actionable OllamaConfigError, not a raw TypeError/AttributeError downstream.
    r = _write(tmp_path, "repo.toml", "base_url = 123\n")
    with pytest.raises(OllamaConfigError) as exc:
        resolve_config(global_path=None, repo_path=r, env={})
    assert "base_url" in str(exc.value)


def test_non_string_api_key_raises_config_error(tmp_path):
    # A TOML `api_key = true` (bool) must never reach header-building
    # (`f"Bearer {api_key}"`) as a non-string.
    r = _write(tmp_path, "repo.toml", "api_key = true\n")
    with pytest.raises(OllamaConfigError) as exc:
        resolve_config(global_path=None, repo_path=r, env={})
    assert "api_key" in str(exc.value)


def test_non_string_model_raises_config_error(tmp_path):
    # A TOML `models.coder = 42` (int) must never reach the backend's `model=` payload
    # field as a non-string.
    r = _write(tmp_path, "repo.toml", "[models]\ncoder = 42\n")
    with pytest.raises(OllamaConfigError) as exc:
        resolve_config(global_path=None, repo_path=r, env={})
    assert "models.coder" in str(exc.value)
