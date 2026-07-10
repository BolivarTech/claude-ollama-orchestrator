# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Config resolver + base_url normalization (idempotent, per-key precedence)."""

import string
import textwrap

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from errors import OllamaConfigError, ValidationError
from ollama_config import (
    CAPABILITIES,
    DEFAULT_BASE_URL,
    DEFAULT_DISABLE_FS_LOCKS,
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
    assert cfg.models["coder"] == DEFAULT_MODELS["coder"]
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
    assert cfg.models["coder"] == DEFAULT_MODELS["coder"]


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


def test_non_string_api_key_does_not_leak_secret_value(tmp_path):
    # Finding 1 (SECURITY): a malformed api_key (array instead of scalar) must raise
    # a REDACTED error -- the real secret must never appear in the exception message
    # (NR3: api_key redacted in every error, not just the happy path).
    r = _write(tmp_path, "repo.toml", 'api_key = ["sk-secret-xyz"]\n')
    with pytest.raises(OllamaConfigError) as exc:
        resolve_config(global_path=None, repo_path=r, env={})
    assert "sk-secret-xyz" not in str(exc.value)
    assert "api_key" in str(exc.value)


def test_shadowed_invalid_base_url_in_losing_layer_is_ignored(tmp_path):
    # Finding 2 (LOGIC): a malformed base_url in a SHADOWED layer (global, here) must
    # never break resolution when a valid higher-precedence layer (repo) wins -- only
    # the winning layer is type-checked.
    g = _write(tmp_path, "global.toml", "base_url = 123\n")
    r = _write(tmp_path, "repo.toml", 'base_url = "http://repo-valid:9999"\n')
    cfg = resolve_config(global_path=g, repo_path=r, env={})
    assert cfg.base_url == "http://repo-valid:9999/v1"


def test_structured_invalid_value_raises(tmp_path):
    r = _write(tmp_path, "repo.toml", '[structured]\ncoder = "maybe"\n')
    with pytest.raises(OllamaConfigError):
        resolve_config(global_path=None, repo_path=r, env={})


def test_generic_ollama_api_key_env_is_fallback():
    # R6: with no OLLAMA_AGENTS_API_KEY / repo / global api_key, the generic
    # OLLAMA_API_KEY env var wins as the last-resort fallback.
    cfg = resolve_config(global_path=None, repo_path=None, env={"OLLAMA_API_KEY": "sk-generic"})
    assert cfg.api_key == "sk-generic"


def test_invalid_max_queued_raises():
    with pytest.raises(OllamaConfigError):
        resolve_config(global_path=None, repo_path=None, env={"OLLAMA_AGENTS_MAX_QUEUED": "-1"})


def test_max_queued_bool_raises(tmp_path):
    r = _write(tmp_path, "repo.toml", "max_queued_agents = true\n")
    with pytest.raises(OllamaConfigError):
        resolve_config(global_path=None, repo_path=r, env={})


def test_max_output_bytes_defaults_to_2_000_000_matching_both_backends():
    # Three-way equality: the config default must match BOTH consumers' own module
    # defaults (backend.py, ollama_stream.py, MS4/Task 4) -- never a fourth, drifted copy.
    from backend import DEFAULT_MAX_OUTPUT_BYTES as BACKEND_DEFAULT
    from ollama_stream import DEFAULT_MAX_OUTPUT_BYTES as STREAM_DEFAULT

    cfg = resolve_config(global_path=None, repo_path=None, env={})
    assert cfg.max_output_bytes == BACKEND_DEFAULT == STREAM_DEFAULT == 2_000_000


def test_max_output_bytes_env_overrides_repo_and_global(tmp_path):
    r = _write(tmp_path, "repo.toml", "max_output_bytes = 500000\n")
    cfg = resolve_config(
        global_path=None,
        repo_path=r,
        env={"OLLAMA_AGENTS_MAX_OUTPUT_BYTES": "750000"},
    )
    assert cfg.max_output_bytes == 750_000  # env wins over repo


def test_max_output_bytes_repo_overrides_global(tmp_path):
    r = _write(tmp_path, "repo.toml", "max_output_bytes = 300000\n")
    g = _write(tmp_path, "global.toml", "max_output_bytes = 900000\n")
    cfg = resolve_config(global_path=g, repo_path=r, env={})
    assert cfg.max_output_bytes == 300_000


def test_max_output_bytes_rejects_zero_and_negative():
    with pytest.raises(OllamaConfigError):
        resolve_config(
            global_path=None, repo_path=None, env={"OLLAMA_AGENTS_MAX_OUTPUT_BYTES": "0"}
        )
    with pytest.raises(OllamaConfigError):
        resolve_config(
            global_path=None, repo_path=None, env={"OLLAMA_AGENTS_MAX_OUTPUT_BYTES": "-1"}
        )


def test_max_output_bytes_rejects_non_integer_string():
    with pytest.raises(OllamaConfigError):
        resolve_config(
            global_path=None,
            repo_path=None,
            env={"OLLAMA_AGENTS_MAX_OUTPUT_BYTES": "not-a-number"},
        )


def test_max_output_bytes_rejects_bool_from_toml(tmp_path):
    # bool IS-A int in Python -- int(True) == 1 succeeds silently without an explicit
    # isinstance guard, coercing a config typo (`max_output_bytes = true`) into a
    # valid-looking-but-wrong 1-byte cap instead of a loud, actionable config error.
    r = _write(tmp_path, "repo.toml", "max_output_bytes = true\n")
    with pytest.raises(OllamaConfigError):
        resolve_config(global_path=None, repo_path=r, env={})


def test_max_parallel_agents_also_rejects_bool_now(tmp_path):
    # The _resolve_int bool-guard is a SHARED fix -- proves it also closes the same
    # latent gap for a PRE-EXISTING key, not just the new one.
    r = _write(tmp_path, "repo.toml", "max_parallel_agents = false\n")
    with pytest.raises(OllamaConfigError):
        resolve_config(global_path=None, repo_path=r, env={})


@pytest.mark.parametrize("value", ["auto", "endpoint", "chat"])
def test_transcribe_transport_accepts_each_valid_value(value):
    cfg = resolve_config(
        global_path=None, repo_path=None, env={"OLLAMA_AGENTS_TRANSCRIBE_TRANSPORT": value}
    )
    assert cfg.transcribe_transport == value


def test_transcribe_transport_defaults_to_auto_when_unset():
    cfg = resolve_config(global_path=None, repo_path=None, env={})
    assert cfg.transcribe_transport == "auto"


def test_transcribe_transport_invalid_value_raises_validation_error():
    with pytest.raises(ValidationError):
        resolve_config(
            global_path=None,
            repo_path=None,
            env={"OLLAMA_AGENTS_TRANSCRIBE_TRANSPORT": "carrier-pigeon"},
        )


def test_transcribe_transport_env_overrides_repo_toml(tmp_path):
    repo = tmp_path / "ollama-agents.toml"
    repo.write_text('transcribe_transport = "chat"\n', encoding="utf-8")
    cfg = resolve_config(
        global_path=None,
        repo_path=str(repo),
        env={"OLLAMA_AGENTS_TRANSCRIBE_TRANSPORT": "endpoint"},
    )
    assert cfg.transcribe_transport == "endpoint"  # env wins over repo (R6 precedence)


# --- MS7 Task 8: disable_fs_locks kill-switch (R7d/R21c operator escape hatch) ---


def test_disable_fs_locks_defaults_to_false():
    cfg = resolve_config(global_path=None, repo_path=None, env={})
    assert cfg.disable_fs_locks is DEFAULT_DISABLE_FS_LOCKS is False


@pytest.mark.parametrize("token", ["1", "true", "TRUE", "True", "yes", "YES"])
def test_disable_fs_locks_env_override_accepts_truthy_tokens(token):
    cfg = resolve_config(
        global_path=None, repo_path=None, env={"OLLAMA_AGENTS_DISABLE_FS_LOCKS": token}
    )
    assert cfg.disable_fs_locks is True


@pytest.mark.parametrize("token", ["0", "false", "FALSE", "no", "NO"])
def test_disable_fs_locks_env_override_accepts_falsy_tokens(token):
    cfg = resolve_config(
        global_path=None, repo_path=None, env={"OLLAMA_AGENTS_DISABLE_FS_LOCKS": token}
    )
    assert cfg.disable_fs_locks is False


def test_disable_fs_locks_repo_overrides_global_and_env_overrides_repo(tmp_path):
    global_path = tmp_path / "global.toml"
    repo_path = tmp_path / "repo.toml"
    global_path.write_text("disable_fs_locks = true\n", encoding="utf-8")
    repo_path.write_text("disable_fs_locks = false\n", encoding="utf-8")
    cfg = resolve_config(global_path=str(global_path), repo_path=str(repo_path), env={})
    assert cfg.disable_fs_locks is False  # repo wins over global
    cfg = resolve_config(
        global_path=str(global_path),
        repo_path=str(repo_path),
        env={"OLLAMA_AGENTS_DISABLE_FS_LOCKS": "true"},
    )
    assert cfg.disable_fs_locks is True  # env wins over both


def test_disable_fs_locks_invalid_env_token_raises_validation_error():
    with pytest.raises(ValidationError):
        resolve_config(
            global_path=None, repo_path=None, env={"OLLAMA_AGENTS_DISABLE_FS_LOCKS": "maybe"}
        )


def test_disable_fs_locks_non_bool_toml_type_raises_validation_error(tmp_path):
    repo_path = tmp_path / "repo.toml"
    repo_path.write_text("disable_fs_locks = 2\n", encoding="utf-8")  # int, not bool/str token
    with pytest.raises(ValidationError):
        resolve_config(global_path=None, repo_path=str(repo_path), env={})


def test_stream_cap_still_accepts_its_original_true_false_tokens_after_the_yes_no_extension():
    """Regression: widening `_coerce_bool`'s accepted tokens (for disable_fs_locks) must not
    change behavior for its existing caller, `stream.<cap>` -- the original true/1/false/0
    tokens keep parsing exactly as MS1 defined them."""
    cfg = resolve_config(
        global_path=None, repo_path=None, env={"OLLAMA_AGENTS_STREAM_CODER": "false"}
    )
    assert cfg.stream["coder"] is False
