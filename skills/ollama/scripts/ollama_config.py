# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Layered, per-key config resolver for the Ollama delegation runtime."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from errors import OllamaConfigError, ValidationError

DEFAULT_BASE_URL = "http://localhost:11434/v1"
# Default semaphore size for concurrent delegations (R21) -- default plan tier is Ollama
# Pro (3 concurrent). Shared with ollama_init.render_template so the scaffolded TOML and
# the resolver default never drift apart.
DEFAULT_MAX_PARALLEL_AGENTS = 3
# Default bound on the queue of delegations waiting for a semaphore slot (R21b) -- an
# anti-DoS backstop, not a resource limit; see spec-behavior.md for the rationale. Shared
# with ollama_init.render_template (see DEFAULT_MAX_PARALLEL_AGENTS above).
DEFAULT_MAX_QUEUED_AGENTS = 32
# Default anti-runaway output cap in UTF-8 bytes (R24c) -- must match
# backend.DEFAULT_MAX_OUTPUT_BYTES / ollama_stream.DEFAULT_MAX_OUTPUT_BYTES (MS4/MS6 Task
# 4) exactly, enforced by a three-way equality test (Task 8), so the layered default never
# drifts from the two consumers' own module defaults.
DEFAULT_MAX_OUTPUT_BYTES = 2_000_000
# MS7 Task 8: operator kill-switch default -- cross-process FS locking (R7d/R21c) is
# fully enforced unless explicitly disabled (per-project/per-env, like every other
# layered key).
DEFAULT_DISABLE_FS_LOCKS = False


def normalize_base_url(raw: str) -> str:
    """Normalize a base URL idempotently.

    Strip a trailing ``/``; prepend ``http://`` if no scheme; append ``/v1``
    **only** when the authority has no path (a value already carrying a path is
    used verbatim, never ``/v1/v1``).

    Args:
        raw: Raw base URL from config/env. Callers going through
            ``resolve_config`` are guaranteed a ``str`` here — every TOML/env
            source of ``base_url`` is type-checked via ``_require_str`` (Task 3)
            before it ever reaches this function, so this function itself does
            not re-check the type, only the empty/whitespace content below.

    Returns:
        The normalized OpenAI-compatible base URL.

    Raises:
        OllamaConfigError: if *raw* is empty/whitespace-only (or collapses to
            empty after stripping trailing slashes) — an explicit empty value is
            a misconfiguration, distinct from an *unset* base_url (which
            resolve_config's per-key precedence already falls back to
            ``DEFAULT_BASE_URL`` for, never reaching this function empty).
        OllamaConfigError: if *raw* has a scheme but no authority/host (e.g.
            ``"http://"``) — stripping trailing slashes from a bare scheme
            would otherwise collapse it to ``"http:"``, which then looks
            scheme-less and gets a bogus ``"http://"`` re-prepended
            (``"http://http:"``). An empty authority is always invalid.
    """
    stripped = raw.strip()
    if not stripped:
        raise OllamaConfigError(
            "base_url must not be empty; leave it unset to use the default "
            f"({DEFAULT_BASE_URL}) or provide a valid host/URL"
        )
    # Decide scheme-presence on the merely-stripped value, *before* trailing
    # slashes are removed — for a bare scheme like "http://" every trailing
    # char is "/", so rstrip below would erase the "://" marker itself and
    # make the value look scheme-less to a check performed afterwards.
    has_scheme = "://" in stripped
    value = stripped.rstrip("/")
    if not value:
        raise OllamaConfigError(
            "base_url must not be empty; leave it unset to use the default "
            f"({DEFAULT_BASE_URL}) or provide a valid host/URL"
        )
    if not has_scheme:
        value = "http://" + value
    parts = urlsplit(value)
    if not parts.netloc:
        raise OllamaConfigError(
            f"base_url {raw!r} has no host (empty authority after the "
            "scheme); provide a value like 'http://localhost:11434'"
        )
    if not parts.path:
        parts = parts._replace(path="/v1")
    return urlunsplit(parts)


CAPABILITIES: tuple[str, ...] = (
    "coder",
    "reviewer",
    "tester",
    "explainer",
    "vision",
    "transcribe",
    "thinking",
)
DEFAULT_MODELS: Mapping[str, str] = MappingProxyType(
    {
        "coder": "kimi-k2.7-code:cloud",
        "reviewer": "glm-5.2:cloud",
        "tester": "deepseek-v4-flash:cloud",
        "explainer": "gpt-oss:120b-cloud",
        "vision": "minimax-m3:cloud",
        "transcribe": "gemma4:cloud",
        "thinking": "deepseek-v4-pro:cloud",
    }
)
DEFAULT_STRUCTURED: Mapping[str, str] = MappingProxyType(
    {c: ("schema" if c in ("reviewer", "tester") else "off") for c in CAPABILITIES}
)
DEFAULT_STREAM: Mapping[str, bool] = MappingProxyType(
    {c: (c not in ("reviewer", "tester")) for c in CAPABILITIES}
)
_STRUCTURED_VALUES = frozenset({"schema", "object", "off"})
# R2 transcribe: the transport to use for the experimental/gated `transcribe` capability
# (see `transcribe.py`). "auto" probes the endpoint then falls back to audio-multimodal
# chat then a gated error; "endpoint"/"chat" force that transport and skip the probe.
_VALID_TRANSCRIBE_TRANSPORTS = frozenset({"auto", "endpoint", "chat"})
_DEFAULT_TRANSCRIBE_TRANSPORT = "auto"


@dataclass(frozen=True)
class OllamaAgentsConfig:
    """Resolved, immutable configuration for the delegation runtime."""

    base_url: str
    api_key: str | None
    models: Mapping[str, str]
    structured: Mapping[str, str]
    stream: Mapping[str, bool]
    max_parallel_agents: int
    max_queued_agents: int
    # R24c: trailing field WITH a default (unlike the two int fields above), so every
    # pre-existing OllamaAgentsConfig(...) construction across MS1-MS7's own fixtures
    # (which predate this field and pass only the original seven fields) keeps compiling
    # unchanged, per standard dataclass trailing-default-field rules.
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    # R2 transcribe: another trailing field WITH a default, for the same reason -- every
    # pre-existing OllamaAgentsConfig(...) construction site predating this field keeps
    # compiling unchanged.
    transcribe_transport: str = _DEFAULT_TRANSCRIBE_TRANSPORT
    disable_fs_locks: bool = False
    """Operator kill-switch (R7d/R21c escape hatch, MS7 Task 8): when True, the
    cross-process `.ollama-slots/`/`.ollama-stdout.lock` coordination (Task 2/3) is
    bypassed entirely -- no lock files are ever created -- and the runtime falls back
    to the in-process `Scheduler` semaphore + the documented single-orchestrator
    invariant (exactly the v0.1 default, pre-R7d/R21c). Use when the per-project temp
    namespace sits on a filesystem that cannot be trusted to honor atomic
    `O_CREAT|O_EXCL` (e.g. certain NFS/SMB configurations) or under persistent
    Windows AV/indexer interference on lock files."""


def _load_toml(path: str | None) -> dict[str, Any]:
    """Return parsed TOML, or ``{}`` if the path is absent/empty.

    Args:
        path: Path to a TOML file, or ``None``.

    Returns:
        The parsed TOML as a dict, or ``{}`` if *path* is falsy or missing.

    Raises:
        OllamaConfigError: if the file exists but is malformed TOML.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise OllamaConfigError(f"Malformed TOML in {path}: {exc}") from exc


def _require_str(value: Any, key: str, *, redact_value: bool = False) -> None:
    """Guard a config value that MUST be a string once present.

    Applied uniformly to every string-typed config key resolved from TOML/env
    (``base_url``, ``api_key``, every ``models.<capability>``, every
    ``structured.<capability>``) so a non-string value (e.g. ``base_url = 123``,
    ``api_key = true``, ``models.coder = 42``) is rejected here, at resolution
    time, with an actionable domain error — instead of silently flowing through
    the ``or``-chain / ``_first_present`` and later blowing up with an uncaught
    ``TypeError``/``AttributeError`` (e.g. inside ``normalize_base_url``'s
    ``raw.strip()`` or an f-string like ``f"{base_url}/chat/completions"``).

    Args:
        value: The candidate value pulled from TOML/env for *key* (``None`` if
            absent at that layer — absence is not an error here, only a wrong
            *type* is; presence-semantics for absence are handled by the
            ``or``-chain / ``_first_present`` callers).
        key: Config key name for the error message (e.g. ``"base_url"``,
            ``"models.coder"``).
        redact_value: When ``True``, the error message omits the offending
            value entirely (only the key name and the wrong type are
            reported). **Must** be ``True`` at every ``api_key`` call site
            (NR3: the key must never be logged/written to artifacts, including
            error messages) — a malformed TOML such as ``api_key =
            ["sk-real-secret"]`` must never embed the secret in the exception
            text via ``{value!r}``. Non-secret keys (``base_url``, ``models``,
            ``structured``) keep the default ``False`` for helpful diagnostics.

    Raises:
        OllamaConfigError: if *value* is present (not ``None``) and not a
            ``str``.
    """
    if value is not None and not isinstance(value, str):
        if redact_value:
            raise OllamaConfigError(f"{key} must be a string, got {type(value).__name__}")
        raise OllamaConfigError(f"{key} must be a string, got {type(value).__name__}: {value!r}")


def _resolve_int(
    env: Mapping[str, str],
    env_key: str,
    repo: dict[str, Any],
    glob: dict[str, Any],
    toml_key: str,
    default: int,
    minimum: int,
) -> int:
    """Resolve an integer config value with env > repo > global > default precedence.

    Args:
        env: Environment mapping.
        env_key: Environment variable name (e.g. ``"OLLAMA_AGENTS_MAX_PARALLEL"``).
        repo: Parsed repo TOML.
        glob: Parsed global TOML.
        toml_key: Key name within the TOML tables (e.g. ``"max_parallel_agents"``).
        default: Built-in default if absent everywhere.
        minimum: Minimum accepted value (inclusive).

    Returns:
        The resolved integer.

    Raises:
        OllamaConfigError: if a present value is not coercible to ``int`` or is
            below *minimum*. A native ``bool`` is rejected (``bool`` is a
            subclass of ``int`` in Python, but ``true``/``false`` is never a
            valid integer config value here).
    """
    for src in (env.get(env_key), repo.get(toml_key), glob.get(toml_key)):
        if src is None or src == "":
            continue
        if isinstance(src, bool):
            raise OllamaConfigError(f"{toml_key} must be an integer, got {src!r}")
        try:
            value = int(src)
        except (TypeError, ValueError) as exc:
            raise OllamaConfigError(f"{toml_key} must be an integer, got {src!r}") from exc
        if value < minimum:
            raise OllamaConfigError(f"{toml_key} must be >= {minimum}, got {value}")
        return value
    return default


def _first_present(*candidates: Any, default: Any) -> Any:
    """Return the first candidate that is present (not None) and non-empty.

    Presence-semantics for string overrides: a present-but-empty value (``""``) is
    treated as *no override at that layer* (an empty model tag / structured value is
    never a valid override), so resolution falls through to the next layer and finally
    *default*. This is deliberate and distinct from ``api_key`` (where present-but-empty
    means ``None``); see ``test_empty_string_model_override_is_not_an_override``.

    Args:
        *candidates: Values to try, in precedence order (highest first).
        default: Value to use if every candidate is absent or empty.

    Returns:
        The first present-and-non-empty candidate, or *default*.
    """
    for value in candidates:
        if value is not None and value != "":
            return value
    return default


def _coerce_bool(value: Any, ctx: str) -> bool:
    """Coerce a config value to bool WITHOUT the ``bool('false') is True`` trap.

    A native TOML bool passes through; a string is parsed case-insensitively
    (``true``/``1``/``yes`` -> True, ``false``/``0``/``no`` -> False -- the
    ``yes``/``no`` tokens added in MS7 Task 8 for ``disable_fs_locks``'s env override,
    backward-compatible with every existing ``stream.<cap>`` value); anything else is
    a config error. Never use bare ``bool(str)`` -- ``bool("false")`` is ``True``,
    which would silently invert a ``stream.<cap> = "false"`` / ``disable_fs_locks =
    "false"`` setting.

    Args:
        value: The resolved config value (native bool or string).
        ctx: Key name for the error message (e.g. ``"stream.reviewer"``,
            ``"disable_fs_locks"``).

    Returns:
        The coerced boolean.

    Raises:
        OllamaConfigError: if *value* is a non-boolean string / unsupported type.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
    raise OllamaConfigError(f"{ctx} must be a boolean (true/false), got {value!r}")


def _resolve_bool(
    env: Mapping[str, str],
    env_key: str,
    repo: dict[str, Any],
    glob: dict[str, Any],
    toml_key: str,
    default: bool,
) -> bool:
    """Resolve a single, non-per-capability boolean key: env > repo > global > default.

    Mirrors ``_resolve_int``'s exact precedence shape and presence-semantics (an
    empty-string override at any layer is treated as ABSENT, falling through to the
    next layer) but coerces through ``_coerce_bool`` instead of ``int(...)``, so a
    native TOML ``true``/``false`` and every string token ``_coerce_bool`` accepts
    both resolve, and anything else raises the same actionable ``OllamaConfigError``
    every other malformed key raises here -- never a silent fallback to *default*.

    Args:
        env: Environment mapping (injected for tests).
        env_key: The env var name (e.g. ``"OLLAMA_AGENTS_DISABLE_FS_LOCKS"``).
        repo: Parsed repo TOML.
        glob: Parsed global TOML.
        toml_key: The TOML key name, also used as ``_coerce_bool``'s error context.
        default: Returned when absent at every layer.

    Returns:
        The resolved boolean.

    Raises:
        OllamaConfigError: if a present value at any layer is not a recognized
            boolean token/type.
    """
    for src in (env.get(env_key), repo.get(toml_key), glob.get(toml_key)):
        if src is None or src == "":
            continue
        return _coerce_bool(src, toml_key)
    return default


def _resolve_transcribe_transport(
    env: Mapping[str, str], repo: dict[str, Any], global_: dict[str, Any]
) -> str:
    """Resolve ``transcribe_transport`` (R2 transcribe): env > repo > global > default.

    Args:
        env: The environment mapping (``OLLAMA_AGENTS_TRANSCRIBE_TRANSPORT``).
        repo: The parsed repo TOML (top-level ``transcribe_transport`` key).
        global_: The parsed global TOML (same key).

    Returns:
        One of ``"auto"``, ``"endpoint"``, ``"chat"``.

    Raises:
        ValidationError: the resolved value is not one of the three valid strings.
    """
    value = (
        env.get("OLLAMA_AGENTS_TRANSCRIBE_TRANSPORT")
        or repo.get("transcribe_transport")
        or global_.get("transcribe_transport")
        or _DEFAULT_TRANSCRIBE_TRANSPORT
    )
    if value not in _VALID_TRANSCRIBE_TRANSPORTS:
        raise ValidationError(
            f"invalid transcribe_transport: {value!r} "
            f"(must be one of {sorted(_VALID_TRANSCRIBE_TRANSPORTS)})"
        )
    return str(value)


def resolve_config(
    *, global_path: str | None, repo_path: str | None, env: Mapping[str, str]
) -> OllamaAgentsConfig:
    """Resolve the layered config with per-key precedence.

    Precedence (per key, highest first): env ``OLLAMA_AGENTS_*`` > repo TOML >
    global TOML > a generic fallback env var (``OLLAMA_HOST``/``OLLAMA_API_KEY``,
    ``base_url``/``api_key`` only) > built-in default.

    Args:
        global_path: Path to ``~/.claude/ollama-agents.toml`` (or ``None``).
        repo_path: Path to ``./.claude/ollama-agents.toml`` (or ``None``).
        env: Environment mapping (injected for tests).

    Returns:
        An immutable :class:`OllamaAgentsConfig`.

    Raises:
        OllamaConfigError: on malformed TOML, on an invalid value (bad int/bool/
            enum), or on a **non-string value for a string-typed key**
            (``base_url``, ``api_key``, any ``models.<cap>``, any
            ``structured.<cap>``) -- every such key is guarded via
            ``_require_str`` so a stray ``base_url = 123`` / ``api_key = true`` /
            ``models.coder = 42`` in TOML never reaches later string operations
            (``normalize_base_url``, header building, the backend payload)
            un-typed. For ``base_url`` only the *winning* layer (per the
            precedence above) is type-checked -- a malformed value in a
            shadowed/losing layer never breaks resolution. ``api_key`` errors
            never embed the offending value (NR3: redacted in every error
            message, not just non-error paths).
    """
    glob = _load_toml(global_path)
    repo = _load_toml(repo_path)

    env_host = env.get("OLLAMA_AGENTS_HOST")
    repo_base = repo.get("base_url")
    glob_base = glob.get("base_url")
    generic_host = env.get("OLLAMA_HOST")
    # Select the WINNING layer first (same truthiness-based precedence as the
    # `or`-chain this replaces: env host > repo > global > generic host >
    # default; a falsy/absent candidate at any layer is skipped, matching
    # presence-semantics for base_url), THEN validate + normalize only that
    # winner. A malformed value in a losing/shadowed layer (e.g. a stray
    # `base_url = 123` in the global TOML that a valid repo `base_url` should
    # shadow) is never inspected and never breaks resolution.
    raw_base = None
    for candidate, label in (
        (env_host, "OLLAMA_AGENTS_HOST"),
        (repo_base, "base_url (repo)"),
        (glob_base, "base_url (global)"),
        (generic_host, "OLLAMA_HOST"),
    ):
        if candidate:
            _require_str(candidate, label)
            raw_base = candidate
            break
    if raw_base is None:
        raw_base = DEFAULT_BASE_URL
    base_url = normalize_base_url(raw_base)

    if "OLLAMA_AGENTS_API_KEY" in env:
        _require_str(env["OLLAMA_AGENTS_API_KEY"], "OLLAMA_AGENTS_API_KEY", redact_value=True)
        api_key = env["OLLAMA_AGENTS_API_KEY"] or None
    elif "api_key" in repo:
        _require_str(repo["api_key"], "api_key (repo)", redact_value=True)
        api_key = repo["api_key"] or None
    elif "api_key" in glob:
        _require_str(glob["api_key"], "api_key (global)", redact_value=True)
        api_key = glob["api_key"] or None
    elif "OLLAMA_API_KEY" in env:
        _require_str(env["OLLAMA_API_KEY"], "OLLAMA_API_KEY", redact_value=True)
        api_key = env["OLLAMA_API_KEY"] or None
    else:
        api_key = None

    repo_models, glob_models = repo.get("models", {}), glob.get("models", {})
    repo_struct, glob_struct = repo.get("structured", {}), glob.get("structured", {})
    repo_stream, glob_stream = repo.get("stream", {}), glob.get("stream", {})
    models: dict[str, str] = {}
    structured: dict[str, str] = {}
    stream: dict[str, bool] = {}
    for cap in CAPABILITIES:
        up = cap.upper()
        # Presence-aware resolution (see _first_present): a present-but-empty override
        # falls through -- it never sets an empty model tag / structured value.
        models[cap] = _first_present(
            env.get(f"OLLAMA_AGENTS_MODEL_{up}"),
            repo_models.get(cap),
            glob_models.get(cap),
            default=DEFAULT_MODELS[cap],
        )
        _require_str(models[cap], f"models.{cap}")
        s = _first_present(
            env.get(f"OLLAMA_AGENTS_STRUCTURED_{up}"),
            repo_struct.get(cap),
            glob_struct.get(cap),
            default=DEFAULT_STRUCTURED[cap],
        )
        _require_str(s, f"structured.{cap}")
        if s not in _STRUCTURED_VALUES:
            raise OllamaConfigError(f"structured.{cap} must be schema|object|off, got {s!r}")
        structured[cap] = s
        # Boolean coercion via _coerce_bool at every layer (env AND TOML): a native
        # bool passes through, a string is parsed true/false, invalid -> ValidationError.
        # NEVER bare bool(str) -- bool("false") is True (the flagged coercion bug).
        env_stream = env.get(f"OLLAMA_AGENTS_STREAM_{up}")
        if env_stream is not None and env_stream != "":
            stream[cap] = _coerce_bool(env_stream, f"stream.{cap}")
        elif cap in repo_stream:
            stream[cap] = _coerce_bool(repo_stream[cap], f"stream.{cap}")
        elif cap in glob_stream:
            stream[cap] = _coerce_bool(glob_stream[cap], f"stream.{cap}")
        else:
            stream[cap] = DEFAULT_STREAM[cap]

    return OllamaAgentsConfig(
        base_url=base_url,
        api_key=api_key,
        models=MappingProxyType(models),
        structured=MappingProxyType(structured),
        stream=MappingProxyType(stream),
        max_parallel_agents=_resolve_int(
            env,
            "OLLAMA_AGENTS_MAX_PARALLEL",
            repo,
            glob,
            "max_parallel_agents",
            DEFAULT_MAX_PARALLEL_AGENTS,
            1,
        ),
        max_queued_agents=_resolve_int(
            env,
            "OLLAMA_AGENTS_MAX_QUEUED",
            repo,
            glob,
            "max_queued_agents",
            DEFAULT_MAX_QUEUED_AGENTS,
            0,
        ),
        max_output_bytes=_resolve_int(
            env,
            "OLLAMA_AGENTS_MAX_OUTPUT_BYTES",
            repo,
            glob,
            "max_output_bytes",
            DEFAULT_MAX_OUTPUT_BYTES,
            1,
        ),
        transcribe_transport=_resolve_transcribe_transport(env, repo, glob),
        disable_fs_locks=_resolve_bool(
            env,
            "OLLAMA_AGENTS_DISABLE_FS_LOCKS",
            repo,
            glob,
            "disable_fs_locks",
            DEFAULT_DISABLE_FS_LOCKS,
        ),
    )
