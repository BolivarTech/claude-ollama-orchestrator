# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""CLI orchestrator for a single Ollama delegation (transactional core)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from typing import Any

from agent_schema import DISCRIMINATOR_KEYS, SCHEMAS
from backend import AgentBackend, DelegationResult, OllamaBackend
from errors import (
    DelegationError,
    InvalidInputError,
    OllamaBackendError,
    OllamaConfigError,
    OllamaPreflightError,
    ValidationError,
)
from ollama_config import CAPABILITIES, OllamaAgentsConfig

# Explicit re-export (mypy strict, no_implicit_reexport): tests reference
# run_ollama.resolve_config directly as a monkeypatch/config seam.
from ollama_config import resolve_config as resolve_config
from ollama_preflight import preflight
from parse_output import parse_agent_output
from token_stats import TokenStats
from validate import MAX_INPUT_FILE_SIZE, validate_output

MAX_HISTORY_RUNS = 5

_RETRY_FEEDBACK = "\n\n---RETRY-FEEDBACK---\nYour previous output failed: {error}\n{schema}\n"
# agents/ live one level up from scripts/ (skills/ollama/agents/).
_AGENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agents")


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse surface for a single delegation.

    Returns:
        The configured parser (positional capability with the 7 choices, input,
        and the R28 flags).
    """
    parser = argparse.ArgumentParser(description="Claude-Ollama-Orchestrator delegation")
    parser.add_argument("capability", choices=CAPABILITIES, help="Delegation capability")
    parser.add_argument("input", help="Path to file or inline text to delegate")
    parser.add_argument("--model", default=None, help="Override the resolved model")
    parser.add_argument("--timeout", type=int, default=900, help="Per-delegation timeout (s)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Caller-owned dir for the raw output artifact (else stdout only)",
    )
    # --keep-runs is validated here (0 rejected, -1 disables); the run-dir cleanup it
    # controls is implemented in MS3 (temp namespace + LRU prune) — a forward reference,
    # not a dead flag. Kept in MS1's surface because R28 specifies the full CLI here.
    parser.add_argument(
        "--keep-runs",
        type=int,
        default=MAX_HISTORY_RUNS,
        help="Max non-live temp run dirs to retain (-1 disables; cleanup in MS3)",
    )
    # --no-status (status display) and --max-parallel (concurrency) are part of R28's CLI
    # surface in MS1, but their behavior is wired in MS3 and MS5 respectively — documented
    # forward references, not silent no-ops.
    parser.add_argument(
        "--no-status",
        dest="show_status",
        action="store_false",
        default=True,
        help="Disable the live status display (wired in MS3)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Override max_parallel_agents (concurrency wired in MS5)",
    )
    parser.add_argument(
        "--warn-input-tokens", type=int, default=150_000, help="Oversize warning threshold (tokens)"
    )
    # --ollama-init is intentionally NOT an argparse flag: it is handled solely by the
    # pre-parse first-token short-circuit in `main` (before argparse ever runs), so there
    # is exactly one handling path for it — never a dual/ambiguous one.
    return parser


def _validate_args(parser: argparse.ArgumentParser, ns: argparse.Namespace) -> None:
    """Reject cross-flag-invalid parsed args via ``parser.error`` (R28 CLI surface)."""
    if ns.keep_runs == 0:
        parser.error("--keep-runs 0 is ambiguous; use -1 to disable cleanup or >= 1")
    if ns.warn_input_tokens <= 0:
        parser.error("--warn-input-tokens must be a positive integer")
    if ns.timeout <= 0:  # 0/negative breaks the socket timeout + R25 deadline math.
        parser.error("--timeout must be a positive integer")


def load_system_prompt(capability: str) -> str:
    """Return the system prompt for *capability* from ``agents/ollama-<cap>.md``.

    Args:
        capability: One of the seven capabilities.

    Returns:
        The prompt file's text.

    Raises:
        OllamaConfigError: if the prompt file is missing/unreadable.
    """
    path = os.path.join(_AGENTS_DIR, f"ollama-{capability}.md")
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError as exc:
        raise OllamaConfigError(f"Missing system prompt for {capability}: {path}") from exc


_PATH_SEPARATORS: tuple[str, ...] = (os.sep,) + ((os.altsep,) if os.altsep else ())


def _looks_like_path(arg: str) -> bool:
    """Heuristically decide whether *arg* has the SHAPE of a filesystem path.

    Used only by :func:`_load_input` to distinguish "a likely typo'd path" from
    "literal prompt text" — never to resolve or validate an actual path. *arg* is
    considered path-shaped when it contains **no whitespace** (free-form prose
    virtually always contains a space; a bare filename/path virtually never does) AND
    at least one of: it is absolute, it contains a path separator, or it ends in a
    recognized file extension (a trailing ``.<alnum>+`` suffix).

    Args:
        arg: The raw CLI input positional.

    Returns:
        True if *arg* looks path-shaped (regardless of whether it actually exists).
    """
    if any(ch.isspace() for ch in arg):
        return False
    if os.path.isabs(arg) or any(sep in arg for sep in _PATH_SEPARATORS):
        return True
    _, ext = os.path.splitext(arg)
    return bool(ext) and ext[1:].isalnum()


def _load_input(arg: str) -> str:
    """Return delegation input: file contents if *arg* is an existing path, else *arg*.

    Explicit disambiguation rule (so a mistyped path never silently becomes literal
    prompt text): if *arg* **exists** as a file, its contents are read (subject to the
    ``MAX_INPUT_FILE_SIZE`` guard, R23 — enforced TOCTOU-safely on the bytes actually
    read, see below) and returned. If *arg* does **not** exist but *looks* path-shaped
    per :func:`_looks_like_path` (a path separator, an absolute path, or a recognized
    file extension, with no whitespace), an actionable WARNING is printed to stderr and
    *arg* is still returned as literal text — warn-and-proceed, consistent with MS1's
    other ambiguous-but-not-fatal cases (e.g. preflight's 404/501 warn-and-proceed). A
    plain text string with no path-like shape (the common case — free-form prose) is
    treated as literal text silently, with no warning.

    Args:
        arg: The raw CLI input positional (a file path or inline text).

    Returns:
        The file's decoded contents, or *arg* itself as literal text.

    Raises:
        ValidationError: if the referenced (existing) file's content exceeds
            ``MAX_INPUT_FILE_SIZE``, checked on the bytes ACTUALLY read (see below),
            never a separate pre-read stat.
    """
    if os.path.isfile(arg):
        # TOCTOU-safe size guard (R23; mirrors MS6's `load_binary` bounded-read
        # pattern): a getsize()-then-open()-then-read() sequence has a race window
        # where the file could grow between the stat and the read, bypassing the cap.
        # There is no separate stat call at all here — read AT MOST
        # `MAX_INPUT_FILE_SIZE + 1` bytes and gate on the length of what was ACTUALLY
        # read (post-read, authoritative), so a concurrently growing file has nothing
        # to race against.
        with open(arg, "rb") as fh:
            raw = fh.read(MAX_INPUT_FILE_SIZE + 1)
        if len(raw) > MAX_INPUT_FILE_SIZE:
            raise ValidationError(f"input file exceeds {MAX_INPUT_FILE_SIZE} bytes: {arg}")
        return raw.decode("utf-8", errors="replace")
    if _looks_like_path(arg):
        print(
            f"WARNING: input {arg!r} looks like a file path but does not exist; "
            "treating it as literal text. If a file was intended, check the path for "
            "typos.",
            file=sys.stderr,
        )
    return arg


def _response_format_for(capability: str, mode: str) -> dict[str, Any] | None:
    """Return the ``response_format`` for *capability* given its structured *mode* (R29).

    *mode* is the resolved per-capability ``[structured]`` setting: ``"schema"`` sends the
    strict per-capability JSON-Schema (only if one exists), ``"object"`` sends a generic
    ``{"type": "json_object"}`` envelope, ``"off"`` sends no ``response_format``.

    Args:
        capability: The capability name.
        mode: The resolved structured mode (``schema`` | ``object`` | ``off``).

    Returns:
        The ``response_format`` dict, or ``None``.
    """
    if mode == "schema":
        schema = SCHEMAS.get(capability)
        if schema is None:
            # Fail LOUD, never degrade silently to json_object/None: a config that asks
            # for strict schema on a capability that has none is a misconfiguration.
            raise ValidationError(
                f"structured={mode!r} for capability {capability!r} but no JSON-Schema is "
                f"defined in agent_schema.SCHEMAS; fix the [structured] config (use "
                f"'object'/'off') or add a schema for {capability!r}."
            )
        return {
            "type": "json_schema",
            "json_schema": {"name": capability, "schema": schema, "strict": True},
        }
    if mode == "object":
        return {"type": "json_object"}
    return None


# Capabilities whose transport is not in MS1 (multimodal/audio → M7). Guarded in
# ``dispatch`` so a binary input is never sent as garbled chat text. Removed in M7.
_MS1_UNSUPPORTED_CAPS = frozenset({"vision", "transcribe"})


def dispatch(
    capability: str,
    prompt: str,
    *,
    backend: AgentBackend,
    model: str,
    timeout: int,
    system_prompt: str,
    config: OllamaAgentsConfig,
    stats: TokenStats | None = None,
) -> DelegationResult:
    """Run one delegation; for the ``schema`` mode parse+validate with one retry.

    The per-capability ``[structured]`` mode from *config* drives the request (R29):
    ``"schema"`` sends the strict JSON-Schema ``response_format`` and parses+validates the
    output (retrying ONCE with a corrective feedback block while a monotonic wall-clock
    deadline (R25) has not elapsed), returning the ``DelegationResult`` with the validated
    dict in ``.parsed``; ``"object"``/``"off"`` send a generic/absent ``response_format``
    and return the ``DelegationResult`` verbatim (``.parsed`` is ``None``, ``.content`` is
    the raw text for Claude to review).

    When *stats* is given, every backend call that COMPLETES (returns a
    ``DelegationResult``) is recorded (``http_calls``) and the logical delegation is
    counted once (``delegations`` via ``counts_as_delegation`` False on the retry). A call
    that raises (connection error/timeout/5xx) propagates the exception unchanged and is
    never recorded — ``http_calls`` reflects completed attempts, not raw connection
    attempts (see ``token_stats.TokenStats``).

    Args:
        capability: The capability name.
        prompt: The user prompt, passed as-is in MS1 (anti-injection sanitization is
            added in M6 — this is a forward reference, MS1 does not sanitize).
        backend: An ``AgentBackend`` (injected).
        model: Resolved model tag.
        timeout: Per-delegation timeout.
        system_prompt: The capability's system prompt.
        config: The resolved config (its ``structured`` mapping drives the mode).
        stats: Optional local token accumulator (R12); every completed backend call is
            recorded into it.

    Returns:
        The ``DelegationResult`` — with the validated dict in ``.parsed`` for schema mode,
        or ``.parsed is None`` and the raw text in ``.content`` for object/off mode.

    Raises:
        OllamaBackendError: if the composite retry deadline is exceeded.
        ValidationError: if the output is still invalid after the retry.
        DelegationError: if *capability* is ``vision``/``transcribe`` (their
            multimodal/audio transport lands in M7 — sending a binary as chat text
            would garble it, so MS1 fails actionably instead).
    """
    # R28 keeps all 7 capabilities in the CLI surface, but MS1 implements only the
    # chat/text ones. vision/transcribe need the multimodal/binary transport added in
    # M7 — guard so a binary input isn't silently garbled. (Removed/replaced in M7.)
    if capability in _MS1_UNSUPPORTED_CAPS:
        raise DelegationError(
            f"capability {capability!r} requires the multimodal/audio transport added "
            "in M7 and is not available in this build"
        )
    mode = config.structured.get(capability, "off")
    response_format = _response_format_for(capability, mode)
    keys = DISCRIMINATOR_KEYS.get(capability)
    # ONE monotonic deadline (R25) bounds the WHOLE delegation and is threaded into every
    # backend.run call, so the backend's 429-backoff loop shares this budget with the
    # parse-retry loop below — neither can independently exceed the delegation's time.
    # ``--timeout`` is the HARD end-to-end budget (parse-retry + 429 backoff included); no
    # hidden slack extends it beyond what the user asked for.
    deadline = time.monotonic() + timeout
    # Parse+validate only when we asked for the strict schema AND the capability has one;
    # "object"/"off" (and any free-text capability) return the content verbatim.
    if mode != "schema" or keys is None:
        # Free-text/object (and schema-without-keys) return the DelegationResult verbatim:
        # `.content` is what Claude reviews, `.parsed` stays None. Record the completed call.
        result = backend.run(
            capability,
            system_prompt,
            prompt,
            model,
            timeout,
            response_format=response_format,
            deadline=deadline,
        )
        if stats is not None:
            stats.record(capability, model, result)
        return result

    attempt_prompt = prompt
    last_error = ""
    for attempt in range(2):  # one retry (R25), sharing the single deadline above.
        if time.monotonic() >= deadline:
            raise OllamaBackendError(
                f"{capability} delegation exceeded its retry deadline ({last_error})"
            )
        result = backend.run(
            capability,
            system_prompt,
            attempt_prompt,
            model,
            timeout,
            response_format=response_format,
            deadline=deadline,
        )
        # Bill EVERY completed backend call (http_calls); the retry does not count as a
        # second logical delegation (counts_as_delegation False on attempt 1).
        if stats is not None:
            stats.record(capability, model, result, counts_as_delegation=(attempt == 0))
        try:
            parsed = validate_output(capability, parse_agent_output(result.content, keys))
            return replace(result, parsed=parsed)
        except (ValidationError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            # Reinject the ACTUAL per-capability JSON-Schema (not just the key list) so the
            # model gets the precise contract on retry — maximizes corrective effectiveness.
            attempt_prompt = prompt + _RETRY_FEEDBACK.format(
                error=last_error,
                schema="Return JSON conforming to this schema:\n" + json.dumps(SCHEMAS[capability]),
            )
    raise ValidationError(f"{capability} output invalid after retry: {last_error}")


def _make_backend(cfg: OllamaAgentsConfig) -> OllamaBackend:
    """Construct the transactional backend (seam for tests)."""
    return OllamaBackend(cfg)


def _global_toml() -> str:
    """Return the path to the global config (``~/.claude/ollama-agents.toml``)."""
    return os.path.join(os.path.expanduser("~"), ".claude", "ollama-agents.toml")


def _repo_toml() -> str:
    """Return the path to the repo config (``./.claude/ollama-agents.toml``)."""
    return os.path.join(os.getcwd(), ".claude", "ollama-agents.toml")


def run_delegation(ns: argparse.Namespace) -> int:
    """Resolve config, preflight, load the prompt, delegate once, print the output.

    Aborts (non-zero, informing) if preflight fails — never silently falls back to Claude
    generation (R14). The delegated output is untrusted data for Claude to review; it is
    printed, never auto-applied.
    """
    try:
        cfg = resolve_config(global_path=_global_toml(), repo_path=_repo_toml(), env=os.environ)
        # R10/R28: resolve the EFFECTIVE model (the --model override when given, else
        # the capability's configured model) BEFORE preflight, and thread it in — so a
        # bad --model override aborts here (actionable), never a later chat-time 404.
        model = ns.model or cfg.models[ns.capability]
        preflight(cfg, capability=ns.capability, effective_model=model)
    except (OllamaConfigError, OllamaPreflightError) as exc:
        print(
            f"Ollama unavailable: {exc}\nNot delegating; resolve the issue and retry "
            "(or generate with Claude explicitly).",
            file=sys.stderr,
        )
        return 2
    system_prompt = load_system_prompt(ns.capability)
    prompt = _load_input(ns.input)
    stats = TokenStats()
    result = dispatch(
        ns.capability,
        prompt,
        backend=_make_backend(cfg),
        model=model,
        timeout=ns.timeout,
        system_prompt=system_prompt,
        config=cfg,
        stats=stats,
    )
    # The reviewable output is the validated structured dict (`.parsed`) when present, else
    # the raw text content — both untrusted data for Claude to review, never auto-applied.
    review = result.parsed if result.parsed is not None else result.content
    rendered = review if isinstance(review, str) else json.dumps(review, indent=2)
    # --output-dir (R28): when given, persist the raw output to a caller-owned dir. No
    # lock/prune here — the temp-managed run-dir lifecycle and the --keep-runs cleanup it
    # controls land in MS3 (documented forward reference). Omitted → stdout only.
    if ns.output_dir is not None:
        os.makedirs(ns.output_dir, exist_ok=True)
        with open(
            os.path.join(ns.output_dir, f"{ns.capability}.raw.json"), "w", encoding="utf-8"
        ) as fh:
            fh.write(rendered)
    # Local token accounting (R12), separate from Claude/Anthropic usage. Best-effort:
    # write into --output-dir when given, else cwd (MS3 replaces this default with the
    # managed temp run dir).
    stats.write(ns.output_dir if ns.output_dir is not None else os.getcwd())
    print(rendered)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point. ``--ollama-init`` short-circuits before positionals are parsed.

    The ``--ollama-init`` short-circuit itself catches TWO distinct failure modes from
    ``write_template``, each actionable and non-zero, never a raw traceback:
    ``FileExistsError`` (refuse-if-exists, R13) and a plain ``OSError`` (disk full,
    permission denied, read-only target dir) — the latter caught SEPARATELY and after
    the former, since ``FileExistsError`` is itself an ``OSError`` subclass.

    The full delegation pipeline (`run_delegation`) is wrapped in a single handler that
    catches the COMPLETE domain-error family — ``ValidationError`` (incl. its
    ``OllamaConfigError``/``OllamaPreflightError`` subclasses), ``OllamaBackendError``,
    ``DelegationError``, ``InvalidInputError`` (sibling of ``ValidationError``), and
    ``OSError`` (incl. ``TimeoutError``, a stdlib ``OSError`` subclass, and an unreadable
    input file from ``_load_input``) — and turns each into an actionable, already-redacted
    (the domain exceptions redact ``api_key`` themselves, R9/NR3) stderr message plus a
    non-zero exit code. **Never a raw traceback for a domain/OS failure.**
    ``KeyboardInterrupt``/``SystemExit`` are intentionally NOT caught here — they propagate
    unchanged so the R27 interrupt-cleanup path (landing in MS3) stays intact.

    Args:
        argv: CLI args (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success; non-zero on any caught domain/OS failure).
    """
    args = sys.argv[1:] if argv is None else argv
    # Detect the flag ONLY as the first token (the documented invocation
    # `run_ollama.py --ollama-init`), so input text that merely contains the
    # literal "--ollama-init" never triggers the scaffold.
    if args and args[0] == "--ollama-init":
        from ollama_init import write_template

        try:
            path = write_template()
        except FileExistsError as exc:
            # A refuse-if-exists is a FAILED scaffold, not silent success → non-zero.
            print(
                f"Refusing to overwrite existing config: {exc} (remove it first to re-scaffold)",
                file=sys.stderr,
            )
            return 1
        except OSError as exc:
            # write_template can ALSO fail for reasons beyond refuse-if-exists — disk
            # full, permission denied, a read-only target directory. FileExistsError IS
            # an OSError subclass, so this generic arm must come SECOND (Python tries
            # except clauses in order) or it would swallow the refuse-if-exists case
            # above. Report actionably and exit non-zero — never a raw traceback for an
            # OS-level scaffold failure.
            print(f"Failed to write config template: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote config template to {path}")
        return 0

    parser = build_parser()
    ns = parser.parse_args(args)
    _validate_args(parser, ns)
    # Comprehensive domain-exception handling (R14/R23/NR8): the delegation pipeline can
    # raise from several distinct call sites — `preflight`/`resolve_config`
    # (OllamaConfigError/OllamaPreflightError, both ValidationError), `load_system_prompt`
    # (OllamaConfigError, e.g. a missing `agents/ollama-<cap>.md`), `_load_input` (a
    # ValidationError on oversize, or a bare OSError if an existing path becomes
    # unreadable/vanishes between the `isfile` check and `open`), `dispatch`
    # (DelegationError for the MS1 vision/transcribe transport guard; ValidationError on a
    # persistently-invalid structured output), and `backend.run` (OllamaBackendError,
    # TimeoutError). Catching the FULL family in one place means every one of those
    # failure modes prints an actionable, already-redacted message (the domain exceptions
    # redact `api_key` themselves) and exits non-zero, never a raw traceback.
    # `TimeoutError` is a stdlib `OSError` subclass, so it is covered by `OSError` here
    # without a separate arm. `InvalidInputError` is a deliberate SIBLING of
    # `ValidationError` (not a subclass, see errors.py) so it must be listed explicitly —
    # it is NOT swallowed by the `ValidationError` arm. `BaseException` (incl.
    # `KeyboardInterrupt`/`SystemExit`) is intentionally NOT caught here: R27's
    # interrupt-cleanup path (MS3) needs it to propagate.
    try:
        return run_delegation(ns)
    except (
        ValidationError,
        OllamaBackendError,
        DelegationError,
        InvalidInputError,
        OSError,
    ) as exc:
        print(f"Delegation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
