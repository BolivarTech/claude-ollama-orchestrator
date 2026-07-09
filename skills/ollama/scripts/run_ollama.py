# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""CLI orchestrator for a single Ollama delegation (transactional core)."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from agent_schema import DISCRIMINATOR_KEYS, SCHEMAS
from backend import AgentBackend, DelegationResult, OllamaBackend
from errors import (
    DelegationError,
    InvalidInputError,
    OllamaBackendError,
    OllamaConfigError,
    ValidationError,
)
from ollama_config import CAPABILITIES, OllamaAgentsConfig

# Explicit re-export (mypy strict, no_implicit_reexport): tests reference
# run_ollama.resolve_config directly as a monkeypatch/config seam.
from ollama_config import resolve_config as resolve_config
from ollama_preflight import preflight
from parse_output import parse_agent_output
from run_lock import remove_lock, staleness_bound_for_timeout, write_lock
from status_display import StatusDisplay
from stderr_shim import buffered_stderr_while
from temp_dirs import (
    cleanup_old_runs,
    create_output_dir,
    project_run_root,
    resolve_project_root,
)
from token_stats import TokenStats
from validate import MAX_INPUT_FILE_SIZE, validate_output

MAX_HISTORY_RUNS = 5

# Retry feedback (R25) is built by explicit concatenation, NOT str.format — the parser/
# validator error text and the JSON-Schema both contain literal braces. Concatenation is
# crash-proof by construction (no format-field parsing of dynamic content) and future-proof
# against a stray brace ever being introduced into a template.
_RETRY_FEEDBACK_PREFIX = "\n\n---RETRY-FEEDBACK---\nYour previous output failed: "
_RETRY_FEEDBACK_SCHEMA_INTRO = "Return JSON conforming to this schema:\n"
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
            # Built by concatenation (not str.format) so braces in last_error / the schema
            # JSON can never be misparsed as format fields.
            attempt_prompt = (
                prompt
                + _RETRY_FEEDBACK_PREFIX
                + last_error
                + "\n"
                + _RETRY_FEEDBACK_SCHEMA_INTRO
                + json.dumps(SCHEMAS[capability])
                + "\n"
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


@contextlib.contextmanager
def managed_run_dir(
    keep_runs: int, timeout: int, *, output_dir: str | None = None
) -> Iterator[str]:
    """Own the run-directory lifecycle for one delegation (R15-R18, R27).

    Extracted out of ``run_delegation`` (SRP: one function, one reason to change), this
    context manager owns exactly the run-directory slice of the lifecycle, so the caller
    reads as a plain ``with`` block around whatever it wants to do with the directory.

    **Managed temp path** (``output_dir is None``, the default): resolves the
    per-project namespace (:func:`resolve_project_root` -> :func:`project_run_root`),
    prunes older runs to ``keep_runs - 1`` (skipped when the namespace degraded to the
    shared ``tempfile.gettempdir()`` -- pruning there would touch *other* projects'
    runs, R15), creates a fresh run dir (:func:`create_output_dir`), and writes the
    process-liveness lock (:func:`write_lock`, sized by
    :func:`staleness_bound_for_timeout`). On exit:

    - **Normal exit (no exception):** the lock is removed (:func:`remove_lock`) -- the
      delegation succeeded, nothing needs to survive for debugging.
    - **`KeyboardInterrupt`/`SystemExit`/`GeneratorExit`:** the in-progress dir is
      ``rmtree``-d (the lock goes with the tree) before the exception is re-raised --
      R27, no orphaned ``ollama-run-*`` dirs on Ctrl-C. `GeneratorExit` is included
      alongside the other two: like them it is a `BaseException`, not an `Exception`,
      and it means this generator-based context manager's `with` body was torn down by
      abandonment (the generator was closed/garbage-collected without a normal
      `__exit__`) rather than by a reported failure -- an abandoned run is being
      discarded, so it gets the same cleanup as an interrupt, not the retain-for-debug
      path below. Re-raising `GeneratorExit` after cleanup satisfies the generator
      close() protocol (a caught `GeneratorExit` must be re-raised, not swallowed).
    - **Any other exception:** the dir *and* its lock are retained, unchanged, and the
      exception is re-raised as-is. The lock stops a concurrent ``cleanup_old_runs``
      from pruning the debug artifacts before its staleness bound elapses (or the
      owning PID dies) -- it is released **only** on the success path above.

    **Explicit `output_dir` path** (`--output-dir`, an advanced override): the directory
    is created (:func:`create_output_dir`) but is otherwise entirely caller-managed --
    no lock is written, no pruning happens, and nothing is ever removed on any exit path
    (success, interrupt, or exception). The caller is responsible for not sharing it
    between concurrent delegations (R28).

    Args:
        keep_runs: The `--keep-runs` budget passed to :func:`cleanup_old_runs` (pruned
            to `keep_runs - 1` so the total lands exactly on `keep_runs` once the new
            dir is created). Ignored when `output_dir` is given.
        timeout: The per-delegation timeout in seconds, used to size the lock's
            staleness bound. Ignored when `output_dir` is given.
        output_dir: An explicit caller-managed directory, or `None` for a managed temp
            run dir (the default).

    Yields:
        The output directory to write artifacts into.

    Raises:
        KeyboardInterrupt: re-raised after `rmtree`-ing the managed temp dir (R27).
        SystemExit: re-raised after `rmtree`-ing the managed temp dir (R27).
        GeneratorExit: re-raised after `rmtree`-ing the managed temp dir (R27) -- an
            abandoned generator means the run is being discarded, same as an interrupt.
        Exception: re-raised unchanged; the managed temp dir and its lock are retained
            for debugging.
    """
    if output_dir is not None:
        yield create_output_dir(output_dir)
        return

    run_root = project_run_root(resolve_project_root())
    # Skip cleanup when project_run_root degraded to the SHARED gettempdir --
    # pruning there would scan/remove OTHER projects' ollama-run-* dirs (R15).
    if run_root != tempfile.gettempdir():
        cleanup_old_runs(keep_runs - 1, run_root)
    run_dir = create_output_dir(None, run_root)
    write_lock(run_dir, staleness_bound_for_timeout(timeout))
    try:
        yield run_dir
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        # R27: an interrupt (or an abandoned/GC'd generator -- GeneratorExit, same
        # BaseException family) orphans the in-progress dir -> rmtree it (the lock goes
        # with the tree). Re-raising satisfies the generator close() protocol for
        # GeneratorExit too (a caught GeneratorExit must be re-raised, never swallowed).
        try:
            shutil.rmtree(run_dir)
        except OSError as exc:
            print(f"WARNING: cleanup failed for {run_dir}: {exc}", file=sys.stderr)
        raise
    else:
        # Reached only on a normal (no-exception) exit -- any OTHER exception skips this
        # `else` entirely and propagates WITHOUT touching the lock or the dir: the lock
        # is RETAINED so a concurrent `cleanup_old_runs` can't prune the debug artifacts
        # before the staleness bound elapses (or the owning PID dies). Released ONLY
        # here, on success.
        remove_lock(run_dir)


def _derive_retried(stats_dict: object) -> list[str]:
    """Return capabilities whose bucket shows more ``http_calls`` than ``delegations``.

    A **structural guard**, MS3-local: ``stats_dict`` is expected to be MS2's
    ``TokenStats.to_dict()`` shape (``{capability: {model: {"http_calls": int,
    "delegations": int, ...}}}``), but that exact shape is not trusted blindly.
    Anything unexpected -- not a dict, a non-dict bucket, missing/non-numeric
    ``http_calls``/``delegations`` -- is skipped rather than raising, so
    ``ollama-report.json``'s ``retried`` field degrades gracefully to ``[]`` instead of
    crashing ``_write_artifacts`` on a shape mismatch.

    Args:
        stats_dict: The result of ``TokenStats.to_dict()`` (or anything else, by
            contract -- this function never raises regardless of input shape).

    Returns:
        List of capability names (insertion order) with at least one retried model
        bucket (empty if the input is malformed or nothing retried).
    """
    if not isinstance(stats_dict, dict):
        return []
    retried: list[str] = []
    for capability, models in stats_dict.items():
        if not isinstance(models, dict):
            continue
        for bucket in models.values():
            if not isinstance(bucket, dict):
                continue
            http_calls = bucket.get("http_calls")
            delegations = bucket.get("delegations")
            if (
                isinstance(http_calls, (int, float))
                and isinstance(delegations, (int, float))
                and not isinstance(http_calls, bool)
                and not isinstance(delegations, bool)
                and http_calls > delegations
            ):
                retried.append(capability)
                break
    return retried


def _safe_display_update(display: "StatusDisplay | None", capability: str, state: str) -> None:
    """Best-effort ``StatusDisplay.update`` -- never raises into the caller's except/finally.

    Used on `run_delegation`'s exception paths (R20) to reflect a failed/timed-out
    delegation in the live status tree before the real exception propagates. A broken
    or already-stopped display must never mask the original failure, so any exception
    *this* raises is itself caught and only warned about.

    Args:
        display: The active ``StatusDisplay``, or ``None`` (``--no-status``) -- a no-op.
        capability: The delegation's capability name (the row to update).
        state: One of ``StatusDisplay.VALID_STATES`` (here: ``"failed"`` or ``"timeout"``).
    """
    if display is None:
        return
    try:
        display.update(capability, state)
    except Exception as exc:  # noqa: BLE001 — must never shadow the real exception in flight.
        print(f"WARNING: status display update failed: {exc}", file=sys.stderr)


def _write_artifacts(
    output_dir: str,
    capability: str,
    ns_input: str,
    result: DelegationResult,
    stats: TokenStats,
    errbuf: io.StringIO,
) -> None:
    """Write the full R18 artifact set for one delegation into *output_dir*.

    ``{cap}.stream.log`` is written by MS4's streaming layer; here the transactional
    content lands in ``{cap}.raw.json`` (+ ``{cap}.parsed.json`` for structured output).
    Every file write is **best-effort**: an ``OSError`` on any one artifact logs a
    warning and the remaining artifacts still get written -- the delegation's primary
    result is already on stdout (R19), so a disk hiccup on a side artifact must never
    crash the run nor abort the rest of the set.

    ``stats.write(output_dir)`` (``token_stats.json``) is wrapped in its own local
    ``try/except OSError`` here too: MS2's ``TokenStats.write`` already returns ``str |
    None`` (``None`` if the dir is unwritable) rather than raising, but
    ``_write_artifacts`` enforces its never-raise contract **locally** -- as
    defense-in-depth -- instead of depending on that internal safety holding across
    milestones.

    The ``retried`` field is derived from ``stats.to_dict()`` defensively via
    :func:`_derive_retried` (MS3-local structural guard; MS2's ``TokenStats.to_dict``
    contract is untouched).

    **Outer guard: contained runtime-failure modes, not a blanket catch-all.** This
    function's entire body is wrapped in one top-level ``except (OSError, TypeError,
    ValueError, RecursionError)`` -- not ``except Exception``. Those four classes are
    exactly the REALISTIC runtime/environment failure modes an artifact-writing pass
    can hit:

    - ``OSError`` -- disk/IO failures (disk full, permission denied, path too long, ...)
      that slip past a narrower per-file guard.
    - ``TypeError`` -- a non-JSON-serializable value inside ``stats.to_dict()``'s
      report (e.g. a bare ``object()``), rejected by ``json.dumps``.
    - ``ValueError`` -- a malformed value ``json.dumps``/formatting rejects.
    - ``RecursionError`` -- ``json.dumps`` hitting Python's recursion limit on a
      pathologically deep/self-referential structure inside ``stats.to_dict()``'s
      report. This is a ``RuntimeError`` subclass, not an ``OSError``/``TypeError``/
      ``ValueError``, so it would otherwise slip past the narrower three-class guard.

    Any of these is caught, logged as one actionable ``WARNING`` to stderr, and the
    function returns. A genuinely unexpected exception (``AttributeError``,
    ``KeyError``, ``IndexError``, or any other bug-shaped failure) is **NOT** caught
    here and is left to propagate: it indicates a real defect in this function or its
    inputs, and surfacing it loudly is more useful than a WARNING that quietly hides
    the bug. ``KeyboardInterrupt``/``SystemExit`` are ``BaseException``, not
    ``Exception`` (and ``RecursionError`` is a ``RuntimeError``, so adding it does not
    blur that boundary either), so neither the narrow nor a blanket form of this guard
    was ever going to catch them -- R27's interrupt-cleanup path is unaffected either
    way.

    Args:
        output_dir: The managed (or caller-owned) run directory to write into.
        capability: The delegation's capability name (artifact filename prefix).
        ns_input: The actual prompt text delegated (written to ``{cap}.prompt.txt``).
        result: The delegation's ``DelegationResult``.
        stats: The run's local token accumulator (R12).
        errbuf: The captured stderr buffer (R18/R20) for ``{cap}.stderr.log``.
    """
    try:

        def _w(name: str, text: str) -> None:
            try:
                with open(os.path.join(output_dir, name), "w", encoding="utf-8") as fh:
                    fh.write(text)
            except OSError as exc:
                print(f"WARNING: could not write artifact {name}: {exc}", file=sys.stderr)

        _w(f"{capability}.prompt.txt", ns_input)
        _w(f"{capability}.raw.json", json.dumps({"content": result.content}, indent=2))
        if result.parsed is not None:
            _w(f"{capability}.parsed.json", json.dumps(result.parsed, indent=2))
        _w(f"{capability}.stderr.log", errbuf.getvalue())
        try:
            stats.write(output_dir)  # token_stats.json
        except OSError as exc:
            print(f"WARNING: could not write token_stats.json: {exc}", file=sys.stderr)
        stats_dict = stats.to_dict()
        # R18 telemetry: `retried` = capabilities whose backend made more http_calls than
        # logical delegations (a parse/schema retry, R25); `timings` = per-delegation
        # wall-clock; `guard` = diff-grounding result (R30), None here (diff-guard lands
        # in MS7).
        retried = _derive_retried(stats_dict)
        report = {
            "capability": capability,
            "tokens_by_model": stats_dict,
            "input_size": {"chars": len(ns_input), "est_tokens": len(ns_input) // 4},
            "estimated": result.estimated,
            "tok_per_s": result.tok_per_s,
            "timings": {capability: result.elapsed_s},
            "retried": retried,
            "guard": None,
        }
        _w("ollama-report.json", json.dumps(report, indent=2))
    except (OSError, TypeError, ValueError, RecursionError) as exc:
        # Contained runtime-failure modes ONLY: disk/IO errors (OSError), a
        # non-JSON-serializable stats.to_dict() value (TypeError), a malformed value
        # json.dumps rejects (ValueError), or json.dumps hitting the recursion limit on
        # a pathologically deep structure (RecursionError -- a RuntimeError subclass
        # that would otherwise slip past the other three) -- every realistic
        # artifact-writing runtime failure. The delegation's primary result is already
        # on stdout (R19) by the time this runs, so a side-artifact failure like this
        # must never crash the run. Deliberately NOT `except Exception`: an
        # AttributeError/KeyError here would mean a real bug in this function or its
        # inputs, and that must surface, not be silently swallowed as a "warning".
        print(
            f"WARNING: artifact write pass failed for capability {capability!r}: {exc}",
            file=sys.stderr,
        )
        return


def run_delegation(ns: argparse.Namespace) -> int:
    """Resolve config, delegate once inside a managed run dir, write artifacts.

    The run-directory lifecycle itself (namespace resolution, pruning, `mkdtemp`, lock,
    and the R27 interrupt/retain/release rules) is entirely owned by
    :func:`managed_run_dir` -- this function only drives one delegation *inside* that
    context: resolve config, run preflight + dispatch under a `StatusDisplay` +
    `buffered_stderr_while`, and write the artifacts.

    Preflight runs INSIDE the stderr shim so its warn-and-proceed warnings (404/501,
    R10) are captured in ``{cap}.stderr.log`` (R18); it still aborts fail-fast on an
    unreachable host or missing model -- that abort is a normal exception, so
    `managed_run_dir` retains the dir with the diagnostic (R27 only `rmtree`s on
    `KeyboardInterrupt`/`SystemExit`/`GeneratorExit`). Before any non-interrupt
    exception propagates, the live display (if up) is marked ``"timeout"`` for a
    ``TimeoutError`` or ``"failed"`` for anything else (R20's state set), via the
    guarded `_safe_display_update` -- so the tree never freezes on ``"running"`` for a
    delegation that actually failed. `KeyboardInterrupt`/`SystemExit` need no dedicated
    `except` clause here: as a `BaseException` it already skips past `except
    TimeoutError`/`except Exception` on its own, still runs `finally: display.stop()`
    (R20 -- the display is restored even on interrupt), and then propagates out to
    `managed_run_dir`'s own interrupt handler, which performs the R27 `rmtree`.
    `buffered_stderr_while`'s `active` flag tracks whether the live display owns the
    terminal: buffer-then-flush when it does, tee (live + captured) when ``--no-status``
    leaves no display up -- so ``{cap}.stderr.log`` is always populated (R18) without
    silencing diagnostics when there is no display to protect.

    R10/R28: the effective model (the `--model` override when given, else the
    capability's configured model) is resolved BEFORE preflight, and threaded into it,
    so a bad `--model` override aborts here (actionable), never a later chat-time 404.

    The reviewable output (`.parsed` when present, else `.content`) is untrusted data
    for Claude to review; it is printed, never auto-applied (R8/R14).
    """
    cfg = resolve_config(global_path=_global_toml(), repo_path=_repo_toml(), env=os.environ)

    with managed_run_dir(ns.keep_runs, ns.timeout, output_dir=ns.output_dir) as output_dir:
        # StatusDisplay captures the REAL sys.stderr at construction (before the shim
        # below), so it still renders live while the shim buffers everyone else's stderr.
        display = None if not ns.show_status else StatusDisplay([ns.capability])
        try:
            # `active` = a live StatusDisplay owns the terminal -> buffer only (protects
            # the ANSI redraw, flush once on exit). `--no-status` -> display is None ->
            # active=False -> tee: diagnostics stay live on the real stderr AND are still
            # captured into `errbuf` for {cap}.stderr.log (R18) -- either mode always
            # yields a usable buffer.
            with buffered_stderr_while(active=(display is not None)) as errbuf:
                model = ns.model or cfg.models[ns.capability]
                # Preflight INSIDE the shim -> its warn-and-proceed warnings (404/501,
                # R10) land in {cap}.stderr.log (R18). Still fail-fast: an unreachable
                # host / missing model aborts here; `managed_run_dir` retains the dir
                # with the stderr.log diagnostic.
                preflight(cfg, capability=ns.capability, effective_model=model)
                system_prompt = load_system_prompt(ns.capability)
                prompt = _load_input(ns.input)
                stats = TokenStats()
                if display is not None:
                    display.update(ns.capability, "running")
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
                _write_artifacts(output_dir, ns.capability, prompt, result, stats, errbuf)
                if display is not None:
                    display.update(ns.capability, "success", tok_per_s=result.tok_per_s)
                # The reviewable output is the validated structured dict (`.parsed`)
                # when present, else the raw text content -- both untrusted data for
                # Claude to review, never auto-applied.
                review = result.parsed if result.parsed is not None else result.content
                rendered = review if isinstance(review, str) else json.dumps(review, indent=2)
                print(rendered)  # Claude reviews stdout before applying via Edit/Write
        except TimeoutError:
            # R20: a timed-out delegation gets its own distinguishable state (not the
            # generic "failed") -- the display update itself never raises (guarded
            # helper).
            _safe_display_update(display, ns.capability, "timeout")
            raise
        except Exception:
            # R20: any other failure (preflight abort, backend error, parse/schema
            # exhaustion, ...) marks the row "failed" before propagating, so the status
            # tree reflects reality instead of freezing on "running". Never masks the
            # real exception (guarded helper). (KeyboardInterrupt/SystemExit are
            # BaseException, not Exception, so they bypass this clause on their own --
            # no dedicated clause is needed for them here; they still run the `finally`
            # below, then propagate to managed_run_dir's own handler.)
            _safe_display_update(display, ns.capability, "failed")
            raise
        finally:
            # R20: flush/restore the live display even on interrupt or a mid-run
            # exception.
            if display is not None:
                display.stop()
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
