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
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any, TextIO

from agent_schema import DISCRIMINATOR_KEYS, SCHEMAS
from backend import AgentBackend, DelegationResult, OllamaBackend
from errors import (
    DelegationError,
    InvalidInputError,
    OllamaBackendError,
    OllamaConfigError,
    OllamaPreflightError,
    SinkError,
    ValidationError,
)
from ollama_config import CAPABILITIES, OllamaAgentsConfig

# Explicit re-export (mypy strict, no_implicit_reexport): tests reference
# run_ollama.resolve_config directly as a monkeypatch/config seam.
from ollama_config import resolve_config as resolve_config
from ollama_preflight import preflight
from ollama_stream import stream_run
from ollama_vision import stream_vision
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


def stdout_sink(text: str) -> None:
    """R7c stdout sink: write a streamed delta to stdout and flush (single delegation only).

    The live, single-in-flight-delegation streaming sink (R7c): every MS4 delegation
    passes this to ``dispatch`` as ``sink`` — MS4 has no fan-out (MS5), so the R7c
    invariant "stdout streams only when exactly one delegation is in flight" trivially
    holds and this is always the active sink for a streaming capability.

    Args:
        text: One delta of streamed content.
    """
    sys.stdout.write(text)
    sys.stdout.flush()


class _FileSink:
    """A per-delegation streaming sink that owns ONE open file handle: opened once,
    written per delta, closed once at stream end. Never reopens the file per token
    (the old design opened+closed on every delta). Callable as a ``Callable[[str], None]``,
    and also supports the context-manager protocol (``with make_file_sink(path) as sink:``)
    so ``close()`` is guaranteed on exit — including when the ``with`` block raises.

    Reserved for the MS5 fan-out (each parallel delegation streams to its OWN file,
    never a shared stdout, R7c); MS4 always streams the single in-flight delegation to
    :func:`stdout_sink` instead.
    """

    def __init__(self, fh: TextIO) -> None:
        """Wrap an ALREADY-OPEN file handle.

        The file is opened by `make_file_sink` — never inside this constructor —
        specifically so a failure anywhere in construction can never produce a
        half-built sink holding an orphan, unreachable handle: `make_file_sink` owns
        the open/close-on-failure pairing (see its docstring), while `_FileSink`
        itself only ever takes ownership of a handle that is guaranteed either fully
        adopted (this constructor returns) or already closed by the caller.

        Args:
            fh: An already-open, writable text file handle.
        """
        self._fh = fh
        self._closed = False

    def __call__(self, text: str) -> None:
        """Write *text* to the open file handle.

        Raises:
            SinkError: called after ``close()`` — use-after-close is a caller bug,
                surfaced as a clear, actionable domain error instead of letting the
                raw ``ValueError: I/O operation on closed file`` from the underlying
                handle leak through unclassified.
        """
        if self._closed:
            raise SinkError("write to a closed _FileSink")
        self._fh.write(text)

    def __enter__(self) -> "_FileSink":
        """Return self — the file is already open from ``__init__``/``make_file_sink``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Guarantee ``close()`` on ``with`` exit, whether or not the block raised."""
        self.close()

    def close(self) -> None:
        """Close the handle (best-effort; **idempotent**).

        A double-close (e.g. ``close()`` called again in a ``finally`` block after
        the stream already closed it, or after ``__exit__`` already closed it) must
        NOT raise ``ValueError: I/O operation on closed file`` — this guard makes the
        second call a no-op.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._fh.close()
        except OSError:
            pass


def make_file_sink(path: str) -> "_FileSink":
    """Open *path* (``{cap}.stream.log``) ONCE and return a `_FileSink` wrapping it.

    The file is opened HERE, not inside `_FileSink.__init__`, and this function owns
    the open/adopt-or-close pairing: if constructing the `_FileSink` around the
    freshly-opened handle fails for ANY reason, the handle is closed before the
    exception propagates — a failed construction can never leak an orphan open file
    handle. On success, ownership of the handle transfers entirely to the returned
    `_FileSink` (its `.close()`/context-manager protocol is thereafter the only way
    to close it).

    The caller MUST ``.close()`` the returned sink in a ``finally`` at stream end —
    OR use it as a context manager (``with make_file_sink(path) as sink:``), whose
    ``__enter__``/``__exit__`` guarantee the close. Used by the MS5 fan-out so each
    parallel delegation streams to its OWN file (never a shared stdout, R7c).

    Raises:
        OSError: the file could not be opened (propagates unchanged; nothing was
            opened, so there is nothing to clean up).
    """
    fh = open(path, "a", encoding="utf-8")  # noqa: SIM115 — ownership transfers to _FileSink
    try:
        return _FileSink(fh)
    except Exception:
        fh.close()  # construction failed after a successful open — never leak the handle
        raise


# Capabilities whose transport is not in MS1 (multimodal/audio → M7). Guarded in
# ``dispatch`` so a binary input is never sent as garbled chat text. Removed in M7.
_MS1_UNSUPPORTED_CAPS = frozenset({"vision", "transcribe"})


def _run_once(
    capability: str,
    system_prompt: str,
    prompt: str,
    model: str,
    timeout: int,
    *,
    backend: AgentBackend,
    config: OllamaAgentsConfig,
    sink: Callable[[str], None] | None,
    response_format: dict[str, Any] | None,
    deadline: float | None = None,
) -> DelegationResult:
    """Pick the streaming or transactional path for one delegation attempt (R7b/R7c).

    Streams via ``ollama_stream.stream_run`` (or ``ollama_vision.stream_vision`` for
    ``vision``) when BOTH the capability's ``[stream]`` config is true AND a *sink* is
    given — no sink means nothing to stream to, so the transactional path always
    applies regardless of the ``[stream]`` setting. Otherwise runs the MS1
    transactional core (``backend.run``) unchanged.

    *deadline* is threaded into ``backend.run`` only: the streaming path
    (``stream_run``/``stream_vision``) derives its OWN deadline internally from
    *timeout* and has no deadline parameter to accept, so passing the caller's shared
    R25 deadline to it would be meaningless. For the transactional path, forwarding it
    preserves MS1's existing guarantee that the parse-retry loop (``dispatch``) and the
    429-backoff loop (``backend.run``) share the SAME time budget.

    Args:
        capability: The capability name.
        system_prompt: The capability's system prompt.
        prompt: The (possibly retry-feedback-augmented) prompt for this attempt.
        model: Resolved model tag.
        timeout: Per-delegation timeout.
        backend: The transactional ``AgentBackend``.
        config: The resolved config (``config.stream`` drives the path choice).
        sink: The delta sink for the streaming path, or ``None`` (never streams).
        response_format: The structured-output shape for this attempt, or ``None``.
        deadline: The shared monotonic deadline (R25), forwarded to ``backend.run``
            only.

    Returns:
        The ``DelegationResult`` for this one attempt.
    """
    if bool(config.stream.get(capability)) and sink is not None:
        fn = stream_vision if capability == "vision" else stream_run
        return fn(
            config,
            system_prompt,
            prompt,
            model,
            timeout,
            sink=sink,
            response_format=response_format,
        )
    return backend.run(
        capability,
        system_prompt,
        prompt,
        model,
        timeout,
        response_format=response_format,
        deadline=deadline,
    )


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
    sink: Callable[[str], None] | None = None,
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

    Each attempt is routed through :func:`_run_once` (R7b/R7c), which picks the
    streaming path (``ollama_stream.stream_run``/``ollama_vision.stream_vision``) when
    BOTH *config*'s per-capability ``[stream]`` setting is true AND *sink* is given,
    else the MS1 transactional core (``backend.run``) — unchanged either way from the
    caller's perspective, since both paths return the same ``DelegationResult`` shape.

    Args:
        capability: The capability name.
        prompt: The user prompt, passed as-is in MS1 (anti-injection sanitization is
            added in M6 — this is a forward reference, MS1 does not sanitize).
        backend: An ``AgentBackend`` (injected).
        model: Resolved model tag.
        timeout: Per-delegation timeout.
        system_prompt: The capability's system prompt.
        config: The resolved config (its ``structured`` mapping drives the mode; its
            ``stream`` mapping drives the streaming-vs-transactional choice, R7b/R7c).
        stats: Optional local token accumulator (R12); every completed backend call is
            recorded into it.
        sink: Optional delta receiver for the streaming path (R7c); ``None`` (the
            default) never streams regardless of *config*'s ``[stream]`` setting —
            callers that want streaming must supply one (e.g. ``stdout_sink`` for the
            single in-flight delegation).

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
        result = _run_once(
            capability,
            system_prompt,
            prompt,
            model,
            timeout,
            backend=backend,
            config=config,
            sink=sink,
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
        result = _run_once(
            capability,
            system_prompt,
            attempt_prompt,
            model,
            timeout,
            backend=backend,
            config=config,
            sink=sink,
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
      exception is re-raised as-is. The lock stops a concurrent ``cleanup_old_runs`` from
      pruning the dir before its staleness bound elapses (or the owning PID dies) -- it is
      released **only** on the success path above. NOTE: :func:`_write_artifacts` runs only
      on the SUCCESS path, so a dir retained after a *failure* holds just the lock -- the
      failure diagnostic is shown live on stderr (via the stderr shim, R19/R20), not
      persisted here. Persisting partial failure-path artifacts (the prompt / stderr
      buffer) for post-mortem debugging is a deliberate future enhancement, NOT implemented
      in this milestone; the retention still serves R27's invariant (a live-marked dir that
      a concurrent cleanup must not prune).

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


def _safe_display_stop(display: "StatusDisplay | None") -> None:
    """Best-effort ``StatusDisplay.stop`` -- never raises into the caller's ``finally``.

    ``run_delegation`` calls this in a ``finally`` that runs while an exception -- possibly
    ``KeyboardInterrupt`` -- is in flight (R27). ``StatusDisplay.stop`` flushes the REAL
    stderr, which can raise ``OSError`` when that stream is broken/closed at teardown
    (Ctrl-C / pipe closure -- exactly R27's scenario). An unguarded raise from a ``finally``
    REPLACES the propagating exception, so ``managed_run_dir``'s
    ``except (KeyboardInterrupt, SystemExit, GeneratorExit)`` rmtree path would miss the
    interrupt (seeing a plain ``OSError`` instead) and wrongly RETAIN the run dir. Any
    failure here is swallowed -- including from the warning ``print`` itself, since stderr
    may be the very stream that is broken -- so the real exception always wins.

    Args:
        display: The active ``StatusDisplay``, or ``None`` (``--no-status``) -- a no-op.
    """
    if display is None:
        return
    try:
        display.stop()
    except Exception as exc:  # noqa: BLE001 — must never shadow the real exception in flight.
        try:
            print(f"WARNING: status display stop failed: {exc}", file=sys.stderr)
        except Exception:  # noqa: BLE001 — stderr itself may be the broken stream.
            pass


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

    `dispatch` is given `sink=stdout_sink` (R7c): for a capability whose per-capability
    `[stream]` config is true, tokens stream live to stdout as they arrive; this CLI
    always drives exactly one delegation at a time (fan-out is MS5), so the R7c
    invariant -- stdout streams only when a single delegation is in flight -- always
    holds here.
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
                    # R7c: this CLI drives exactly ONE delegation at a time (concurrency
                    # is MS5) -- the single-in-flight case always gets the stdout sink.
                    sink=stdout_sink,
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
            # exception. Guarded (R27): a raising stop() must never REPLACE the exception
            # in flight (esp. KeyboardInterrupt), or managed_run_dir's rmtree path is missed.
            _safe_display_stop(display)
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
    except OllamaPreflightError as exc:
        # R14: a preflight failure (unreachable host / missing model) is genuinely "Ollama
        # unavailable" → abort with an ACTIONABLE message and a DISTINCT exit code 2, never
        # a generic "Delegation failed", never a silent fall-back to Claude generation. The
        # domain exc already carries the specific remedy (ollama pull / signin / edit TOML);
        # this frames it and offers the explicit-Claude alternative. MUST precede the
        # ValidationError arm below (OllamaPreflightError is a ValidationError subclass;
        # Python tries except clauses in order). A plain OllamaConfigError (bad TOML, or a
        # missing agent prompt from load_system_prompt) is NOT "Ollama unavailable" — it
        # falls through to the generic actionable handler below.
        print(
            f"Ollama unavailable: {exc}\nNot delegating; resolve the issue and retry "
            "(or generate with Claude explicitly).",
            file=sys.stderr,
        )
        return 2
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
