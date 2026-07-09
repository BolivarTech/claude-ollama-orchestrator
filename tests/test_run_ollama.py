# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""CLI orchestrator: argparse surface, --ollama-init short-circuit, single dispatch."""

import json
import os
import tempfile
from dataclasses import replace
from types import MappingProxyType

import pytest

import run_ollama
from backend import DelegationResult
from errors import OllamaBackendError, OllamaPreflightError, ValidationError
from run_ollama import _validate_args, build_parser


def test_capability_positional_rejects_invalid_choice():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["notacap", "some input"])  # argparse `choices` → SystemExit


# --keep-runs/--warn-input-tokens/--timeout are cross-validated in `_validate_args`
# (not by argparse types), so these exercise that function after a clean parse.
def test_keep_runs_zero_is_rejected():
    parser = build_parser()
    ns = parser.parse_args(["coder", "in", "--keep-runs", "0"])
    with pytest.raises(SystemExit):
        _validate_args(parser, ns)


def test_warn_input_tokens_must_be_positive():
    parser = build_parser()
    ns = parser.parse_args(["coder", "in", "--warn-input-tokens", "0"])
    with pytest.raises(SystemExit):
        _validate_args(parser, ns)


def test_timeout_must_be_positive():
    parser = build_parser()
    ns = parser.parse_args(["coder", "in", "--timeout", "0"])
    with pytest.raises(SystemExit):
        _validate_args(parser, ns)


def test_valid_args_parse():
    parser = build_parser()
    ns = parser.parse_args(["coder", "write a function", "--timeout", "120"])
    assert ns.capability == "coder"
    assert ns.timeout == 120


def test_ollama_init_short_circuit_scaffolds_and_exits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from run_ollama import main

    rc = main(["--ollama-init"])
    assert rc == 0
    assert (tmp_path / ".claude" / "ollama-agents.toml").exists()


def test_ollama_init_on_existing_config_returns_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    from run_ollama import main

    assert main(["--ollama-init"]) == 0  # first scaffold succeeds
    rc = main(["--ollama-init"])  # second refuses
    assert rc != 0  # refuse-if-exists is a FAILURE
    assert "refusing to overwrite" in capsys.readouterr().err.lower()


def test_ollama_init_flag_in_input_text_does_not_scaffold(tmp_path, monkeypatch):
    # The flag triggers ONLY as the first token; the literal in the input text must not.
    monkeypatch.chdir(tmp_path)
    import run_ollama

    monkeypatch.setattr(run_ollama, "run_delegation", lambda ns: 0)  # skip real delegation
    run_ollama.main(["coder", "please run --ollama-init later"])  # flag not the first token
    assert not (tmp_path / ".claude" / "ollama-agents.toml").exists()


def test_ollama_init_flag_not_first_token_is_unrecognized_arg(tmp_path, monkeypatch):
    # --ollama-init is NOT an argparse flag (removed from build_parser) — the pre-parse
    # short-circuit is the ONLY handling path, and it fires only when args[0] is the
    # flag. As a later token it is neither the init flag nor a valid positional: argparse
    # rejects it as an unrecognized argument (SystemExit), never a silent no-op.
    monkeypatch.chdir(tmp_path)
    import run_ollama

    with pytest.raises(SystemExit):
        run_ollama.main(["coder", "--ollama-init"])
    assert not (tmp_path / ".claude" / "ollama-agents.toml").exists()


def test_ollama_init_oserror_is_actionable_nonzero_not_a_crash(tmp_path, monkeypatch, capsys):
    # write_template can raise a plain OSError BEYOND the already-handled
    # FileExistsError (refuse-if-exists) — disk full, permission denied, a read-only
    # target dir. Since FileExistsError IS an OSError subclass, main must catch it as a
    # SEPARATE, more-specific arm before a generic `except OSError`, or the generic arm
    # would swallow the refuse-if-exists case. Either way, this must be an actionable,
    # non-zero exit — never a raw, unhandled traceback.
    monkeypatch.chdir(tmp_path)
    import ollama_init
    import run_ollama

    def _boom(repo_root=None):
        raise OSError("disk full")

    monkeypatch.setattr(ollama_init, "write_template", _boom)
    rc = run_ollama.main(["--ollama-init"])
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "failed to write config template" in err
    assert "disk full" in err


class _FakeBackend:
    """Records each call; returns queued contents (or raises a queued exc) in order."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls: list = []

    def run(
        self,
        capability,
        system_prompt,
        prompt,
        model,
        timeout,
        *,
        response_format=None,
        deadline=None,
    ):
        self.calls.append(
            {"capability": capability, "prompt": prompt, "response_format": response_format}
        )
        item = self.scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        # MS2: backend.run now returns a DelegationResult (content + token metrics),
        # not a bare str. Scripted items are the `content`; wrap them (estimated
        # metrics, unused by these MS1-era dispatch tests) so `.content` reads back.
        return DelegationResult(item, 0, 0, True, 0.0)


_GOOD_REVIEW = (
    '{"capability": "reviewer", "findings": [{"severity": "info", "title": "t", "detail": "d"}]}'
)

# Default config (structured: reviewer/tester="schema", others="off") + a helper to
# override the per-capability structured mode for the mode-driven dispatch tests (R29).
_CFG = run_ollama.resolve_config(global_path=None, repo_path=None, env={})


def _cfg_with_structured(**overrides):
    return replace(_CFG, structured=MappingProxyType({**_CFG.structured, **overrides}))


def test_dispatch_free_text_returns_content_and_sends_no_schema():
    be = _FakeBackend(["def f(): pass"])
    out = run_ollama.dispatch(
        "coder", "write f", backend=be, model="m", timeout=10, system_prompt="sys", config=_CFG
    )  # coder default "off"
    assert out.content == "def f(): pass"
    assert out.parsed is None  # free-text: no structured payload
    assert be.calls[0]["response_format"] is None  # off: no schema


def test_dispatch_structured_sends_per_capability_json_schema():
    be = _FakeBackend([_GOOD_REVIEW])
    run_ollama.dispatch(
        "reviewer", "review", backend=be, model="m", timeout=10, system_prompt="sys", config=_CFG
    )  # reviewer default "schema"
    rf = be.calls[0]["response_format"]
    assert rf["type"] == "json_schema"  # real schema (R29), not json_object
    assert rf["json_schema"]["schema"] == run_ollama.SCHEMAS["reviewer"]


def test_dispatch_structured_off_returns_content_verbatim_without_schema():
    # R29: structured.reviewer="off" suppresses the schema AND the parse/validate path.
    be = _FakeBackend(["free-form review text"])
    out = run_ollama.dispatch(
        "reviewer",
        "review",
        backend=be,
        model="m",
        timeout=10,
        system_prompt="sys",
        config=_cfg_with_structured(reviewer="off"),
    )
    assert out.content == "free-form review text"  # off: content verbatim
    assert be.calls[0]["response_format"] is None  # off: no response_format


def test_dispatch_object_mode_sends_generic_json_envelope():
    # R29: structured.coder="object" sends the generic JSON envelope, content verbatim.
    be = _FakeBackend(['{"any": "json"}'])
    out = run_ollama.dispatch(
        "coder",
        "emit json",
        backend=be,
        model="m",
        timeout=10,
        system_prompt="sys",
        config=_cfg_with_structured(coder="object"),
    )
    assert out.content == '{"any": "json"}'  # object: content verbatim (no validate)
    assert out.parsed is None  # object mode: no schema validation → nothing parsed
    assert be.calls[0]["response_format"] == {"type": "json_object"}


def test_dispatch_rejects_vision_transcribe_in_ms1():
    # MS1 has no multimodal/binary transport (lands in M7). Dispatching vision/transcribe
    # must fail actionably with DelegationError, not send a binary as garbled chat text.
    from errors import DelegationError

    be = _FakeBackend(["unused"])
    for cap in ("vision", "transcribe"):
        with pytest.raises(DelegationError):
            run_ollama.dispatch(
                cap,
                "img-or-audio",
                backend=be,
                model="m",
                timeout=10,
                system_prompt="sys",
                config=_CFG,
            )
    assert be.calls == []  # backend never invoked


def test_dispatch_structured_schema_without_a_schema_fails_loud():
    # A capability set to structured="schema" but lacking a JSON-Schema must FAIL LOUD
    # (actionable ValidationError), never silently degrade to json_object/None.
    be = _FakeBackend(["unused"])
    with pytest.raises(ValidationError):
        run_ollama.dispatch(
            "coder",
            "x",
            backend=be,
            model="m",
            timeout=10,
            system_prompt="sys",
            config=_cfg_with_structured(coder="schema"),
        )


def test_dispatch_retries_once_with_feedback_on_schema_failure():
    be = _FakeBackend(['{"capability": "reviewer"}', _GOOD_REVIEW])  # 1st fails, 2nd ok
    out = run_ollama.dispatch(
        "reviewer", "review", backend=be, model="m", timeout=10, system_prompt="sys", config=_CFG
    )
    assert out.parsed["findings"][0]["severity"] == "info"
    assert "---RETRY-FEEDBACK---" in be.calls[1]["prompt"]  # feedback reinjected
    assert '"const": "reviewer"' in be.calls[1]["prompt"]  # ACTUAL JSON-Schema, not just keys


def test_dispatch_deadline_exceeded_raises_backend_error(monkeypatch):
    # Monotonic wall-clock deadline (R25): start -> attempt-1 within budget -> retry past it.
    # Ticks: 0.0 sets deadline=1.0; 0.5 is attempt-1's pre-check (still within budget, so
    # attempt-1 actually RUNS and fails schema validation, triggering the retry); 1.5 is
    # attempt-2's pre-check, now past the deadline, so it raises WITHOUT a second backend
    # call. (_FakeBackend.run itself never consumes a tick.)
    ticks = iter([0.0, 0.5, 1.5])
    monkeypatch.setattr(run_ollama.time, "monotonic", lambda: next(ticks))
    be = _FakeBackend(['{"capability": "reviewer"}'])  # first attempt fails schema
    with pytest.raises(OllamaBackendError):
        run_ollama.dispatch(
            "reviewer", "review", backend=be, model="m", timeout=1, system_prompt="sys", config=_CFG
        )
    assert len(be.calls) == 1  # attempt-1 ran; the retry never did


def test_dispatch_free_text_records_one_delegation_and_returns_result():
    from token_stats import TokenStats

    class _RecordingBackend:
        def run(
            self,
            capability,
            system_prompt,
            prompt,
            model,
            timeout,
            *,
            response_format=None,
            deadline=None,
        ):
            return DelegationResult("hello", 10, 5, False, 0.5)

    stats = TokenStats()
    out = run_ollama.dispatch(
        "coder",
        "write",
        backend=_RecordingBackend(),
        model="kimi-k2.7-code:cloud",
        timeout=10,
        system_prompt="sys",
        config=_CFG,
        stats=stats,
    )
    assert out.content == "hello" and out.parsed is None
    d = stats.to_dict()["coder"]["kimi-k2.7-code:cloud"]
    assert d["http_calls"] == 1 and d["delegations"] == 1
    assert d["prompt_tokens"] == 10 and d["completion_tokens"] == 5


def test_dispatch_structured_returns_delegation_result_with_parsed():
    from token_stats import TokenStats

    class _StructuredBackend:
        def run(
            self,
            capability,
            system_prompt,
            prompt,
            model,
            timeout,
            *,
            response_format=None,
            deadline=None,
        ):
            return DelegationResult(_GOOD_REVIEW, 10, 5, False, 0.5)

    stats = TokenStats()
    out = run_ollama.dispatch(
        "reviewer",
        "review",
        backend=_StructuredBackend(),
        model="glm-5.2:cloud",
        timeout=10,
        system_prompt="sys",
        config=_CFG,
        stats=stats,
    )
    assert isinstance(out, DelegationResult)
    assert out.parsed is not None and out.parsed["capability"] == "reviewer"
    d = stats.to_dict()["reviewer"]["glm-5.2:cloud"]
    assert d["http_calls"] == 1 and d["delegations"] == 1


def test_dispatch_structured_retry_counts_two_http_calls_one_delegation():
    from token_stats import TokenStats

    class _FlakyBackend:
        def __init__(self):
            self.calls = 0

        def run(
            self,
            capability,
            system_prompt,
            prompt,
            model,
            timeout,
            *,
            response_format=None,
            deadline=None,
        ):
            self.calls += 1
            content = "not json" if self.calls == 1 else _GOOD_REVIEW
            return DelegationResult(content, 10, 5, False, 0.5)

    stats = TokenStats()
    out = run_ollama.dispatch(
        "reviewer",
        "review",
        backend=_FlakyBackend(),
        model="glm-5.2:cloud",
        timeout=10,
        system_prompt="sys",
        config=_CFG,
        stats=stats,
    )
    assert out.parsed is not None
    d = stats.to_dict()["reviewer"]["glm-5.2:cloud"]
    assert d["http_calls"] == 2 and d["delegations"] == 1  # retry billed, one delegation
    assert d["prompt_tokens"] == 20 and d["completion_tokens"] == 10  # BOTH attempts' tokens


def test_dispatch_does_not_record_stats_for_a_call_that_raises_before_returning():
    # http_calls counts backend calls that COMPLETE and produce a DelegationResult.
    # A call that raises (connection/timeout/5xx) before returning is never recorded —
    # tracking failed attempts is the per-model circuit breaker's job (R14b/MS5).
    from token_stats import TokenStats

    class _FailingBackend:
        def run(
            self,
            capability,
            system_prompt,
            prompt,
            model,
            timeout,
            *,
            response_format=None,
            deadline=None,
        ):
            raise OllamaBackendError("connection refused")

    stats = TokenStats()
    with pytest.raises(OllamaBackendError):
        run_ollama.dispatch(
            "coder",
            "write",
            backend=_FailingBackend(),
            model="kimi-k2.7-code:cloud",
            timeout=10,
            system_prompt="sys",
            config=_CFG,
            stats=stats,
        )
    assert stats.to_dict() == {}


def test_load_system_prompt_reads_agent_file(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "ollama-coder.md").write_text("CODER PROMPT", encoding="utf-8")
    monkeypatch.setattr(run_ollama, "_AGENTS_DIR", str(agents))
    assert run_ollama.load_system_prompt("coder") == "CODER PROMPT"


def test_load_input_rejects_oversize_file(tmp_path, monkeypatch):
    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * 16)
    monkeypatch.setattr(run_ollama, "MAX_INPUT_FILE_SIZE", 4)
    with pytest.raises(ValidationError):
        run_ollama._load_input(str(big))


def test_load_input_size_check_is_post_read_not_a_separate_stat_toctou(tmp_path, monkeypatch):
    # TOCTOU guard (R23): the cap must be enforced on the ACTUALLY-read bytes via a
    # bounded read (MS6's `load_binary` pattern), never a getsize()-then-open() pair —
    # a file that grows between a stat and a read would otherwise bypass the cap. Assert
    # the read itself is bounded to MAX_INPUT_FILE_SIZE + 1, never unbounded.
    p = tmp_path / "in.txt"
    p.write_text("hello", encoding="utf-8")
    monkeypatch.setattr(run_ollama, "MAX_INPUT_FILE_SIZE", 100)
    seen: dict[str, int] = {}
    real_open = open

    class _BoundRecordingFile:
        def __init__(self, fh):
            self._fh = fh

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self._fh.close()
            return False

        def read(self, n=-1):
            seen["n"] = n
            return self._fh.read(n)

    def _fake_open(path, mode="rb", **kwargs):
        return _BoundRecordingFile(real_open(path, mode, **kwargs))

    monkeypatch.setattr(run_ollama, "open", _fake_open, raising=False)
    assert run_ollama._load_input(str(p)) == "hello"
    assert seen["n"] == 101  # MAX_INPUT_FILE_SIZE + 1 — never -1 / unbounded


def test_load_input_inline_text_passthrough():
    assert run_ollama._load_input("just inline text") == "just inline text"


def test_load_input_existing_path_reads_file_content(tmp_path):
    p = tmp_path / "input.txt"
    p.write_text("file contents here", encoding="utf-8")
    assert run_ollama._load_input(str(p)) == "file contents here"


def test_load_input_path_shaped_missing_arg_warns_and_still_treats_as_text(capsys):
    # A typo'd path (has path shape — separator + extension, no whitespace — but does
    # NOT exist) must not be silently swallowed as if it were intentional literal text:
    # an actionable WARNING is printed to stderr, and the documented behavior is
    # warn-and-proceed (still literal text), consistent with MS1's other
    # ambiguous-but-not-fatal cases (e.g. preflight 404/501 warn-and-proceed).
    missing = os.path.join("some", "typo'd", "path.py")
    out = run_ollama._load_input(missing)
    assert out == missing  # defined behavior: still literal text
    err = capsys.readouterr().err.lower()
    assert "warning" in err and "does not exist" in err


def test_load_input_plain_prose_with_no_path_shape_does_not_warn(capsys):
    # Free-form prose (contains whitespace) is never mistaken for a path shape — no
    # warning, silent literal-text passthrough (the common case).
    text = "please refactor this function to be faster"
    assert run_ollama._load_input(text) == text
    assert capsys.readouterr().err == ""


def test_looks_like_path_detects_separator_extension_and_absolute():
    assert run_ollama._looks_like_path(os.path.join("a", "b.py")) is True
    assert run_ollama._looks_like_path("script.py") is True
    assert run_ollama._looks_like_path(os.path.abspath("x")) is True
    assert run_ollama._looks_like_path("please write a function") is False  # has spaces
    assert run_ollama._looks_like_path("just-a-word") is False  # no sep/ext


def test_main_runs_full_pipeline_and_prints_output(capsys, monkeypatch, tmp_path):
    # Isolate cwd: run_delegation writes token_stats.json to os.getcwd() (MS2 interim, no
    # --output-dir here) — chdir to tmp_path so the suite never touches the real repo root.
    monkeypatch.chdir(tmp_path)
    # Use the REAL OllamaAgentsConfig (via _cfg_with_structured) — a hand-rolled stub with
    # only `models` would AttributeError once dispatch reads `config.structured` (coder→"off").
    cfg = _cfg_with_structured()
    monkeypatch.setattr(run_ollama, "resolve_config", lambda **kw: cfg)
    # **kw absorbs preflight's capability=/effective_model= kwargs (R10/R28 fix) — the
    # stub only needs to no-op regardless of what run_delegation now threads through.
    monkeypatch.setattr(run_ollama, "preflight", lambda cfg, **kw: None)
    monkeypatch.setattr(run_ollama, "load_system_prompt", lambda cap: "sys")
    monkeypatch.setattr(run_ollama, "_make_backend", lambda cfg: _FakeBackend(["hello world"]))
    rc = run_ollama.main(["coder", "write hello", "--no-status"])
    assert rc == 0
    assert "hello world" in capsys.readouterr().out


def test_main_model_override_reaches_backend(monkeypatch, tmp_path):
    # R28: --model overrides the config default and reaches backend.run as `model`.
    monkeypatch.chdir(tmp_path)  # isolate cwd: run_delegation writes token_stats.json there
    seen: dict[str, str] = {}

    class _CapBackend:
        def run(
            self,
            capability,
            system_prompt,
            prompt,
            model,
            timeout,
            *,
            response_format=None,
            deadline=None,
        ):
            seen["model"] = model
            return DelegationResult("ok", 0, 0, True, 0.0)

    cfg = _cfg_with_structured()
    monkeypatch.setattr(run_ollama, "resolve_config", lambda **kw: cfg)
    # **kw absorbs preflight's capability=/effective_model= kwargs (R10/R28 fix) — the
    # stub only needs to no-op regardless of what run_delegation now threads through.
    monkeypatch.setattr(run_ollama, "preflight", lambda cfg, **kw: None)
    monkeypatch.setattr(run_ollama, "load_system_prompt", lambda cap: "sys")
    monkeypatch.setattr(run_ollama, "_make_backend", lambda cfg: _CapBackend())
    run_ollama.main(["coder", "write hi", "--no-status", "--model", "custom-model:cloud"])
    assert seen["model"] == "custom-model:cloud"  # override wins over cfg.models["coder"]


def test_main_threads_the_model_override_into_preflight_not_just_the_backend(monkeypatch, tmp_path):
    # R10/R28 fix: a --model override must reach preflight's model-existence check, not
    # only backend.run — otherwise an override to a nonexistent model would silently
    # bypass preflight and only surface as a chat-time 404. This asserts preflight is
    # called with the OVERRIDE (as effective_model, for capability="coder"), never the
    # stale config default.
    monkeypatch.chdir(tmp_path)  # isolate cwd: run_delegation writes token_stats.json there
    seen_preflight: dict[str, object] = {}

    def _preflight(cfg, **kw):
        seen_preflight.update(kw)

    cfg = _cfg_with_structured()
    monkeypatch.setattr(run_ollama, "resolve_config", lambda **kw: cfg)
    monkeypatch.setattr(run_ollama, "preflight", _preflight)
    monkeypatch.setattr(run_ollama, "load_system_prompt", lambda cap: "sys")
    monkeypatch.setattr(run_ollama, "_make_backend", lambda cfg: _FakeBackend(["ok"]))
    run_ollama.main(["coder", "write hi", "--no-status", "--model", "custom-model:cloud"])
    assert seen_preflight["capability"] == "coder"
    assert seen_preflight["effective_model"] == "custom-model:cloud"
    assert seen_preflight["effective_model"] != cfg.models["coder"]  # not the stale default


def test_main_aborts_when_preflight_fails(monkeypatch):
    # A real config (via _cfg_with_structured) is required here now: run_delegation
    # resolves the effective model (`ns.model or cfg.models[ns.capability]`) BEFORE
    # calling preflight (R10/R28 fix), so the mocked config must have a real `.models`
    # mapping to index into — a bare string stand-in would AttributeError before
    # preflight even runs.
    monkeypatch.setattr(run_ollama, "resolve_config", lambda **kw: _cfg_with_structured())

    def _boom(cfg, **kw):
        raise OllamaPreflightError("host down")

    monkeypatch.setattr(run_ollama, "preflight", _boom)
    monkeypatch.setattr(
        run_ollama,
        "_make_backend",
        lambda cfg: (_ for _ in ()).throw(AssertionError("must not delegate")),
    )
    rc = run_ollama.main(["coder", "write hello", "--no-status"])
    assert rc != 0  # abort, non-zero, no delegation


def test_main_preflight_failure_gives_actionable_r14_message_and_exit_2(
    monkeypatch, capsys, tmp_path
):
    # R14: an unreachable host / missing model / bad config aborts with an ACTIONABLE
    # "Ollama unavailable ... Not delegating; resolve and retry (or generate with Claude
    # explicitly)" message and a DISTINCT exit code 2 — never a generic "Delegation
    # failed", never a silent fall-back to Claude. Pins the R14 messaging so the run-dir
    # lifecycle wiring cannot silently regress it.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_ollama, "resolve_config", lambda **kw: _cfg_with_structured())

    def _boom(cfg, **kw):
        raise OllamaPreflightError("host unreachable at http://x:11434/v1")

    monkeypatch.setattr(run_ollama, "preflight", _boom)
    monkeypatch.setattr(
        run_ollama,
        "_make_backend",
        lambda cfg: (_ for _ in ()).throw(AssertionError("must not delegate")),
    )
    rc = run_ollama.main(["coder", "write hi", "--no-status"])
    assert rc == 2  # R14 distinct exit code, not the generic 1
    err = capsys.readouterr().err
    assert "Ollama unavailable" in err and "Not delegating" in err


def test_output_dir_writes_raw_artifact(tmp_path, monkeypatch):
    # R28: --output-dir persists the raw output to a caller-owned dir. Since MS3 (Task 5),
    # {cap}.raw.json follows the canonical R18 artifact format (`_write_artifacts`, a JSON
    # envelope `{"content": ...}`) for BOTH a managed run dir and an explicit --output-dir —
    # the interim MS1/MS2 verbatim-text dump is superseded by that standardized format.
    cfg = _cfg_with_structured()  # real config; coder→"off" free-text → content written verbatim
    monkeypatch.setattr(run_ollama, "resolve_config", lambda **kw: cfg)
    # **kw absorbs preflight's capability=/effective_model= kwargs (R10/R28 fix) — the
    # stub only needs to no-op regardless of what run_delegation now threads through.
    monkeypatch.setattr(run_ollama, "preflight", lambda cfg, **kw: None)
    monkeypatch.setattr(run_ollama, "load_system_prompt", lambda cap: "sys")
    monkeypatch.setattr(run_ollama, "_make_backend", lambda cfg: _FakeBackend(["hi there"]))
    rc = run_ollama.main(["coder", "write hi", "--no-status", "--output-dir", str(tmp_path)])
    assert rc == 0
    raw = json.loads((tmp_path / "coder.raw.json").read_text(encoding="utf-8"))
    assert raw["content"] == "hi there"


def test_main_rejects_vision_as_actionable_nonzero_not_a_traceback(monkeypatch, capsys):
    # `dispatch` raises DelegationError for the MS1 vision/transcribe transport guard
    # (multimodal lands in M7). `main` must catch it — never an uncaught traceback — and
    # the backend must NEVER be invoked (the guard fires before any backend.run call).
    class _BackendMustNotBeCalled:
        def run(self, *args, **kwargs):
            raise AssertionError(
                "backend.run must not be invoked for an MS1-unsupported capability"
            )

    cfg = _cfg_with_structured()
    monkeypatch.setattr(run_ollama, "resolve_config", lambda **kw: cfg)
    # **kw absorbs preflight's capability=/effective_model= kwargs (R10/R28 fix) — the
    # stub only needs to no-op regardless of what run_delegation now threads through.
    monkeypatch.setattr(run_ollama, "preflight", lambda cfg, **kw: None)
    monkeypatch.setattr(run_ollama, "load_system_prompt", lambda cap: "sys")
    monkeypatch.setattr(run_ollama, "_make_backend", lambda cfg: _BackendMustNotBeCalled())
    rc = run_ollama.main(["vision", "img.png", "--no-status"])
    assert rc != 0
    assert "delegation failed" in capsys.readouterr().err.lower()


def test_main_handles_missing_agent_prompt_as_actionable_nonzero_not_a_traceback(
    monkeypatch,
    capsys,
):
    # `load_system_prompt` raises OllamaConfigError (a ValidationError subclass) when the
    # capability's agent prompt file is missing. `main` must catch it and print
    # actionably instead of propagating a traceback; the backend must never be reached
    # (the failure happens before `_make_backend`/`dispatch` are invoked).
    cfg = _cfg_with_structured()
    monkeypatch.setattr(run_ollama, "resolve_config", lambda **kw: cfg)
    # **kw absorbs preflight's capability=/effective_model= kwargs (R10/R28 fix) — the
    # stub only needs to no-op regardless of what run_delegation now threads through.
    monkeypatch.setattr(run_ollama, "preflight", lambda cfg, **kw: None)
    monkeypatch.setattr(run_ollama, "_AGENTS_DIR", "/nonexistent/agents/dir/xyz")
    monkeypatch.setattr(
        run_ollama,
        "_make_backend",
        lambda cfg: (_ for _ in ()).throw(AssertionError("must not delegate")),
    )
    rc = run_ollama.main(["coder", "write something", "--no-status"])
    assert rc != 0
    assert "delegation failed" in capsys.readouterr().err.lower()


def test_main_handles_unreadable_input_file_as_actionable_nonzero_not_a_traceback(
    monkeypatch,
    capsys,
):
    # `_load_input` treats an existing path as a file to read; if it "exists" per
    # `os.path.isfile` but `open()` fails (permission denied / vanished after the check —
    # a TOCTOU race), it raises a bare OSError. `main` must catch it (the plain `OSError`
    # arm, distinct from the domain `ValidationError` family) and print actionably rather
    # than crash with a traceback.
    ghost_path = "definitely-not-a-real-file-xyz.txt"
    real_isfile = run_ollama.os.path.isfile
    monkeypatch.setattr(
        run_ollama.os.path, "isfile", lambda p: True if p == ghost_path else real_isfile(p)
    )
    cfg = _cfg_with_structured()
    monkeypatch.setattr(run_ollama, "resolve_config", lambda **kw: cfg)
    # **kw absorbs preflight's capability=/effective_model= kwargs (R10/R28 fix) — the
    # stub only needs to no-op regardless of what run_delegation now threads through.
    monkeypatch.setattr(run_ollama, "preflight", lambda cfg, **kw: None)
    monkeypatch.setattr(run_ollama, "load_system_prompt", lambda cap: "sys")
    monkeypatch.setattr(
        run_ollama,
        "_make_backend",
        lambda cfg: (_ for _ in ()).throw(AssertionError("must not delegate")),
    )
    rc = run_ollama.main(["coder", ghost_path, "--no-status"])
    assert rc != 0
    assert "delegation failed" in capsys.readouterr().err.lower()


# --- Task 5: managed run-dir lifecycle + artifacts + interrupt cleanup ---


def _wire(monkeypatch, tmp_path, result, *, container=None):
    """Point the run-dir namespace at an isolated container and stub the pipeline."""
    container = container or str(tmp_path / "runs")
    os.makedirs(container, exist_ok=True)
    # run_ollama does `from temp_dirs import resolve_project_root, project_run_root`, so those
    # names live in run_ollama's namespace — patch THERE (where they're looked up), not temp_dirs.
    monkeypatch.setattr(run_ollama, "resolve_project_root", lambda start=None: str(tmp_path))
    monkeypatch.setattr(run_ollama, "project_run_root", lambda root: container)

    class _FakeBackend:
        def run(
            self,
            capability,
            system_prompt,
            prompt,
            model,
            timeout,
            *,
            response_format=None,
            deadline=None,
        ):
            return result

    monkeypatch.setattr(run_ollama, "_make_backend", lambda cfg: _FakeBackend())
    # **kw absorbs preflight's capability=/effective_model= kwargs (R10/R28 fix, already
    # exercised elsewhere in this file) — the stub only needs to no-op regardless of what
    # run_delegation threads through.
    monkeypatch.setattr(run_ollama, "preflight", lambda cfg, **kw: None)
    monkeypatch.setattr(run_ollama, "load_system_prompt", lambda cap: "sys")
    return container


def _run_dir(container):
    dirs = [
        os.path.join(container, x) for x in os.listdir(container) if x.startswith("ollama-run-")
    ]
    assert dirs, "no run dir created"
    return dirs[0]


def _delegate(argv):
    """Parse *argv* and drive ``run_delegation`` directly, bypassing ``main()``'s
    exception-to-exit-code conversion.

    ``main()`` deliberately catches the full domain-error family (incl.
    ``OllamaBackendError``/``OSError``, the latter covering ``TimeoutError``) and turns
    each into an actionable stderr message plus a non-zero exit code — that CLI-facing
    contract is established, tested MS1 behavior (``test_main_aborts_when_preflight_fails``
    et al.) and is not part of Task 5's scope. A few of Task 5's own tests need the RAW
    exception itself (to assert a side effect of ``run_delegation``/``managed_run_dir``'s
    exception handling — the status-display state update, or the run dir being
    retained/rmtree'd) rather than main()'s converted exit code, so those drive
    ``run_delegation`` directly instead of going through ``main()``.
    """
    parser = run_ollama.build_parser()
    ns = parser.parse_args(argv)
    run_ollama._validate_args(parser, ns)
    return run_ollama.run_delegation(ns)


def test_managed_run_writes_full_artifacts_and_removes_lock(tmp_path, monkeypatch):
    container = _wire(monkeypatch, tmp_path, DelegationResult("hi", 3, 2, False, 0.1))
    assert run_ollama.main(["coder", "write hi", "--no-status"]) == 0
    d = _run_dir(container)
    for name in ("coder.prompt.txt", "coder.raw.json", "token_stats.json", "ollama-report.json"):
        assert os.path.exists(os.path.join(d, name)), name
    assert not os.path.exists(os.path.join(d, ".ollama-lock"))  # removed on success


def test_ollama_report_has_r18_telemetry_fields(tmp_path, monkeypatch):
    container = _wire(monkeypatch, tmp_path, DelegationResult("hi", 3, 2, False, 0.1))
    assert run_ollama.main(["coder", "write hi", "--no-status"]) == 0
    report = json.loads(
        open(os.path.join(_run_dir(container), "ollama-report.json"), encoding="utf-8").read()
    )
    for field in ("tokens_by_model", "input_size", "timings", "retried", "guard"):
        assert field in report, field


def test_stderr_log_written_even_with_no_status(tmp_path, monkeypatch):
    # R18: {cap}.stderr.log is persisted regardless of the live display (--no-status).
    container = _wire(monkeypatch, tmp_path, DelegationResult("hi", 3, 2, False, 0.1))
    assert run_ollama.main(["coder", "write hi", "--no-status"]) == 0
    assert os.path.exists(os.path.join(_run_dir(container), "coder.stderr.log"))


def test_structured_delegation_writes_parsed_json(tmp_path, monkeypatch):
    res = DelegationResult(
        '{"capability":"reviewer","findings":[]}',
        3,
        2,
        False,
        0.1,
        parsed={"capability": "reviewer", "findings": []},
    )
    container = _wire(monkeypatch, tmp_path, res)
    assert run_ollama.main(["reviewer", "review x", "--no-status"]) == 0
    assert os.path.exists(os.path.join(_run_dir(container), "reviewer.parsed.json"))


def test_write_artifacts_survives_one_file_oserror(tmp_path, monkeypatch):
    # A disk error on one artifact must not crash the run nor block the rest (R18).
    # The patched `open` intercepts ONLY the one target filename; every other call
    # (including any the test harness/pytest itself makes during `main()`) is passed
    # straight through to the real `open` — narrower than a blanket global patch.
    container = _wire(monkeypatch, tmp_path, DelegationResult("hi", 3, 2, False, 0.1))
    real_open = open

    def _flaky_open(path, *a, **k):
        try:
            is_target = os.path.basename(os.fspath(path)) == "coder.raw.json"
        except TypeError:
            is_target = False  # non-path argument (e.g. an fd) — never our target
        if is_target:
            raise OSError("disk full (simulated)")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", _flaky_open)
    assert run_ollama.main(["coder", "write hi", "--no-status"]) == 0  # no exception propagates
    d = _run_dir(container)
    assert not os.path.exists(os.path.join(d, "coder.raw.json"))  # the flaky one is skipped
    assert os.path.exists(os.path.join(d, "coder.prompt.txt"))  # others still written
    assert os.path.exists(os.path.join(d, "ollama-report.json"))


def test_write_artifacts_survives_token_stats_write_oserror(tmp_path, monkeypatch):
    # `stats.write(output_dir)` (token_stats.json) must not break `_write_artifacts`'s
    # never-raise contract either — a local try/except guards it as defense-in-depth,
    # independent of MS2's `TokenStats.write` already returning `str | None` internally.
    from token_stats import TokenStats

    container = _wire(monkeypatch, tmp_path, DelegationResult("hi", 3, 2, False, 0.1))

    def _raising_write(self, output_dir):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(TokenStats, "write", _raising_write)
    assert run_ollama.main(["coder", "write hi", "--no-status"]) == 0  # no exception propagates
    d = _run_dir(container)
    assert not os.path.exists(os.path.join(d, "token_stats.json"))  # the flaky one is skipped
    assert os.path.exists(os.path.join(d, "coder.prompt.txt"))  # others still written
    assert os.path.exists(os.path.join(d, "ollama-report.json"))


def test_write_artifacts_survives_non_serializable_stats_dict(tmp_path, monkeypatch):
    # The DEFINITIVE belt (finding 1, round 3; narrowed in round 4): a value inside
    # stats.to_dict() that json.dumps rejects (e.g. a bare object()) raises a TypeError
    # from the report serialization step, and must not propagate out of
    # _write_artifacts at all. The outer `except (OSError, TypeError, ValueError,
    # RecursionError)` around the whole function body catches that TypeError
    # specifically, warns, and returns; artifacts already written before the crash
    # point (prompt/raw/token_stats) stay on disk, but ollama-report.json itself
    # never lands.
    from token_stats import TokenStats

    container = _wire(monkeypatch, tmp_path, DelegationResult("hi", 3, 2, False, 0.1))
    monkeypatch.setattr(
        TokenStats,
        "to_dict",
        lambda self: {"coder": {"some-model": object()}},  # object() is not JSON-serializable
    )
    assert run_ollama.main(["coder", "write hi", "--no-status"]) == 0  # no exception propagates
    d = _run_dir(container)
    assert os.path.exists(os.path.join(d, "coder.prompt.txt"))  # written before the crash
    assert os.path.exists(os.path.join(d, "coder.raw.json"))  # written before the crash
    assert not os.path.exists(os.path.join(d, "ollama-report.json"))  # report never landed


def test_retried_derivation_degrades_gracefully_on_malformed_stats_dict(tmp_path, monkeypatch):
    # `retried` must not be derived by blindly trusting TokenStats.to_dict()'s exact shape
    # (MS3-local structural guard — MS2's plan/contract is untouched). A malformed/empty
    # result degrades to `retried == []` instead of raising KeyError/TypeError.
    from token_stats import TokenStats

    container = _wire(monkeypatch, tmp_path, DelegationResult("hi", 3, 2, False, 0.1))
    monkeypatch.setattr(TokenStats, "to_dict", lambda self: {"coder": "not-a-dict"})
    assert run_ollama.main(["coder", "write hi", "--no-status"]) == 0
    report = json.loads(
        open(os.path.join(_run_dir(container), "ollama-report.json"), encoding="utf-8").read()
    )
    assert report["retried"] == []


def test_retried_derivation_flags_capability_with_more_http_calls_than_delegations(
    tmp_path, monkeypatch
):
    # Normal, well-shaped to_dict() result: a capability whose http_calls exceed its
    # delegations (a parse/schema retry, R25) appears in `retried`.
    from token_stats import TokenStats

    container = _wire(monkeypatch, tmp_path, DelegationResult("hi", 3, 2, False, 0.1))
    monkeypatch.setattr(
        TokenStats,
        "to_dict",
        lambda self: {"coder": {"kimi-k2.7-code:cloud": {"http_calls": 2, "delegations": 1}}},
    )
    assert run_ollama.main(["coder", "write hi", "--no-status"]) == 0
    report = json.loads(
        open(os.path.join(_run_dir(container), "ollama-report.json"), encoding="utf-8").read()
    )
    assert report["retried"] == ["coder"]


def test_interrupt_rmtrees_dir_but_plain_exception_keeps_it(tmp_path, monkeypatch):
    from errors import OllamaBackendError

    container = _wire(monkeypatch, tmp_path, DelegationResult("x", 1, 1, False, 0.1))

    monkeypatch.setattr(
        run_ollama, "dispatch", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        run_ollama.main(["coder", "x", "--no-status"])
    assert not [x for x in os.listdir(container) if x.startswith("ollama-run-")]  # R27: cleaned

    monkeypatch.setattr(
        run_ollama, "dispatch", lambda *a, **k: (_ for _ in ()).throw(OllamaBackendError("boom"))
    )
    # Drive run_delegation directly: main()'s comprehensive except clause (established MS1
    # behavior) would otherwise catch OllamaBackendError and convert it to a non-zero exit
    # code — this test needs the raw exception to observe managed_run_dir's retain-on-
    # exception side effect, not main()'s CLI-facing conversion (see _delegate's docstring).
    with pytest.raises(OllamaBackendError):
        _delegate(["coder", "x", "--no-status"])
    assert [x for x in os.listdir(container) if x.startswith("ollama-run-")]  # kept for debug


def test_status_display_stop_called_even_on_interrupt(tmp_path, monkeypatch):
    # R20: the live display is restored (stop()) even when the run is interrupted.
    _wire(monkeypatch, tmp_path, DelegationResult("x", 1, 1, False, 0.1))
    stopped = {"n": 0}

    class _SpyDisplay:
        def __init__(self, agents, **kw):
            pass

        def update(self, *a, **k):
            pass

        def stop(self):
            stopped["n"] += 1

    monkeypatch.setattr(run_ollama, "StatusDisplay", _SpyDisplay)
    monkeypatch.setattr(
        run_ollama, "dispatch", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        run_ollama.main(["coder", "x"])  # display active (no --no-status)
    assert stopped["n"] == 1  # stop() ran in the finally despite the interrupt


def test_failed_delegation_marks_display_failed(tmp_path, monkeypatch):
    # R20: a delegation that raises must mark its row "failed" in the live status
    # tree before the exception propagates — never left frozen on "running".
    from errors import OllamaBackendError

    _wire(monkeypatch, tmp_path, DelegationResult("x", 1, 1, False, 0.1))
    updates: list[tuple[str, str]] = []

    class _SpyDisplay:
        def __init__(self, agents, **kw):
            pass

        def update(self, agent, state, tok_per_s=None):
            updates.append((agent, state))

        def stop(self):
            pass

    monkeypatch.setattr(run_ollama, "StatusDisplay", _SpyDisplay)
    monkeypatch.setattr(
        run_ollama, "dispatch", lambda *a, **k: (_ for _ in ()).throw(OllamaBackendError("boom"))
    )
    # Drive run_delegation directly (see _delegate's docstring): main()'s established
    # except clause would otherwise convert this into a non-zero exit code before this
    # test can observe the raw exception.
    with pytest.raises(OllamaBackendError):
        _delegate(["coder", "x"])  # display active (no --no-status)
    assert ("coder", "failed") in updates


def test_timeout_delegation_marks_display_timeout(tmp_path, monkeypatch):
    # TimeoutError is cleanly distinguishable from a generic failure — it gets its
    # own "timeout" state (R20's state set) instead of the generic "failed".
    _wire(monkeypatch, tmp_path, DelegationResult("x", 1, 1, False, 0.1))
    updates: list[tuple[str, str]] = []

    class _SpyDisplay:
        def __init__(self, agents, **kw):
            pass

        def update(self, agent, state, tok_per_s=None):
            updates.append((agent, state))

        def stop(self):
            pass

    monkeypatch.setattr(run_ollama, "StatusDisplay", _SpyDisplay)
    monkeypatch.setattr(
        run_ollama, "dispatch", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timed out"))
    )
    # Drive run_delegation directly (see _delegate's docstring): main()'s established
    # except clause catches OSError (TimeoutError's parent) and would otherwise convert
    # this into a non-zero exit code before this test can observe the raw exception.
    with pytest.raises(TimeoutError):
        _delegate(["coder", "x"])
    assert ("coder", "timeout") in updates
    assert ("coder", "failed") not in updates


def test_preflight_warning_captured_in_stderr_log(tmp_path, monkeypatch):
    # R18: preflight runs INSIDE the shim, so its warn-and-proceed output lands in stderr.log.
    import sys as _sys

    container = _wire(monkeypatch, tmp_path, DelegationResult("x", 1, 1, False, 0.1))
    monkeypatch.setattr(
        run_ollama,
        "preflight",
        lambda cfg, **kw: print("WARNING: /models returned 404; skipping check", file=_sys.stderr),
    )
    assert run_ollama.main(["coder", "x", "--no-status"]) == 0
    log = open(os.path.join(_run_dir(container), "coder.stderr.log"), encoding="utf-8").read()
    assert "404" in log


def test_keep_runs_zero_is_rejected_via_main_with_message(capsys):
    # Distinct from the earlier `test_keep_runs_zero_is_rejected` (which exercises
    # `_validate_args` directly): this drives the rejection through the full `main()`
    # entry point and asserts the actionable message reaches stderr.
    with pytest.raises(SystemExit):
        run_ollama.main(["coder", "x", "--keep-runs", "0"])
    assert "keep-runs" in capsys.readouterr().err.lower()


def test_gettempdir_fallback_skips_cross_project_cleanup(tmp_path, monkeypatch):
    # When the per-project container can't be made, project_run_root degrades to the
    # SHARED gettempdir → cleanup must be skipped (never prune other projects' runs).
    monkeypatch.setattr(run_ollama, "resolve_project_root", lambda start=None: str(tmp_path))
    monkeypatch.setattr(run_ollama, "project_run_root", lambda root: tempfile.gettempdir())
    calls = {"cleanup": 0}
    monkeypatch.setattr(
        run_ollama,
        "cleanup_old_runs",
        lambda *a, **k: calls.__setitem__("cleanup", calls["cleanup"] + 1),
    )
    monkeypatch.setattr(
        run_ollama,
        "_make_backend",
        lambda cfg: type(
            "B",
            (),
            {"run": staticmethod(lambda *a, **k: DelegationResult("x", 1, 1, False, 0.1))},
        )(),
    )
    monkeypatch.setattr(run_ollama, "preflight", lambda cfg, **kw: None)
    monkeypatch.setattr(run_ollama, "load_system_prompt", lambda cap: "sys")
    run_ollama.main(["coder", "x", "--no-status"])
    assert calls["cleanup"] == 0  # no cross-project prune in the gettempdir fallback


# --- managed_run_dir (extracted lifecycle context manager) — unit-level, direct ---
# These exercise `managed_run_dir` on its own, one level below the `main()`-driven
# integration tests above (which still cover the same three outcomes end-to-end via
# `test_managed_run_writes_full_artifacts_and_removes_lock` and
# `test_interrupt_rmtrees_dir_but_plain_exception_keeps_it`). The assertions here are
# the same ones those tests already make (lock present/absent, dir present/absent) —
# just targeted at the context manager directly instead of through the whole CLI.


def _wire_namespace(monkeypatch, tmp_path):
    """Point the run-dir namespace at an isolated, per-test container."""
    container = str(tmp_path / "runs")
    os.makedirs(container, exist_ok=True)
    monkeypatch.setattr(run_ollama, "resolve_project_root", lambda start=None: str(tmp_path))
    monkeypatch.setattr(run_ollama, "project_run_root", lambda root: container)
    return container


def test_managed_run_dir_removes_lock_on_success(tmp_path, monkeypatch):
    _wire_namespace(monkeypatch, tmp_path)
    with run_ollama.managed_run_dir(keep_runs=5, timeout=30) as output_dir:
        assert os.path.exists(os.path.join(output_dir, ".ollama-lock"))
    assert os.path.exists(output_dir)  # dir itself stays
    assert not os.path.exists(os.path.join(output_dir, ".ollama-lock"))  # lock removed on success


def test_managed_run_dir_rmtrees_on_interrupt(tmp_path, monkeypatch):
    _wire_namespace(monkeypatch, tmp_path)
    captured: dict[str, str] = {}
    with pytest.raises(KeyboardInterrupt):
        with run_ollama.managed_run_dir(keep_runs=5, timeout=30) as output_dir:
            captured["dir"] = output_dir
            raise KeyboardInterrupt()
    assert not os.path.exists(captured["dir"])  # R27: rmtree'd, no orphaned ollama-run-*


def test_managed_run_dir_rmtrees_on_generator_exit(tmp_path, monkeypatch):
    # GeneratorExit is a BaseException, same family as KeyboardInterrupt/SystemExit
    # (R27): an abandoned/GC'd generator means the run is being discarded, so it must
    # get the same rmtree cleanup, not the retain-for-debug path. `cm.gen.close()`
    # (contextlib's underlying generator) is exactly how a real abandonment throws
    # GeneratorExit at the suspended `yield`.
    _wire_namespace(monkeypatch, tmp_path)
    cm = run_ollama.managed_run_dir(keep_runs=5, timeout=30)
    output_dir = cm.__enter__()
    cm.gen.close()  # simulates generator abandonment -> GeneratorExit at the yield
    assert not os.path.exists(output_dir)  # R27: rmtree'd, same as an interrupt


def test_managed_run_dir_retains_dir_and_lock_on_plain_exception(tmp_path, monkeypatch):
    _wire_namespace(monkeypatch, tmp_path)
    captured: dict[str, str] = {}
    with pytest.raises(ValueError):
        with run_ollama.managed_run_dir(keep_runs=5, timeout=30) as output_dir:
            captured["dir"] = output_dir
            raise ValueError("boom")
    assert os.path.exists(captured["dir"])  # retained for debug
    assert os.path.exists(os.path.join(captured["dir"], ".ollama-lock"))  # lock retained too


def test_managed_run_dir_with_explicit_output_dir_skips_lock_and_prune(tmp_path, monkeypatch):
    # --output-dir is caller-managed: no lock, no pruning, nothing removed on any exit path.
    calls = {"cleanup": 0}
    monkeypatch.setattr(
        run_ollama,
        "cleanup_old_runs",
        lambda *a, **k: calls.__setitem__("cleanup", calls["cleanup"] + 1),
    )
    caller_dir = str(tmp_path / "caller-owned")
    with run_ollama.managed_run_dir(keep_runs=5, timeout=30, output_dir=caller_dir) as output_dir:
        assert output_dir == caller_dir
        assert not os.path.exists(os.path.join(output_dir, ".ollama-lock"))
    assert os.path.exists(output_dir)  # never removed by managed_run_dir
    assert calls["cleanup"] == 0  # a caller-managed dir is never pruned
