# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""CLI orchestrator: argparse surface, --ollama-init short-circuit, single dispatch."""

import os
from dataclasses import replace
from types import MappingProxyType

import pytest

import run_ollama
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
        return item


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
    assert out == "def f(): pass"
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
    assert out == "free-form review text"  # off: content verbatim
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
    assert out == '{"any": "json"}'  # object: content verbatim (no validate)
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
    assert out["findings"][0]["severity"] == "info"
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


def test_main_runs_full_pipeline_and_prints_output(capsys, monkeypatch):
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


def test_main_model_override_reaches_backend(monkeypatch):
    # R28: --model overrides the config default and reaches backend.run as `model`.
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
            return "ok"

    cfg = _cfg_with_structured()
    monkeypatch.setattr(run_ollama, "resolve_config", lambda **kw: cfg)
    # **kw absorbs preflight's capability=/effective_model= kwargs (R10/R28 fix) — the
    # stub only needs to no-op regardless of what run_delegation now threads through.
    monkeypatch.setattr(run_ollama, "preflight", lambda cfg, **kw: None)
    monkeypatch.setattr(run_ollama, "load_system_prompt", lambda cap: "sys")
    monkeypatch.setattr(run_ollama, "_make_backend", lambda cfg: _CapBackend())
    run_ollama.main(["coder", "write hi", "--no-status", "--model", "custom-model:cloud"])
    assert seen["model"] == "custom-model:cloud"  # override wins over cfg.models["coder"]


def test_main_threads_the_model_override_into_preflight_not_just_the_backend(monkeypatch):
    # R10/R28 fix: a --model override must reach preflight's model-existence check, not
    # only backend.run — otherwise an override to a nonexistent model would silently
    # bypass preflight and only surface as a chat-time 404. This asserts preflight is
    # called with the OVERRIDE (as effective_model, for capability="coder"), never the
    # stale config default.
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


def test_output_dir_writes_raw_artifact(tmp_path, monkeypatch):
    # R28: --output-dir persists the raw output to a caller-owned dir (cleanup/lock is MS3).
    cfg = _cfg_with_structured()  # real config; coder→"off" free-text → content written verbatim
    monkeypatch.setattr(run_ollama, "resolve_config", lambda **kw: cfg)
    # **kw absorbs preflight's capability=/effective_model= kwargs (R10/R28 fix) — the
    # stub only needs to no-op regardless of what run_delegation now threads through.
    monkeypatch.setattr(run_ollama, "preflight", lambda cfg, **kw: None)
    monkeypatch.setattr(run_ollama, "load_system_prompt", lambda cap: "sys")
    monkeypatch.setattr(run_ollama, "_make_backend", lambda cfg: _FakeBackend(["hi there"]))
    rc = run_ollama.main(["coder", "write hi", "--no-status", "--output-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "coder.raw.json").read_text(encoding="utf-8") == "hi there"


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
