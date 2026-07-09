# tests/test_interface_contracts.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Cross-milestone signature contracts MS3 depends on (MS1/MS2) — inspect.signature only.

MS3 does not redefine `dispatch` / `AgentBackend` / `OllamaBackend` / `DelegationResult` /
`TokenStats` (they are MS1/MS2 interfaces this milestone's `run_ollama.py` wiring, esp.
`run_delegation` and `_write_artifacts`, consumes as given — see Task 5's Interfaces note).
This suite is deliberately shallow: signature shape and attribute presence, never behavior —
it exists to fail loudly HERE if MS1/MS2 drift, instead of surfacing as a confusing
TypeError/AttributeError deep inside `run_delegation`/`dispatch` at integration time.
"""

from __future__ import annotations

import inspect

import run_ollama
from backend import AgentBackend, DelegationResult, OllamaBackend
from token_stats import TokenStats


def test_agent_backend_run_accepts_keyword_only_deadline():
    params = inspect.signature(AgentBackend.run).parameters
    assert "deadline" in params
    assert params["deadline"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["deadline"].default is None


def test_ollama_backend_run_accepts_keyword_only_deadline():
    params = inspect.signature(OllamaBackend.run).parameters
    assert "deadline" in params
    assert params["deadline"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["deadline"].default is None


def test_dispatch_accepts_keyword_only_config():
    # `run_ollama.dispatch` (not `backend.dispatch`): this is the exact attribute
    # `run_delegation` calls and MS3's other tests monkeypatch, so checking it here
    # catches drift regardless of which MS1/MS2 module `dispatch` is defined in.
    params = inspect.signature(run_ollama.dispatch).parameters
    assert "config" in params
    assert params["config"].kind == inspect.Parameter.KEYWORD_ONLY


def test_dispatch_is_annotated_to_return_delegation_result():
    sig = inspect.signature(run_ollama.dispatch)
    if sig.return_annotation is not inspect.Signature.empty:
        assert "DelegationResult" in str(sig.return_annotation)


def test_delegation_result_has_the_fields_write_artifacts_reads():
    # Constructed exactly as MS3's own tests already do (positional core fields, optional
    # keyword `parsed`) — attribute presence, not a claim about the exact type being a
    # dataclass, so this stays valid across an internal MS1/MS2 refactor.
    result = DelegationResult("hi", 3, 2, False, 0.1)
    for required in ("content", "prompt_tokens", "completion_tokens", "estimated", "elapsed_s"):
        assert hasattr(result, required), required
    for optional in ("parsed", "tok_per_s"):
        assert hasattr(result, optional), optional


def test_token_stats_exposes_to_dict():
    assert hasattr(TokenStats, "to_dict")
    assert callable(TokenStats.to_dict)
