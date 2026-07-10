# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Tolerant extractor: fences, <think> recovery, fail-closed on ambiguity, DoS bounds."""

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from parse_output import parse_agent_output, strip_think

_KEYS = ("agent", "verdict")


def test_clean_json_parses():
    assert parse_agent_output('{"agent": "reviewer", "verdict": "ok"}', _KEYS)["verdict"] == "ok"


def test_code_fence_is_stripped():
    raw = '```json\n{"agent": "reviewer", "verdict": "ok"}\n```'
    assert parse_agent_output(raw, _KEYS)["agent"] == "reviewer"


def test_code_fence_not_at_string_start_is_stripped():
    # A ```json fence that begins on a later LINE (not char 0) must still be recognized —
    # the fence regex is compiled with re.MULTILINE so ^```lang anchors to any line start.
    raw = 'Result below:\n```json\n{"agent": "reviewer", "verdict": "ok"}\n```'
    assert parse_agent_output(raw, _KEYS)["verdict"] == "ok"


def test_think_leak_is_recovered():
    raw = '<think>let me reason...</think>\n{"agent": "reviewer", "verdict": "ok"}'
    assert parse_agent_output(raw, _KEYS)["verdict"] == "ok"


def test_ambiguous_two_objects_is_fail_closed():
    raw = '{"agent": "a", "verdict": "x"} and also {"agent": "b", "verdict": "y"}'
    with pytest.raises(json.JSONDecodeError):
        parse_agent_output(raw, _KEYS)  # ≥2 discriminator-matching objects → not parseable


def test_object_without_discriminator_keys_is_ignored():
    raw = '{"unrelated": 1}\n{"agent": "a", "verdict": "x"}'
    assert parse_agent_output(raw, _KEYS)["agent"] == "a"  # only the real one qualifies


def test_blob_over_lenient_recovery_max_chars_is_fail_closed():
    # No clean top-level parse, and the blob exceeds LENIENT_RECOVERY_MAX_CHARS,
    # so the slow brace-scanning recovery must be skipped entirely (anti-DoS bound).
    from parse_output import LENIENT_RECOVERY_MAX_CHARS

    raw = "x" * (LENIENT_RECOVERY_MAX_CHARS + 1) + '{"agent": "a", "verdict": "x"}'
    with pytest.raises(json.JSONDecodeError):
        parse_agent_output(raw, _KEYS)


def test_brace_probes_are_capped():
    # More than MAX_BRACE_PROBES non-decodable '{' occurrences before a real object —
    # the scan must bail out via the probe cap rather than examine every one, and the
    # qualifying object beyond the cap is never found (fail-closed, not slow-but-correct).
    from parse_output import MAX_BRACE_PROBES

    noise = "{ " * (MAX_BRACE_PROBES + 10)
    raw = noise + '{"agent": "a", "verdict": "x"}'
    with pytest.raises(json.JSONDecodeError):
        parse_agent_output(raw, _KEYS)


def test_deep_nesting_recursion_error_is_mapped_to_json_decode_error():
    # A pathologically deep nested-array literal blows Python's json recursion limit;
    # RecursionError must be caught and mapped to a handled JSONDecodeError, never escape.
    deep = "[" * 100_000 + "]" * 100_000
    with pytest.raises(json.JSONDecodeError):
        parse_agent_output(deep, _KEYS)


def test_non_dict_top_level_is_ignored_not_crashed():
    # A bare JSON array (or scalar) at the top level does not qualify as a discriminator
    # match; recovery finds nothing and fails closed rather than raising a TypeError.
    raw = "[1, 2, 3]"
    with pytest.raises(json.JSONDecodeError):
        parse_agent_output(raw, _KEYS)


@given(st.text(max_size=500))
def test_parse_never_crashes_only_raises_jsondecodeerror(s):
    try:
        parse_agent_output(s, _KEYS)
    except json.JSONDecodeError:
        pass  # the only allowed failure mode


# --- Task 6: thinking capability — <think> conclusion recovery (BDD-21) ---


def test_strip_think_yields_the_conclusion():
    raw = "<think>weighing the trade-offs, a heap vs a sorted list...</think>\nUse a heap."
    assert strip_think(raw) == "Use a heap."


def test_strip_think_handles_multiple_blocks_and_is_case_insensitive():
    raw = "<THINK>a</THINK>keep1<think>b</think>keep2"
    assert strip_think(raw) == "keep1keep2"


def test_strip_think_without_a_block_is_identity():
    assert strip_think("just an answer") == "just an answer"
