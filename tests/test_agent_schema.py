# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Per-capability structured schema + validator, in lockstep (bidirectional)."""

import pytest

from agent_schema import DISCRIMINATOR_KEYS, SCHEMAS
from errors import ValidationError
from validate import clean_title, validate_output


def test_reviewer_schema_and_discriminators_exist():
    assert "reviewer" in SCHEMAS and "tester" in SCHEMAS
    assert DISCRIMINATOR_KEYS["reviewer"] == ("capability", "findings")


def test_valid_reviewer_output_passes():
    obj = {
        "capability": "reviewer",
        "findings": [{"severity": "warning", "title": "x", "detail": "y"}],
    }
    assert validate_output("reviewer", obj)["findings"][0]["severity"] == "warning"


def test_reviewer_output_missing_key_fails():
    with pytest.raises(ValidationError):
        validate_output("reviewer", {"capability": "reviewer"})  # no findings


def test_reviewer_bad_severity_enum_fails():
    obj = {
        "capability": "reviewer",
        "findings": [{"severity": "nope", "title": "x", "detail": "y"}],
    }
    with pytest.raises(ValidationError):
        validate_output("reviewer", obj)


def test_clean_title_strips_zero_width_bidi_but_keeps_hyphen():
    zwsp, zwnj, rlo, bom = chr(0x200B), chr(0x200C), chr(0x202E), chr(0xFEFF)
    dirty = "a" + zwsp + "b" + zwnj + rlo + "c" + bom + "-d"
    assert clean_title(dirty) == "abc-d"  # invisibles stripped, hyphen preserved
    with pytest.raises(ValidationError):
        clean_title(zwsp + "   " + zwsp)  # only invisibles + spaces → empty → reject


# Accept/reject corpora per structured capability. Lockstep property (bidirectional):
# the domain validator accepts exactly the objects a strict JSON-Schema check would —
# schema-valid <=> validator-accepts, in BOTH directions.
_CORPUS = {
    "reviewer": {
        "accept": [
            {"capability": "reviewer", "findings": []},
            {
                "capability": "reviewer",
                "findings": [{"severity": "info", "title": "t", "detail": "d"}],
            },
        ],
        "reject": [
            {"capability": "reviewer"},  # no findings
            {
                "capability": "reviewer",
                "findings": [{"severity": "nope", "title": "t", "detail": "d"}],
            },  # bad enum
            {
                "capability": "reviewer",
                "findings": [{"severity": "info", "title": "t"}],
            },  # missing detail
            {"capability": "reviewer", "findings": [], "extra": 1},  # extra top-level key
            {
                "capability": "reviewer",
                "findings": [{"severity": "info", "title": "t", "detail": "d", "x": 1}],
            },  # extra item key
            {"capability": "coder", "findings": []},  # wrong capability const
            {"capability": "reviewer", "findings": {"severity": "info"}},  # findings not a list
            {
                "capability": "reviewer",
                "findings": [{"severity": "info", "title": 123, "detail": "d"}],
            },  # title not a string
            {
                "capability": "reviewer",
                "findings": [{"severity": "info", "title": "t", "detail": ["x"]}],
            },  # detail not a string
            {
                "capability": "reviewer",
                "findings": [{"severity": "info", "title": "t", "detail": "d", "line": 0}],
            },  # line below the 1-based minimum (schema minimum: 1)
            {
                "capability": "reviewer",
                "findings": [{"severity": "info", "title": "t", "detail": "d", "line": -3}],
            },  # negative line number
        ],
    },
    "tester": {
        "accept": [
            {"capability": "tester", "tests": []},
            {"capability": "tester", "tests": [{"name": "t", "code": "assert True"}]},
        ],
        "reject": [
            {"capability": "tester"},  # no tests
            {"capability": "tester", "tests": [{"name": "t"}]},  # missing code
            {"capability": "tester", "tests": [], "bogus": 1},  # extra top-level key
            {
                "capability": "tester",
                "tests": [{"name": "t", "code": "c", "z": 1}],
            },  # extra item key
            {"capability": "coder", "tests": []},  # wrong capability const
            {"capability": "tester", "tests": "not-a-list"},  # tests not a list
            {"capability": "tester", "tests": [{"name": 1, "code": "c"}]},  # name not a string
            {"capability": "tester", "tests": [{"name": "t", "code": None}]},  # code not a string
        ],
    },
}


def test_lockstep_schema_and_validator_agree_bidirectionally():
    # Every structured capability has both a schema and a corpus; the validator accepts
    # exactly the schema-valid objects and rejects the schema-invalid ones (both ways).
    assert set(SCHEMAS) == set(_CORPUS)
    for cap, corpus in _CORPUS.items():
        for ok in corpus["accept"]:
            validate_output(cap, ok)  # schema-valid  => accepted
        for bad in corpus["reject"]:
            with pytest.raises(ValidationError):
                validate_output(cap, bad)  # schema-invalid => rejected


def test_schemas_and_discriminator_keys_are_in_lockstep():
    # The two constant maps must never drift apart: identical structured-capability sets.
    assert set(SCHEMAS) == set(DISCRIMINATOR_KEYS) == {"reviewer", "tester"}


def test_non_structured_capability_has_no_validator_branch():
    with pytest.raises(ValidationError):
        validate_output("coder", {"anything": 1})  # coder is free-text, not structured


@pytest.mark.parametrize("non_dict", [["not", "a", "dict"], None, "a string", 42])
def test_reviewer_non_dict_top_level_raises_validation_error(non_dict):
    # R23 fail-closed branch: a non-dict top-level must raise ValidationError (the
    # domain exception), never an unhandled AttributeError/TypeError from downstream
    # dict-only operations like `.keys()` or `in obj`.
    with pytest.raises(ValidationError):
        validate_output("reviewer", non_dict)


@pytest.mark.parametrize("non_dict", [["not", "a", "dict"], None, "a string", 42])
def test_tester_non_dict_top_level_raises_validation_error(non_dict):
    with pytest.raises(ValidationError):
        validate_output("tester", non_dict)


def test_invisibles_stripped_from_all_structured_string_fields():
    # R23: EVERY structured string field is sanitized, not just reviewer.title.
    zwsp, rlo, bom = chr(0x200B), chr(0x202E), chr(0xFEFF)
    rev = validate_output(
        "reviewer",
        {
            "capability": "reviewer",
            "findings": [
                {
                    "severity": "info",
                    "title": "t" + zwsp + "-1",
                    "detail": "de" + rlo + "tail" + bom,
                }
            ],
        },
    )
    f = rev["findings"][0]
    assert f["title"] == "t-1"  # title cleaned, hyphen survives
    assert f["detail"] == "detail"  # invisibles stripped from detail too
    tst = validate_output(
        "tester",
        {
            "capability": "tester",
            "tests": [{"name": "test" + zwsp + "-a", "code": "line1\nli" + zwsp + "ne2"}],
        },
    )
    t = tst["tests"][0]
    assert t["name"] == "test-a"  # name cleaned, hyphen survives
    assert t["code"] == "line1\nline2"  # invisibles stripped, newline PRESERVED


def test_bidi_isolation_marks_stripped_from_structured_output():
    # R23 (Trojan Source, CVE-2021-42574 class): the bidi ISOLATE marks LRI/RLI/FSI/PDI
    # (U+2066-U+2069) must be stripped from untrusted structured output, not just the
    # zero-width/BOM/embedding chars already covered. Use the ACTUAL code points (not a
    # description of them) embedded in both a free-text field (tester.code, newline-bearing)
    # and an identity field (reviewer.title, single-line) to prove both _strip_invisibles
    # and clean_title cover the isolate range.
    lri, rli, fsi, pdi = chr(0x2066), chr(0x2067), chr(0x2068), chr(0x2069)

    tst = validate_output(
        "tester",
        {
            "capability": "tester",
            "tests": [
                {
                    "name": "t",
                    "code": "line1" + lri + "-hidden" + pdi + "\nline2" + rli + fsi + "end",
                }
            ],
        },
    )
    code = tst["tests"][0]["code"]
    assert code == "line1-hidden\nline2end"  # isolates gone; hyphen + newline survive intact
    assert not any(c in code for c in (lri, rli, fsi, pdi))

    rev = validate_output(
        "reviewer",
        {
            "capability": "reviewer",
            "findings": [{"severity": "info", "title": lri + "t-1" + pdi, "detail": "d"}],
        },
    )
    title = rev["findings"][0]["title"]
    assert title == "t-1"  # clean_title strips isolates, keeps the hyphen
    assert not any(c in title for c in (lri, rli, fsi, pdi))


def test_validate_output_does_not_mutate_its_input():
    # The caller's dict (and its nested items) must be untouched — a NEW object is returned.
    zwsp = chr(0x200B)
    src = {
        "capability": "reviewer",
        "findings": [{"severity": "info", "title": "t" + zwsp, "detail": "d" + zwsp}],
    }
    out = validate_output("reviewer", src)
    assert src["findings"][0]["title"] == "t" + zwsp  # input UNCHANGED (still has zwsp)
    assert out["findings"][0]["title"] == "t"  # returned object is cleaned
    assert out is not src and out["findings"] is not src["findings"]


# --- MS7 Task 7: reviewer findings can optionally carry a location claim (`file`/`line`),
# so R30's diff_guard has something to ground. Additive, backward-compatible: not added
# to "required", so MS1's original finding shape (no location claim) keeps validating.


def test_reviewer_finding_optional_file_and_line_validate_when_present():
    obj = {
        "capability": "reviewer",
        "findings": [
            {"severity": "warning", "title": "t", "detail": "d", "file": "src/app.py", "line": 11}
        ],
    }
    out = validate_output("reviewer", obj)
    assert out["findings"][0]["file"] == "src/app.py"
    assert out["findings"][0]["line"] == 11


def test_reviewer_finding_without_file_or_line_still_validates():
    # Backward compatibility: MS1's original finding shape (no location claim) is
    # untouched by this additive schema change.
    obj = {
        "capability": "reviewer",
        "findings": [{"severity": "info", "title": "t", "detail": "d"}],
    }
    assert validate_output("reviewer", obj)["findings"][0].get("file") is None


def test_reviewer_finding_bad_line_type_fails():
    obj = {
        "capability": "reviewer",
        "findings": [
            {"severity": "info", "title": "t", "detail": "d", "file": "x.py", "line": "11"}
        ],
    }
    with pytest.raises(ValidationError):
        validate_output("reviewer", obj)  # line must be an integer, not a string
