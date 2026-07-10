# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Domain validation of structured capability output (lockstep with agent_schema)."""

from __future__ import annotations

import re
from typing import Any

from agent_schema import SCHEMAS, SEVERITIES
from errors import ValidationError

# Anti-DoS bound (R23) shared with the input-reading path; enforced in run_ollama's
# input loader (a file larger than this is rejected BEFORE it is read).
MAX_INPUT_FILE_SIZE = 10 * 1024 * 1024

# Zero-width / joiners / BOM / soft-hyphen / bidi controls, built from EXPLICIT integer
# code points (never literal invisibles in source — those are fragile and a mistyped
# range can silently collapse to matching a literal hyphen). Covers ZWSP-ZWJ
# (200B-200D), LRM/RLM (200E-200F), bidi embeddings/overrides (202A-202E), the
# word-joiner block (2060-2064), the bidi ISOLATE marks LRI/RLI/FSI/PDI
# (2066-2069 — the exact code points used in "Trojan Source" attacks; R23 requires
# these be stripped from untrusted structured output), BOM/ZWNBSP (FEFF), and soft
# hyphen (00AD).
_ZW_RANGES = (
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x2064),
    (0x2066, 0x2069),
    (0xFEFF, 0xFEFF),
    (0x00AD, 0x00AD),
)
_ZERO_WIDTH = re.compile(
    "[" + "".join(chr(c) for lo, hi in _ZW_RANGES for c in range(lo, hi + 1)) + "]"
)
_CONTROL_WS = re.compile(r"[\t\n\v\f\r\x85]")


def _strip_invisibles(raw: str) -> str:
    """Strip zero-width/bidi/BOM/soft-hyphen chars, PRESERVING normal formatting.

    The anti-smuggling half of R23 for free-text structured fields (``detail``,
    ``code``) where newlines/tabs are meaningful: it removes only the invisible/bidi
    code points, never collapsing legitimate whitespace (unlike :func:`clean_title`,
    which is for single-line identity fields). Applied to EVERY structured string
    field so untrusted output can't smuggle length/layout via invisibles anywhere.

    Args:
        raw: An untrusted string leaf from model output.

    Returns:
        The string with invisible/bidi characters removed.
    """
    return _ZERO_WIDTH.sub("", raw)


def clean_title(raw: str) -> str:
    """Strip zero-width/bidi/control chars and collapse whitespace; reject empty.

    For single-line identity fields (``title``, ``name``): strips invisibles, then
    collapses control whitespace to a single space and rejects an empty result.

    Args:
        raw: The untrusted title string from model output.

    Returns:
        The cleaned title.

    Raises:
        ValidationError: if the title is empty/whitespace after cleaning.
    """
    cleaned = _CONTROL_WS.sub(" ", _strip_invisibles(raw)).strip()
    if not cleaned:
        raise ValidationError("title is empty after sanitization")
    return cleaned


def _truncate_utf8_bytes(text: str, max_bytes: int) -> tuple[str, bool]:
    """Truncate *text* to at most *max_bytes* UTF-8 bytes, never splitting a
    multi-byte character (R24c anti-runaway output cap).

    THE single canonical implementation of this algorithm: both `backend.py`
    (transactional path) and `ollama_stream.py` (streaming path, MS4) import it from
    here instead of each carrying its own copy.

    Args:
        text: The text to (possibly) truncate.
        max_bytes: The maximum length of the UTF-8-encoded result, in bytes.

    Returns:
        A ``(result_text, was_truncated)`` tuple. ``result_text`` equals *text*
        unchanged when it already fits; otherwise it is the longest prefix of *text*
        whose UTF-8 encoding is at most *max_bytes* bytes, cut on a whole-character
        boundary (never a bare ``0b10xxxxxx`` continuation byte), so
        ``result_text.encode("utf-8")`` always succeeds and never raises. A
        non-positive *max_bytes* (``<= 0``) has no valid non-negative slice, so it
        returns ``("", True)`` (fully truncated) rather than risking a wrong or
        negative-length slice (defensive guard, Caspar residual).
    """
    if max_bytes <= 0:
        return "", True
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    cut = max_bytes
    while cut > 0 and (encoded[cut] & 0xC0) == 0x80:  # 0x80 = continuation-byte marker
        cut -= 1
    return encoded[:cut].decode("utf-8"), True


def _reject_extra_keys(obj: dict[str, Any], allowed: set[str], ctx: str) -> None:
    """Raise if *obj* carries keys outside *allowed* — mirrors the JSON-Schema's
    ``additionalProperties: false`` so the validator stays in lockstep with the schema
    (R29). Without this, the validator would accept objects the schema rejects.

    Raises:
        ValidationError: if any unexpected key is present.
    """
    extra = set(obj) - allowed
    if extra:
        raise ValidationError(f"{ctx}: unexpected keys {sorted(extra)}")


def validate_output(capability: str, obj: dict[str, Any]) -> dict[str, Any]:
    """Validate *obj* against the capability's structured contract.

    Enforces required keys, enums, the ``capability`` const, list-typed array fields
    (fail-closed on a non-list `findings`/`tests`), AND ``additionalProperties: false`` at
    every object level (top-level and each array item) so the domain validator accepts
    exactly the objects the strict JSON-Schema accepts — the bidirectional lockstep of R29.

    Args:
        capability: One of the structured capabilities (``reviewer``/``tester``).
        obj: The parsed model output.

    Returns:
        A NEW validated, cleaned object; the caller's *obj* is never mutated.

    Raises:
        ValidationError: if *capability* has no schema, or *obj* violates it.
    """
    schema = SCHEMAS.get(capability)
    if schema is None:
        raise ValidationError(f"capability {capability!r} has no structured schema")
    if not isinstance(obj, dict):  # fail-closed on a non-dict top-level type (R23).
        raise ValidationError(f"{capability}: output is not a JSON object")
    required = schema["required"]
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValidationError(f"{capability}: missing keys {missing}")
    _reject_extra_keys(obj, set(schema["properties"]), capability)  # additionalProperties: false
    # Enforce the schema's ``capability`` const so the validator stays in lockstep with the
    # schema (R29): a payload whose capability field != the expected const is rejected.
    expected = schema["properties"]["capability"]["const"]
    if obj.get("capability") != expected:
        raise ValidationError(f"{capability}: capability field must be {expected!r}")
    # Build a NEW cleaned object — validate_output NEVER mutates the caller's input dict.
    cleaned = dict(obj)
    if capability == "reviewer":
        findings = obj["findings"]
        if not isinstance(findings, list):  # fail-closed: never iterate a non-list (R23).
            raise ValidationError("reviewer: findings must be a list")
        new_findings = []
        for f in findings:
            # NOTE: this allowed/required key set MUST be kept in lockstep with
            # SCHEMAS["reviewer"]["properties"]["findings"]["items"]["properties"] in
            # agent_schema.py (no jsonschema engine here, R29 is stdlib-only) — the
            # bidirectional corpus test (test_agent_schema.py) is the safety net that
            # catches drift if one side is edited without the other. ``file``/``line``
            # (MS7 Task 7, R30) are OPTIONAL: allowed but not required, so a finding
            # that omits them (MS1's original shape) still validates unchanged.
            if not isinstance(f, dict) or {"severity", "title", "detail"} - f.keys():
                raise ValidationError("reviewer: malformed finding")
            _reject_extra_keys(
                f, {"severity", "title", "detail", "file", "line"}, "reviewer finding"
            )
            if f["severity"] not in SEVERITIES:
                raise ValidationError(f"reviewer: bad severity {f['severity']!r}")
            # Lockstep with the schema's ``type: "string"`` (R29): reject a non-string
            # instead of ``str()``-coercing it (which would accept ints/None the schema rejects).
            if not isinstance(f["title"], str) or not isinstance(f["detail"], str):
                raise ValidationError("reviewer: 'title' and 'detail' must be strings")
            cleaned_finding: dict[str, Any] = {
                "severity": f["severity"],
                "title": clean_title(f["title"]),  # identity field: reject empty
                "detail": _strip_invisibles(f["detail"]),  # free text: keep formatting (R23)
            }
            if "file" in f:
                # Lockstep with the schema's ``type: "string"`` (R29): reject non-strings.
                if not isinstance(f["file"], str):
                    raise ValidationError("reviewer: 'file' must be a string")
                cleaned_finding["file"] = _strip_invisibles(f["file"])
            if "line" in f:
                # Lockstep with the schema's ``type: "integer"`` (R29): reject a bool
                # (a `bool` is an `int` subclass in Python) and any non-int, e.g. the
                # string "11" a model might emit instead of the bare integer 11.
                if isinstance(f["line"], bool) or not isinstance(f["line"], int):
                    raise ValidationError("reviewer: 'line' must be an integer")
                # Lockstep with the schema's ``"minimum": 1`` (R29): a source line is
                # 1-based, so reject 0/negative rather than passing a semantically
                # impossible location downstream to diff_guard.
                if f["line"] < 1:
                    raise ValidationError("reviewer: 'line' must be >= 1")
                cleaned_finding["line"] = f["line"]
            new_findings.append(cleaned_finding)
        cleaned["findings"] = new_findings
    elif capability == "tester":
        tests = obj["tests"]
        if not isinstance(tests, list):  # fail-closed: never iterate a non-list (R23).
            raise ValidationError("tester: tests must be a list")
        new_tests = []
        for t in tests:
            # NOTE: this allowed/required key set MUST be kept in lockstep with
            # SCHEMAS["tester"]["properties"]["tests"]["items"]["properties"] in
            # agent_schema.py (no jsonschema engine here, R29 is stdlib-only) — the
            # bidirectional corpus test (test_agent_schema.py) is the safety net that
            # catches drift if one side is edited without the other.
            if not isinstance(t, dict) or {"name", "code"} - t.keys():
                raise ValidationError("tester: malformed test")
            _reject_extra_keys(t, {"name", "code"}, "tester test")
            # Lockstep with the schema's ``type: "string"`` (R29): reject non-strings.
            if not isinstance(t["name"], str) or not isinstance(t["code"], str):
                raise ValidationError("tester: 'name' and 'code' must be strings")
            new_tests.append(
                {
                    "name": clean_title(t["name"]),  # identity field: reject empty
                    "code": _strip_invisibles(t["code"]),  # code: strip invisibles, keep \n\t
                }
            )
        cleaned["tests"] = new_tests
    return cleaned
