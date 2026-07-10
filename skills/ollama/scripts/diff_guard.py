# skills/ollama/scripts/diff_guard.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Diff-grounded hallucination guard (stdlib unified-diff parser, no git).

Grounds a model's ``file:line`` claims against a diff PROVIDED BY CLAUDE (R30): a
finding on a file absent from the diff is hard-dropped (fabricated), one outside the
changed range is soft-annotated. Inherent limit (accepted, INFO): it can only ground
against the given diff — a real bug outside the diff's scope is not something this guard
can confirm or deny; it validates *claims about the changed lines*, not global truth.
Stdlib-only, total (never raises into the orchestrator).

Testing note (accepted, INFO — deferred to implementation time): this plan's unit tests
use small hand-written diff fixtures (Task 1, Step 1); corpus/property tests against REAL
``git diff`` / ``git diff --binary`` output (varied hunk shapes, mode changes, multi-file
renames, combined diffs) should be added once the module is actually implemented, to catch
parser edge cases synthetic fixtures miss.

Known parser limitation (accepted, documented, low-impact): the parser does not track a
hunk's declared line COUNT, so an ADDED line whose own content starts with ``++ `` (the
full line reads ``+++ ...``) or a REMOVED line whose content starts with ``-- ``
(``--- ...``) is misread as a ``+++``/``---`` file-header, registering a phantom file. The
effect is strictly a FALSE NEGATIVE — the guard may FAIL TO DROP a fabricated-file finding,
never dropping a real one — and the guard is defense-in-depth over Claude's own review
(diff-grounding is optional/scope-gated, R30). A fully robust fix tracks the
``@@ -a,b +c,d @@`` counts to know exactly when a hunk body ends; deferred with the corpus
tests above as it requires that hunk-length bookkeeping.
"""

from __future__ import annotations

import re
from typing import Any

# A hunk header: `@@ -<old> +<newStart>[,<newCount>] @@[ <section heading>]`. Only the
# new-file start is captured. **CRITICAL fix:** a REAL unified diff often carries trailing
# context after the closing `@@` — e.g. `@@ -1,3 +10,4 @@ def foo():` (git's "funcname"
# hint naming the enclosing function/section) — so the pattern must NOT require the line to
# END at the closing `@@`; an optional ` <anything>` trailer is allowed via `(?: .*)?$`. The
# earlier `$`-anchored-at-`@@` form silently rejected every hunk header carrying a section
# heading, which meant the diff failed to ground at all for a large share of real-world diffs.
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@(?: .*)?$")
# The new-file path line `+++ b/<path>` (the `b/` prefix is git's; `diff -u` omits it).
_PLUSFILE = re.compile(r"^\+\+\+ (?:b/)?(.+?)\s*$")
# `Binary files a/<x> and b/<y> differ` — capture the new-side (b/) path.
_BINARY = re.compile(r"^Binary files .+? and (?:b/)?(.+?) differ\s*$")
# `rename to <new path>` — the destination path of a rename (may carry no hunk body).
_RENAME_TO = re.compile(r"^rename to (.+?)\s*$")


def parse_diff(diff: str) -> tuple[set[str], dict[str, set[int]]]:
    """Return the set of touched files and the added line numbers per file.

    Handles text hunks (added-line tracking), **binary** files (``Binary files ...
    differ`` → registered as touched with empty ranges — never treated as fabricated),
    **renames** (``rename to <new>`` → the new path is registered even without a
    ``+++``/hunk body), the ``\\ No newline at end of file`` marker (skipped — it is
    not content and must never advance the new-file line counter, or every added-line
    number after it would be off by one), and a hunk header carrying a **trailing
    section heading** (``@@ -1,3 +10,4 @@ def foo():`` — git's "funcname" hint): the
    hunk-header pattern allows an optional trailer after the closing ``@@`` instead of
    requiring the line to end there, since real diffs commonly carry one. Total: any
    unexpected error degrades to the partial result accumulated so far, never raising.

    Args:
        diff: A unified diff (as produced by ``git diff`` / ``diff -u``).

    Returns:
        ``(files, ranges)`` where ``ranges[file]`` is the set of added line numbers
        (empty for a binary/rename-only file that carries no hunk body).
    """
    files: set[str] = set()
    ranges: dict[str, set[int]] = {}
    current: str | None = None
    line_no = 0
    try:
        for line in diff.splitlines():
            m_bin = _BINARY.match(line)
            if m_bin:  # binary file: touched, no line info
                path = m_bin.group(1)
                files.add(path)
                ranges.setdefault(path, set())
                current = None  # no hunk body follows a binary line
                continue
            m_rename = _RENAME_TO.match(line)
            if m_rename:  # rename destination: touched
                path = m_rename.group(1)
                files.add(path)
                ranges.setdefault(path, set())
                continue
            m_file = _PLUSFILE.match(line)
            if m_file and line.startswith("+++"):
                current = m_file.group(1)
                if current != "/dev/null":
                    files.add(current)
                    ranges.setdefault(current, set())
                continue
            m_hunk = _HUNK.match(line)
            if m_hunk:
                line_no = int(m_hunk.group(1))
                continue
            if current is None:
                continue
            if line.startswith("\\ "):
                # e.g. "\ No newline at end of file" — a marker, not content. Must NOT
                # advance the new-file line counter (falling through to the "context
                # line" branch below would shift every subsequent added-line number
                # off by one).
                continue
            if line.startswith("+") and not line.startswith("+++"):
                ranges.setdefault(current, set()).add(line_no)
                line_no += 1
            elif line.startswith("-") and not line.startswith("---"):
                continue  # removed line: does not advance the new-file counter
            else:
                line_no += 1  # context line advances the new-file counter
    except Exception:  # noqa: BLE001 — total: degrade to the partial result, never raise.
        pass
    return files, ranges


def validate_findings(
    findings: list[dict[str, Any]], diff: str, *, margin: int = 3
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ground *findings* against *diff*: drop fabricated files, annotate out-of-range.

    Args:
        findings: Model findings, each with ``file`` and (optionally) ``line``.
        diff: The unified diff provided by Claude (empty → no-op).
        margin: Lines of slack around a changed range before annotating.

    Returns:
        ``(kept, dropped)``. Total: on any unexpected error the findings are returned
        unchanged (kept) rather than raising — a guard failure must never break a
        review, only forgo grounding.
    """
    try:
        if not diff.strip():
            return findings, []
        files, ranges = parse_diff(diff)
        kept: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        for f in findings:
            path = f.get("file")
            if path is not None and path not in files:
                # A finding CLAIMING a file that is ABSENT from the diff is a fabrication →
                # hard-drop. A finding making NO file claim (`file` omitted — it is optional
                # in the reviewer schema, MS7 Task 7) is a legitimate general observation:
                # it falls through here and is kept (the line check below is a no-op for it,
                # since `ranges.get(None)` is empty). R30 grounds CLAIMS, never penalizes the
                # absence of a claim.
                dropped.append(f)
                continue
            line = f.get("line")
            # Range-check only a finding that BOTH names a (touched) file and a line — a
            # fileless finding (`path is None`) has no changed-line range to check against
            # and is kept as-is (no annotation).
            if path is not None and isinstance(line, int):
                changed = ranges.get(path, set())
                if changed and not any(abs(line - c) <= margin for c in changed):
                    f = {**f, "annotation": "[outside changed range]"}
            kept.append(f)
        return kept, dropped
    except Exception:  # noqa: BLE001 — total: forgo grounding, never break the review.
        return findings, []
