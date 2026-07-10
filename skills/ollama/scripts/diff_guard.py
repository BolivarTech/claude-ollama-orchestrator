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

The parser tracks each hunk's declared line COUNT from its ``@@ -a,b +c,d @@`` header
(hunk-length bookkeeping), so it knows exactly when a hunk body ends. That means an ADDED
line whose own content starts with ``++ `` (the full line reads ``+++ ...``) or a REMOVED
line whose content starts with ``-- `` (``--- ...``) is correctly read as body content while
inside the hunk, never misread as a ``+++``/``---`` file header -- file/binary/rename/hunk
headers are only matched OUTSIDE a hunk body. (An earlier revision lacked this bookkeeping and
could register a phantom file from such a line -- a false negative that let the guard miss a
fabricated-file finding; that is now closed.)
"""

from __future__ import annotations

import re
from typing import Any

# A hunk header: `@@ -<oldStart>[,<oldCount>] +<newStart>[,<newCount>] @@[ <section heading>]`.
# All four numbers are captured (counts optional, defaulting to 1 for a single-line hunk):
# group 1 = old start, 2 = old count, 3 = new start, 4 = new count. The old/new COUNTS drive
# hunk-length bookkeeping in :func:`parse_diff` -- knowing exactly how many body lines a hunk
# spans lets an added/removed line whose CONTENT starts with `++ `/`-- ` (diff line `+++ `/
# `--- `) be read as body content, never confused with a `+++`/`---` file header. A REAL
# unified diff often carries trailing context after the closing `@@` (git's "funcname" hint,
# e.g. `@@ -1,3 +10,4 @@ def foo():`), so an optional ` <anything>` trailer is allowed via
# `(?: .*)?$` rather than anchoring at the closing `@@`.
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$")
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
    # Hunk-length bookkeeping: how many old-side / new-side lines the CURRENT hunk body still
    # has, from its `@@ -a,b +c,d @@` header. While EITHER is positive we are inside the body,
    # so a line is classified by its FIRST char (content), never as a file header -- that is
    # what lets a `+++ `/`--- ` whose text merely starts with `++ `/`-- ` be read as an
    # added/removed line instead of a phantom `+++`/`---` file header. Header matching (file,
    # binary, rename, next hunk) only runs OUTSIDE a hunk body.
    old_remaining = 0
    new_remaining = 0
    try:
        for line in diff.splitlines():
            if old_remaining > 0 or new_remaining > 0:
                # INSIDE a hunk body -> classify by first char, decrement the budgets.
                if line.startswith("+"):
                    if current is not None:
                        ranges.setdefault(current, set()).add(line_no)
                    line_no += 1  # added line advances the new-file counter
                    new_remaining -= 1
                elif line.startswith("-"):
                    old_remaining -= 1  # removed line does NOT advance the new-file counter
                elif line.startswith("\\"):
                    continue  # "\ No newline at end of file": a marker, consumes no budget
                else:
                    # context line (leading space, or a bare empty line) -> both sides.
                    line_no += 1
                    old_remaining -= 1
                    new_remaining -= 1
                continue
            m_hunk = _HUNK.match(line)
            if m_hunk:
                old_remaining = int(m_hunk.group(2) or 1)
                line_no = int(m_hunk.group(3))
                new_remaining = int(m_hunk.group(4) or 1)
                continue
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
            # Any other line OUTSIDE a hunk body (`--- a/...`, `index ...`, `diff --git ...`,
            # a blank separator) carries no added-line info -> ignore.
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
