# tests/test_diff_guard.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Diff-grounded hallucination guard: hard-drop fabricated files, annotate out-of-range,
tolerate binary diffs / renames / malformed input (total, never raises)."""

from diff_guard import parse_diff, validate_findings

_DIFF = """\
diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -10,2 +10,3 @@
 context
+added line one
+added line two
"""


def test_parse_diff_extracts_files_and_added_lines():
    files, ranges = parse_diff(_DIFF)
    assert "src/app.py" in files
    assert {11, 12} <= ranges["src/app.py"]  # the two added lines


def test_finding_on_fabricated_file_is_hard_dropped():
    kept, dropped = validate_findings(
        [{"file": "does/not/exist.py", "line": 5, "title": "ghost"}], _DIFF
    )
    assert kept == [] and len(dropped) == 1


def test_finding_in_range_is_kept():
    kept, _ = validate_findings([{"file": "src/app.py", "line": 11, "title": "real"}], _DIFF)
    assert kept[0]["title"] == "real"
    assert "annotation" not in kept[0]


def test_finding_out_of_range_is_soft_annotated_but_kept():
    kept, _ = validate_findings(
        [{"file": "src/app.py", "line": 99, "title": "far"}], _DIFF, margin=3
    )
    assert kept[0]["annotation"] == "[outside changed range]"


def test_no_diff_is_noop():
    findings = [{"file": "x", "line": 1, "title": "t"}]
    kept, dropped = validate_findings(findings, "")
    assert kept == findings and dropped == []


def test_binary_diff_file_finding_is_not_hard_dropped():
    # A binary file has no +++/hunk body — only a `Binary files ... differ` line. It is a
    # TOUCHED file, so a finding on it must NOT be misparsed as fabricated and hard-dropped.
    diff = (
        "diff --git a/assets/logo.png b/assets/logo.png\n"
        "Binary files a/assets/logo.png and b/assets/logo.png differ\n"
    )
    files, ranges = parse_diff(diff)
    assert "assets/logo.png" in files  # registered as touched
    assert ranges["assets/logo.png"] == set()  # no added-line info for a binary
    kept, dropped = validate_findings(
        [{"file": "assets/logo.png", "line": 1, "title": "meta"}], diff
    )
    assert dropped == [] and kept[0]["title"] == "meta"  # kept, not hard-dropped


def test_renamed_file_findings_validate_against_the_new_path():
    diff = (
        "diff --git a/old_name.py b/new_name.py\n"
        "similarity index 95%\n"
        "rename from old_name.py\n"
        "rename to new_name.py\n"
        "--- a/old_name.py\n"
        "+++ b/new_name.py\n"
        "@@ -1,1 +1,2 @@\n"
        " ctx\n"
        "+added on the renamed file\n"
    )
    files, ranges = parse_diff(diff)
    assert "new_name.py" in files  # the NEW path is what findings reference
    kept, dropped = validate_findings([{"file": "new_name.py", "line": 2, "title": "ok"}], diff)
    assert kept and kept[0]["title"] == "ok" and dropped == []


def test_pure_rename_without_hunk_still_registers_the_new_path():
    diff = "diff --git a/a.py b/b.py\nsimilarity index 100%\nrename from a.py\nrename to b.py\n"
    files, _ = parse_diff(diff)
    assert "b.py" in files  # tracked even with no +++/hunk


def test_malformed_diff_never_raises():
    for junk in (
        "@@ garbage no numbers @@",
        "+++ ",
        "Binary files broken\n",
        "@@ -x +y @@\n+z",
        "rename to\n",
        "\x00\x01\x02",
    ):
        files, ranges = parse_diff(junk)  # must not raise
        assert isinstance(files, set) and isinstance(ranges, dict)
        kept, dropped = validate_findings([{"file": "z", "line": 1}], junk)  # must not raise
        assert isinstance(kept, list) and isinstance(dropped, list)


def test_no_newline_marker_does_not_shift_subsequent_line_numbers():
    # `\ No newline at end of file` is a MARKER, not content — it must not advance the
    # new-file line counter, or every added line after it would be miscounted by one.
    diff = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -10,1 +10,3 @@\n"
        "+first added line\n"
        "\\ No newline at end of file\n"
        "+second added line\n"
    )
    files, ranges = parse_diff(diff)
    assert ranges["src/app.py"] == {10, 11}  # NOT {10, 12} — the marker must not count


def test_parse_diff_context_lines_advance_line_numbers():
    # A context line consumes a new-file line number just like an added line does —
    # only a REMOVED line (`-`) fails to advance the new-file counter. Header
    # `@@ -1,3 +10,4 @@` starts the new file at line 10; one context line consumes
    # line 10, so the two additions that follow must land at 11 and 12, not 10/11.
    diff = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,3 +10,4 @@\n"
        " context\n"
        "+added one\n"
        "+added two\n"
    )
    files, ranges = parse_diff(diff)
    assert ranges["src/app.py"] == {11, 12}


def test_hunk_header_with_trailing_section_heading_is_still_parsed():
    # CRITICAL: real unified diffs commonly carry the enclosing section/function after
    # the closing `@@` (git's "funcname" hint), e.g. `@@ -1,3 +10,4 @@ def foo():`. A
    # hunk-header regex that requires the line to END at `@@` would reject this header
    # entirely and silently fail to ground the diff. The added-line numbers must still
    # be extracted correctly despite the trailing text.
    diff = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,3 +10,4 @@ def foo():\n"
        " context\n"
        "+added one\n"
        "+added two\n"
    )
    files, ranges = parse_diff(diff)
    assert ranges["src/app.py"] == {11, 12}


def test_validate_findings_keeps_a_finding_that_makes_no_file_claim():
    # R30 grounds findings that CLAIM a file ABSENT from the diff -- it must NOT drop a
    # legitimate general finding that cites no file at all (file/line are OPTIONAL in the
    # reviewer schema, MS7 Task 7). A None-path finding falls through the file check and is
    # kept, unannotated (there is no line to range-check).
    from diff_guard import validate_findings

    diff = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,2 @@\n line\n+added\n"
    fileless = {"severity": "info", "title": "overall approach is sound", "detail": "..."}
    real = {"severity": "warning", "title": "x", "detail": "y", "file": "f.py", "line": 2}
    kept, dropped = validate_findings([fileless, real], diff)
    assert fileless in kept  # a finding with no `file` key is NOT dropped
    assert real in kept
    assert dropped == []
