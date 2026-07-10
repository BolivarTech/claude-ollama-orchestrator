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


def test_added_content_line_starting_with_plus_plus_is_not_misread_as_file_header():
    """Hunk-length bookkeeping: an ADDED line whose own CONTENT starts with '++ ' (so the diff
    line reads '+++ ...') sits INSIDE the hunk body and must be counted as an added line, not
    misread as a '+++ b/<file>' file header -- which would register a PHANTOM file and miss the
    added line. The @@ header's declared new-count tells the parser exactly how many body lines
    to consume, so a '+++ '/'--- ' inside a hunk is never confused with a file header."""
    diff = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -10,1 +10,3 @@\n"
        " context\n"
        "+++ an added line whose text starts with plus-plus\n"
        "+another added line\n"
    )
    files, ranges = parse_diff(diff)
    assert files == {"src/app.py"}  # NO phantom file from the '+++ ...' body line
    assert {11, 12} <= ranges["src/app.py"]  # 11='+++ ...' added, 12='+another' added


def test_git_quoted_unicode_path_is_unquoted_so_real_findings_are_not_dropped():
    """Git quotes a path with special/high bytes (core.quotePath on, the default):
    '"b/caf\\303\\251.py"' for 'café.py'. parse_diff must UN-quote it so the files set holds
    the REAL path -- otherwise a model's finding on 'café.py' would be wrongly HARD-DROPPED as
    fabricated (a real finding lost), which is worse than a false negative."""
    diff = (
        'diff --git "a/caf\\303\\251.py" "b/caf\\303\\251.py"\n'
        '--- "a/caf\\303\\251.py"\n'
        '+++ "b/caf\\303\\251.py"\n'
        "@@ -1,1 +1,2 @@\n"
        " context\n"
        "+added\n"
    )
    files, ranges = parse_diff(diff)
    assert "café.py" in files  # unquoted to the real path, not the escaped/quoted form
    kept, dropped = validate_findings([{"file": "café.py", "line": 2, "title": "real"}], diff)
    assert dropped == []  # a real finding on the unicode path is KEPT, not hard-dropped


def test_removed_content_line_starting_with_minus_minus_is_not_misread_as_file_header():
    """Symmetric to the '+++' case: a REMOVED line whose content starts with '-- ' (diff line
    '--- ...') inside a hunk body is a removed line, not a '--- a/<file>' header. It must not
    register a phantom file nor shift the new-file line counter (removed lines don't advance it)."""
    diff = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -10,2 +10,1 @@\n"
        " context\n"
        "--- a removed line whose text starts with minus-minus\n"
    )
    files, ranges = parse_diff(diff)
    assert files == {"src/app.py"}  # no phantom file from the '--- ...' body line


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


# --- Corpus: realistic multi-file `git diff` output (mode changes, index lines, multiple
# hunks, a rename+modify, and a full file addition against /dev/null). Balthasar residual:
# exercise the parser on shapes closer to real `git diff` than the minimal fixtures above. ---
_GIT_CORPUS = """\
diff --git a/src/mod_a.py b/src/mod_a.py
index 1a2b3c4..5d6e7f8 100644
--- a/src/mod_a.py
+++ b/src/mod_a.py
@@ -3,2 +3,3 @@ def existing():
 unchanged one
-old removed line
+new line at four
+another new line at five
@@ -20,1 +21,2 @@ class Foo:
 ctx
+appended at twenty-two
diff --git a/src/created.py b/src/created.py
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/src/created.py
@@ -0,0 +1,2 @@
+brand new one
+brand new two
diff --git a/old/name.py b/new/name.py
similarity index 92%
rename from old/name.py
rename to new/name.py
index 111..222 100644
--- a/old/name.py
+++ b/new/name.py
@@ -8,1 +8,2 @@
 kept
+added after rename
"""


def test_corpus_multi_file_git_diff_parses_each_file_and_its_added_lines():
    files, ranges = parse_diff(_GIT_CORPUS)
    assert {"src/mod_a.py", "src/created.py", "new/name.py"} <= files
    # mod_a.py: first hunk new-start 3 -> ctx@3, added@4, added@5; second hunk new-start 21 ->
    # ctx@21, added@22.
    assert {4, 5, 22} <= ranges["src/mod_a.py"]
    # created.py: new-start 1 -> added@1, added@2 (full addition against /dev/null).
    assert {1, 2} <= ranges["src/created.py"]
    # renamed file's post-rename path carries the added line at new-start 8 -> ctx@8, added@9.
    assert 9 in ranges["new/name.py"]
    # `/dev/null` is never registered as a real file.
    assert "/dev/null" not in files


def test_corpus_findings_are_grounded_against_the_real_paths():
    # A finding on a real changed line is kept; one on a fabricated file is hard-dropped.
    kept, dropped = validate_findings(
        [
            {"file": "src/created.py", "line": 1, "title": "real add"},
            {"file": "src/ghost.py", "line": 1, "title": "fabricated"},
        ],
        _GIT_CORPUS,
    )
    kept_files = {f["file"] for f in kept}
    assert "src/created.py" in kept_files and "src/ghost.py" not in kept_files
    assert any(f["file"] == "src/ghost.py" for f in dropped)


def test_corpus_out_of_range_finding_on_a_real_file_is_soft_annotated_not_dropped():
    kept, dropped = validate_findings(
        [{"file": "src/mod_a.py", "line": 999, "title": "far out of range"}], _GIT_CORPUS
    )
    assert dropped == []  # real file -> never hard-dropped
    assert kept[0].get("annotation") == "[outside changed range]"


def test_copy_to_directive_registers_the_new_path_like_rename():
    # git emits `copy to <path>` under copy detection (-C); the destination is a touched file.
    diff = (
        "diff --git a/src/orig.py b/src/copy.py\n"
        "similarity index 100%\n"
        "copy from src/orig.py\n"
        "copy to src/copy.py\n"
    )
    files, _ = parse_diff(diff)
    assert "src/copy.py" in files
    kept, dropped = validate_findings([{"file": "src/copy.py", "line": 1, "title": "ok"}], diff)
    assert dropped == []  # a finding on the copy destination is grounded, not dropped


def test_line_claim_on_a_file_with_no_added_lines_is_annotated():
    # A binary/rename-only file has NO added lines (empty range). A model finding that still
    # claims a specific LINE there cannot be grounded -> annotate it (not hard-drop: the file
    # IS in the diff), so Claude sees the claim is unsupported.
    diff = (
        "diff --git a/assets/logo.png b/assets/logo.png\n"
        "Binary files a/assets/logo.png and b/assets/logo.png differ\n"
    )
    kept, dropped = validate_findings(
        [{"file": "assets/logo.png", "line": 7, "title": "suspect line claim"}], diff
    )
    assert dropped == []  # in the diff -> not fabricated -> not dropped
    assert kept[0].get("annotation") == "[outside changed range]"  # but flagged as ungroundable
