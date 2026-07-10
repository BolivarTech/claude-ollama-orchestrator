# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-10
"""Shared UTF-8-byte-safe truncation primitive (R24c), canonical home in validate.py.

MS4 (`ollama_stream.py`, streaming path) and MS6's transactional path (`backend.py`)
each carried their own private copy of the identical UTF-8-continuation-byte back-off
algorithm; this suite covers the ONE canonical implementation both now import.
"""

from validate import _truncate_utf8_bytes


def test_truncate_utf8_bytes_leaves_short_text_unchanged():
    assert _truncate_utf8_bytes("short", 100) == ("short", False)


def test_truncate_utf8_bytes_cuts_ascii_on_the_boundary():
    assert _truncate_utf8_bytes("a" * 20, 8) == ("a" * 8, True)


def test_truncate_utf8_bytes_exact_fit_is_not_truncated():
    text = "abc"
    assert _truncate_utf8_bytes(text, len(text.encode("utf-8"))) == (text, False)


def test_truncate_utf8_bytes_never_splits_a_multibyte_cjk_character():
    # Each CJK char below is 3 UTF-8 bytes; a naive code-point-count-based cut would
    # slice mid-character and the result would fail to re-encode/round-trip cleanly.
    cjk = "語" * 10  # 10 code points, 30 UTF-8 bytes
    result, truncated = _truncate_utf8_bytes(cjk, 9)
    assert truncated is True
    assert len(result.encode("utf-8")) <= 9
    result.encode("utf-8")  # must not raise: cut only on a char boundary


def test_truncate_utf8_bytes_with_zero_max_bytes_returns_empty_and_truncated():
    # CRITICAL fix (Caspar residual): max_bytes=0 has no valid non-negative slice —
    # the continuation-byte back-off loop's `cut > 0` guard would otherwise return
    # b"" anyway for 0, but a NEGATIVE max_bytes falls through to a wrong/negative
    # slice (see next test). Both are guarded explicitly rather than relying on
    # incidental loop behavior.
    assert _truncate_utf8_bytes("hello", 0) == ("", True)


def test_truncate_utf8_bytes_with_negative_max_bytes_returns_empty_and_truncated():
    # A negative max_bytes must never produce a negative-length/garbage slice — it is
    # treated the same as "no budget at all" (fully truncated to empty).
    assert _truncate_utf8_bytes("hello", -1) == ("", True)
    assert _truncate_utf8_bytes("hello", -100) == ("", True)
