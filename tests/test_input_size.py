# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Token estimation + oversize flag (conservative for CJK/Arabic)."""

from input_size import MAX_INPUT_FILE_SIZE, check_input_size, estimate_tokens


def test_ascii_uses_chars_over_four():
    assert estimate_tokens("a" * 400) == 100


def test_non_ascii_heavy_uses_conservative_divisor():
    cjk = "語" * 100  # all non-ASCII → chars/2 → 50 (> chars/4 = 25)
    assert estimate_tokens(cjk) == 50


def test_check_input_size_flags_oversize():
    est, over = check_input_size("a" * 4000, threshold=500)  # 1000 tokens > 500
    assert est == 1000 and over is True


def test_max_input_file_size_is_10mb():
    assert MAX_INPUT_FILE_SIZE == 10 * 1024 * 1024
