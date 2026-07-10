# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Stdlib token estimation + oversize flag (biased to warn for non-Latin scripts)."""

from __future__ import annotations

MAX_INPUT_FILE_SIZE = 10 * 1024 * 1024

_NON_ASCII_RATIO_THRESHOLD = 0.3
_ASCII_DIVISOR = 4
_CONSERVATIVE_DIVISOR = 2


def estimate_tokens(text: str) -> int:
    """Estimate token count as ``chars/4`` (``chars/2`` when non-ASCII-heavy).

    The conservative divisor for non-Latin scripts (CJK/Arabic) biases the
    estimate toward warning of *more*, never fewer, tokens. Total; never raises.

    Rationale for the ``30%`` non-ASCII ratio / ``chars/2`` divisor (INFO, documented
    empirical basis, not a calibrated tokenizer): ``chars/4`` is a reasonable rough
    estimate for English/Latin-script text, where a "token" is typically several
    characters. CJK text is roughly 1 code point ≈ 1 token — i.e. close to ``chars/1``,
    not ``chars/4`` — so applying the English divisor there would UNDER-count by
    roughly 4x, silently suppressing a warning that should have fired (the opposite of
    R24's fail-open-toward-warning intent). ``chars/2`` is a deliberately simple,
    single conservative midpoint between the two regimes: it still under-counts
    pure-CJK content somewhat (a precise CJK tokenizer would count closer to
    ``chars/1``), but it warns far sooner than the unmodified ``chars/4`` would, which
    is the direction R24 explicitly wants to bias toward (over-warning, never
    under-warning). The ``30%`` non-ASCII-ratio threshold is picked so that ordinary
    English text with occasional accented characters, emoji, or a few non-Latin words
    stays on the ``chars/4`` path (it doesn't cross 30% non-ASCII), while text that is
    substantially CJK/Arabic (which will be at or near 100% non-ASCII) reliably crosses
    it and gets the conservative divisor. Neither constant claims tokenizer-level
    precision — both are heuristics whose only job is to bias toward warning more
    often, never less, which is exactly what a non-blocking advisory guard (R24) needs.

    Args:
        text: Raw input text to estimate. May be empty.

    Returns:
        Estimated token count (``0`` for empty input). Never raises.
    """
    if not text:
        return 0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    divisor = (
        _CONSERVATIVE_DIVISOR
        if non_ascii > len(text) * _NON_ASCII_RATIO_THRESHOLD
        else _ASCII_DIVISOR
    )
    return len(text) // divisor


def check_input_size(text: str, threshold: int) -> tuple[int, bool]:
    """Estimate input size and flag whether it exceeds a warning threshold.

    Args:
        text: Raw input text to estimate.
        threshold: Token-count threshold above which the input is oversize.

    Returns:
        A ``(estimated_tokens, exceeds_threshold)`` tuple.
    """
    est = estimate_tokens(text)
    return est, est > threshold
