# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Tolerant extraction of a JSON object from noisy model output."""

from __future__ import annotations

import json
import re
from typing import Any

LENIENT_RECOVERY_MAX_CHARS = 1_000_000
MAX_BRACE_PROBES = 2000
# A fence is a whole line: optional indent, ```, optional language tag, optional trailing
# space. re.MULTILINE anchors ^/$ to EVERY line, so an opening ```json fence that is NOT at
# char 0 (e.g. after a leading prose line) is still stripped, not just one at string start.
_FENCE = re.compile(r"^[ \t]*```[a-zA-Z0-9]*[ \t]*$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    """Remove Markdown code-fence lines (```json / ```) anywhere in *text* (re.MULTILINE).

    Args:
        text: Raw model output that may contain Markdown code fences wrapping JSON.

    Returns:
        *text* with fence lines removed and surrounding whitespace stripped.
    """
    return _FENCE.sub("", text).strip()


def _qualifies(obj: object, keys: tuple[str, ...]) -> bool:
    """Return True if *obj* is a dict carrying every key in *keys*.

    Args:
        obj: A candidate value decoded from JSON (may be any JSON type).
        keys: The discriminator keys that identify the real object for a capability.

    Returns:
        True only if *obj* is a ``dict`` and contains all of *keys*.
    """
    return isinstance(obj, dict) and all(k in obj for k in keys)


def parse_agent_output(raw: str, discriminator_keys: tuple[str, ...]) -> dict[str, Any]:
    """Extract the single JSON object carrying *discriminator_keys* from *raw*.

    Fast path: strict ``json.loads`` on the fence-stripped text (handles clean
    JSON and JSON preceded/followed only by fence lines). Recovery: scan ``{``
    positions with ``json.JSONDecoder().raw_decode`` (bounded by
    :data:`MAX_BRACE_PROBES`, skipped entirely for blobs over
    :data:`LENIENT_RECOVERY_MAX_CHARS` to avoid O(n^2) behavior) and collect
    objects that carry all discriminator keys — this is what recovers a
    leading ``<think>...</think>`` block or other surrounding prose.
    **Fail-closed on ambiguity:** if the recovery scan does not find *exactly
    one* qualifying object, raise ``json.JSONDecodeError`` (the caller treats
    this as "not parseable" and retries) rather than guess which one is real.
    Deep nesting that would overflow Python's recursion limit is mapped to a
    handled ``JSONDecodeError`` — ``RecursionError`` never escapes this
    function.

    Args:
        raw: The raw model output (already decoded to str).
        discriminator_keys: Keys that identify the real object for this capability.

    Returns:
        The single qualifying JSON object.

    Raises:
        json.JSONDecodeError: on no/ambiguous match, an oversized blob, or
            deep nesting that would otherwise raise ``RecursionError``.
    """
    text = _strip_fences(raw)
    try:
        obj: Any = json.loads(text)
        if _qualifies(obj, discriminator_keys):
            return dict(obj)
    except (json.JSONDecodeError, RecursionError):
        pass

    if len(text) > LENIENT_RECOVERY_MAX_CHARS:
        raise json.JSONDecodeError("blob too large for recovery", text, 0)

    decoder = json.JSONDecoder()
    matches: list[dict[str, Any]] = []
    probes = 0
    index = 0
    while index < len(text) and probes < MAX_BRACE_PROBES:
        brace = text.find("{", index)
        if brace == -1:
            break
        probes += 1
        try:
            obj, end = decoder.raw_decode(text, brace)
        except (json.JSONDecodeError, RecursionError):
            index = brace + 1
            continue
        if _qualifies(obj, discriminator_keys):
            matches.append(obj)
        index = end if end > brace else brace + 1

    if len(matches) == 1:
        return matches[0]
    raise json.JSONDecodeError(
        f"expected exactly one object with {discriminator_keys}, found {len(matches)}",
        text,
        0,
    )
