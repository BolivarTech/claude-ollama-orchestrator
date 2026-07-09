# skills/ollama/scripts/status_display.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Live status tree for concurrent delegations (stdlib-only)."""

from __future__ import annotations

import sys
from typing import TextIO

VALID_STATES: frozenset[str] = frozenset(
    {"pending", "running", "retrying", "success", "failed", "timeout"}
)
# Nicer UTF-8 glyphs when the stream encoding supports them; ASCII otherwise.
_UTF8_GLYPHS = {
    "pending": "·",
    "running": "◐",
    "retrying": "↻",
    "success": "✓",
    "failed": "✗",
    "timeout": "⏱",
}
_ASCII_GLYPHS = {
    "pending": ".",
    "running": "*",
    "retrying": "~",
    "success": "+",
    "failed": "x",
    "timeout": "!",
}
_CURSOR_UP = "\x1b[{n}A"  # move cursor up n lines
_CLEAR_LINE = "\x1b[2K"  # erase the entire line


def _stream_supports_utf8(stream: TextIO) -> bool:
    """True if *stream*'s encoding can represent the UTF-8 glyphs."""
    enc = getattr(stream, "encoding", None) or ""
    try:
        "✓◐↻·✗⏱".encode(enc)
        return True
    except (LookupError, UnicodeEncodeError):
        return False


class StatusDisplay:
    """Render per-agent delegation state to *stream*.

    Auto-detects TTY + encoding: on a TTY it redraws the whole tree in place with
    ANSI cursor controls (one row per agent); on a non-TTY it emits one plain line
    per :meth:`update`. UTF-8 glyphs are used when the stream encoding supports
    them, ASCII glyphs (cp1252 Windows consoles) otherwise. stdlib-only.
    """

    def __init__(self, agents: list[str], *, stream: TextIO | None = None) -> None:
        self._agents = list(agents)
        self._stream = stream if stream is not None else sys.stderr
        self._state: dict[str, str] = {a: "pending" for a in self._agents}
        self._rate: dict[str, float | None] = {a: None for a in self._agents}
        self._use_ansi = bool(getattr(self._stream, "isatty", lambda: False)())
        self._glyphs = _UTF8_GLYPHS if _stream_supports_utf8(self._stream) else _ASCII_GLYPHS
        self._drawn = 0  # rows written in the previous ANSI frame

    def _row(self, agent: str) -> str:
        state = self._state[agent]
        glyph = self._glyphs.get(state, "?")
        rate = self._rate[agent]
        suffix = f" {rate:.0f} tok/s" if rate is not None else ""
        return f"[{glyph}] {agent:<12} {state}{suffix}"

    def update(self, agent: str, state: str, tok_per_s: float | None = None) -> None:
        """Record *agent*'s new *state* (+ optional tok/s) and render.

        Raises:
            ValueError: if *state* is not in :data:`VALID_STATES`.
        """
        if state not in VALID_STATES:
            raise ValueError(f"invalid state {state!r}")
        self._state[agent] = state
        if tok_per_s is not None:
            self._rate[agent] = tok_per_s
        if self._use_ansi:
            self._redraw()
        else:
            self._stream.write(self._row(agent) + "\n")
            self._stream.flush()

    def _redraw(self) -> None:
        """Redraw the full tree in place (cursor up over the last frame, rewrite)."""
        if self._drawn:
            self._stream.write(_CURSOR_UP.format(n=self._drawn))
        for agent in self._agents:
            self._stream.write(_CLEAR_LINE + self._row(agent) + "\n")
        self._drawn = len(self._agents)
        self._stream.flush()

    def stop(self) -> None:
        """Finalize the display (flush)."""
        self._stream.flush()
