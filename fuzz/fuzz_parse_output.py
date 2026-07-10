#!/usr/bin/env python3
# fuzz/fuzz_parse_output.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Atheris fuzzing harness for parse_output.parse_agent_output (R23).

MILESTONE/CI GATE — NOT part of the Red-Green-Refactor loop (see CLAUDE.local.md §0.3):
run this manually before a release or wire it into CI; never during a TDD phase, and it
is not required for `make verify`.

Requires `pip install atheris` and a working clang toolchain (Linux/WSL; atheris is not
supported on native Windows — run this under WSL there).

Usage:
    python fuzz/fuzz_parse_output.py [-atheris_runs=200000]

Property under test: for ANY input bytes, `parse_agent_output` must never raise
anything other than `json.JSONDecodeError` (its documented "not parseable" signal) —
"arbitrary content ⇒ never crash, always handled or recovered".
"""

from __future__ import annotations

import json
import os
import sys

import atheris

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "skills", "ollama", "scripts"))

with atheris.instrument_imports():
    from parse_output import parse_agent_output

_DISCRIMINATOR_KEYS = ("agent", "verdict")


def _fuzz_one_input(data: bytes) -> None:
    """Feed *data* to the tolerant parser; only `json.JSONDecodeError` may propagate.

    Any OTHER exception (TypeError, RecursionError, MemoryError, UnicodeError, ...)
    escaping this call is a FUZZING FAILURE: the parser must map every malformed or
    hostile input to its one documented failure mode, never crash the process.
    """
    text = data.decode("utf-8", errors="replace")
    try:
        parse_agent_output(text, _DISCRIMINATOR_KEYS)
    except json.JSONDecodeError:
        pass  # the ONLY allowed failure mode


def main() -> None:
    atheris.Setup(sys.argv, _fuzz_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
