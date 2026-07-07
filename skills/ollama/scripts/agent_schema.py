# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""JSON-Schema constants per structured capability, in lockstep with validate.py."""

from __future__ import annotations

from typing import Any

_SEVERITY = ["critical", "warning", "info"]

SCHEMAS: dict[str, dict[str, Any]] = {
    "reviewer": {
        "type": "object",
        "additionalProperties": False,
        "required": ["capability", "findings"],
        "properties": {
            "capability": {"const": "reviewer"},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["severity", "title", "detail"],
                    "properties": {
                        "severity": {"enum": _SEVERITY},
                        "title": {"type": "string"},
                        "detail": {"type": "string"},
                    },
                },
            },
        },
    },
    "tester": {
        "type": "object",
        "additionalProperties": False,
        "required": ["capability", "tests"],
        "properties": {
            "capability": {"const": "tester"},
            "tests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "code"],
                    "properties": {
                        "name": {"type": "string"},
                        "code": {"type": "string"},
                    },
                },
            },
        },
    },
}

DISCRIMINATOR_KEYS: dict[str, tuple[str, ...]] = {
    "reviewer": ("capability", "findings"),
    "tester": ("capability", "tests"),
}
_SEVERITIES = frozenset(_SEVERITY)
