# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Structural checks for the SKILL.md frontmatter and the 7 agent prompts."""

import os

_SKILL = os.path.join("skills", "ollama", "SKILL.md")
_AGENTS = os.path.join("skills", "ollama", "agents")
_CAPS = ("coder", "reviewer", "tester", "explainer", "vision", "transcribe", "thinking")


def test_skill_frontmatter_names_ollama():
    with open(_SKILL, encoding="utf-8") as fh:
        head = fh.read(400)
    assert head.startswith("---")
    assert "name: ollama" in head


def test_all_seven_agent_prompts_exist_and_nonempty():
    for cap in _CAPS:
        path = os.path.join(_AGENTS, f"ollama-{cap}.md")
        assert os.path.exists(path), path
        assert os.path.getsize(path) > 0


def test_skill_documents_hybrid_delegation_and_positional_capability():
    with open(_SKILL, encoding="utf-8") as fh:
        body = fh.read()
    assert "/ollama" in body
    assert "capacidad" in body or "capability" in body   # positional routing (R1b)
    assert "delega" in body.lower()                        # hybrid heuristic (R1c)
