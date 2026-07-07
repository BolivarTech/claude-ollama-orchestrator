# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Scaffold ./.claude/ollama-agents.toml from defaults; refuse to overwrite."""

import os
import tomllib

import pytest

from ollama_init import REPO_CONFIG_RELPATH, render_template, write_template


def test_rendered_template_parses_and_carries_all_defaults():
    data = tomllib.loads(render_template())
    assert data["base_url"] == "http://localhost:11434/v1"
    assert data["max_parallel_agents"] == 3
    assert data["max_queued_agents"] == 32
    assert set(data["models"]) == {
        "coder",
        "reviewer",
        "tester",
        "explainer",
        "vision",
        "transcribe",
        "thinking",
    }
    assert data["models"]["thinking"] == "deepseek-v4-pro:cloud"
    assert data["structured"]["reviewer"] == "schema"
    assert data["structured"]["coder"] == "off"
    assert data["stream"]["reviewer"] is False
    assert data["stream"]["coder"] is True


def test_write_creates_file_then_refuses_overwrite(tmp_path):
    path = write_template(repo_root=str(tmp_path))
    assert path == os.path.join(str(tmp_path), REPO_CONFIG_RELPATH)
    assert os.path.exists(path)
    with pytest.raises(FileExistsError):
        write_template(repo_root=str(tmp_path))
