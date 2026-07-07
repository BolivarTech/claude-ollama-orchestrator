# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Scaffold ./.claude/ollama-agents.toml from canonical defaults (refuse-if-exists)."""

from __future__ import annotations

import os

from ollama_config import (
    CAPABILITIES,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_PARALLEL_AGENTS,
    DEFAULT_MAX_QUEUED_AGENTS,
    DEFAULT_MODELS,
    DEFAULT_STREAM,
    DEFAULT_STRUCTURED,
)

REPO_CONFIG_RELPATH = os.path.join(".claude", "ollama-agents.toml")


def render_template() -> str:
    """Return the TOML template text populated from the canonical defaults.

    Returns:
        The full TOML text (``base_url``, ``max_parallel_agents``,
        ``max_queued_agents``, and the ``[models]``/``[structured]``/``[stream]``
        tables, each enumerating all seven capabilities explicitly). The
        ``api_key`` line is emitted commented-out — never a real key.
    """
    lines: list[str] = [
        "# Claude-Ollama-Orchestrator config (./.claude/ollama-agents.toml)",
        "# Precedence per key: env > this file (repo) > ~/.claude/ollama-agents.toml > built-in",
        "",
        f'base_url = "{DEFAULT_BASE_URL}"   # path verbatim e idempotente; host:port pelado -> /v1',
        '# api_key = "sk-..."                 # solo nube/auth; local no necesita (chmod 600)',
        f"max_parallel_agents = {DEFAULT_MAX_PARALLEL_AGENTS}"
        "             # delegaciones corriendo a la vez (semaforo)",
        f"max_queued_agents   = {DEFAULT_MAX_QUEUED_AGENTS}"
        "            # tope de la cola de espera (backstop anti-DoS)",
        "",
        "[models]",
    ]
    for cap in CAPABILITIES:
        lines.append(f'{cap:<10} = "{DEFAULT_MODELS[cap]}"')
    lines += [
        "",
        '# "schema" = JSON-Schema | "object" = cualquier JSON | "off" = texto libre',
        "[structured]",
    ]
    for cap in CAPABILITIES:
        lines.append(f'{cap:<10} = "{DEFAULT_STRUCTURED[cap]}"')
    lines += ["", "# true -> SSE con tok/s en vivo, false -> transaccional", "[stream]"]
    for cap in CAPABILITIES:
        lines.append(f"{cap:<10} = {str(DEFAULT_STREAM[cap]).lower()}")
    return "\n".join(lines) + "\n"


def write_template(repo_root: str | None = None) -> str:
    """Write the template to ``<repo_root>/.claude/ollama-agents.toml``.

    The target is created atomically with ``O_CREAT | O_EXCL`` -- there is no
    check-then-write window: if the file already exists, the ``open`` call
    itself raises ``FileExistsError`` and the existing file is never touched
    (never truncated, never partially overwritten).

    Args:
        repo_root: Repository root. Defaults to ``os.getcwd()``.

    Returns:
        The absolute path written.

    Raises:
        FileExistsError: if the target already exists (never clobbers).
    """
    if repo_root is None:
        repo_root = os.getcwd()
    path = os.path.join(repo_root, REPO_CONFIG_RELPATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Atomic refuse-if-exists (R13): O_CREAT|O_EXCL raises FileExistsError if the
    # target already exists -- no TOCTOU window between an exists-check and the open.
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(render_template())
    return path
