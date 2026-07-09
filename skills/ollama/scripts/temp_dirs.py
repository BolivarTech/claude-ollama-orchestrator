# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Per-project temp-directory namespace + LRU housekeeping (stdlib-only)."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile

from run_lock import is_dir_live

OLLAMA_DIR_PREFIX = "ollama-run-"
OLLAMA_RUNS_CONTAINER = "ollama-runs"
_MARKERS = (".git", "pyproject.toml")


def resolve_project_root(start: str | None = None) -> str:
    """Walk up from *start* (or cwd) to a repo marker; else return realpath(cwd).

    Stdlib-only (no ``git`` subprocess): stable for any subdirectory of a repo.
    """
    cur = os.path.realpath(start or os.getcwd())
    while True:
        if any(os.path.exists(os.path.join(cur, m)) for m in _MARKERS):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.realpath(start or os.getcwd())
        cur = parent


def project_run_root(project_root: str) -> str:
    """Return (creating) the per-project container ``<tmp>/ollama-runs/<hash>/``."""
    norm = os.path.normcase(os.path.realpath(project_root))
    key = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
    root = os.path.join(tempfile.gettempdir(), OLLAMA_RUNS_CONTAINER, key)
    try:
        os.makedirs(root, exist_ok=True)
    except OSError as exc:
        print(f"WARNING: could not create run root {root}: {exc}", file=sys.stderr)
        return tempfile.gettempdir()
    return root


def create_output_dir(output_dir: str | None, run_root: str | None = None) -> str:
    """Create and return the output dir (explicit path, or a unique temp run dir)."""
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    if run_root is None:
        run_root = tempfile.gettempdir()
    return tempfile.mkdtemp(prefix=OLLAMA_DIR_PREFIX, dir=run_root)


def _safe_prefix(root: str) -> str:
    p = os.path.normcase(os.path.realpath(root))
    return p if p.endswith(os.sep) else p + os.sep


def cleanup_old_runs(keep: int, run_root: str | None = None) -> None:
    """Remove oldest ``ollama-run-*`` dirs under *run_root*, keeping *keep* non-live.

    Live (locked) dirs are excluded from both the survivor budget and deletion.
    ``keep < 0`` disables cleanup. Total: scan/stat/rmtree errors degrade to
    no-op/warning, never raising.

    Caller precondition (ENFORCED AT RUNTIME below, not just documented): *run_root*
    MUST be a per-project container (i.e. the return value of :func:`project_run_root`),
    never the shared ``tempfile.gettempdir()`` — this function prunes every
    ``ollama-run-*`` dir found directly under *run_root*, so passing the shared root
    would prune other projects' runs. ``project_run_root``'s ``gettempdir()``
    degradation path (R15) is deliberately never passed here in normal operation (see
    ``run_ollama.run_delegation``'s skip-cleanup guard) — but if it (or any other
    caller) ever does pass the shared root, this function detects it and no-ops with a
    warning instead of silently pruning other projects' runs.
    """
    if keep < 0:
        return
    if run_root is None:
        run_root = tempfile.gettempdir()
    if os.path.realpath(run_root) == os.path.realpath(tempfile.gettempdir()):
        # Runtime enforcement of the caller precondition above: never scan/prune the
        # SHARED temp root, which would touch other projects' ollama-run-* dirs.
        print(
            "WARNING: cleanup_old_runs refused to run against the shared "
            f"gettempdir() ({run_root}); pass project_run_root()'s return value "
            "instead. Skipping cleanup.",
            file=sys.stderr,
        )
        return
    try:
        entries = [
            (e.stat().st_mtime, e.path)
            for e in os.scandir(run_root)
            # follow_symlinks=False: a symlink named `ollama-run-*` is anomalous (real run
            # dirs come from mkdtemp, never symlinks) and must never even be a deletion
            # candidate — it is NOT a directory for our purposes, so it is excluded here.
            if e.is_dir(follow_symlinks=False) and e.name.startswith(OLLAMA_DIR_PREFIX)
        ]
    except OSError:
        return
    candidates = [(m, p) for (m, p) in entries if not is_dir_live(p)]
    if len(candidates) <= keep:
        return
    candidates.sort(key=lambda x: (-x[0], x[1]))
    safe = _safe_prefix(run_root)
    for _mtime, path in candidates[keep:]:
        # Never rmtree a symlink: a run dir is a real mkdtemp directory, never a symlink,
        # so skip one entirely rather than following it. Belt-and-suspenders with the
        # realpath-containment check below and scandir's follow_symlinks=False above — and
        # it closes a TOCTOU where a real dir is swapped to a symlink between scandir and
        # here (shutil.rmtree also refuses a top-level symlink, but skipping is cleaner).
        if os.path.islink(path):
            continue
        if os.path.normcase(os.path.realpath(path)).startswith(safe):
            try:
                shutil.rmtree(path)
            except OSError as exc:
                print(f"WARNING: failed to remove {path}: {exc}", file=sys.stderr)
