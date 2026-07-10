# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Pre-dispatch hardening: console IO, config-permission and transport warnings, and
the input-size warning — one small, single-responsibility function per concern
(R26, NR3, NR9, R24).

Extracted out of `run_ollama.py` (SRP, MS6 self-review finding): `run_ollama.py`'s job
is CLI parsing + dispatch orchestration (argparse, the MS5 semaphore/queue, retry,
output-dir lifecycle, interrupt cleanup); the four concerns here are cross-cutting
startup/pre-dispatch checks with no dependency on dispatch itself, so they are grouped
in their OWN module rather than inlined into (or bundled as private helpers inside)
the orchestrator. `run_ollama.py` calls each of these by name from `main`/
`run_delegation` and adds no logic of its own around them.
"""

from __future__ import annotations

import ipaddress
import os
import sys
from urllib.parse import urlsplit

from input_size import check_input_size

# Conventional non-IP local hostname (NOT an IP literal, so `ipaddress.ip_address` cannot
# classify it — handled as a separate, explicit fallback in `_is_local_host` below).
_LOCAL_HOSTNAMES = frozenset({"localhost"})


def _is_local_host(hostname: str | None) -> bool:
    """True if *hostname* is loopback — the FULL range, not a fixed enumeration (NR9).

    [WARNING -> fix, re-hardening this revision] The prior implementation checked
    membership in a fixed four-item allow-list (`localhost`, `127.0.0.1`, `::1`,
    `0:0:0:0:0:0:0:1`), which under-recognized the REST of the IPv4 loopback block —
    `127.0.0.2` through `127.255.255.254` are equally loopback addresses (RFC 5735 /
    RFC 6890), sometimes used to disambiguate multiple local services bound to distinct
    loopback aliases on one host, and were incorrectly treated as remote.

    Args:
        hostname: `urlsplit(base_url).hostname` — already unbracketed and lower-cased
            for a bracketed IPv6 URL host by `SplitResult.hostname`; `None` if *base_url*
            has no host component at all.

    Returns:
        `True` if *hostname* parses as an IP address (v4 or v6) whose
        `ipaddress.ip_address(...).is_loopback` is `True` — covering the entire
        `127.0.0.0/8` IPv4 block and every IPv6 loopback notation (`::1`, its expanded
        form, ...) in ONE check, via the stdlib `ipaddress` module rather than an
        enumerated address list. Falls back to a conventional-hostname check
        (`localhost`, case-insensitive) when *hostname* does NOT parse as an IP
        (`ipaddress.ip_address` raises `ValueError` for a DNS name) — that `ValueError`
        is expected and handled, never a crash.
    """
    if hostname is None:
        return False
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return hostname.lower() in _LOCAL_HOSTNAMES


def enable_utf8_console_io() -> None:
    """On win32, reconfigure stdout/stderr to UTF-8 with backslashreplace (R26).

    SCOPE NOTE: `run_ollama.py` runs as a SEPARATE CLI subprocess, invoked via Bash by
    `SKILL.md` — this mutates ONLY that subprocess's own `sys.stdout`/`sys.stderr`
    file objects. It cannot reach, and has no effect on, the host Claude Code
    runtime's own streams: an accepted, scoped side effect local to the subprocess's
    process image, not a global-state hazard for the host.

    ``backslashreplace`` is always ASCII-encodable, so writing an already-hard-to-encode
    message can never itself raise. A stream without ``.reconfigure`` (e.g. a test
    double, or an already-substituted non-``TextIOWrapper`` stream) is skipped, never
    raising. No-op on non-Windows platforms.
    """
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def warn_if_config_world_readable(path: str) -> None:
    """POSIX-only, best-effort warning (NR3): *path* (the TOML config, which may hold
    ``api_key``) is group/world-readable.

    Windows ACL semantics differ substantially from POSIX mode bits, so this is a
    documented no-op there rather than a misleading partial enforcement.

    Args:
        path: The config file path. A missing path or a failed ``stat`` is silently
            skipped (best-effort; never raises).
    """
    if os.name != "posix" or not os.path.isfile(path):
        return
    try:
        mode = os.stat(path).st_mode
    except OSError:
        return
    if mode & 0o077:
        print(
            f"WARNING: {path} is group/world-readable and may contain api_key; "
            f"run `chmod 600 {path}`.",
            file=sys.stderr,
        )


def warn_if_remote_http_with_api_key(base_url: str, api_key: str | None) -> None:
    """Warn (NR9) if *base_url* is a non-local ``http://`` endpoint AND an ``api_key``
    is configured — the key would travel in the ``Authorization`` header in clear text.

    "Local" is decided by :func:`_is_local_host` — the full loopback range (any
    ``127.0.0.0/8`` IPv4 address, any IPv6 loopback notation) via stdlib ``ipaddress``,
    falling back to the conventional ``localhost`` hostname check for a non-IP host.

    Args:
        base_url: The resolved base_url.
        api_key: The resolved api_key (or ``None``); never echoed in the warning.
    """
    if not api_key:
        return
    parts = urlsplit(base_url)
    if parts.scheme == "http" and not _is_local_host(parts.hostname):
        print(
            f"WARNING: api_key is sent in clear text over http to a remote host "
            f"({parts.hostname}); use https for LAN/cloud endpoints.",
            file=sys.stderr,
        )


def warn_if_oversize(text: str, threshold: int) -> None:
    """Warn (R24) when *text*'s estimated token count exceeds *threshold*.

    Delegates the estimate itself to :func:`input_size.check_input_size`; this
    function's only responsibility is composing the actionable stderr message. Never
    blocks — the guard is advisory only.

    Args:
        text: The raw (pre-sanitization) input text.
        threshold: The ``--warn-input-tokens`` threshold.
    """
    est, over = check_input_size(text, threshold)
    if over:
        print(
            f"WARNING: input is very large (~{est} estimated tokens > "
            f"{threshold}); consider splitting it.",
            file=sys.stderr,
        )
