# Claude-Ollama-Orchestrator — Delegate Generation to Ollama from Claude Code

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Runtime](https://img.shields.io/badge/runtime-stdlib--only-success.svg)](#requirements)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange.svg)](https://docs.astral.sh/ruff/)
[![Typecheck](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Release](https://img.shields.io/badge/release-v0.2.1-brightgreen.svg)](https://github.com/BolivarTech/claude-ollama-orchestrator/releases)

A Claude Code plugin that lets Claude **orchestrate a set of Ollama subagents** and
**delegate** concrete generation tasks to them — code, review, tests, explanation,
image/UI analysis, and audio transcription.

Claude stays the **orchestrator and decision-maker**; the token-heavy generation runs
on **local / LAN / cloud open-weight models** served by an OpenAI-compatible Ollama
endpoint. Claude **reviews** the delegated output before applying it with its own
Edit/Write tools. The goal: **cut Anthropic token cost** and keep bulk generation
**on-device**, without ceding Claude's judgment.

---

## Why Delegate to Ollama?

Generating code and tests is token-heavy and recurrent — the exact workload where a
premium orchestration model is overkill for the bulk output but ideal for judgment.
Two-tier delegation splits those roles:

| Concern | How the plugin addresses it |
|---------|-----------------------------|
| **Cost** | Bulk generation runs on open-weight models → Anthropic token cost drops to ~0 marginal (local) or flat (cloud), reserving Claude's budget for orchestration and review |
| **On-device / privacy** | Code and prompts can stay on the user's hardware (local daemon or LAN endpoint) |
| **Visible streaming** | Unlike an opaque MCP call, delegation **streams every token to the terminal** with live speed (tok/s) and token counts — the user watches progress, not a final block. Built as a **decoupled layer** over a transactional core (toggle `stream`): the core works with or without it; token metrics come from the `usage` object either way |
| **Right model per task** | A distinct model per capability (coder / reviewer / tester / explainer / vision / ASR), resolved from layered config — not one model for everything |
| **Judgment preserved** | Delegated output is **reviewed by Claude** before any file is written; generation never silently bypasses Claude |

The differentiator versus a plain MCP integration is **two-tier delegation with visible
streaming**: Claude decides *what* and *whether* to delegate, Ollama generates, and
Claude reviews the result before it touches your files.

---

## Capabilities

Seven delegable capabilities, each with its own system prompt and its own default
model, resolved from layered config:

| Capability | Task | Default model |
|------------|------|---------------|
| **`ollama-coder`** | Write / fix / refactor code | `kimi-k2.7-code:cloud` |
| **`ollama-reviewer`** | Security & code review | `glm-5.2:cloud` |
| **`ollama-tester`** | Unit / integration test generation | `deepseek-v4-flash:cloud` |
| **`ollama-explainer`** | Code explanation | `gpt-oss:120b-cloud` |
| **`ollama-vision`** | Image / UI analysis | `minimax-m3:cloud` |
| **`ollama-transcribe`** | Audio transcription (+ text/image) | `gemma4:cloud` |
| **`ollama-thinking`** | Deep analysis / extended reasoning | `deepseek-v4-pro:cloud` |

> All defaults are `:cloud` tags (mode A: `ollama signin`, no weight download), overridable
> via config. The Ollama catalog evolves — verify current tags.

---

## Usage

### Getting started

1. **Install the plugin** — see [Installation](#installation).

2. **Sign in for cloud models** (the defaults are `:cloud`). Run this **once** on your local
   Ollama daemon — it does **not** download any weights:

   ```bash
   ollama signin
   ```

3. **Scaffold the config** (optional — the built-in defaults work out of the box). In Claude
   Code, run:

   ```
   /ollama --ollama-init
   ```

   This writes `./.claude/ollama-agents.toml` from the defaults and exits without delegating
   (it refuses to overwrite an existing file). Edit the TOML to change models, the endpoint, or
   the concurrency limits (see [Configuration](#configuration)) — or skip this step entirely to
   use the defaults.

   > Running from a local checkout instead of an installed plugin? The underlying CLI is
   > `python skills/ollama/scripts/run_ollama.py --ollama-init`.

4. **Delegate** — call `/ollama <capability> <request>` in Claude Code (next section).

### Delegating with `/ollama`

The first token picks one of the seven capabilities; omit it and Claude auto-classifies the
request and picks the fitting one. An invalid capability errors with the list of valid ones.

```
/ollama <capability> <request>     # explicit capability
/ollama <request>                  # capability omitted → auto-classified
```

**Text capabilities** (`coder`, `reviewer`, `tester`, `explainer`, `thinking`) take a free-text
request:

```
/ollama coder      write a retry decorator with exponential backoff
/ollama reviewer   review the auth changes in this diff for security issues
/ollama thinking   should we shard by tenant or by region? weigh the trade-offs
/ollama            add unit tests for the parser        # no capability → auto-routed
```

**Media capabilities** (`vision`, `transcribe`) take a **file path** (checked by the file's
magic bytes, not its extension):

```
/ollama vision      path/to/screenshot.png      # analyze an image / UI
/ollama transcribe  path/to/recording.mp3       # transcribe audio (experimental)
```

- `vision` needs a multimodal model (default `minimax-m3:cloud`); a text-only model is
  rejected with a clear error.
- `transcribe` is **experimental**: if the endpoint can't handle audio it fails with an
  actionable message (never a crash), and the other six capabilities keep working.

**The request can also be a file.** For the text capabilities, the request argument is
*path-or-text*: if it names an existing file, the plugin reads that file's contents as the
prompt (up to 10 MB); otherwise it is used as literal text. This is the way to send a long,
detailed brief — a full spec with the tests pasted in — without fighting the shell's argument
length limit (~32 KB on Windows). If the argument *looks* like a path but does not exist, the
plugin warns on stderr and proceeds with it as literal text, so watch for that warning when a
delegation returns something unexpected.

In every case, Ollama generates the output and **Claude reviews it before writing anything to
your files** — delegation never bypasses Claude's judgment.

For how to write a delegation brief that comes back correct, and the full CLI surface, see the
[User Guide](docs/user-guide.md).

---

## Installation

### From GitHub (for users)

```bash
# 1. Add this repo as a marketplace source
/plugin marketplace add BolivarTech/claude-ollama-orchestrator

# 2. Install the plugin
/plugin install claude-ollama-orchestrator@bolivartech-ollama-orchestrator

# 3. For :cloud models, run `ollama signin` once on your daemon — then just use /ollama
```

Update after new versions are published:

```bash
/plugin marketplace update
```

### Local Development

```bash
# Option 1: plugin flag
claude --plugin-dir /path/to/claude-ollama-orchestrator
```

For symlink auto-discovery into `.claude/skills/`, **on Windows use a junction** — true
symlinks need `SeCreateSymbolicLinkPrivilege` and otherwise silently fall back to a
*copy*:

```powershell
# Option 2: junction (Windows) — NOT `mklink /D` and NOT git-bash `ln -s`
New-Item -ItemType Junction -Path .claude\skills\ollama -Target ..\..\skills\ollama
```

Verify it is a real junction (`LinkType: Junction`) and byte-identical (`fc` / `diff -q`).

---

## Configuration

Configuration is **optional** — the built-in defaults work out of the box. To customize
models, the endpoint, or concurrency, create `./.claude/ollama-agents.toml` (gitignored, so
the file and any API key are never tracked). Only the keys you set override the defaults:

```toml
base_url = "http://localhost:11434/v1"   # OpenAI base (path verbatim; bare host:port → /v1)
# api_key = "sk-..."                       # cloud/auth only; local needs none
max_parallel_agents = 3                    # delegations running at once (semaphore; default = Ollama Pro plan)
max_queued_agents   = 32                   # queue cap beyond the running set (DoS backstop; overflow → reject)

[models]
coder      = "kimi-k2.7-code:cloud"     # default (cloud)
reviewer   = "glm-5.2:cloud"            # alt: "glm-5.1:cloud" | "nemotron-3-ultra:cloud"
tester     = "deepseek-v4-flash:cloud"  # alt: "qwen3-coder:480b-cloud"
explainer  = "gpt-oss:120b-cloud"       # alt: "minimax-m3:cloud"
vision     = "minimax-m3:cloud"         # alt: "qwen3.5" | "gemma4:cloud"
transcribe = "gemma4:cloud"             # text+image and audio transcription in cloud
thinking   = "deepseek-v4-pro:cloud"    # default (cloud)

# Output format asked of the model, PER CAPABILITY (response_format):
#   "schema" = JSON matching a strict JSON-Schema | "object" = any JSON | "off" = free text.
# ALL capabilities listed explicitly (even "off") to avoid ambiguity.
[structured]
coder      = "off"       # free text (raw code)
reviewer   = "schema"    # structured findings
tester     = "schema"    # structured test cases
explainer  = "off"       # prose
vision     = "off"       # free text
transcribe = "off"       # free text
thinking   = "off"       # free text (extended reasoning)

# Visible streaming PER CAPABILITY: true → live SSE + tok/s, false → transactional
# (same content + metrics). Long generation → true; structured/short → false.
[stream]
coder      = true
reviewer   = false
tester     = false
explainer  = true
vision     = true
transcribe = true
thinking   = true
```

> **All seven models must be available.** Preflight validates the model of **every**
> capability before any delegation runs (fail-fast), so each entry under `[models]` must
> resolve to a model your endpoint can serve — even capabilities you don't currently use.
> If you only have some models enabled, repoint the others to models you do have (or ensure
> they're reachable via `ollama signin` / `ollama pull`); otherwise every `/ollama` call
> aborts at preflight naming the missing ones.

### Layered precedence (per key)

Config is merged **per key**, so a repo file can override just `base_url` while inheriting
models from the global file:

```
built-in defaults  <  ~/.claude/ollama-agents.toml  <  ./.claude/ollama-agents.toml  <  env
```

| Key | Resolution order |
|-----|------------------|
| `base_url` | `OLLAMA_AGENTS_HOST` → repo → global → `OLLAMA_HOST` → `http://localhost:11434/v1` |
| `api_key` | `OLLAMA_AGENTS_API_KEY` → repo → global → `OLLAMA_API_KEY` → `None` |
| `models.<cap>` | `OLLAMA_AGENTS_MODEL_<CAP>` → repo `[models]` → global `[models]` → default |
| `max_parallel_agents` | `OLLAMA_AGENTS_MAX_PARALLEL` → repo → global → `3` (Ollama Pro default) |
| `max_queued_agents` | `OLLAMA_AGENTS_MAX_QUEUED` → repo → global → `32` |
| `structured.<cap>` | `OLLAMA_AGENTS_STRUCTURED_<CAP>` → repo `[structured]` → global → `off` |
| `stream.<cap>` | `OLLAMA_AGENTS_STREAM_<CAP>` → repo `[stream]` → global → default |

**Bounded concurrency.** `max_parallel_agents` caps how many delegations run at once —
a semaphore ensures no more than N Ollama agents are in flight, matching the maximum
your subscription or load allows. Must be an integer ≥ 1 (invalid → actionable
`ValidationError`). Work beyond that queues, itself capped by `max_queued_agents`
(integer ≥ 0); overflow is rejected fail-closed rather than growing without bound.

### When Ollama retires a model tag

All seven defaults are `:cloud` tags, and Ollama retires tags on short notice. Because
preflight validates **every** configured model, a retired tag on a capability you never use
(say `transcribe`) will still abort a `coder` run. The failure is loud and immediate, not
silent — and the fix is one line:

1. Read the preflight error; it names the missing models and their count.
2. Edit that capability's entry under `[models]` in `./.claude/ollama-agents.toml` to point at
   a model your endpoint serves.
3. Re-run. No code change, no reinstall.

> Do **not** conflate this config (`ollama-agents.toml`) with MAGI's (`magi-ollama.toml`).
> This plugin is the **delegation runtime**; MAGI is the **review gate**.

### Cloud modes

- **Mode A — daemon + signin:** local daemon + `ollama signin`; `:cloud`-tagged models run
  in the cloud without downloading weights, no `api_key` needed.
- **Mode B — direct cloud API:** point `base_url` at a cloud `/v1` endpoint and set
  `api_key`; the request carries `Authorization: Bearer <key>`.

`Authorization` is sent **only** when an `api_key` is present. Keys are never logged, never
written to artifacts, and are **redacted** in error messages.

---

## How It Works

1. **Classify & gate** — Claude decides whether to delegate and which of the seven
   capabilities fits (explicit via `/ollama <capability>`, or auto-classified).
2. **Resolve config** — layered, per-key merge (`env > repo TOML > global TOML > default`).
3. **Preflight (fail-fast)** — a quick check that the host is reachable and that **every
   model configured for the seven capabilities is available**, not only the one you
   invoked. The whole config is validated up front on purpose, so a missing model surfaces
   immediately instead of interrupting you mid-task. If any is absent it aborts with an
   actionable message naming the missing models and their count (`ollama signin` for
   `:cloud` tags / `ollama pull <model>` for local / repoint them in the config). Models
   are never auto-downloaded — you must have all configured models enabled for the plugin
   to run.
4. **Delegate & stream** — an OpenAI-compatible chat request streams tokens to your terminal
   with live speed and count; structured capabilities (reviewer/tester) request strict JSON.
5. **Account locally** — token usage is recorded per capability/model in a local file,
   separate from your Claude/Anthropic usage.
6. **Review & apply** — **Claude reviews the delegated output before applying any change**
   with its own Edit/Write tools. If Ollama is unavailable, the plugin reports it and does
   **not** silently fall back to Claude without your authorization.

**Output & concurrency.** A single delegation streams live to your terminal (tok/s + count);
status and warnings go to stderr; the raw output and token accounting are written to per-run
files. Multiple delegations can run in parallel, capped by `max_parallel_agents` (default 3)
with extra work queued up to `max_queued_agents` (default 32); when more than one runs at once
each streams to its own file (never interleaved on the terminal). Runs are isolated per project,
so a concurrent session never interferes.

---

## Requirements

| Component | Required | Notes |
|-----------|----------|-------|
| Ollama | Yes | Local daemon, LAN endpoint, or `ollama signin` for `:cloud` models |
| Python 3.12+ | Yes | Uses `tomllib`, `asyncio`, `dict[str, Any]` syntax |
| Runtime dependencies | **None** | stdlib-only: `urllib`, `json`, `tomllib` |

---

## Documentation

| Document | What it covers |
|----------|----------------|
| [User Guide](docs/user-guide.md) | Setup, the seven capabilities, the full CLI surface, where run artifacts land, and what each common error means |
| [Architecture](docs/architecture.md) | How the runtime is built: the module map, the three output sinks, bounded concurrency, the temp/lock lifecycle, and the untrusted-content hardening |
| [Delegation best practices](docs/ollama-delegation-best-practices.md) | How to write a delegation brief that comes back correct, and how to review the output |

---

## References

- **Engineering standard (hard reference):**
  [`MAGI-Claude`](https://github.com/BolivarTech/magi-claude) — this repo mirrors its plugin
  layout, layered Ollama config, stdlib-only transport, preflight, structured-output
  handling, TDD tooling, dual license, and release conventions. MAGI is the **review gate**;
  this plugin is the **delegation runtime** — two plugins that share the Ollama backend idea.
- **Behavioral reference (what to build):**
  [`PratikHotchandani22/claude-ollama-agents`](https://github.com/PratikHotchandani22/claude-ollama-agents)
  — Claude Code subagents that stream delegation to Ollama with visible token/speed metrics
  and local token accounting.

---

## License

Dual licensed under [MIT](LICENSE) OR [Apache-2.0](LICENSE-APACHE), at your option.
