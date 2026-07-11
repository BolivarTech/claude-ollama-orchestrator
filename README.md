# Claude-Ollama-Orchestrator — Delegate Generation to Ollama from Claude Code

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Runtime](https://img.shields.io/badge/runtime-stdlib--only-success.svg)](#requirements)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange.svg)](https://docs.astral.sh/ruff/)
[![Typecheck](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Release](https://img.shields.io/badge/release-v0.0.7-brightgreen.svg)](https://github.com/BolivarTech/claude-ollama-orchestrator/releases)

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

3. **Create the config file** from the built-in defaults (it refuses to overwrite an existing
   one):

   ```bash
   python skills/ollama/scripts/run_ollama.py --ollama-init
   ```

   This writes `./.claude/ollama-agents.toml`. Edit it to change the models, the endpoint, or
   the concurrency limits (see [Configuration](#configuration)) — or leave it as-is to use the
   defaults.

4. **Delegate** — call `/ollama` in Claude Code (next section).

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

In every case, Ollama generates the output and **Claude reviews it before writing anything to
your files** — delegation never bypasses Claude's judgment.

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

**Bounded concurrency.** `max_parallel_agents` caps how many delegations run at once —
a semaphore ensures no more than N Ollama agents are in flight, matching the maximum
your subscription or load allows. Must be an integer ≥ 1 (invalid → actionable
`ValidationError`).

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
3. **Preflight (fail-fast)** — a quick check that the host is reachable and the configured
   model is available; otherwise it aborts with an actionable message (`ollama pull` /
   `ollama signin` / edit the config). Models are never auto-downloaded.
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
