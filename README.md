# Claude-Ollama-Orchestrator — Delegate Generation to Ollama from Claude Code

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Runtime](https://img.shields.io/badge/runtime-stdlib--only-success.svg)](#requirements)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange.svg)](https://docs.astral.sh/ruff/)
[![Typecheck](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Status](https://img.shields.io/badge/status-spec%20phase-yellow.svg)](#project-status)

A Claude Code plugin that lets Claude **orchestrate a set of Ollama subagents** and
**delegate** concrete generation tasks to them — code, review, tests, explanation,
image/UI analysis, and audio transcription.

Claude stays the **orchestrator and decision-maker**; the token-heavy generation runs
on **local / LAN / cloud open-weight models** served by an OpenAI-compatible Ollama
endpoint. Claude **reviews** the delegated output before applying it with its own
Edit/Write tools. The goal: **cut Anthropic token cost** and keep bulk generation
**on-device**, without ceding Claude's judgment.

> [!IMPORTANT]
> ### Project Status
> **Spec phase.** This repository is being built under the SBTDD workflow (Spec +
> Behavior + Test-Driven Development). The authoritative description of *what* to build
> lives in `sbtdd/`. The tree, tables, and commands below describe the **intended
> design**; production code lands as the spec is implemented test-first.

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

| Subagent | Task | Model grade (tags fixed in the spec) |
|----------|------|--------------------------------------|
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

The plugin is invoked with **`/ollama`** — the slash command is the skill name
(`name: ollama`), the same way `/magi` works (no `commands/` directory).

```
/ollama <capability> [context]     # explicit capability (positional)
/ollama [context]                  # capability omitted → the skill auto-classifies
```

The first token after `/ollama` selects one of the seven capabilities — analogous to
`/magi <mode>`. Omit it and the orchestrator classifies the task and picks the fitting
capability; an invalid capability errors with the valid list.

```
/ollama coder      write a retry decorator with exponential backoff
/ollama vision     <attach a screenshot> what UI pattern is this?
/ollama thinking   should we shard by tenant or by region? weigh the trade-offs
/ollama            <request>            # no capability → auto-routed
```

Scaffold the config once, then delegate:

```bash
# Generate ./.claude/ollama-agents.toml from defaults (refuses to overwrite)
python skills/ollama/scripts/run_ollama.py --ollama-init
```

---

## Installation

### From GitHub (for users)

```bash
# 1. Add this repo as a marketplace source
/plugin marketplace add BolivarTech/claude-ollama-orchestrator

# 2. Install the plugin
/plugin install claude-ollama-orchestrator@bolivartech-ollama-orchestrator

# 3. Scaffold the config, then delegate
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

The runtime is stdlib-only; config lives in `./.claude/ollama-agents.toml` (gitignored —
the file and any API key are never tracked). Scaffold it from defaults with the plugin's
init (does **not** overwrite an existing file).

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

# Visible streaming PER CAPABILITY (R7b): true → live SSE + tok/s, false → transactional
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

```
Claude (orchestrator)
  |  classifies task -> picks capability
  v
SKILL.md (gate: delegate? which capability?)
  |
  v
resolve config (ollama_config)  ->  preflight (ollama_preflight, fail-fast)
  |
  v
backend.run(capability, system_prompt, prompt, model)   # OpenAI-compatible /v1, streaming
  |         \__ stream tokens -> stdout (tok/s + count) __\
  v                                                        v
choices[0].message.content                     token_stats (local accounting)
  |
  v
Claude reviews the output  ->  applies with Edit/Write
```

### Step by Step

1. **Classify & gate** — the orchestrator skill decides whether to delegate and which of
   the seven capabilities fits (explicit via `/ollama <capability>`, or auto-classified).
2. **Resolve config** — layered, per-key merge (`env > repo TOML > global TOML > default`).
3. **Preflight (fail-fast)** — `GET {base_url}/models` with a short timeout. Host
   unreachable or a configured model missing → **abort** with an actionable message
   (`ollama pull` / `ollama signin` / edit the TOML). `401/403` → abort (auth). `404/501` on
   `/models` → warn-and-proceed. **No auto-pull.**
4. **Delegate & stream** — an OpenAI-compatible `POST {base_url}/chat/completions` streams
   tokens to stdout with live speed and count. HTTP calls do not block the event loop
   (`asyncio.to_thread`).
5. **Structured output + backstop** — request JSON structured output where supported; an
   HTTP 400 rejecting `response_format` triggers **one** retry without it (downgrade),
   backed by a tolerant parser that recovers prose / `<think>` leaks.
6. **Account locally** — token usage is recorded per capability/model in a local artifact,
   **separate** from Claude/Anthropic usage.
7. **Review & apply** — Claude reviews the delegated output and applies changes with its own
   Edit/Write tools. If Ollama is unavailable, the plugin **reports** and does **not**
   silently fall back to generating with Claude without explicit authorization.

### Run isolation, output & concurrency (MAGI-level)

Each run writes its artifacts into a unique directory under a **per-project temp
namespace** — `<tempdir>/ollama-runs/<sha256(project_root)[:16]>/ollama-run-*` created
with `mkdtemp` (atomic, collision-free). A `.ollama-lock` (PID + ISO start time +
staleness bound, written atomically) marks a run **live**, so a concurrent session's LRU
cleanup never prunes an in-progress run; the liveness probe is cross-platform and
conservative. `cleanup_old_runs` keeps the most recent `--keep-runs` non-live dirs,
excludes live ones, and refuses to delete anything outside the temp root.

Output is split across **three sinks**:

| Sink | Content |
|------|---------|
| **stdout** | The live token stream (tok/s + count) — **only when a single delegation is in flight** (stdout is serial; streaming N parallel agents there would interleave) |
| **stderr** | Live status tree — the fan-out view: per-agent state + tok/s (ANSI redraw on a TTY, one plain line per update on a pipe, ASCII on cp1252), preflight, warnings |
| **file** | Per-agent `{cap}.stream.log` / `.raw.json` + JSON accounting (`ollama-report.json` / `token_stats.json`) — never dumped to stdout |

Multiple delegations may run in parallel, bounded by a semaphore of size
`max_parallel_agents` (**default 3**) so the plugin never exceeds the concurrency your
Ollama subscription/load allows. Work beyond the running set **queues**, itself hard-capped
by `max_queued_agents` (**default 32**); overflow is rejected fail-closed as a **DoS
backstop** against a runaway fan-out (total ceiling = 3 running + 32 queued = 35).

**When more than one runs at once, streaming to stdout is suppressed** — each delegation
streams to its own file in the run dir and the live view is the status display on stderr;
stdout carries a single delegation's stream only. That "one streamer" invariant holds even
across **independent processes** (e.g. a background delegation + a new request) via a
project-level **`.ollama-stdout.lock`** — the holder streams to the terminal, any concurrent
delegation is file-only until the token frees (self-healing if the holder dies). If a new
request interrupts the turn instead, the running delegation is cleaned up (SIGINT) and the
token is released.

---

## Project Structure (target)

```
.claude-plugin/
  plugin.json                 -- Plugin manifest (name, version, author, repository)
  marketplace.json            -- Local marketplace config (bolivartech-ollama-orchestrator)
skills/ollama/                -- dir == skill name == command (MAGI standard)
  SKILL.md                    -- Orchestrator (name: ollama → /ollama): classification, delegation, fallback
  agents/
    ollama-coder.md           -- write / fix / refactor code
    ollama-reviewer.md        -- security & code review
    ollama-tester.md          -- unit / integration test generation
    ollama-explainer.md       -- code explanation
    ollama-vision.md          -- image / UI analysis
    ollama-transcribe.md      -- audio transcription
    ollama-thinking.md        -- deep analysis / extended reasoning
  scripts/
    __init__.py               -- Python package marker
    run_ollama.py             -- CLI orchestrator (capability positional arg; --ollama-init)
    ollama_init.py            -- renders ollama-agents.toml from defaults (refuse-if-exists)
    ollama_stream.py          -- streaming text delegation (token/speed metrics)
    ollama_vision.py          -- streaming vision delegation
    ollama_config.py          -- layered config resolver (env > repo > global > default)
    ollama_preflight.py       -- fail-fast host + model-availability check
    backend.py                -- OpenAI-compatible /v1 transport (chat/completions)
    token_stats.py            -- local token accounting (independent of Claude usage)
    temp_dirs.py              -- per-project temp namespace, unique run dirs, LRU cleanup
    run_lock.py               -- .ollama-lock (PID/liveness/bound), cross-platform probe
    status_display.py         -- live status tree (ANSI/TTY, plain on pipe, ASCII fallback)
    stderr_shim.py            -- buffers real stderr while the status display renders
    sanitize.py               -- 4-layer anti-prompt-injection + nonce delimiters
    parse_output.py           -- tolerant JSON extractor (<think> recovery, fail-closed, DoS bounds)
    validate.py               -- domain ValidationError + guards + clean_title
    agent_schema.py           -- per-capability JSON-Schema (lockstep with validate.py)
    input_size.py             -- token estimate (chars/4) + oversize warning
    diff_guard.py             -- (optional) unified-diff parser + hallucination guard
tests/                        -- pytest suite (BDD-named, HTTP mocked at the urllib edge)
docs/                         -- user & architecture docs
pyproject.toml                -- Python >= 3.12, dual license, dev deps, tool config
conftest.py                   -- tdd-guard pytest plugin + sys.path setup
Makefile                      -- verify, test, lint, format, typecheck targets
```

> The exact module split is fixed by the spec-base — treat the tree as the intended shape,
> not a contract, until the spec lands.

---

## Testing

Test-first (SBTDD): a pytest suite with BDD-style names, **all HTTP mocked at the `urllib`
edge** (zero network). Property-based checks (`hypothesis`) cover the config resolver,
`base_url` normalization, and `api_key` redaction; fuzzing (`atheris`) hardens the tolerant
parser against untrusted model output.

```bash
python -m pytest tests/ -v    # unit + BDD suite

make verify                   # lockcheck + test + lint + format + typecheck
make test | lint | format | typecheck
```

`make verify` must be clean before every release.

---

## Requirements

| Component | Required | Notes |
|-----------|----------|-------|
| Ollama | Yes | Local daemon, LAN endpoint, or `ollama signin` for `:cloud` models |
| Python 3.12+ | Yes | Uses `tomllib`, `asyncio`, `dict[str, Any]` syntax |
| Runtime dependencies | **None** | stdlib-only: `urllib`, `json`, `tomllib` |

### Dev Dependencies

```bash
pip install pytest pytest-asyncio ruff mypy hypothesis
```

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
