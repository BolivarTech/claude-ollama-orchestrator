# User Guide

How to set up, invoke, and troubleshoot the plugin. The [README](../README.md) is the
overview; this is the working reference. For the architecture behind it, see
[architecture.md](architecture.md).

---

## 1. Setup

### Install

```bash
/plugin marketplace add BolivarTech/claude-ollama-orchestrator
/plugin install claude-ollama-orchestrator@bolivartech-ollama-orchestrator
```

### Enable the models

Every default model is a `:cloud` tag, which runs in Ollama's cloud **without downloading
weights**. Run this once against your local daemon:

```bash
ollama signin
```

If instead you want models running locally, `ollama pull` each one and repoint the config
(next section). Either way, the plugin **never downloads a model for you** — a model that is
not enabled is a preflight failure, not an auto-install.

### Configure (optional)

The built-in defaults work with no config file. To customize:

```
/ollama --ollama-init
```

That writes `./.claude/ollama-agents.toml` from the defaults and exits **without delegating**.
It refuses to overwrite an existing file. The file is under `.claude/`, which is gitignored —
so your endpoint and any API key are never committed.

Config is merged **per key** across four layers, so you can override one thing and inherit the
rest:

```
built-in defaults  <  ~/.claude/ollama-agents.toml  <  ./.claude/ollama-agents.toml  <  env vars
```

The full key list and env-var names are in the README's [Configuration](../README.md#configuration)
section. The keys you are most likely to touch:

| Key | Meaning |
|-----|---------|
| `base_url` | The OpenAI-compatible endpoint. A bare `host:port` gets `/v1` appended; a URL that already has a path is used verbatim. |
| `api_key` | Only for a cloud endpoint that authenticates (mode B). Sent as `Authorization: Bearer` and **only** when present. Never logged, never written to artifacts, redacted in errors. |
| `[models]` | One model per capability. All seven must resolve — see the preflight note below. |
| `max_parallel_agents` | How many delegations run at once (default 3, the Ollama Pro cap). |
| `max_queued_agents` | How many may wait beyond those (default 32). Overflow is rejected, not queued forever. |
| `[structured]` | Per capability: `schema` (strict JSON), `object` (any JSON), `off` (free text). |
| `[stream]` | Per capability: `true` streams tokens live, `false` uses the transactional path (same content, same metrics). |

> Do not confuse `ollama-agents.toml` (this plugin, the **delegation runtime**) with MAGI's
> `magi-ollama.toml` (a separate plugin, the **review gate**). They are different files.

---

## 2. Delegating

```
/ollama <capability> <request>     # explicit
/ollama <request>                  # capability omitted → Claude classifies it
```

| Capability | Use it for | Default model |
|------------|-----------|---------------|
| `coder` | Write, fix, or refactor code | `kimi-k2.7-code:cloud` |
| `reviewer` | Security and quality review | `glm-5.2:cloud` |
| `tester` | Unit / integration test generation | `deepseek-v4-flash:cloud` |
| `explainer` | Explain existing code | `gpt-oss:120b-cloud` |
| `vision` | Analyze an image or a UI screenshot | `minimax-m3:cloud` |
| `transcribe` | Transcribe audio (**experimental**) | `gemma4:cloud` |
| `thinking` | Deep analysis, weighing trade-offs | `deepseek-v4-pro:cloud` |

An invalid capability errors with the list of valid ones. When the capability is omitted and
the request is genuinely ambiguous, the plugin will not guess a capability that *writes* code —
it defaults to a read-only one or asks.

### The request can be a file

For the five text capabilities the request argument is **path-or-text**: if it names an
existing file, that file's contents become the prompt (up to **10 MB**); otherwise the
argument is used as literal text. Passing a file is how you send a long brief — a full
contract with the tests pasted in — without hitting the shell's argument-length limit
(~32 KB on Windows).

If the argument *looks* like a path but no such file exists, you get a warning on stderr and
the text is delegated literally. A delegation that returns something surprising is often a
typo'd path, so read that warning.

For `vision` and `transcribe` the argument is **always a media file path**, validated by the
file's magic bytes rather than its extension (images: PNG/JPEG/WebP; audio: WAV/MP3/FLAC/OGG),
with a 20 MB cap.

### Writing a request that comes back correct

The model cannot see your repository and cannot run tools: **whatever you leave out, it
invents.** The single highest-leverage habit is to paste what it must not guess — the current
file if you are asking for an edit, and the tests if you have them.
[Delegation best practices](ollama-delegation-best-practices.md) covers this in full.

---

## 3. What you get back

The delegated output is printed to **stdout**, wrapped in untrusted-output markers. Claude
reads it, reviews it, and applies it with Edit/Write. **Nothing is auto-applied** — that gate
is the whole point of the two-tier design, and it is what keeps a misrouted or malicious
generation from touching your files.

Alongside stdout, each run writes artifacts to a **run directory**:

| Artifact | Contents |
|----------|----------|
| `{cap}.prompt.txt` | The exact prompt that was sent |
| `{cap}.stream.log` | The token stream as it arrived |
| `{cap}.raw.json` | The raw content returned by the model |
| `{cap}.parsed.json` | The parsed object (structured capabilities only) |
| `{cap}.stderr.log` | Warnings and diagnostics from the run |
| `token_stats.json` | Tokens consumed, per capability and model |
| `ollama-report.json` | Run report: tokens, tok/s, timings, input size, whether a retry fired |

By default that directory is a **managed temp dir**
(`<tempdir>/ollama-runs/<project-hash>/ollama-run-XXXX/`), isolated per project, protected by
a lock so a concurrent session never prunes it, and eventually pruned LRU (`--keep-runs`).
**Its path is not printed**, so if you want the artifacts at a location you can find, pass
`--output-dir <path>` — that opts out of the lock and the pruning, and it is then your job not
to point two concurrent delegations at the same directory.

The token accounting is **local and separate from your Claude/Anthropic usage** — it tells you
what the delegation cost on the Ollama side, nothing more.

---

## 4. CLI reference

The skill invokes this for you; you only need it when running from a checkout or debugging.

```bash
python skills/ollama/scripts/run_ollama.py <capability> <input> [options]
python skills/ollama/scripts/run_ollama.py --ollama-init      # scaffold config, then exit
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | resolved from config | Override the model for this delegation |
| `--timeout` | `900` | Per-delegation timeout, seconds (must be > 0) |
| `--output-dir` | managed temp dir | Write artifacts to a directory you own (no lock, no pruning) |
| `--keep-runs` | built-in | Non-live temp run dirs to retain; `-1` disables cleanup, `0` is rejected as ambiguous |
| `--no-status` | display on | Turn off the live status tree on stderr |
| `--max-parallel` | from config | Override `max_parallel_agents` |
| `--warn-input-tokens` | `150000` | Warn (do not block) above this estimated input size |
| `--diff` | none | A unified diff (path or inline) to ground `reviewer` findings against |

`--diff` is the anti-hallucination guard: findings that cite a file the diff does not contain
are dropped, and ones outside the changed line range are annotated. Without it, the guard is a
no-op.

---

## 5. Troubleshooting

**"Not all configured models are available… Missing N of M."**
Preflight validates the model of **every** capability, not just the one you invoked — so a
retired or unavailable model on a capability you never use will still block a `coder` run.
This is deliberate: it fails loudly and immediately rather than mid-task. Fix it by pointing
that capability's `[models]` entry at a model your endpoint serves, or by enabling the model
(`ollama signin` for `:cloud`, `ollama pull` for local). Ollama retires `:cloud` tags on short
notice, so this is the error you are most likely to meet; the fix is one line of TOML, never a
code change.

**The endpoint is unreachable.**
Preflight aborts before any delegation. The plugin **reports and stops** — it does not silently
fall back to having Claude generate the code, because that would quietly spend the Anthropic
tokens you were trying to save. Start the daemon, fix `base_url`, or authorize a fallback
explicitly.

**A 401 or 403.**
An auth problem: either a missing/invalid `api_key` for a cloud endpoint (mode B), or a daemon
that never ran `ollama signin` for `:cloud` tags (mode A).

**"queue full" (`DelegationError`).**
More delegations were enqueued than `max_parallel_agents + max_queued_agents` allows. This is a
runaway backstop, and the rejection is **per delegation** — the others keep running. Reduce the
fan-out, retry later, or raise the caps.

**`transcribe` fails with "endpoint does not support audio".**
Expected, not a bug: `transcribe` is experimental and gated on what the endpoint actually
exposes. The other six capabilities are unaffected.

**The output looks like it answered a different question.**
Check stderr for the "looks like a file path but does not exist" warning — a typo'd path is
delegated as literal text.

**A rate limit (429).**
Backed off automatically, honoring `Retry-After` when present, with jitter so parallel
delegations do not retry in lockstep. Persistent 429s mean the plan's concurrency is lower than
`max_parallel_agents` — lower it.

---

## 6. One constraint worth knowing

If you also run the MAGI review plugin against the same Ollama endpoint, **the two share one
agent cap** and do not coordinate. MAGI's three-mage panel occupies all three slots, so a
delegation dispatched at the same time will be starved (and a mage starved by a delegation
invalidates the MAGI run). Serialize them: either MAGI is running, or `/ollama` is — never both.
