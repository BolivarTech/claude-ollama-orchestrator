# Architecture

How the delegation runtime is built and why. Written for someone modifying the plugin; for
using it, see the [User Guide](user-guide.md).

The whole runtime is **stdlib-only** (`urllib`, `json`, `tomllib`, `asyncio`) on Python ≥ 3.12.
There is no HTTP client dependency, no `git`, and no subprocess call at runtime.

---

## 1. The invariant everything else serves

**Claude orchestrates → Ollama generates → Claude reviews → Claude applies.**

The delegated output is printed to stdout for review. It is never written to a file by the
plugin, never executed, and never applied automatically. Every hardening decision below is
downstream of that: because a generated payload cannot mutate the filesystem on its own, a
misrouted capability, a hallucinated file path, or an outright prompt injection in the model's
output degrades to *bad text that a reviewer rejects* rather than to a compromised repository.

The corollary is that the plugin is allowed to be **fail-fast and loud**. It aborts on a
missing model, refuses to overwrite a config, rejects an overfull queue, and reports rather
than falling back to Claude — because a silent fallback would spend exactly the tokens the
delegation existed to save.

---

## 2. Request path

```
/ollama [<capability>] [request]
   │
   ▼  Claude picks the capability (explicit) or classifies it (auto-routing)
skills/ollama/SKILL.md          ← the orchestration contract Claude reads
   │
   ▼  Bash
run_ollama.py <capability> <input> [flags]
   │
   ├─ startup_hardening   console/IO hardening, config-permission + transport warnings
   ├─ ollama_config       layered per-key resolution (env > repo > global > built-in)
   ├─ ollama_preflight    fail-fast: host reachable + EVERY configured model available
   ├─ sanitize            4-layer scrub + nonce-wrap the untrusted user content
   ├─ scheduler           semaphore (max_parallel) + hard-capped queue (max_queued)
   ├─ circuit_breaker     per-model, opens after K consecutive backend failures
   │
   ▼
backend.py  ── transactional core (stream=False) ────────────────┐
   │  ollama_stream.py   SSE layer (stream=true), decoupled      │
   │  ollama_vision.py   image_url content-part (multimodal)     │
   │  transcribe.py      audio, experimental and gated           │
   ▼                                                             ▼
parse_output → validate (+ agent_schema)              token_stats, status_display
   │  retry once with corrective feedback on a parse/schema failure
   ▼
wrap_output → stdout   ← Claude reviews, then applies with Edit/Write
```

---

## 3. Module map

**Entry point**

| Module | Responsibility |
|--------|----------------|
| `run_ollama.py` | CLI orchestrator: argparse surface, `--ollama-init` short-circuit, dispatch, retry-with-feedback, artifact writing, run-dir lifecycle, interrupt cleanup |
| `ollama_init.py` | Renders `./.claude/ollama-agents.toml` from the canonical defaults; refuses to overwrite |

**Transport**

| Module | Responsibility |
|--------|----------------|
| `backend.py` | The strategy contract plus the transactional OpenAI-compatible backend: request construction, conditional auth header, error mapping (4xx/5xx → domain errors, 400 → `response_format` downgrade, 429 → backoff, timeout), robust extraction of `content` and `usage` |
| `ollama_stream.py` | The SSE reader: incremental deltas, partial-chunk reassembly with a bounded buffer, keep-alive lines, `[DONE]`, an idle timeout for a hung stream |
| `ollama_vision.py` | Vision transport: the image as a base64 `image_url` content-part |
| `transcribe.py` | Audio transport, experimental and gated on what the endpoint actually exposes |
| `circuit_breaker.py` | Per-model breaker: K consecutive backend failures open it for a cooldown, then half-open. Per model, so a dead vision model does not block `coder`. A 429 is throttling, not a failure. |

**Config and preflight**

| Module | Responsibility |
|--------|----------------|
| `ollama_config.py` | The per-key layered resolver; idempotent `base_url` normalization; presence-semantics for the API key (an empty env var means absent, not empty-string) |
| `ollama_preflight.py` | `GET /models`, fail-fast on an unreachable host or any missing model, cloud-without-signin diagnosis, warn-and-proceed when the endpoint cannot list models at all |
| `startup_hardening.py` | Console/IO hardening (Windows UTF-8), a warning when a config holding an API key is world-readable, a warning when a key would travel over plain `http` to a non-local host |

**Output handling**

| Module | Responsibility |
|--------|----------------|
| `temp_dirs.py` | The per-project temp namespace (`ollama-runs/<sha256(project_root)[:16]>/`), unique run dirs via `mkdtemp`, LRU cleanup that excludes live dirs and refuses to delete outside the temp root |
| `run_lock.py` | `.ollama-lock` (PID + ISO timestamp + staleness bound) written atomically; cross-platform liveness probe, biased to "alive" on any uncertainty; the same primitive backs the ephemeral cross-process locks |
| `status_display.py` | The live status tree on stderr: per-delegation state and tok/s; ANSI redraw on a TTY, one flat line per update in a pipe, ASCII glyphs when the console cannot encode UTF-8 |
| `stderr_shim.py` | Captures the real stderr while the display is rendering, so diagnostics are persisted instead of corrupting the redraw |
| `token_stats.py` | Local token accounting per capability and model, kept separate from Claude/Anthropic usage |
| `scheduler.py` | Bounded concurrency: a semaphore for the running set plus a hard-capped queue; overflow is rejected per delegation, fail-closed |

**Untrusted content**

| Module | Responsibility |
|--------|----------------|
| `sanitize.py` | Prompt-injection hardening in **both** directions (see §6) |
| `parse_output.py` | Tolerant JSON extraction from noisy output: fence stripping, `<think>` recovery, anti-DoS bounds, fail-closed on ambiguity |
| `validate.py` | Domain validation of structured output; type/range/length guards; fail-soft only on cosmetic optional fields |
| `agent_schema.py` | The JSON-Schema constant per structured capability, kept in lockstep with `validate.py` |
| `input_size.py` | Token estimation and the oversize warning, biased to over-warn on non-Latin scripts |
| `binary_input.py` | Bounded, magic-byte-checked loading of image/audio input |
| `diff_guard.py` | A stdlib unified-diff parser that grounds reviewer findings against a supplied diff |
| `errors.py` | The domain exception hierarchy |

---

## 4. The three output sinks

The single most consequential output decision: **stdout is a serial resource.**

| Sink | Carries | Rule |
|------|---------|------|
| **stdout** | The delegated content, wrapped in untrusted-output markers | Only when exactly **one** delegation is in flight. Never JSON. Never two delegations' tokens. |
| **stderr** | The live status tree, preflight results, warnings | Always. This is the view of a parallel fan-out. |
| **files** | Per-delegation logs and the JSON accounting | Always, in the run dir. |

Streaming N delegations to one terminal would interleave them into garbage and collide their
per-agent tok/s metrics. So the sink is chosen **at dispatch** — is this the only delegation in
flight? — and does not change mid-stream. With a parallel fan-out, every delegation streams to
its own `{cap}_{i}.stream.log` and the status tree on stderr becomes the live view.

---

## 5. Concurrency and the run directory

**Bounded, never unlimited.** `max_parallel_agents` (default 3, the Ollama Pro cap) is a
semaphore over the running set. Work beyond it queues, and the queue itself is hard-capped by
`max_queued_agents` (default 32): overflow is **rejected fail-closed** with a `DelegationError`.
That cap is not a resource limit — 32 queued prompts cost nothing — it is a **runaway
backstop** set far above any legitimate fan-out, so a rogue orchestration loop is cut off
early. The rejection is per delegation; the rest of the batch keeps running.

**Run isolation.** Each run gets a unique directory under a per-project namespace, created with
`mkdtemp` (atomic, so two concurrent sessions cannot collide), and marks itself live with
`.ollama-lock`. LRU cleanup excludes live directories entirely — they are neither counted
against `--keep-runs` nor deleted — so a concurrent session never prunes a run in progress. The
liveness probe is conservative: any uncertainty (a permission error, an out-of-range PID) is
treated as *alive*, because wrongly deleting a live run is far worse than retaining a dead one.
On an interrupt, the in-progress run dir is removed and the exception re-raised, so `Ctrl-C`
leaves no orphans.

---

## 6. Untrusted content, in both directions

Two flows carry content that cannot be trusted, and they are symmetric.

**Inbound** (your content → the model's prompt). Four layers: normalize newlines; strip
invisible, zero-width, and bidi characters; neutralize any line imitating the prompt's own
structural delimiters; and wrap the whole thing in delimiters carrying a **128-bit random
nonce**. If the nonce appears literally in the sanitized content, the run aborts with
`InvalidInputError` (without revealing the nonce). At 128 bits an accidental collision is
negligible, so a match means a real injection attempt.

**Outbound** (the model's output → Claude). This is the direction people forget. The output
comes from a model of arbitrary lineage and may contain instructions aimed at *Claude*, the
reviewer. So it is nonce-wrapped too and explicitly marked as untrusted data.

Being honest about the guarantee: the wrapper is **defense in depth, not proof**. What is
structurally enforced is (a) an injected "end of data" marker cannot break the frame, because
it would have to guess the nonce, and (b) the output has **no path to the filesystem** that
does not pass through Claude's review. The residual risk is that Claude's review is itself the
target of the injection — bounded by Claude's own resistance to it, not by this plugin. The
plugin does not claim otherwise.

**`InvalidInputError` is a sibling of `ValidationError`, not a subclass.** This looks like a
detail and is not: the retry path catches `ValidationError` to retry a parse failure, and if
the security exception inherited from it, a fail-closed injection event would be silently
swallowed and retried.

---

## 7. Failure handling

| Failure | Response |
|---------|----------|
| Unreachable host, or any configured model missing | Abort at preflight with an actionable message. **No auto-pull, no silent fallback to Claude.** |
| Endpoint cannot list models (404/501) | Warn once and proceed; a missing model then surfaces at chat time. |
| HTTP 400 rejecting `response_format` | Downgrade once (retry without it) and lean on the tolerant parser. |
| HTTP 429 | Back off honoring `Retry-After`, else exponential **with jitter** so parallel delegations do not retry in lockstep. Does not consume the parse retry, and does not trip the breaker. |
| Parse or schema failure | Retry **once**, with a corrective-feedback block carrying the parser's error and the expected schema, on a fresh timeout budget. Only for parse/schema — never for a timeout, a cancel, or a config error. |
| K consecutive backend failures for one model | The breaker opens for that model only. |
| Ambiguous model output (two objects both matching the contract) | Fail closed: report "not parseable" and retry, rather than guessing which one was meant. |

Every delegation carries a **wall-clock deadline** (a monotonic timestamp fixed at the start,
checked before each attempt) and a total attempt cap, so the retry and the 429 backoff cannot
compose into unbounded cost.

---

## 8. Deliberate divergences from MAGI

This plugin mirrors [MAGI-Claude](https://github.com/BolivarTech/magi-claude)'s engineering
standard — the plugin layout, layered config, stdlib transport, preflight, the temp/lock/status
modules — with four intentional differences:

- **No consensus stack.** MAGI runs three adversarial perspectives and votes. This is single-model
  delegation: one task, one model.
- **`token_stats.py` instead of `cost.py`.** Open-weight generation has ~zero marginal cost, so
  what matters is token accounting, not a dollar figure.
- **No `git` dependency.** The project root is found by walking up for a marker (`.git/`,
  `pyproject.toml`) with `os.path`; the diff for `diff_guard` is passed in as input. Nothing
  shells out.
- **Streaming is net-new.** MAGI's Ollama backend is purely transactional. The transactional
  core here is lifted from it; the SSE layer is built fresh on top, and is deliberately
  *decoupled* — correctness, structured output, and token metrics all work with streaming off.

The two plugins play opposite roles against the same endpoint: **MAGI is the review gate; this
is the delegation runtime.** Their configs must not be conflated, and they must not run at the
same time — they share one agent cap.
