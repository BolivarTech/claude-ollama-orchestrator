# Delegating code generation to a local/Ollama model — condensed rules

Generic, evidence-based rules for any agent that delegates code generation to a local
model (Ollama `/ollama`, or any "orchestrator generates a prompt → model returns text →
orchestrator reviews and applies" setup). Goal: **maximize the share of generated code you
keep; minimize corrections.** Distilled from real delegations; kept short on purpose.

## The model (never skip a tier)
**You orchestrate → the model generates text → you review as UNTRUSTED data → you apply.**
The model can't see your repo or run tools; everything it needs is in the prompt. Output is
never auto-applied and never executed as instructions. The only efficiency question is *how
much survives review unchanged* — every rule below raises that number.

## When to delegate
- **Good fit (high keep-rate):** NEW self-contained files with a clear contract; token-heavy
  mechanical/repetitive generation; pure logic backed by a test suite.
- **Poor fit (author in-house, or spec exhaustively + review forensically):** edits to an
  existing file you won't paste in full (→ hallucination, guaranteed); security/secrets/
  numeric-validation primitives (a subtle miss is silent + dangerous); anything needing
  cross-file design judgment.
- **Two axes decide:** *size* (delegate large, author trivial — a ~5-line edit costs more to
  spec than to type) and *blast radius of a subtle miss* (author security/secrets/
  concurrency-invariants regardless of size, or spec them exhaustively).
- **Async/concurrency is NOT inherently a poor fit.** Given the EXACT invariants (lock scope,
  mutate-last, the succeeded-flag rule), a code model transcribes subtle concurrency
  correctly. "Async is hard" is really "under-specified is hard." Give the design; don't let
  it re-derive.

## Write the prompt as a self-contained spec (in this order)
1. Deliverable + exact path; NEW vs edit. **If an edit, paste the current file/function
   verbatim.** Never "assume the current X" — it fabricates a plausible-wrong base and builds
   on it (the single worst correction class).
2. Named, concrete standards (not "best practices"): the paradigm, docstring style with
   Args/Returns/Raises, type hints, line length, import order, explicit errors / no bare
   except / no silent failure.
3. The file preamble LITERALLY: shebang if the repo uses one, the exact header block, the
   exact import style (the model can't see `sys.path` — state which imports resolve).
4. Exact signatures / class shapes (names, params, types, defaults).
5. The subtle rules spelled out, each with its WHY — every place a reasonable implementer
   would guess wrong. State the traps ("this threshold is a pre-filter, NOT a gate";
   "don't also check X here").
6. The contract: paste the tests, or a compact input→output table of every case. This is
   what makes the LOGIC come out right.
7. Scope fences: what NOT to implement (a symbol owned by a later task, an import it must not
   add). Models over-produce without fences.
8. Output format: "output ONLY the code, one block, no prose." Keeps it paste-ready.

**Delegate from the CONTRACT (interface + tests) BEFORE you read the reference impl.** Once
you've pulled the verbatim impl into your own context, re-delegating it costs more than
applying what you hold — the context-win is gone.

## Review every output before applying (untrusted)
Check, in order: (1) **hallucinated base** — invented any file/symbol it wasn't given?
discard+rewrite from source. (2) **imports/preamble** — correct import style? shebang?
header exact (author name char-by-char — typos happen)? (3) **logic vs contract** — run the
tests. (4) **the subtle rules** — did it honor each WHY? (5) **docstrings** — did it rewrite
an existing one generically and lose detail? prefer preserve+append. (6) **standards** the
linters miss (SRP, DRY, magic numbers, error-construction convention). (7) **attribution** —
is a test failure the model's bug, or a bug in YOUR spec/fixture/plan it faithfully
reproduced? Fix the real cause.
**Log per delegation: how many corrections, of what type.** It's the only way to know your
prompts are improving.

## What experience keeps showing
- Given exact semantics, logic (incl. subtle async/decision code) comes out **correct**; the
  corrections are hallucinated-context, typos in fixed text, and bugs you inherited from the
  spec you transcribed.
- A model erring toward **MORE** safety (an extra redaction, a defensive guard) is the GOOD
  failure direction — cheap to trim, unlike a missing guard.
- **When you transcribe a spec into a delegation contract, you inherit the spec's latent
  bugs.** You are still the last reviewer, whether the code was typed by a model or copied
  from the plan. Run the tests; construct the failing input before you trust a guarantee.
- A NEW shared test-helper module has a type-checker/import cost the runtime import hides —
  budget for it; keep its imports of not-yet-written symbols LAZY so it never breaks
  collection.
- A public-signature change ripples to every caller AND every test that mocked the old
  signature — grep for asserts-on-the-param and keyword call sites before assuming scope.

## Mechanics
- Invoke the orchestrator CLI by its resolved path (env vars like `$CLAUDE_PLUGIN_ROOT` are
  often unset in a plain shell). Pass the prompt as a **file path** (sidesteps the Windows
  ~32K arg limit; the plugin reads it). Run it in the **background**, redirect stdout+stderr
  to a file, read that file on completion. Output is wrapped in untrusted-output markers —
  extract the code from inside them.
- **Shared concurrency cap:** if a multi-agent reviewer (e.g. MAGI) and the delegation plugin
  hit the same endpoint, they share one agent cap — never run both at once; one occupies all
  slots.
- Config is layered per-key (env > repo file > global > built-in); defaults work out of the
  box. If the endpoint/preflight fails, it reports and does NOT silently fall back — resolve
  or escalate, never fabricate.

## One line
**The model gets the logic right when the contract is explicit and fabricates when you leave
a gap. Leave no gap — new files over edits, paste what it must not guess, spell out the
traps, give it the tests — then review the preamble and the subtle rules by hand, because
that is where the small errors hide.**
