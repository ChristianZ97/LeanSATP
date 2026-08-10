import LeanCopilot
import Aesop
import LeanSATP.RulesetInit

set_option maxHeartbeats 0

-- Register LeanCopilot's tacGen as an Aesop rule SCOPED TO `bfs` ruleset
-- only. A bare `aesop?` (rule_sets defaults to `[default, builtin]`) does
-- NOT see tacGen — this is the load-bearing invariant that keeps the
-- paper's `aesop_plain` mode free of vLLM side-channels.
--
-- `bfsaesop` (below) explicitly opts in with `rule_sets := [bfs, ...]`,
-- so every `bfsaesop` closure is attributable to the 7B BFS-Prover
-- tacGen. Do not relax this scoping.
@[aesop 100% (rule_sets := [bfs])] def tacGen := LeanCopilot.tacGen

open Lean Meta LeanCopilot

-- Runtime containment for the sampled search: a vLLM-sampled candidate
-- can blow `maxRecDepth` during eval, and runtime exceptions skip every
-- plain `try`/`first` on the way out (`Core.tryCatch` rethrows them) —
-- aborting the whole gap and, in cascade wraps, shadowing later
-- branches. Convert depth blowups (only) to a regular recoverable
-- failure; heartbeats, ordinary failures, and interrupts propagate
-- unchanged (see the handler below).
open Elab Tactic in
elab "containRuntime " t:tacticSeq : tactic => do
  -- Message watermark: a nested `by` inside a sampled candidate that
  -- blows maxRecDepth is ADMITTED via sorryAx with the error merely
  -- LOGGED (Lean SyntheticMVars runTactic, errToSorry=true) — no
  -- exception ever escapes for the catch below to see. A "success"
  -- that logged new error-severity messages fails the file (rc=1)
  -- anyway, so fail the branch and let `first`'s restore scrub them.
  let errCount : MessageLog → Nat := fun l =>
    (l.toList.filter (fun m => m.severity matches .error)).length
  let msgsBefore ← Core.getMessageLog
  match ← tryCatchRuntimeEx (Except.ok <$> evalTactic t)
      (fun e => do
        -- Only depth blowups are converted to a retryable failure.
        -- Heartbeat exhaustion is monotonic across backtracking and
        -- ordinary failures keep their identity — both rethrow.
        if e.isMaxRecDepth then pure (Except.error e.toMessageData)
        else throw e) with
  | .ok _ =>
      if errCount (← Core.getMessageLog) > errCount msgsBefore then
        throwError "bfsaesop: emitted error diagnostics contained"
  | .error msg => throwError "bfsaesop: runtime exception contained: {msg}"

-- Default model identity = Hugging Face path of the BFS-Prover model
-- we evaluated against. Swap this string and re-register below to point
-- `bfsaesop` at a different vLLM-backed generator; the macro expansion
-- re-sets `LeanCopilot.suggest_tactics.model` to the same literal, so
-- both sides must stay in sync. `host` / `port` target whatever HTTP
-- server you expose (our eval uses a BFS-Prover aggregator on
-- `localhost:23338`).
def BFS : ExternalGenerator := {
  name := "ByteDance-Seed/BFS-Prover-V2-7B"
  host := "localhost"
  port := 23338
}

-- `initialize` runs on module load, so every `lake env lean` that
-- imports LeanSATP.BFS registers the generator before any tactic
-- fires. The registered name MUST match `BFS.name` above (and the
-- literal in the `bfsaesop` macro below) — LeanCopilot looks up
-- generators by this string at tactic execution time.
initialize registerGenerator "ByteDance-Seed/BFS-Prover-V2-7B" (.external BFS)

-- Paper-citable `bfsaesop` macro.
--   maxGoals := 64     — cite aesop docs (search budget cap)
--   bfsScore := true   — cite aesop (BFS search algorithm)
--   terminal := true   — cite aesop (correctness: only accept closing proofs)
--   rule_sets := [bfs, -builtin, -default]
--     cite LeanCopilot: only tacGen rule in scope. No aesop built-ins,
--     no Mathlib @[aesop] attrs.
--
-- The `set_option LeanCopilot.suggest_tactics.model "..." in` prefix
-- pins the generator at tactic expansion time, so `bfsaesop` always
-- resolves tacGen to the same model regardless of the caller's current
-- options. LeanCopilot reads the option via `getOptions` at runtime
-- (see LeanCopilot/Options.lean); a top-level `set_option` in this
-- file would not propagate through `import`. LeanCopilot's own
-- examples use the same `set_option ... in` scoping pattern.
--
-- To swap generators, edit the three occurrences of
-- `"ByteDance-Seed/BFS-Prover-V2-7B"` (this macro + `BFS.name` above +
-- `registerGenerator` above) to your model identifier, and point
-- `BFS.host` / `BFS.port` at your server. The rest of the pipeline is
-- model-agnostic.
macro "bfsaesop" : tactic =>
  `(tactic|
      containRuntime
        set_option LeanCopilot.suggest_tactics.model "ByteDance-Seed/BFS-Prover-V2-7B" in
        aesop?
          (config := { enableSimp := false, enableUnfold := false, maxGoals := 64, bfsScore := true, terminal := true })
          (rule_sets := [bfs, -builtin, -default]))
