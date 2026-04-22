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

def BFS : ExternalGenerator := {
  name := "BFS-Prover-API"
  host := "localhost"
  port := 23338
}

-- `initialize` runs on module load, so every `lake env lean` that
-- imports LeanSATP.BFS registers the generator before any tactic
-- fires. Required for `bfsaesop`'s tacGen path to reach the BFS
-- aggregator on port 23338.
initialize registerGenerator "BFS-Prover" (.external BFS)

-- Paper-citable `bfsaesop` macro.
--   maxGoals := 64     — cite aesop docs (search budget cap)
--   bfsScore := true   — cite aesop (BFS search algorithm)
--   terminal := true   — cite aesop (correctness: only accept closing proofs)
--   rule_sets := [bfs, -builtin, -default]
--     cite LeanCopilot: only tacGen rule in scope. No aesop built-ins,
--     no Mathlib @[aesop] attrs.
--
-- The `set_option LeanCopilot.suggest_tactics.model "BFS-Prover" in`
-- prefix binds the generator model at tactic expansion time, so
-- `bfsaesop` in any calling context resolves tacGen to BFS-Prover even
-- if the caller's file never set the option. LeanCopilot reads the
-- option via `getOptions` at runtime (see LeanCopilot/Options.lean);
-- a top-level `set_option` in THIS file would not propagate through
-- import. LeanCopilot's own examples use the same `set_option ... in`
-- scoping pattern.
macro "bfsaesop" : tactic =>
  `(tactic|
      set_option LeanCopilot.suggest_tactics.model "BFS-Prover" in
      aesop?
        (config := { maxGoals := 64, bfsScore := true, terminal := true })
        (rule_sets := [bfs, -builtin, -default]))
