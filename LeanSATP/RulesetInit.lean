import Aesop

-- Declare the `bfs` ruleset so `@[aesop 100% (rule_sets := [bfs])]`
-- in LeanSATP.BFS can register `tacGen` into a dedicated ruleset that
-- bare `aesop?` does NOT see (bare aesop defaults to `[default, builtin]`).
-- This keeps `aesop_plain` mode free of vLLM side-channels.
declare_aesop_rule_sets [bfs]
