import LeanSATP.Bridge

-- BFS tree search (`bfsaesop`) is opt-in: `import LeanSATP.BFS`. It
-- depends on LeanCopilot, which downstream packages must add to their
-- own lakefile. Keeping BFS off the top-level re-export keeps
-- `import LeanSATP` self-contained to Mathlib-only for the SATP tactic
-- users who don't need vLLM tree search.
