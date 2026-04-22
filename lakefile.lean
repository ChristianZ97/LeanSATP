import Lake

open Lake DSL

-- Mathlib + LeanCopilot (dsp+) + aesop (dsp+, ships `bfsScore`) all
-- come transitively from caochenrui/mathlib4's own lakefile. Pinning
-- a commit (not a branch tip) so builds stay reproducible across
-- upstream branch updates. See
--   https://github.com/caochenrui/mathlib4/tree/dsp+
-- for the fork's README; the DSP-Plus paper cites this same fork.
require mathlib from git
  "https://github.com/caochenrui/mathlib4.git" @ "8b97781329"

package LeanSATP where

@[default_target]
lean_lib LeanSATP
