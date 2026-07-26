import Lake

open Lake DSL

-- v4.27 standalone: mathlib-dsp+ = official stock v4.27 Mathlib
-- (leanprover-community @ a3a10db0e9) + aesop-dsp+ (official aesop @ cb837cc +
-- our bfsScore patch) + LeanCopilot (official lean-dojo @ v4.27.0), all built
-- into deps/ by ./setup.sh (self-contained; caochenrui's dsp+ fork only exists
-- at v4.17, so we replicate its lakefile structure on v4.27 OFFICIAL sources).
require mathlib from "deps/mathlib4"

package LeanSATP where

@[default_target]
lean_lib LeanSATP
