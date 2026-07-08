import Lake

open Lake DSL

-- v4.26 standalone: mathlib-dsp+ = official stock v4.26 Mathlib
-- (leanprover-community @ 2df2f0150c) + aesop-dsp+ (official aesop @ 2f6d238 +
-- our bfsScore patch) + LeanCopilot (official lean-dojo @ v4.26.0), all built
-- into deps/ by ./setup.sh (self-contained; caochenrui's dsp+ fork only exists
-- at v4.17, so we replicate its lakefile structure on v4.26 OFFICIAL sources).
require mathlib from "deps/mathlib4"

package LeanSATP where

@[default_target]
lean_lib LeanSATP
