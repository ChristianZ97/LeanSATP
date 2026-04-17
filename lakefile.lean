import Lake

open Lake DSL

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.26.0"

package LeanSATP where

@[default_target]
lean_lib LeanSATP
