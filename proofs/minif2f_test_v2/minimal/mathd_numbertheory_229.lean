import Mathlib

theorem mathd_numbertheory_229 : 5 ^ 30 % 7 = 1 := by
  simp_all only [Nat.reducePow, Nat.reduceMod]
