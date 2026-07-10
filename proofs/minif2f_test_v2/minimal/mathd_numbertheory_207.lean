import Mathlib

theorem mathd_numbertheory_207 : 8 * 9 ^ 2 + 5 * 9 + 2 = 695 := by
  simp_all only [Nat.reducePow, Nat.reduceMul, Nat.reduceAdd]
