import Mathlib

theorem mathd_numbertheory_85 : 1 * 3 ^ 3 + 2 * 3 ^ 2 + 2 * 3 + 2 = 53 := by
  simp_all only [Nat.reducePow, one_mul, Nat.reduceMul, Nat.reduceAdd]
