import Mathlib

theorem mathd_numbertheory_769 : (129 ^ 34 + 96 ^ 38) % 11 = 9 := by
  simp_all only [Nat.reducePow, Nat.reduceAdd, Nat.reduceMod]
