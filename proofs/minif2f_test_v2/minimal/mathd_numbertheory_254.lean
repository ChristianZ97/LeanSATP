import Mathlib

theorem mathd_numbertheory_254 : (239 + 174 + 83) % 10 = 6 := by
  simp_all only [Nat.reduceAdd, Nat.reduceMod]
