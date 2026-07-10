import Mathlib

theorem mathd_numbertheory_235 : (29 * 79 + 31 * 81) % 10 = 2 := by
  simp_all only [Nat.reduceMul, Nat.reduceAdd, Nat.reduceMod]
