import Mathlib

theorem mathd_numbertheory_299 : 1 * 3 * 5 * 7 * 9 * 11 * 13 % 10 = 5 := by
  simp_all only [one_mul, Nat.reduceMul, Nat.reduceMod]
