import Mathlib

theorem mathd_numbertheory_212 : 16 ^ 17 * 17 ^ 18 * 18 ^ 19 % 10 = 8 := by
  simp_all only [Nat.reducePow, Nat.reduceMul, Nat.reduceMod]
