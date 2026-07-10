import Mathlib

theorem mathd_numbertheory_517 : 121 * 122 * 123 % 4 = 2 := by
  simp_all only [Nat.reduceMul, Nat.reduceMod]
