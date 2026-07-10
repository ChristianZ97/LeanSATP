import Mathlib

theorem mathd_algebra_296 : abs ((3491 - 60) * (3491 + 60) - 3491 ^ 2 : ℤ) = 3600 := by
  simp only [Int.reduceSub, Int.reduceAdd, Int.reduceMul, Int.reducePow, abs_neg, Nat.abs_ofNat] at *
