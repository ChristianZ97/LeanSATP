import Mathlib

theorem mathd_algebra_275 (x : ℝ) (h : ((11 : ℝ) ^ (1 / 4)) ^ (3 * x - 3) = 1 / 5) :
    ((11 : ℝ) ^ (1 / 4)) ^ (6 * x + 2) = 121 / 25 := by
  simp_all only [Nat.reduceDiv, pow_zero, Real.one_rpow, one_div, one_eq_inv, OfNat.ofNat_ne_one]
