import Mathlib

theorem mathd_algebra_302 : (Complex.I / 2) ^ 2 = -(1 / 4) := by
  simp_all only [one_div]
  (ring)
  simp_all only [Complex.I_sq, one_div, neg_mul, one_mul]
  (bound )
