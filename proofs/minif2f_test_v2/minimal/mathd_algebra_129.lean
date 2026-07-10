import Mathlib

theorem mathd_algebra_129 (a : ℝ) (h₀ : a ≠ 0) (h₁ : 8⁻¹ / 4⁻¹ - a⁻¹ = 1) : a = -2 := by
  simp_all only [ne_eq, div_inv_eq_mul]
  (grind only [div_eq_one])
