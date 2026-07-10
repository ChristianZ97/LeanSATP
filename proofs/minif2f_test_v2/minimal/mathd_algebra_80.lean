import Mathlib

theorem mathd_algebra_80 (x : ℝ) (h₀ : x ≠ -1) (h₁ : (x - 9) / (x + 1) = 2) : x = -11 := by
  simp_all only [ne_eq]
  (grind only [div_eq_one])
