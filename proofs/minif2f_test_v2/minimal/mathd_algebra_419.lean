import Mathlib

theorem mathd_algebra_419 (a b : ℝ) (h₀ : a = -1) (h₁ : b = 5) : -a - b ^ 2 + 3 * (a * b) = -39 := by
  simp_all only [neg_neg, neg_mul, one_mul, mul_neg]
  (linarith)
