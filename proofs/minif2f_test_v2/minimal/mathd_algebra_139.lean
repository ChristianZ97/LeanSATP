import Mathlib

theorem mathd_algebra_139 (s : ℝ → ℝ → ℝ)
    (h₀ : ∀ (x) (_ : x ≠ 0), ∀ (y) (_ : y ≠ 0), s x y = (1 / y - 1 / x) / (x - y)) :
    s 3 11 = 1 / 33 := by
  simp_all only [ne_eq, one_div, OfNat.ofNat_ne_zero, not_false_eq_true]
  (linarith)
