import Mathlib

theorem aime_1990_p4 (x : ℝ) (h₀ : 0 < x) (h₁ : x ^ 2 - 10 * x - 29 ≠ 0)
    (h₂ : x ^ 2 - 10 * x - 45 ≠ 0) (h₃ : x ^ 2 - 10 * x - 69 ≠ 0)
    (h₄ : 1 / (x ^ 2 - 10 * x - 29) + 1 / (x ^ 2 - 10 * x - 45) - 2 / (x ^ 2 - 10 * x - 69) = 0) :
    x = 13 := by
  simp_all only [ne_eq, one_div]
  (bound )
  (field_simp [*] at *)
  simp_all only [one_div]
  (bound )
  simp_all only
  (field_simp [*] at *)
  simp_all only [zero_mul]
  (nlinarith)
