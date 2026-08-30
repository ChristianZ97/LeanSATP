import Mathlib

theorem mathd_algebra_208 : Real.sqrt 1000000 - 1000000 ^ ((1 : ℝ) / 3) = 900 := by
  simp_all only [one_div]
  (linarith)
