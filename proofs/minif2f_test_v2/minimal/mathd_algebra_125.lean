import Mathlib

theorem mathd_algebra_125 (x y : ℕ) (h₀ : 0 < x ∧ 0 < y) (h₁ : 5 * x = y)
    (h₂ : ↑x - (3 : ℤ) + (y - (3 : ℤ)) = 30) : x = 6 := by
  subst h₁
  simp_all only [Nat.ofNat_pos, mul_pos_iff_of_pos_left, and_self, Nat.cast_mul, Nat.cast_ofNat]
  (linarith)
