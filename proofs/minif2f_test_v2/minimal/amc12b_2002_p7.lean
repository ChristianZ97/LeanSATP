import Mathlib

theorem amc12b_2002_p7 (a b c : ℕ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c) (h₁ : b = a + 1) (h₂ : c = b + 1)
    (h₃ : a * b * c = 8 * (a + b + c)) : a ^ 2 + (b ^ 2 + c ^ 2) = 77 := by
  simp_all only [lt_add_iff_pos_left, add_pos_iff, zero_lt_one, or_true, Nat.ofNat_pos, and_self, and_true]
  (nlinarith)
