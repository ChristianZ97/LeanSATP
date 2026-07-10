import Mathlib

theorem mathd_algebra_314 (n : ℕ) (h₀ : n = 11) : (1 / 4) ^ (n + 1) * 2 ^ (2 * n) = 1 / 4 := by
  simp_all only [Nat.reduceDiv, Nat.reduceAdd, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, Nat.reduceMul, Nat.reducePow, zero_mul]
