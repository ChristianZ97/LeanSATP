import Mathlib

theorem mathd_numbertheory_314 (r n : ℕ) (h₀ : r = 1342 % 13) (h₁ : 0 < n) (h₂ : 1342 ∣ n)
    (h₃ : n % 13 < r) : 6710 ≤ n := by
  subst h₀
  simp_all only [Nat.reduceMod]
  (grind only [Nat.mod_lt])
