import Mathlib

theorem mathd_algebra_188 (σ : Equiv ℝ ℝ) (h : σ.1 2 = σ.2 2) : σ.1 (σ.1 2) = 2 := by
  simp_all only [Equiv.toFun_as_coe, Equiv.invFun_as_coe, Equiv.apply_symm_apply]
