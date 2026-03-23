import Mathlib
import LeanSATP

example (P : Prop) (h : P) : P := by
  satp

example : True := by
  satp

example (P : Prop) (h : P) : P := by
  have h' : P := by
    satp
  exact h'

example (P Q : Prop) (hP : P) (hQ : Q) : P ∧ Q := by
  constructor
  · satp
  · satp

example (P Q : Prop) (hP : P) : Q → P := by
  intro hQ
  have h' : P := by
    satp
  exact h'
