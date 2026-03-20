import Mathlib
import LeanSATP

example (P : Prop) (h : P) : P := by
  satp

example : True := by
  satp
