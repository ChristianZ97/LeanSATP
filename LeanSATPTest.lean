import Mathlib
import LeanSATP

/-! Smoke tests for the `satp` / `satp?` tactics. -/

-- Basic: `satp` dispatches a simple propositional goal.
example (P : Prop) (h : P) : P := by
  satp

example : True := by
  satp

-- Mathlib-aware arithmetic: the SATP model returns Mathlib tactics
-- (ring / omega / nlinarith / positivity) and they must elaborate.
example (n : Nat) (h : n > 0) : n + 1 > 1 := by
  satp

example (a b : Nat) : a + b = b + a := by
  satp

-- Extra user-supplied lemmas after the list are appended as unsafe rules.
example (a b : Nat) : a + b = b + a := by
  satp [Nat.add_comm]

-- `satp?` prints a Try-this suggestion with the concrete tactic it ran.
example (a b : Nat) (h : a = b) : a + 1 = b + 1 := by
  satp?

-- `satp` used inside `have`: the collected formal statement has the
-- outer binders but the have's target becomes the goal.
example (n : Nat) (hn : n > 0) : n * 2 ≥ 2 := by
  have h1 : n ≥ 1 := by satp
  linarith

-- Chained `have`s all become binders of the next satp invocation.
example (n : Nat) (hn : n > 0) : n * 2 > 1 := by
  have h1 : n ≥ 1 := hn
  have h2 : n * 2 ≥ 2 := by satp
  satp

-- Nested constructor + satp on each subgoal.
example (P Q : Prop) (hP : P) (hQ : Q) : P ∧ Q := by
  constructor
  · satp
  · satp

example (P Q : Prop) (hP : P) : Q → P := by
  intro hQ
  have h' : P := by satp
  exact h'
