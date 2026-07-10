import Mathlib

theorem mathd_numbertheory_345 : (2000 + 2001 + 2002 + 2003 + 2004 + 2005 + 2006) % 7 = 0 := by
  simp_all only [Nat.reduceAdd, Nat.reduceMod]
