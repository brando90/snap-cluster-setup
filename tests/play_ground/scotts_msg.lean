-- https://csphdcommunity.slack.com/archives/C045D60CYGL/p1696363892506999?thread_ts=1696293186.383839&cid=C045D60CYGL
import Mathlib.Tactic

#synth Add Nat -- instAddNat
#check instAddNat

#check (. + .) -- use goto def
#synth HAdd Nat Nat Nat
#check instHAdd

#check Nat.comm -- after typing `m`, wait for autocompletion popup

example [Ring α] : ∀ (a b : α), a + b = b + a := by exact?