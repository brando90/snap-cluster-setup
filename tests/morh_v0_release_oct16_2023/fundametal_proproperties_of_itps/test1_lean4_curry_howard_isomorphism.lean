/-
Give me the Lean 4 command for inspecting definitions. 
In particular I want to see or print the induction principle for my 
definition of (unary) atural numbers (which I called u_nat).

-- My definition of unary nat in lean 4
inductive u_nat : Type
| zero : u_nat -- base case zero == 0
| succ : u_nat -> u_nat -- inductive case: succ n == n + 1
-/

inductive u_nat : Type
| zero : u_nat -- base case zero == 0
| succ : u_nat -> u_nat -- inductive case: succ n == n + 1

#print u_nat.rec
