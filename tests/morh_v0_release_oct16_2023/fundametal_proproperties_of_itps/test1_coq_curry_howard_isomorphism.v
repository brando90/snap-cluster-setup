(* 

Give me the Coq command for inspecting definitions. 
In particular I want to see or print the induction principle for
natural numbers. 
*)
Inductive u_nat : Type :=
| zero : u_nat (* zero == 0 *)
| succ : u_nat -> u_nat (* succ n == n + 1 *)
.

#Print u_nat_nat