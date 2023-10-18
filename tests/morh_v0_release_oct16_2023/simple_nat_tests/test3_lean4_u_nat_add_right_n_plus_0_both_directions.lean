/-
----------------
Define unary natural numbers inductively in Lean 4, call the type u_nat.
-/

-- -- Now, we'll define the unary natural numbers.
-- inductive u_nat : Type
-- | zero : u_nat                -- base case
-- | succ : u_nat → u_nat        -- inductive case

-- open u_nat

-- /-
-- Define right addition for unary nats by recursing on the right argument, call add_right in Lean 4.
-- -/

-- -- Function to add (right) two unary natural numbers.
-- def add_right : u_nat → u_nat → u_nat
-- | n, zero     => n                           -- base case: n +_R 0 = n
-- | n, succ m   => succ (add_right n m)              -- inductive step: n +_R succ(m) = succ(n +_R m)


/-
----------------

Let's prove that 0 is a right identity for add_right. i.e., let's prove for all n, add_right n 0 = n, in Lean 4.

Here is the starter code in Lean4:

```lean4
-- Now, we'll define the unary natural numbers.
inductive u_nat : Type
| zero : u_nat                -- base case
| succ : u_nat → u_nat        -- inductive case

-- Function to add (right) two unary natural numbers.
def add_right : u_nat → u_nat → u_nat
| n, zero     => n                           -- base case: n +_R 0 = n
| n, succ m   => succ (add_right n m)              -- inductive step: n +_R succ(m) = succ(n +_R m)
```
Complete the code and produce the theorem in Lean 4 with detailed comments.
-/

-- Now, we'll define the unary natural numbers.
inductive u_nat : Type
| zero : u_nat                -- base case
| succ : u_nat → u_nat        -- inductive case
open u_nat

-- Function to add (right) two unary natural numbers.
def add_right : u_nat → u_nat → u_nat
| n, zero     => n                           -- base case: n +_R 0 = n
| n, succ m   => succ (add_right n m)              -- inductive step: n +_R succ(m) = succ(n +_R m)

-- theorem add_right_zero_eq : ∀ (n : u_nat), add_right n 0 = n :=
-- begin
--   intro n,
--   induction n with n ih,
--   { refl },
--   { rw [add_right, ih], refl }
-- end


theorem add_right_zero_eq : ∀ (n : u_nat), add_right n zero = n :=
  intro n,
  induction n with n ih,
  { refl },
  { rw [add_right, ih], refl }
end



/-
Proof: we want to show (wts) add_right n 0 = n. By the base case of add_right we have that add_right n 0 rewrites to n.
Which completes the proof. QED.
-/

/-
Let's prove that 0 is also the left identity for add_right. i.e., let's prove for all n, add_right 0 n = n, in Lean 4.
-/

/-
Proof: proof goes by induction. Consider any n of type unary nat. We want to show (wts) add_right 0 n = n. Since we
have no rewrite rules from add_right for the 1st argument we need to proceed by cases or induciton. 
We choose induction because it gives us the inductive hypothesis for free. i.e., we get to assume that add_right 0 n = n.
So let's show that add_right 0 (succ n) = succ n in any way we construct a unary natural number.
For the base case we have add_right 0 0 and wts add_right 0 0 = 0. In this case we rewrite add_right 0 0 to 0 by using 
the rewrite rule of the base case of add_right i.e. add_right n 0 = n where n = 0. This completes the base case.
For the inductive case we want to show that if we already have a unary nat n' that has the property P we want 
(i.e., add_right n' 0 = n'), then if we use the (remaining) constructors we can construct a new unary nat n that also 
has P we want i.e., add_right 0 n = add_right n.
Therefore, assume the induction hypothesis (IH) i.e., assume add_right 0 n' = n'. 
And thus we want to show that using succ on n' to construct the next unary nat, we still have the property P i.e.
wts add_right 0 (succ n') = succ n'.
Thus start with the goal add_right 0 (succ n'). We can rewrite this to succ (add_right 0 n') by using the rewrite rule of
add_right. Now the inside of succ is the statement we are assuming by the IH.
Therefore we have succ (add_right 0 n') rewrite to succ n' by using the IH. 
Therefore we have add_right 0 (succ n') = succ n'.
This completes the inductive case. QED.
Therefore we have shown any way to construct unary nats gives us the property we want.
-/