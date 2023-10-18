inductive u_nat: Type
| zero: u_nat
| succ: u_nat -> u_nat

open u_nat

def add_right: u_nat -> u_nat -> u_nat
| n, zero => n
| n, succ m' => succ (add_right n m')

def add_right2: u_nat -> u_nat -> u_nat :=
  fun n m =>
    match m with
      | zero => n
      | succ m' => succ (add_right2 n m')

/-
by start tactic mode, ow your writing the proof term.
-/
theorem zero_is_right_identity0: ∀ n: u_nat, add_right n zero = n := by intro n; rfl
theorem zero_is_right_identity1: ∀ n: u_nat, add_right n zero = n := by simp[add_right]
theorem zero_is_right_identity2: ∀ n: u_nat, add_right n zero = n := by intro n; unfold add_right; rfl

-- #print u_nat.rec
-- #print u_nat.induction_on

theorem zero_is_left_identity: ∀ n: u_nat, add_right zero n = n := by 
  intro n
  induction n with 
  | zero => rfl
  -- | succ n' ih => simp[add_right, ih]
  | succ n' ih => 
    unfold add_right
    rw[ih]

