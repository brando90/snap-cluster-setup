(* 
Define unary natural numbers inductively in Coq, call the type u_nat. 

ref: https://coq.vercel.app/scratchpad.html
*)
Inductive u_nat : Type :=
| zero: u_nat
| succ: u_nat -> u_nat.

Print u_nat.

Fixpoint add_right (n m : u_nat) : u_nat :=
  match m with
  | zero => n  (* if m is zero, return n *)
  | succ m' => succ (add_right n m')  (* if m is the successor of m', return the successor of (add_right n m') *)
  end.
  
Print add_right.

Theorem zero_right_identity:
  forall n : u_nat, add_right n zero = n.
Proof.
  intros n.
  unfold add_right.
  reflexivity.
Qed.

Theorem add_right_zero : forall n : u_nat, add_right n zero = n.
Proof.
  intro n.  (* Consider an arbitrary u_nat n *)
  simpl.
  reflexivity.
Qed.

Theorem zero_left_identity:
  forall n : u_nat, add_right zero n = n.
Proof.
    intros n.
    induction n as [| n' IHn'].
    - reflexivity.
    - simpl. rewrite -> IHn'. reflexivity.
Qed.

Theorem zero_right_identity2_af_proof : forall (n : u_nat), 
add_right n zero = n.
Proof.
  intros n.
  unfold add_right.
  reflexivity.
Qed.