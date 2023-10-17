(* 
https://coq.vercel.app/scratchpad.html

ref: https://chat.openai.com/c/199d77c8-95d9-4e45-b8c0-6dc38820ad8e
*)

(*
Define unary natural numbers inductively in Coq, call the type u_nat.
*)
Inductive u_nat : Type :=
| zero: u_nat
| succ: u_nat -> u_nat.

Print u_nat.

(*
Define right addition for unary nats by recursing on the right argument, call add_right in Coq.
*)
Fixpoint add_right (n m : u_nat) : u_nat :=
  match m with
  | zero => n  (* if m is zero, return n *)
  | succ m' => succ (add_right n m')  (* if m is the successor of m', return the successor of (add_right n m') *)
  end.
  
Print add_right.

(*
Let's prove that 0 is a right identity for add_right. i.e., let's prove for all n, add_right n 0 = n, in Coq.
*)
(* My answer *)
Theorem zero_right_identity:
  forall n : u_nat, add_right n zero = n.
Proof.
  intros n.
  unfold add_right.
  reflexivity.
Qed.

(* GPT4's answer *)
Theorem add_right_zero : forall n : u_nat, add_right n zero = n.
Proof.
  intro n.  (* Consider an arbitrary u_nat n *)
  simpl.
  reflexivity.
Qed.

(*
Translate the following theorem and proof into Coq:

Theorem zero_right_identity2_af_proof: for all n, add_right n 0 = n.
Proof: 
we want to show (wts) add_right n 0 = n. 
By the base case of add_right we have that add_right n 0 rewrites to n.
Which completes the proof. QED.

Here is the starting script:
Inductive u_nat : Type :=
| zero: u_nat
| succ: u_nat -> u_nat.

Fixpoint add_right (n m : u_nat) : u_nat :=
  match m with
  | zero => n  (* if m is zero, return n *)
  | succ m' => succ (add_right n m')  (* if m is the successor of m', return the successor of (add_right n m') *)
  end.

ref: https://chat.openai.com/c/d24f06ab-fd60-45bc-b1f1-39d22c26779b
*)

(*GPT4 answer*)
(* First, we define our unary natural numbers. *)
Inductive u_nat : Type :=
  | zero : u_nat
  | succ : u_nat -> u_nat.

(* Next, we define a function for addition on the right. *)
Fixpoint add_right (n m : u_nat) : u_nat :=
  match m with
  | zero => n  (* if m is zero, return n *)
  | succ m' => succ (add_right n m')  (* if m is the successor of m', return the successor of (add_right n m') *)
  end.

(* Now, we state and prove the theorem that zero is a right identity for add_right. *)
Lemma zero_right_identity2_af_proof : forall n, add_right n zero = n.
Proof.
  intros n. (* Introduce the arbitrary number n *)
  simpl. (* Simplify the goal using the definition of add_right *)
  reflexivity. (* Since both sides of the equality are the same, we can use reflexivity to complete the proof *)
Qed.


(* 
Let's prove that 0 is also the left identity for add_right. i.e., let's prove for all n, add_right 0 n = n, in Coq.
*)

(* My answer *)
Theorem zero_left_identity:
  forall n : u_nat, add_right zero n = n.
Proof.
    intros n.
    induction n as [| n' IHn'].
    - reflexivity.
    - simpl. rewrite -> IHn'. reflexivity.
Qed.

(*
Translate the following theorem and proof into Coq:

Theorem add_right_zero : forall n : u_nat, add_right zero n = n.
Proof: 
proof goes by induction. 
Consider any n of type unary nat and use 0 as short hand for zero of type u_nat. 
We want to show (wts) add_right 0 n = n. 
Since we have no rewrite rules from add_right for the 1st argument we need to proceed by cases or induciton. 
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

ref: https://chat.openai.com/c/cda00fe7-c668-4458-8dbd-cbba26ce8e69
*)

(* Defining unary natural numbers *)
Inductive u_nat : Type :=
  | zero : u_nat
  | succ : u_nat -> u_nat.

(* Defining addition for unary natural numbers *)
Fixpoint add_right (n m : u_nat) : u_nat :=
  match m with
  | zero => n (* this is the base case for add_right *)
  | succ m' => succ (add_right n m') (* this is the recursive case *)
  end.

(* Theorem statement *)
Theorem add_right_zero : forall n : u_nat, add_right zero n = n.
Proof.
  intro n. (* Consider any n of type u_nat *)
  induction n as [| n' IHn']. (* Proceed by induction on n *)
  
  - (* Base case: n = zero *)
    simpl. (* simplifies the expression, using the base case of add_right *)
    reflexivity. (* This shows that add_right zero zero = zero, thus proving the base case *)
    
  - (* Inductive case: n = succ n' where the induction hypothesis IHn' is add_right zero n' = n' *)
    simpl. (* simplifies the expression, using the recursive case of add_right *)
    rewrite IHn'. (* Using the induction hypothesis, we know add_right zero n' = n', so we rewrite to succ n' *)
    reflexivity. (* This shows that add_right zero (succ n') = succ n', thus proving the inductive step *)
Qed.


