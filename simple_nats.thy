(*
ref: https://chat.openai.com/c/6881a873-3345-4883-86ba-4184a2f11fc4
ref: https://claude.ai/chat/afba3ae8-6425-4919-881b-d867eb1c7162
ref: https://discord.com/channels/1117623339456933940/1161025563708903445/1162113353657880616
*)
theory simple_nats
imports Main
begin

(* Define a unary representation of natural numbers *)
datatype u_nat = Zero | Suc u_nat

(* Define a function for addition of our unary natural numbers *)
fun add :: "u_nat ⇒ u_nat ⇒ u_nat" where
"add Zero     n = n" |  (* Adding zero to any number n gives n *)
"add (Suc m) n = Suc (add m n)"  (* Recursive case: peeling off one Suc and add it later *)

(* Let's do some evaluations to check our understanding *)
value "Zero"  (* Evaluate 0 *)
value "Suc Zero"  (* Evaluate 1 *)
value "Suc (Suc Zero)"  (* Evaluate 2 *)

(* Evaluate addition of some simple unary numbers *)
value "add Zero (Suc Zero)"  (* Evaluate 0 + 1 *)
value "add (Suc Zero) (Suc Zero)"  (* Evaluate 1 + 1 *)

(* Prove that adding zero to a number results in the same number: n + 0 = n *)
lemma add_n_0: "add n Zero = n"
proof (induct n)
  (* Base Case: Prove for Zero. So we need to show: add Zero Zero = Zero.
     This follows directly from the definition of add. *)
  show "add Zero Zero = Zero" by simp
  (* Inductive Step: Assume hypothesis (IH) holds for n = m. i.e., add m Zero = m.
     Now show it holds for n = Suc m. i.e., add (Suc m) Zero = Suc m. *)
next
  fix m assume IH: "add m Zero = m"  (* Inductive Hypothesis *)
  (* Need to show: add (Suc m) Zero = Suc m.
     We can use our function definition of add to rewrite the LHS to Suc (add m Zero).
     Then apply IH. *)
  show "add (Suc m) Zero = Suc m" using IH by simp
qed

(* Prove that zero added to a number results in the same number: 0 + n = n *)
lemma add_0_n: "add Zero n = n"
  by simp  (* This proof follows directly from the definition of add. *)

end
