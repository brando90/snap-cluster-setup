-- - test1, lazy formalization only task, no sketch of soln, no explicit COT. Task: ask it for the lean formalization of nats.
-- 	- prompt:
-- 	- @morph-lean-aide-v0#3025  Give me the formalization in the Lean theorem prover of natural numbers.
-- ref: https://discord.com/channels/1117623339456933940/1161025563708903445

inductive u_nat : Type
| zero : u_nat
| succ : u_nat -> u_nat

-- def add (a b : nat) : nat :=
-- match a, b with
-- | zero, _ => _
-- | succ _, _ => _
-- end

-- def add (a b : Nat) : Nat :=
-- match a, b with
-- | zero, _ => b
-- | succ _, _ => succ (add a.succ b)
-- end

-- fixed to use our nat definiion and missing match statements for correct recursion
-- def add (a b : nat) : nat :=
-- match a, b with
-- | zero, b => b
-- | succ a', b => succ (add a' b)
-- end

-- fixed to use our nat definiion and missing match statements for correct recursion
-- def add (a b : u_nat) : u_nat :=
-- match b with
-- | zero, b => b
-- | succ a', b => succ (add a' b)

-- def add: u_nat -> u_nat -> u_nat :=
-- | zero, b => b
-- | succ a', b => succ (add a' b)

-- -- gpt4 still doesn't work, doesn't know to remove end, likely due to lean 3 syntax https://chat.openai.com/c/4e4a12f9-e4d5-4453-8f76-6b9249e695a2
-- -- def add (a b : u_nat) : u_nat :=
-- --   match a with
-- --   | u_nat.zero => b
-- --   | u_nat.succ a' => u_nat.succ (add a' b)
-- --   end
-- def add (a b : u_nat) : u_nat :=
--   match a with
--   | u_nat.zero => b
--   | u_nat.succ a' => u_nat.succ (add a' b)


def add: u_nat -> u_nat -> u_nat
  | u_nat.zero, b => b
  | u_nat.succ a', b => u_nat.succ (add a' b)


-- def add : u_nat → u_nat → u_nat
-- | n, zero => n
-- | n, succ n' => succ (add n n')


-- GPT4 ChatGPT output: https://chat.openai.com/c/4976cd02-9ef7-43b5-96d9-b0a1bed4107f
-- prompt:
-- Give me the formalization in the Lean theorem prover of natural numbers.

-- only code, it did give a nice code block with comments

-- This is usually not necessary as Lean already has a built-in definition of natural numbers, called `nat`.
-- The `nat` type and its basic operations and properties are defined in Lean's standard library.
-- But for explanatory purposes, we can look at how it's fundamentally approached.

-- First, we would declare a namespace to avoid name conflicts

open u_nat

def add2 : u_nat → u_nat → u_nat
| zero, n => n
| succ m, n => succ (add m n)

-- -- Proving properties about this addition operation would follow.
-- -- As an example, let's prove right identity of our addition.
-- theorem add_zero (m : my_nat) : add m zero = m :=
-- match m with
-- | zero    => rfl
-- | succ m' => congr_arg succ (add_zero m')
-- end

-- -- Similarly, we could define multiplication, etc.

-- -- We can also prove properties about these operations, like the associativity of addition.
-- theorem add_assoc (m n k : nat) : add (add m n) k = add m (add n k) :=
-- begin
--   -- proof goes here, using induction and the definitions above
--   -- Lean's powerful automation and standard library can help a lot with these kinds of proofs.
-- end

-- end of the namespace
