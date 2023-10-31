-- -- ref: https://chat.openai.com/c/86d847e3-fc4c-428b-b87b-664f955558ea
-- -- ref: https://claude.ai/chat/f0a965f6-f782-440d-a7c2-3624ca4fcf8e
-- -- ref: 
-- -- Skipping importing default nat libraries
-- -- import data.nat.basic -- imports standard nats in lean

-- ---- 1. Defining unary natural numbers
-- inductive u_nat : Type
-- | zero : u_nat -- #0
-- | succ : u_nat → u_nat -- #1
-- -- Open the namespace to use `zero` and `succ` without prefix. So you don't have to do unary_nat.zero or unary_nat.succ
-- open u_nat


-- -- ---- 2. let's construct some natural numbers using u_nat, 0, 1, 2, 3.
-- #reduce zero
-- -- #eval succ zero
-- #reduce succ zero
-- -- #eval succ(succ zero)
-- #reduce succ (succ zero)  -- should reduce to: succ (succ zero)

-- ------ 3. Let's define (left) addition recursively
-- def add : u_nat → u_nat → u_nat 
-- -- def add (n m : unary_nat) : unary_nat :=
-- | zero m => m
-- | (succ n') m => succ (add n' m)

-- -- -- define familiar notation! under the hood it's doing add
-- -- local infix (name := add) ` + ` := add

-- -- ---- 4. Let's prove n + 0 = n, what such a simple thing needs a proof?!
-- -- -- theorem: for all n ∈ Nats, n + 0 = n (NL)
-- -- theorem thm_nat_add_zero : ∀ (n : u_nat), n + zero = n -- (FL)
-- -- :=
-- -- -- theorem add_zero (n : nat) : n + zero = n :=  -- syntax of functions also works since forall n : nat is equivalent to ∀ n : nat
-- -- begin
-- --   -- tactics do local proofs steps!
-- --   -- tactic intro n, removes the forall to make new goal n + zero = n
-- --   intro n,
-- --   -- tactic refl (for reflexivity) simplifies and ends 
-- --   refl,
-- -- end

-- -- ---- 5. [Bonus] Proof by induction (Q: can you figure out why this needs induction? What if we defined add in reverse (i.e. left addition)?
-- -- theorem add_zero : ∀ (n : u_nat), zero + n = n :=
-- -- begin
-- --   -- Introduce variable n and assume arbitrary u_nat.
-- --   intro n,
-- --   -- Induction on n.
-- --   induction n with n' hd,
-- --   { -- Base case: n = zero
-- --     -- This is true by the definition of add.
-- --     refl },
-- --   { -- Inductive step: Assume it's true for d, prove for succ d.
-- --     -- Again, this follows directly from the definition of add.
-- --     simp [add, hd] },
-- -- end

-- -- Creating an inductive type named `u_nat` for our natural numbers.
-- inductive u_nat : Type
-- | zero : u_nat  -- The base case, zero.
-- | succ : u_nat → u_nat  -- The successor function, constructing n + 1 from n.

-- -- Now, let's construct some basic natural numbers and reduce them.
-- namespace u_nat

-- -- Constructing 0, 1, 2 using `zero` and `succ`.
-- def one := succ zero  -- one is the successor of zero
-- def two := succ one   -- two is the successor of one

-- -- Let's reduce the expressions.
-- -- #reduce zero -- This will simply be u_nat.zero.
-- -- #reduce succ zero -- This will be u_nat.succ u_nat.zero, which we've defined to be one.

-- -- Defining addition recursively on the left argument.
-- def add : u_nat → u_nat → u_nat
-- | zero     m := m  -- base case: 0 + m = m
-- | (succ n) m := succ (add n m)  -- inductive step: (n+1) + m = succ (n + m)

-- -- Let's create infix notation for our addition.
-- infix ` +_ `: 65 := add

-- -- Proof 1: 0 + n = n
-- theorem zero_add (n : u_nat) : zero +_ n = n :=
-- begin
--   -- This proof is straightforward because by our definition of add,
--   -- the base case (zero + m = m) directly gives us the desired equality.
--   refl,
-- end

-- -- Proof 2: n + 0 = n
-- -- For this proof, we'll use induction on n.
-- theorem add_zero (n : u_nat) : n +_ zero = n :=
-- begin
--   -- Applying induction on n.
--   induction n with d hd,
--   { 
--     -- Base case: show that 0 + 0 = 0, which is true by our add definition.
--     refl,
--   },
--   {
--     -- Inductive step: we assume hd : d + zero = d and need to show
--     -- (succ d) + zero = succ d. We have:
--     -- (succ d) + zero = succ (d + zero) [by our add definition]
--     --               = succ d [using the inductive hypothesis hd]
--     rw [add, hd],
--   },
-- end

-- end u_nat

/-!
Definition of (unary) natural numbers using inductive types and calling it u_nat.
-/
inductive u_nat : Type
| zero : u_nat
| succ : u_nat → u_nat

-- Open the namespace to use `zero` and `succ` without prefix. So you don't have to do unary_nat.zero or unary_nat.succ
open u_nat

/-!
Constructing 0,1,2 using zero and succ.
-/
def O : u_nat := zero --- 0 = zero
def one : u_nat := succ zero --- 1 == succ zero
def two : u_nat := succ one --- 2 == succ (succ zero)
#reduce O
#reduce one
#reduce two 

/-!
Definition of addition by recursing on the right argument like Lean4 mathlib does https://github.com/leanprover/lean4/blob/a62d2fd4979671b76b8ab13ccbe4fdf410ec0d9d/src/Init/Prelude.lean#L1443-L1445
-/
def add : u_nat → u_nat → u_nat
| n, zero => n
| n, succ n' => succ (add n n')

#reduce add O O
#reduce add one one

--- TODO: continue tomorrow with GPT4 vs morph: https://chat.openai.com/c/833bf10e-7f34-4fbc-a97d-ecc8d1525173
/-!
Proving 0+n=n using the definition of addition.
-/
-- theorem add_zero : ∀ n : u_nat, add n zero = n :=
-- begin

/-!
Proving 0+n=n using the definition of addition.
-/
-- theorem zero_add : ∀ n : u_nat, add zero n = n
-- | zero := rfl
-- | (u_nat.succ m) := congr_arg succ (zero_add m)
