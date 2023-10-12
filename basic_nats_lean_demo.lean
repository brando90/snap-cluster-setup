-- ref: https://chat.openai.com/c/86d847e3-fc4c-428b-b87b-664f955558ea
-- ref: 
-- Skipping importing default nat libraries
-- import data.nat.basic -- imports standard nats in lean

---- 1. Defining unary natural numbers
inductive u_nat : Type
| zero : u_nat -- #0
| succ : u_nat → u_nat -- #1
-- Open the namespace to use `zero` and `succ` without prefix. So you don't have to do unary_nat.zero or unary_nat.succ
open u_nat

-- ---- 2. let's construct some natural numbers using u_nat, 0, 1, 2, 3.
#reduce zero
-- #eval succ zero
#reduce succ zero
-- #eval succ(succ zero)
-- -- #reduce succ(succ zero)

def one : u_nat := succ zero         -- #1
def two : u_nat := succ one          -- #2
def three : u_nat := succ two        -- #3
def four : u_nat := succ three       -- #4

#reduce one

-- ------ 3. Let's define addition recursively (on the right)
-- def add : u_nat → u_nat → u_nat 
-- -- def add (n m : unary_nat) : unary_nat :=
-- | m zero     := m
-- | m (succ n') := succ (add m n')

-- -- define familiar notation! under the hood it's doing add
-- local infix (name := add) ` + ` := add

-- ---- 4. Let's prove n + 0 = n, what such a simple thing needs a proof?!
-- -- theorem: for all n ∈ Nats, n + 0 = n (NL)
-- theorem thm_nat_add_zero : ∀ (n : u_nat), n + zero = n -- (FL)
-- :=
-- -- theorem add_zero (n : nat) : n + zero = n :=  -- syntax of functions also works since forall n : nat is equivalent to ∀ n : nat
-- begin
--   -- tactics do local proofs steps!
--   -- tactic intro n, removes the forall to make new goal n + zero = n
--   intro n,
--   -- tactic refl (for reflexivity) simplifies and ends 
--   refl,
-- end

-- ---- 5. [Bonus] Proof by induction (Q: can you figure out why this needs induction? What if we defined add in reverse (i.e. left addition)?
-- theorem add_zero : ∀ (n : u_nat), zero + n = n :=
-- begin
--   -- Introduce variable n and assume arbitrary u_nat.
--   intro n,
--   -- Induction on n.
--   induction n with n' hd,
--   { -- Base case: n = zero
--     -- This is true by the definition of add.
--     refl },
--   { -- Inductive step: Assume it's true for d, prove for succ d.
--     -- Again, this follows directly from the definition of add.
--     simp [add, hd] },
-- end
