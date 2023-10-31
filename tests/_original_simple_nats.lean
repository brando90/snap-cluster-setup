-- https://leanprover-community.github.io/lean-web-editor/#code=--%20Skipping%20importing%20default%20nat%20libraries%0A--%20import%20data.nat.basic%20--%20imports%20standard%20nats%20in%20lean%0A%0A----%201.%20Defining%20unary%20natural%20numbers%20%28could%20have%20defined%20it%20as%20my_nat%29%0Ainductive%20u_nat%20%3A%20Type%0A%7C%20zero%20%3A%20u_nat%20--%20%230%0A%7C%20succ%20%3A%20u_nat%20%E2%86%92%20u_nat%20--%20%231%0A--%20Open%20the%20namespace%20to%20use%20%60zero%60%20and%20%60succ%60%20without%20prefix.%20So%20you%20don't%20have%20to%20do%20unary_nat.zero%20or%20unary_nat.succ%0Aopen%20u_nat%0A%0A----%202.%20let's%20construct%20some%20natural%20numbers!%0A%23eval%20zero%0A--%20%23reduce%20zero%0A%23eval%20succ%20zero%0A--%20%23reduce%20succ%20zero%0A%23eval%20succ%28succ%20zero%29%0A--%20%23reduce%20succ%28succ%20zero%29%0A%0A------%203.%20Let's%20define%20addition%20recursively%20%28on%20the%20right%29%0Adef%20add%20%3A%20u_nat%20%E2%86%92%20u_nat%20%E2%86%92%20u_nat%20%0A--%20def%20add%20%28n%20m%20%3A%20unary_nat%29%20%3A%20unary_nat%20%3A%3D%0A%7C%20m%20zero%20%20%20%20%20%3A%3D%20m%0A%7C%20m%20%28succ%20n'%29%20%3A%3D%20succ%20%28add%20m%20n'%29%0A%0A--%20define%20familiar%20notation!%20under%20the%20hood%20it's%20doing%20add%0Alocal%20infix%20%28name%20%3A%3D%20add%29%20%60%20%2B%20%60%20%3A%3D%20add%0A%0A----%204.%20Let's%20prove%20n%20%2B%200%20%3D%20n%2C%20what%20such%20a%20simple%20thing%20needs%20a%20proof%3F!%0A--%20theorem%3A%20for%20all%20n%20%E2%88%88%20Nats%2C%20n%20%2B%200%20%3D%20n%20%28NL%29%0Atheorem%20thm_nat_add_zero%20%3A%20%E2%88%80%20%28n%20%3A%20u_nat%29%2C%20n%20%2B%20zero%20%3D%20n%20--%20%28FL%29%0A%3A%3D%0A--%20theorem%20add_zero%20%28n%20%3A%20nat%29%20%3A%20n%20%2B%20zero%20%3D%20n%20%3A%3D%20%20--%20syntax%20of%20functions%20also%20works%20since%20forall%20n%20%3A%20nat%20is%20equivalent%20to%20%E2%88%80%20n%20%3A%20nat%0Abegin%0A%20%20--%20tactics%20do%20local%20proofs%20steps!%0A%20%20--%20tactic%20intro%20n%2C%20removes%20the%20forall%20to%20make%20new%20goal%20n%20%2B%20zero%20%3D%20n%0A%20%20intro%20n%2C%0A%20%20--%20tactic%20refl%20%28for%20reflexivity%29%20simplifies%20and%20ends%20%0A%20%20refl%2C%0Aend%0A%0A----%205.%20%5BBonus%5D%20Proof%20by%20induction%20%28Q%3A%20can%20you%20figure%20out%20why%20this%20needs%20induction%3F%20What%20if%20we%20defined%20add%20in%20reverse%20%28i.e.%20left%20addition%29%3F%0Atheorem%20add_zero%20%3A%20%E2%88%80%20%28n%20%3A%20u_nat%29%2C%20zero%20%2B%20n%20%3D%20n%20%3A%3D%0Abegin%0A%20%20--%20Introduce%20variable%20n%20and%20assume%20arbitrary%20u_nat.%0A%20%20intro%20n%2C%0A%20%20--%20Induction%20on%20n.%0A%20%20induction%20n%20with%20n'%20hd%2C%0A%20%20%7B%20--%20Base%20case%3A%20n%20%3D%20zero%0A%20%20%20%20--%20This%20is%20true%20by%20the%20definition%20of%20add.%0A%20%20%20%20refl%20%7D%2C%0A%20%20%7B%20--%20Inductive%20step%3A%20Assume%20it's%20true%20for%20d%2C%20prove%20for%20succ%20d.%0A%20%20%20%20--%20Again%2C%20this%20follows%20directly%20from%20the%20definition%20of%20add.%0A%20%20%20%20simp%20%5Badd%2C%20hd%5D%20%7D%2C%0Aend
-- Skipping importing default nat libraries
-- import data.nat.basic -- imports standard nats in lean

---- 1. Defining unary natural numbers
inductive u_nat : Type
| zero : u_nat -- #0
| succ : u_nat → u_nat -- #1
-- Open the namespace to use `zero` and `succ` without prefix. So you don't have to do unary_nat.zero or unary_nat.succ
open u_nat

---- 2. let's construct some natural numbers!
#eval zero
-- #reduce zero
#eval succ zero
-- #reduce succ zero
#eval succ(succ zero)
-- #reduce succ(succ zero)

------ 3. Let's define addition recursively (on the right)
def add : u_nat → u_nat → u_nat 
-- def add (n m : unary_nat) : unary_nat :=
| m zero     := m
| m (succ n') := succ (add m n')

-- define familiar notation! under the hood it's doing add
local infix (name := add) ` + ` := add

---- 4. Let's prove n + 0 = n, what such a simple thing needs a proof?!
-- theorem: for all n ∈ Nats, n + 0 = n (NL)
theorem thm_nat_add_zero : ∀ (n : u_nat), n + zero = n -- (FL)
:=
-- theorem add_zero (n : nat) : n + zero = n :=  -- syntax of functions also works since forall n : nat is equivalent to ∀ n : nat
begin
  -- tactics do local proofs steps!
  -- tactic intro n, removes the forall to make new goal n + zero = n
  intro n,
  -- tactic refl (for reflexivity) simplifies and ends 
  refl,
end

---- 5. [Bonus] Proof by induction (Q: can you figure out why this needs induction? What if we defined add in reverse (i.e. left addition)?
theorem add_zero : ∀ (n : u_nat), zero + n = n :=
begin
  -- Introduce variable n and assume arbitrary u_nat.
  intro n,
  -- Induction on n.
  induction n with n' hd,
  { -- Base case: n = zero
    -- This is true by the definition of add.
    refl },
  { -- Inductive step: Assume it's true for d, prove for succ d.
    -- Again, this follows directly from the definition of add.
    simp [add, hd] },
end
