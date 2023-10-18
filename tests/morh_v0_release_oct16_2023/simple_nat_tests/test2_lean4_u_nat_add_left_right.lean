/-
tried this 
Define (unary) nats inductively in Lean4 of unary nat (name it u_nat), then addition recursively on the right argument (so 2nd argument), and then the proofs of 0 + n = n and n + 0 = n..
but ultimately only manually fixed the answers to get add left and right
-/

-- inductive u_nat : Type
-- | zero : u_nat
-- | succ : u_nat → u_nat

-- def add_left : u_nat → u_nat → u_nat
-- | u_nat.zero, n => n
-- | u_nat.succ n', n => u_nat.succ (add_left n' n)

-- def add_right: u_nat -> u_nat -> u_nat
-- | a, u_nat.zero => a
-- | a, u_nat.succ b' => u_nat.succ (add_right a b')

-- main conclusion from my addition experiments:
-- It seems morph doesn't know how to reason using add left to define add right, even if given the definition of add right. It's stubborn.
-- however, it's explanation of add_right was detailed and correct. Implying it's better at informalization than autoformalizaiton.
-- it still needs help to get lean3 vs lean4 right. I helped it compile.
-- ref: https://discord.com/channels/1117623339456933940/1161025563708903445/1163632255646834891

inductive u_nat : Type
| zero : u_nat
| succ : u_nat → u_nat

def add_left : u_nat → u_nat → u_nat
| u_nat.zero, n => n
| u_nat.succ n', n => u_nat.succ (add_left n' n)

def add_right : u_nat → u_nat → u_nat
| m, u_nat.zero => m
| m, u_nat.succ n => u_nat.succ (add_right m n)