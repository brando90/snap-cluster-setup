# Prompts for AutoFormalization (AF)

## 

```prompt
Give me a detailed excellent explanation of what a class is in Lean 4.

Give me a detailed, excellent explanation of what a class is in Lean 4. What is it for and give me the most simple and best example of how to use it.
```


# TODOs

- what is the best way to learn about the syntax of lean e.g., `-- structure == inductive, but structure has one constructor` and `class ≈ structure + other machinery`.
- TODO: structures ~ classes + 
```lean
-- structure == inductive, but structure has one constructor
-- class ≈ structure + other machinery
/-
structure foo_s where
  x : Nat
  y : Nat
#check foo_s.mk
inductive foo_i
| mk (x y : Nat)

def foo_i.x : foo_i → Nat
| mk x _ => x

#check foo_s.x
#check foo_i.x
-/
```