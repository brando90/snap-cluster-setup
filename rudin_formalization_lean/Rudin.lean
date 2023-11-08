/-

Specs:
- use reasonable amount mathlib, so it's ok to re-use the rationls from mathlib

Proofs
0. p² = 2 has no rational solution
1. Showing A = {p ∈ ℚ | p² < 2} has no maximum element.

Thm: ∀p ∈ A, ∃ q ∈ A, p < q.
q = p + e

WTS: i.e. we want to choose a q = p + e such that
  p < (p + e)² < 2
(or we want to enforce it somehow by choosing e (witness) appropriately)

Proof:
To show this let's do some algebra to create the witness we want
p² + 2pe + e²
p² + pe + pe + e²
intuitively want make e subject
p² + pe + e(p + e)
observe that p + e < 2 (lemma)
p² + pe + 2e < 2
p² + e(p + 2) < 2
e(p + 2) < 2 - p²
e < (2 - p²) / (p + 2)
-- plug e back into WTS to show its square is less than 2, though it should be clear it works because we constructed it so that it would work
So consider such e & show it's less than 2 (and thus its in A)
p + e
< p + (2 - p²) / (p + 2)
= p (p + 2) / (p + 2) + (2 - p²) / (p + 2)
= ((p² + 2p) + (2 - p²)) / (p + 2)
= (2p + 2) / (p + 2)
now let's square it
(p + e)² < (2p + 2)² / (p + 2)²

-/

import Mathlib.Data.Rat.Basic

def A : set ℚ := { p : ℚ | p² < 2 }

-- theorem A_has_no_max : ∀ p : ℚ, p ∈ A → ∃ q ∈ A, p < q :=
theorem A_has_no_max : ∀ p ∈ A → ∃ q ∈ A, p < q :=
