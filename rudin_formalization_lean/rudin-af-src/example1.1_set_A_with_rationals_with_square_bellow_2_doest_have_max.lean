/-

Specs:
- use reasonable amount mathlib, so it's ok to re-use the rationls from mathlib

Proofs
0. p² = 2 has no rational solution
1. Showing A = {p ∈ ℚ | p² < 2} has no maximum element.

Thm: ∀p ∈ A, ∃ q ∈ A, p < q.
q = p + e

WTS: (p + e)² < 2
p² + 2pe + e²
p² + pe + pe + e²
intuitively want make e subject
p² + pe + e(p + e)
observe that p + e < 2 (lemma)
p² + pe + 2e < 2
p² + e(p + 2) < 2
e < 2 - p² / (p + 2)
-- plug e back into WTS to show it's true

-/

import Mathlib.Data.Rat.Basic

def A : set ℚ := { p : ℚ | p² < 2 }

-- theorem A_has_no_max : ∀ p : ℚ, p ∈ A → ∃ q ∈ A, p < q :=
theorem A_has_no_max : ∀ p ∈ A → ∃ q ∈ A, p < q :=
