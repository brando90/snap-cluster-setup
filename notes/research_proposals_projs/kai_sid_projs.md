# Project Options

Three Standard benchmarks to evaluate on:
1. AF: ProofNet
2. informal maths: MATH
3. proving accuracy: MiniF2F

Project Options (always thinking how to improve the above 3 benchmarks + what is the new conceptual novel contribution):
1. formal proofs -> informal proofs (main: improve ProofNet, MATH)
   1. Contribution: a. show formal proofs improve informal reasoning 2. easy way to get more formal, formal data for AF that is high quality
   - tasks:
      1. From already existing formal proofs generate AF pairs (e.g., execute formal proof to get `full_formal_proof=<thm, ps, tac, ...>` and informalize it, also at each step too, [extra proof term version too])
      2. generates informal proofs to train for MATH `full_informal_proof=LLM(full_formal_proof, "informalize formal proof ")`
      3. how does it improve MiniF2F?
      4. Compute alignments/quality to the above target benchmarks (let's precompute task2vec embedding of the (val) target benchmarks)
2. informal problems/thms `i_thm` -> formal problem/thm `f_thm` -> formal proofs `f_pf` (say resticted to IMO so to improve IMO reasoning or restricted to higher level mathematics but that needs a new benchmark)
   1. Contribution: demonstrate data (thms/pfs,problems/solns) from textbook improve language models at: the 3 benchmarks above (& beat Tony's current work AF work)
   - tasks:
      1. extract textbook level problems/solutions (e.g., AMPS, IMO, higher maths level textbooks) and do: `i_thm` -> `f_thm` -> `f_pf` -> `i_pf`.
         - build an neural theorem prover NTP for the above with expert iteration (start from something easy to use e.g., LeanDojo's ReProver?)
         - find a way to reliably get `i_thm` -> `f_thm` (e.g., see MiniF2F and use mathlib)
         - then once we get proofs, decide which `f_pf` to train on (the ones that compile, type check, proof the theorem, train on the ones with high back translation to informal language)
      2. generate formal proofs for MATH's APS and repeat above
      3. Report improvement on all 3: AF, MATH, TP
      4. Study data alignment/quality
3. In depth study of different data domains on 3 benchmarks & quantitative data quality/alignment metrics
   1. coq, isabelle, pair, unpaired, mixed but unpaired, textbook, python-docstrings, etc
   2. I feel we can give up this part and instead do 1 & 2 and put the data metrics study at the end
4. Formalize a textbook autonomously + improve AF on a new textbook AF based benchmark

idea of ps, tact -> informal docs informal 
Remark: 
   - How to do `f_pf` -> `i_pf`
   - For every proof step we have `proof_state, tactic` (short `ps, tac`):
      - We create `full_formal_proof=<thm, ps, tac, ...>`
      - We informalize step by step: 
         - `LLM(ps, "informalie") -> i_ps`
         - `LLM(tac, "informalize") -> i_reasoning_step = i_rea_step`. Here the prompt engineering is important. We use few-shot example + importantly, the observation tactics are really hard to make human/informal interpretable. So we use the change in the proof state + the documentation of the tactic to produce a high quality informalization.
         - then concantenate to produce a proof/solution that AF based (not exactly synthetic)
         - (we can always improve diversity via MetaMath's techniques)

# Wish list

- Better benchmarks
   - split mathlib for a new thm proving benchmark & AF benchmark
   - textbook based AF (and thm but harder to guarantee thms are right), extract thm, proofs for a solid benchmark of about 5K test examples (that's the size of MATH)