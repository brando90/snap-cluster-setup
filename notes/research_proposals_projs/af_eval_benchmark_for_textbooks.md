# Massive Autoformalization Eval Benchmark For Textbook Mathematics with ML metrics
Essential Goal: Auotoformalization from mathematics (NL) to formal mathematics (FL).

Idea: 
1. Extract all nl theorems from textbooks (say using an LLM)
2. Evaluate an LLM (say GPT4, Morph, Llema) how good they are on ProofNet & IsaProofNet
3. Autoformalize the nl textbooks statements to Lean, Coq, Isabelle (even Lean it's fine)
4. Then only with ML metrics (ppl, CE, token acc, LLM reward model for equiv) how similar the two FL statements are.
5. Then keep training/developing models and see how good it is on ITP-Textbook-Proof-Net eval benchmark

note:
- if AF LLM is closed like GPT4 you **cannot** train this model usually (see their terms)
- if none ML metrics like equiv using an ITP env + prover is ideal, but background theory complicates if this harsh metric is truly measuring progressing because it might be to harsh of a metric (i.e., a false emergent metric)
