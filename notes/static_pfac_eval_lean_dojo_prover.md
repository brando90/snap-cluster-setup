# Static eval for Proving Accuracy (PfAc) using Lean Dojo with a Prover

# Goal
Create a trust worthy eval benchmark evaluates a LLM models capabilities of proving theorems on a benchmark using Lean Dojo.

# Idea
The idea is to create an eval benchmark where we can measure reliably if a model + prover is capable of proving theorems in an standard ML for theorem proving eval benchmarks.

ML4TP eval Benchmarks options:
1. MiniF2F: https://github.com/openai/miniF2F
2. ...

The components we need are:
1. Eval data set for benchmarking ML4TP (e.g., MiniF2F)
2. Prover + Model (e.g., DSP + LLM, Parsel + LLM, DSP2 + LLM, ReProver + LLM)
3. Thus given:
  a. Eval ds (minif2f)
  b. Prover for TP &
  c. LLM as ML model
  d. Env (LeanDojo) to evaluate proof accuracy (number of theorems prover)

# Plan Experiment 1: MiniF2F + ReProver + LLM + LeanDojo
