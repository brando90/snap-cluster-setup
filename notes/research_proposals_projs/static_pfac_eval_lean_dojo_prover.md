# Development of an Evaluation Static Benchmark for Large Language Model-Based Theorem Proving using Lean Dojo

Brando Miranda: brando9@stanford.edu
AI/ML
Aut_win_spr, 2023-2024 Academic Year
Course credit
Up to 5 students

# Project Description
This project aims to develop a reliable and static benchmark to evaluate the theorem-proving capabilities of Large Language Models (LLMs) using Lean Dojo. The goal is to create a trustworthy evaluation benchmark that assesses an LLM model's ability to prove theorems on standard ML for theorem proving evaluation benchmarks like MiniF2F. The components needed for this project include an evaluation dataset for benchmarking ML4TP, a prover + model, and an environment (LeanDojo) to evaluate proof accuracy. The project will also explore autoformalization as a technique for high-quality data augmentation to improve the number of theorems proved, i.e., proof accuracy. This endeavor is motivated by the aspiration to build an automated mathematician capable of unlocking the vast knowledge encapsulated in mathematical textbooks written in natural language, contributing to advancements in mathematics, scientific discovery, and AI safety.

Recommended Background:
Interested candidates are encouraged to share their background when reaching out. A strong foundation in Python is essential, and knowledge in theorem proving using Lean, Coq, or Isabelle is preferred but not mandatory. A passion or intense curiosity about mathematics, formalization/verification of mathematics, AI safety/alignment, or software verification & verified program synthesis would be ideal.

Prerequisites / Preparation:
Participants will be expected to make direct contributions to the project and should be comfortable coding in Python. Familiarity with theorem proving and a keen interest in mathematics or software verification would be advantageous.

# Key Citations:
1. MiniF2F: https://github.com/openai/miniF2F
2. Autoformalization (AF): https://arxiv.org/abs/2205.12615
3. AF video: https://youtu.be/_pqJYnQua58?si=jVliUTqqXTjpeods&t=1
4. DSP (Draft Sketch Prove): https://openreview.net/forum?id=SMa9EAovKMC
5. ProofNet: https://arxiv.org/abs/2302.12433
6. LeanDojo: https://github.com/lean-dojo
7. Parsel: https://github.com/ezelikman/parsel

# Goal
Create a trust worthy eval benchmark evaluates a LLM models capabilities of proving theorems on a benchmark using Lean Dojo.
Eventually, getting Autoformalization as the main technique for high quality data augmentation for improving number of theorems prooved (i.e., what we call proof accuracy).

# Idea
The idea is to create an eval benchmark where we can measure reliably if a model + prover/search method is capable of proving theorems in an standard ML for theorem proving eval benchmarks.

ML4TP eval Benchmarks options:
1. MiniF2F: https://github.com/openai/miniF2F
2. Autoformalization (AF): https://arxiv.org/abs/2205.12615
3. AF video: https://youtu.be/_pqJYnQua58?si=jVliUTqqXTjpeods&t=1
4. DSP (Draft Sketch Prove): https://openreview.net/forum?id=SMa9EAovKMC
5. ProofNet: https://arxiv.org/abs/2302.12433
6. LeanDojo: https://github.com/lean-dojo
7. Parsel: https://github.com/ezelikman/parsel

The components we need are:
1. Eval data set for benchmarking ML4TP (e.g., MiniF2F)
2. Prover + Model (e.g., DSP + LLM, Parsel + LLM, DSP2 + LLM, ReProver + LLM, DSP2 + LLM)
3. Thus given:
  a. Eval ds (minif2f)
  b. Prover for TP &
  c. LLM as ML model
  d. Env (LeanDojo) to evaluate proof accuracy (number of theorems prover)
4. Autoformalization as way for high quality data augmentation 

# Plan Experiment 1: Implement evaluation benchmark with MiniF2F/ProofNet + ReProver + LLM + LeanDojo

# Plan Experiment 2: Implement evaluation benchmark with MiniF2F/ProofNet + DSP + LLM + LeanDojo

# Plan Experiment 3: Implement evaluation benchmark with MiniF2F/ProofNet + Parsel + LLM + LeanDojo

# Plan Experiment 4: Use Autoformalization to get more proofs/theoms to train model, perhaps filter for quality e.g., which parse etc.
