# Static eval for AutoFormalization (AF) using Lean Dojo with a Prover

# Goal
Create a trust worthy benchmark that evaluating autoformalization LLM models using Lead Dojo

# Idea
The idea is to create a eval benchmark where we can measure reliably if a model is capable of translating natural language specificiations to formally verifiable specificiations (in the ITP Lean).
Thus the task is:

> Task AF: can a model create a formalization that is (formally) semantically equivalent to the target formalization?

The main components we will need are:
1. A benchmark with ground truth pairs of informal statements to formal statements (specifying Task AF via examples) e.g., https://huggingface.co/datasets/brando/debug1_af
2. An **equivalence** function to be used as a score/loss function. It tells us **perfectly** if a traslated/autoformalize informal statement is equivalent to the target formal statement.
3. Full pipeline code that runs eval given:
   a. LLM model (that know how to do AF)
   b. Equivalence function with a prover capable of proving equivalences e.g., `fs1 === fs2 ? | Prover, ITP`
   c. An ITP (Interactive Theorem Prover, Lean). In this case LeanDojo.

So final code looks as follos
```python
af_score = eval_af_static(model=af_model, equi_score=equivalence, env=LeanDojo)
print(f'Autoformalization eval performance: {af_score=}
```

# Plan/Experiment 1: Static eval for AutoFormalization (AF) using NLP equivalence score/loss
Goal: first plan will be to use the AF data https://huggingface.co/datasets/brando/debug1_af to evaluate a models capabilities in Autoformalizing using a standard NLP loss function as the equivalence function. 

See dummy code here: https://github.com/brando90/evals-for-autoformalization/blob/main/src/nlp_eval/af_ppl_eval2.py

# Plan/Experiment 2: Static eval for AutoFormalization (AF) using Prover based equivalence score/loss
Goal: evaluate using the LeanDojo Lean proving env and ReProver in LeanDojo

dummy code: todo, but I suggest we try to edit/add an eval loss function using LeanDojo as ITP + ReProver in LeanDojo and push the eval metric to the HF `evaluate` library: https://huggingface.co/docs/evaluate/creating_and_sharing
