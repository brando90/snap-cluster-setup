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

# Plan/Experiment 1: Static eval for AutoFormalization (AF) using NLP equivalence loss
Goal: first plan will be to use the AF data https://huggingface.co/datasets/brando/debug1_af to evaluate a models capabilities in Autoformalizing using a standard NLP loss function as the equivalence function. 

Final code should look like:
```python
def ppl(predicted_formal_stmt, target_formal_stmt):
  """Perplexity equivalence function.
    High score if the formal statement is far from the target statement
  """


af_score = eval_af_static(model=af_model, equi_score=ppl, env=LeanDojo)
print(f'Autoformalization eval performance: {af_score=}
```
