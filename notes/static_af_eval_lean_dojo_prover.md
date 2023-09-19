# Static eval for AutoFormalization (AF) using Lean Dojo with a Prover

# Goal
Create a trust worthy benchmark that evaluating autoformalization LLM models using Lead Dojo

# Idea
The idea is to create a eval benchmark where we can measure reliably if a model is capable of translating natural language specificiations to formally verifiable specificiations (in the ITP Lean).
Thus the task is:

> Task = AF (AutoFormalization) =: can a ml model create a formalization (from an informal statement) that is (formally) semantically equivalent to the target formalization? `AF == i_stmt -> f_stmt`

The main components we will need are:
1. A benchmark with ground truth pairs of informal statements to formal statements (specifying Task AF via examples) see my current public hf data set https://huggingface.co/datasets/brando/debug1_af 
2. An **equivalence** function to be used as a score/loss function. It tells (ideally) us **perfectly** if a traslated/autoformalize informal statement is equivalent to the target formal statement.
3. Full pipeline code that runs eval given:
   - a. (AF) LLM model
   - b. Equivalence score/loss function with a prover capable of proving true equivalences e.g., `fs1 === fs2 ? | Prover, ITP`
   - c. An ITP (Interactive Theorem Prover, Lean). In this case LeanDojo.

So final code looks as follows
```python
af_score = eval_af_static(model=af_model, equi_score=equivalence, env=LeanDojo)
print(f'Autoformalization eval performance: {af_score=}
```

# Plan/Experiment 1: Static eval for AutoFormalization (AF) using NLP equivalence score/loss
Goal: first plan will be to use the AF data https://huggingface.co/datasets/brando/debug1_af to evaluate a models capabilities in Autoformalizing using a standard NLP loss function as the equivalence function. 

See dummy code here: https://github.com/brando90/evals-for-autoformalization/blob/main/src/nlp_eval/af_ppl_eval2.py

```python
  af_score = eval_af_static(model, equi_score_or_loss, eval_dataset, env=LeanDojo)
  print(f'Autoformalization eval performance: {af_score=}')
```

# Plan/Experiment 2: Static eval for AutoFormalization (AF) using Prover based equivalence score/loss
Goal: evaluate using the LeanDojo Lean proving env and ReProver in LeanDojo

starter code TODO: https://github.com/brando90/evals-for-autoformalization/blob/main/src/nlp_eval/af_re_prover_eval.py

Suggested plan:
- figure out what the right way to use LeanDojo is https://github.com/lean-dojo maybe an import statement isn't the right way. Need to read through the git repos and figure out what's needed
- create an eval score/loss `metric = evaluate.load("re_prover_lean_dojo")` to evaluate AF using the default reprover prover and lean dojo lean env. Upload to https://huggingface.co/docs/evaluate/creating_and_sharing
- then run eval benchmark on `debug1_af` (https://huggingface.co/datasets/brando/debug1_af/tree/main) and see the score of our AF model this way

