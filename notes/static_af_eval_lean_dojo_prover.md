# Static eval for AutoFormalization (AF) using Lean Dojo with a Prover

# High Level Motivation
The dream is to build an automated mathematician that is powerful enough to automate mathematics, sciencific discovery, and AI safety with an automated mathematician. 
I conjecture formal maths is the only way to create safe AGI because safety requires a "for all" quantifier saying "there is no way AGI will kill humanity". That type of statement are impossible to guarantee empirically and only a mathematical formal proof can guarantee it. 
Hence why I think building an automated mathematician is the only way for safe AGI.

With this in mind, there is tremendous amount of information that hasn't been unlocked in mathematical textbooks written in natural language (informal langauge).
In addition, to guarantee correctness we want to leverage the power of formal languages and theorem proving.
Overall the combination of these two worlds (informal and formal) is through the task of Autoformalization -- training AI systems to translate between informal languages (e.g., English) and formal languages (e.g., PyThon, Lean). 

# Background

Read/skim references:
- https://arxiv.org/abs/2205.12615
- https://youtu.be/_pqJYnQua58?si=jVliUTqqXTjpeods&t=1

Then write a reflection.
Have GPT4/Claude evaluate your reflection and interact with it for a little to consolidate your understanding through a discussion.

# Goal
Create a trust worthy benchmark for evaluating Autoformalization LLM models using Lead Dojo.

## Extra background (option)
- --- extra ---
- https://leanprover-community.github.io/archive/stream/219941-Machine-Learning-for-Theorem-Proving/topic/Paper.3A.20Autoformalization.20with.20Large.20Language.20Models.html
- https://arxiv.org/abs/2206.01962
- --- extra ---

# Idea
The idea is to create a eval benchmark where we can measure reliably if a model is capable of translating natural language specificiations to formally verifiable specificiations (in the ITP Lean).
Thus the task is:

> Task = AF (AutoFormalization) =: can a ml model create a formalization (from an informal statement) that is (formally) semantically equivalent to the target formalization? `AF == i_stmt -> f_stmt`

The main components we will need are:
1. A benchmark with ground truth pairs of informal statements to formal statements (specifying Task AF via examples) see my current public hf data set [debug1](https://huggingface.co/datasets/brando/debug1_af) or [ProofNet](https://huggingface.co/datasets/hoskinson-center/proofnet)
2. An **equivalence** function to be used as a score/loss function. It tells us (ideally) **perfectly** if a traslated/autoformalize informal statement is equivalent to the target formal statement.
3. Full pipeline code that runs eval given:
   - a. (AF) LLM model
   - b. Equivalence score/loss function with a prover capable of proving true equivalences e.g., `fs1 === fs2 ? | Prover, ITP`
   - c. An ITP (Interactive Theorem Prover, Lean). In this case LeanDojo.

So final call to code looks as follows:
```python
af_score = eval_af_static(model=af_model, equi_score_or_loss=equivalence, env=LeanDojo)
print(f'Autoformalization eval performance: {af_score=}
```

# Plan/Experiment 1: Static eval for AutoFormalization (AF) using NLP equivalence score/loss
Goal: first plan will be to use the AF data in [ProoNet](https://huggingface.co/datasets/hoskinson-center/proofnet) or my [debug1](https://huggingface.co/datasets/brando/debug1_af) to evaluate a models capabilities in Autoformalizing using a standard NLP loss function as the equivalence function. 

See dummy code here: https://github.com/brando90/evals-for-autoformalization/blob/main/src/nlp_eval/af_ppl_eval.py

```python
af_score = eval_af_static(model, equi_score_or_loss, eval_dataset, env=LeanDojo)
print(f'Autoformalization eval performance: {af_score=}')
```

# Plan/Experiment 2: Static eval for Autoformalization using a Lean Automation/tactic/Prover based equivalence score/loss
Goal: evaluate a AF model using the LeanDojo proving env with the the simplest prover we can use -- an out of the box automation/tactic/prover in Lean.
The idea is to try to **prove** equivalences between the autoformalized statement done by our model and the ground truth target formal statement i.e., `model(i_stmt) ===_tactic f_stmt*`.
The simplest prover to try here would be a powerful tactic like [`linarith`](https://github.com/phlippe/Lean_hammer/issues/2) or [`LeanHammer`](https://github.com/phlippe/Lean_hammer).

Conceptually the api/pseudo-code would look something like this:
```python
def equivalence_basic_lean_prover(formal_stmt: str, target_formal_stmt: str, prover = linarith) -> bool:
   equivalent: bool = LeanDojo.env(formal_stmt, target_formal_stmt)
   return equivalent

af_score = eval_af_static(model=af_model, equi_score_or_loss=equivalence_basic_lean_prover, env=LeanDojo)
print(f'Autoformalization eval performance: {af_score=}
```

starter code TODO: https://github.com/brando90/evals-for-autoformalization/blob/main/src/nlp_eval/af_re_prover_eval.py

Suggested plan:
- figure out what the right way to use LeanDojo is https://github.com/lean-dojo maybe an import statement isn't the right way. Maybe we need to proof things in bulk e.g., an entire data set. What is the right way to use LeanDojo here for our purposes? Need to read through the git repos and figure out what's needed. Also see https://github.com/lean-dojo/LeanDojo/discussions/68
- create an eval score/loss `metric = evaluate.load("linarith")` or `metric = evaluate.load("LeanHammer")` to evaluate AF using the default reprover prover and lean dojo lean env. Upload to https://huggingface.co/docs/evaluate/creating_and_sharing
- then run eval benchmark on [ProofNet](hoskinson-center/proofnet) or [debug1_af](https://huggingface.co/datasets/brando/debug1_af/tree/main) and see the score of our AF model this way

# Plan/Experiment 3: Static eval for AutoFormalization (AF) using Prover based equivalence score/loss
Goal: evaluate a AF model using the LeanDojo proving env with the Prover called ReProver. 
The cruz is the implementation of the `equivance_score` score/loss function for scoring if `mld(i_stmt) ===_re_prover f_stmt`. 

starter code TODO: https://github.com/brando90/evals-for-autoformalization/blob/main/src/nlp_eval/af_re_prover_eval.py

Suggested plan:
- figure out what the right way to use LeanDojo is https://github.com/lean-dojo maybe an import statement isn't the right way. Need to read through the git repos and figure out what's needed
- create an eval score/loss `metric = evaluate.load("re_prover_lean_dojo")` to evaluate AF using the default reprover prover and lean dojo lean env. Upload to https://huggingface.co/docs/evaluate/creating_and_sharing
- then run eval benchmark on [debug1_af](https://huggingface.co/datasets/brando/debug1_af/tree/main) and see the score of our AF model this way

