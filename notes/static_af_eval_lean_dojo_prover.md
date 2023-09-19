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

```python
"""
Initial code demoing how compute the ppl eval loss on a HF data set.

refs:
https://chat.openai.com/c/a04d1841-c0e5-47e0-963f-9670afbce157
https://claude.ai/chat/eb9b05c7-6aa9-4dca-9bd2-6c95621b0def
https://huggingface.co/docs/evaluate/transformers_integrations
https://huggingface.co/docs/evaluate/package_reference/evaluator_classes
"""

def eval_af_static(model, equi_score, eval_dataset, env=LeanDojo, per_device_eval_batch_size=16):
  """ """
  compute_metrics = equi_score
  
  # Define training arguments with a reasonable batch size for evaluation
  training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    per_device_eval_batch_size=per_device_eval_batch_size,  # Adjust based on your GPU memory; you can try 32, 64, etc.
    logging_dir='./logs',
  )

  # Initialize the Trainer with the subset
  trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_subset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # Use the compute_metrics function defined above
  )

  # 5. Evaluate the model to compute perplexity
  # ------------------------------
  results = trainer.evaluate()
  af_score = results['eval_perplexity']
  print(f'Autoformalization eval performance: {af_score=}')
  return af_score

def main_af_ppl_eval_hf_ds():
  """ Main fun to eval AF using PPL score/loss using hf dataset. """
  # --- Define a function to compute perplexity using the evaluate library
  # Load the perplexity metric from the evaluate library
  metric = evaluate.load("perplexity")  # TODO: this is the function to change to have Lean Dojo?
  def compute_metrics(eval_pred, metric):
    logits, labels = eval_pred
    # Convert logits to probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    # Compute perplexity
    return metric.compute(probs=probs, references=labels)

  # 4. Select a subset for evaluation and setup Trainer
  # ------------------------------
  # Select 10,000 samples from the tokenized validation dataset
  eval_subset = tokenized_datasets["validation"].shuffle(seed=42).select(range(10000))

  af_score = eval_af_static(model, equi_score, eval_dataset, env=LeanDojo, per_device_eval_batch_size=16)
  print(f'Autoformalization eval performance: {af_score=}')

if __name__ == '__main__':
    main_af_ppl_eval_hf_ds()
    print(f'Done!\a')
```

# Plan/Experiment 2: Static eval for AutoFormalization (AF) using Prover based equivalence score/loss
Goal: evaluate using the LeanDojo Lean proving env and ReProver in LeanDojo

dummy code: todo, but I suggest we try to edit/add an eval loss function using LeanDojo as ITP + ReProver in LeanDojo and push the eval metric to the HF `evaluate` library: https://huggingface.co/docs/evaluate/creating_and_sharing

```python
TODO
```
