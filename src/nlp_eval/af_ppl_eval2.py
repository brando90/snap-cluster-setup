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
