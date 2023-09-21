"""
Initial code demoing how compute the reprover eval loss on a HF data set.

refs:
https://chat.openai.com/c/a04d1841-c0e5-47e0-963f-9670afbce157
https://claude.ai/chat/eb9b05c7-6aa9-4dca-9bd2-6c95621b0def
https://huggingface.co/docs/evaluate/transformers_integrations
https://huggingface.co/docs/evaluate/package_reference/evaluator_classes
"""
LeanDojo = None  # TODO

def eval_af_static(model, 
                   equi_score_or_loss, 
                   eval_dataset, 
                   env=LeanDojo, 
                   per_device_eval_batch_size=16,  # Adjust based on your GPU memory; you can try 32, 64, etc.
                  ):
  """ """
  compute_metrics = equi_score_or_loss
  
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
  seed = 0
  
  # 1. Load and tokenize the dataset
  # ------------------------------
  # Load the dataset
  dataset_name = "brando/debug1_af"
  dataset = load_dataset(dataset_name)

  # Load the tokenizer for GPT-2
  model_name = "gpt2-medium"  # TODO: custom AF model or code llama2 
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name)  # TODO: load to gpu

  # Define a function to tokenize the dataset
  def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
  # Tokenize the dataset
  tokenized_eval_datasets = dataset.map(tokenize_function, batched=True)

  # 2. Setup evaluation using the evaluate library
  # ------------------------------
  # Load the perplexity metric from the evaluate library
  metric = evaluate.load("perplexity")  # TODO: this is the function to change to have Lean Dojo + ReProver
  def compute_metrics(eval_pred, metric):
    logits, labels = eval_pred
    # Convert logits to probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    # Compute perplexity
    return metric.compute(probs=probs, references=labels)
  equi_score_or_loss = compute_metrics

  # 3. Select a subset for evaluation and setup Trainer
  # ------------------------------
  # Select num_samples samples from the tokenized validation dataset
  num_eval_samples: int = 10000
  eval_subset = tokenized_datasets["validation"].shuffle(seed=seed).select(range(num_eval_samples))  # use take if using streamling = True

  # 4. Run AF eval on using equi_score
  af_score = eval_af_static(model, equi_score_or_loss, tokenized_eval_datasets, env=LeanDojo, per_device_eval_batch_size=16)
  print(f'Autoformalization eval performance: {af_score=}')

if __name__ == '__main__':
    main_af_ppl_eval_hf_ds()
    print(f'Done!\a')
