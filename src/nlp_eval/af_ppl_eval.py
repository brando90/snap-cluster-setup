"""
Initial code demoing how compute the ppl eval loss on a HF data set.

refs:
https://chat.openai.com/c/a04d1841-c0e5-47e0-963f-9670afbce157
https://claude.ai/chat/eb9b05c7-6aa9-4dca-9bd2-6c95621b0def
https://huggingface.co/docs/evaluate/transformers_integrations
https://huggingface.co/docs/evaluate/package_reference/evaluator_classes
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import numpy as np
import evaluate

# 1. Load and tokenize the dataset
# ------------------------------
# Load the dataset
dataset_name = "wikitext-103-raw-v1"   #TODO: use https://huggingface.co/datasets/brando/debug1_af
dataset = load_dataset(dataset_name)

# Load the tokenizer for GPT-2
model_name = "gpt2-medium"  # TODO: custom AF model or
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define a function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 2. Setup evaluation using the evaluate library
# ------------------------------
# Load the perplexity metric from the evaluate library
metric = evaluate.load("perplexity")

# Define a function to compute perplexity using the evaluate library
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits to probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    # Compute perplexity
    return metric.compute(probs=probs, references=labels)

# 3. Load the GPT-2 model
# ------------------------------
# Load the GPT-2 model
model = AutoModelForCausalLM.from_pretrained(model_name)

# 4. Select a subset for evaluation and setup Trainer
# ------------------------------
# Select 10,000 samples from the tokenized validation dataset
eval_subset = tokenized_datasets["validation"].shuffle(seed=42).select(range(10000))

# Define training arguments with a reasonable batch size for evaluation
training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    per_device_eval_batch_size=16,  # Adjust based on your GPU memory; you can try 32, 64, etc.
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

# Print the perplexity
print(f"Perplexity: {results['eval_perplexity']}")
