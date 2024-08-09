"""
https://chatgpt.com/c/f8c4b5fe-bec5-4f53-a503-00b91a69c8b4
"""
import os
import wandb
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Initialize W&B with local server URL
os.environ["WANDB_BASE_URL"] = "http://localhost:8080"
os.environ["WANDB_API_KEY"] = "local"  # Using "local" as a dummy API key
os.environ["WANDB_MODE"] = "offline"

# Start a new W&B run
print('Start a new W&B run')
wandb.init(project="local-gpt2-test", name="synthetic-data-run")
print('Done with wandb.init(...)')

# Set up the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Generate synthetic data (200 examples of repeating sequences)
data = ["This is a synthetic example." for _ in range(30)]
encoded_data = tokenizer(data, return_tensors="pt", padding=True, truncation=True)

# Create a Dataset object from the synthetic data
dataset = Dataset.from_dict({
    "input_ids": encoded_data["input_ids"],
    "attention_mask": encoded_data["attention_mask"],
    "labels": encoded_data["input_ids"]
})

# Define training arguments, logging to W&B
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    logging_dir="./logs",
    logging_steps=10,
    report_to="wandb",  # Log to W&B
    run_name="synthetic-data-run"
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train the model
trainer.train()

# Finish the W&B run
wandb.finish()
