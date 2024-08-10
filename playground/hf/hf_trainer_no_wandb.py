import os
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

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
    report_to="none",  # Log to W&B
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

