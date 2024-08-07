"""
Trying to debug this: https://stackoverflow.com/questions/78841125/how-to-fix-segmentation-fault-when-training-gpt-2-model-using-hugging-face-trans
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from typing import Dict
import os

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the pad_token to the eos_token
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)

# Move model to GPU and enable bf16 precision
model = model.to(device=device, dtype=torch.bfloat16)

def preprocess_function_proofnet_simple(examples: Dict[str, list], tokenizer: GPT2Tokenizer, max_length: int = 512) -> Dict[str, torch.Tensor]:
    """
    Preprocess the input data for the proofnet dataset.

    Args:
    examples: The examples to preprocess.
    tokenizer: The tokenizer for encoding the texts.

    Returns:
    The processed model inputs.
    """
    inputs = [f"{examples['nl_statement'][i]}{tokenizer.eos_token}{examples['formal_statement'][i]}" for i in range(len(examples['nl_statement']))]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = model_inputs.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

# Load the dataset
dataset_path = "hoskinson-center/proofnet"
dataset = load_dataset(dataset_path)

# Select only 10 examples for training and validation
small_train_dataset = dataset['validation'].select(range(10))
small_val_dataset = dataset['test'].select(range(10))

# Preprocess the dataset
train_dataset = small_train_dataset.map(lambda examples: preprocess_function_proofnet_simple(examples, tokenizer), batched=True, remove_columns=["nl_statement", "formal_statement"])
val_dataset = small_val_dataset.map(lambda examples: preprocess_function_proofnet_simple(examples, tokenizer), batched=True, remove_columns=["nl_statement", "formal_statement"])

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# # Training arguments
# training_args = TrainingArguments(
#     output_dir=os.path.expanduser("~/tmp/gpt2_trainer"),
#     overwrite_output_dir=True,
#     max_steps=2,  # TODO get rid of this in favour of 1 or 2 or 3 epochs
#     # num_train_epochs=3,  # Train for 3 epochs
#     gradient_accumulation_steps=2,  # based on alpaca https://github.com/tatsu-lab/stanford_alpaca, allows to process effective_batch_size = gradient_accumulation_steps * batch_size, num its to accumulate before opt update step
#     gradient_checkpointing = True,  # TODO depending on hardware set to true?
#     per_device_train_batch_size=2,
#     save_steps=10_000,
#     save_total_limit=2,
#     bf16=True,  # Enable bf16 training only
#     logging_dir=os.path.expanduser("~/tmp/gpt2_trainer/logs"),
#     logging_steps=200,
#     report_to="none"  # Disable logging to WandB
# )

# Training arguments
from pathlib import Path
output_dir_train: Path = Path('~/tmp').expanduser()
output_dir_train.mkdir(parents=True, exist_ok=True)
training_args = TrainingArguments(
    output_dir=output_dir_train,
    max_steps=2,  # TODO get rid of this in favour of 1 or 2 or 3 epochs
    # num_train_epochs=num_train_epochs, 
    gradient_accumulation_steps=2,  # based on alpaca https://github.com/tatsu-lab/stanford_alpaca, allows to process effective_batch_size = gradient_accumulation_steps * batch_size, num its to accumulate before opt update step
    # gradient_checkpointing = True,  # TODO depending on hardware set to true?
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-5,
    weight_decay=0.01, 
    max_grad_norm=1.0, # TODO once real training change?
    lr_scheduler_type='cosine',  # TODO once real training change? using what I've seen most in vision 
    warmup_ratio=0.01,
    optim='paged_adamw_32bit',
    # logging_strategy='epoch', # TODO
    save_steps=100, # Save checkpoint every 500 steps
    save_total_limit=3, # save last 3
    logging_steps=10,  # Frequency of logging steps
    logging_first_step=True,
    logging_dir=output_dir_train,
    # evaluation_strategy='no',  # "no"`: No evaluation is done during training. no can be good to avoid memory issues.
    eval_strategy='no',  # "no"`: No evaluation is done during training. no can be good to avoid memory issues.
    # evaluation_strategy="steps",  # TODO Evaluate model at specified steps
    # eval_steps=110,  # TODO Evaluate every 100 steps
    # remove_unused_columns=False,  # TODO https://stackoverflow.com/questions/76879872/how-to-use-huggingface-hf-trainer-train-with-custom-collate-function/76929999#76929999 , https://claude.ai/chat/475a4638-cee3-4ce0-af64-c8b8d1dc0d90
    report_to='none',  # options I recommend: 'none', 'wandb'
    fp16=False,  # never ever set to True
    bf16=torch.cuda.is_bf16_supported(),
    # full_determinism=True,  # TODO periphery, Ensure reproducibility
    # torchdynamo="nvfuser",  # TODO periphery, Use NVFuser backend for optimized torch operations
    # dataloader_prefetch_factor=2,  # TODO periphery, Number of batches to prefetch
    # dataloader_pin_memory=True,  # TODO periphery, Pin memory in data loaders for faster transfer to GPU
    # dataloader_num_workers=16,  # TODO Number of subprocesses for data loading
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained(os.path.expanduser("~/tmp/gpt2_trainer/final_model"))
tokenizer.save_pretrained(os.path.expanduser("~/tmp/gpt2_trainer/final_model"))

print('Done!\a')