import torch
print(torch.backends.cudnn.version())
"""
>>> import torch

>>> print(torch.backends.cudnn.version())
90100
"""
#%%
import torch
import torch.nn as nn

def test_matrix_multiplication():
    # Create dummy data on CUDA
    a = torch.randn(1000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()

    # Perform matrix multiplication
    result = torch.mm(a, b)
    print("Matrix multiplication result:", result)

def test_linear_layer():
    # Create a dummy linear layer
    linear_layer = nn.Linear(1000, 1000).cuda()

    # Create dummy input data
    input_data = torch.randn(1000, 1000).cuda()

    # Perform forward pass
    output = linear_layer(input_data)
    print("Linear layer output:", output)

# if __name__ == "__main__":
print("Testing matrix multiplication on CUDA")
test_matrix_multiplication()
print("Testing linear layer on CUDA")
test_linear_layer()

#%%
import torch
from transformers import GPT2Tokenizer, GPT2Model

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

# Move model to GPU and enable bf16 precision
model = model.to(device=device, dtype=torch.bfloat16)

# Generate random tokens
input_text = "Hello, how are you doing today?"
input_tokens = tokenizer(input_text, return_tensors='pt').to(device=device)

# Perform forward pass with bf16 precision
with torch.no_grad():
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(**input_tokens)

# Print the shape of the last hidden states
print("Last hidden states shape:", outputs.last_hidden_state.shape)
print("Last hidden states shape:", outputs.last_hidden_state.sum())

#%%
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Move model to GPU and enable bf16 precision
model = model.to(device=device, dtype=torch.bfloat16)

# Generate random input data
input_text = "Hello, how are you doing today?"
inputs = tokenizer(input_text, return_tensors='pt').to(device=device)

# Generate random target data (same as input for simplicity)
target_text = "Hello, how are you doing today?"
targets = tokenizer(target_text, return_tensors='pt').to(device=device)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss()

# Training loop
model.train()
epochs = 1
for epoch in range(epochs):
    optimizer.zero_grad()
    
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(**inputs, labels=targets['input_ids'])
    
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    
    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

# Save the model
import os
os.makedirs(os.path.expanduser("~/tmp/"), exist_ok=True)
model.save_pretrained(os.path.expanduser("~/tmp/gpt2_finetuned"))
tokenizer.save_pretrained(os.path.expanduser("~/tmp/gpt2_finetuned"))

# test model we just saved
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned model and tokenizer
model_path = os.path.expanduser("~/tmp/gpt2_finetuned")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Move model to GPU and enable bf16 precision
model = model.to(device=device, dtype=torch.bfloat16)

# Generate some sample input data
input_text = "Hello, how are you doing today?"
inputs = tokenizer(input_text, return_tensors='pt').to(device=device)

# Perform forward pass with bf16 precision
with torch.no_grad():
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(**inputs)

# Print the shape of the last hidden states
print(f'{outputs=}')
# print("Last hidden states shape:", outputs.last_hidden_states.shape)
# print("Last hidden states sum:", outputs.last_hidden_states.sum().item())
print("Logits sum:", outputs.logits.sum().item())
 


#%%
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
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

# Generate some dummy data
data = [
    {"text": "Hello, how are you doing today?"},
    {"text": "The weather is great today."},
    {"text": "Let's learn about AI and ML."},
    {"text": "Transformers are powerful models."}
]

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(data)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=32)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir=os.path.expanduser("~/tmp/gpt2_trainer"),
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    # fp16=True,  # Enable mixed precision training
    bf16=True,  # Enable bf16 training
    logging_dir=os.path.expanduser("~/tmp/gpt2_trainer/logs"),
    logging_steps=200,
    report_to='none',
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained(os.path.expanduser("~/tmp/gpt2_trainer/final_model"))
tokenizer.save_pretrained(os.path.expanduser("~/tmp/gpt2_trainer/final_model"))

#%%
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

# Preprocess the dataset
# train_dataset = dataset['train'].map(lambda examples: preprocess_function_proofnet_simple(examples, tokenizer), batched=True, remove_columns=["nl_statement", "formal_statement"])
train_dataset = dataset['validation'].map(lambda examples: preprocess_function_proofnet_simple(examples, tokenizer), batched=True, remove_columns=["nl_statement", "formal_statement"])
test_dataset = dataset['test'].map(lambda examples: preprocess_function_proofnet_simple(examples, tokenizer), batched=True, remove_columns=["nl_statement", "formal_statement"])

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir=os.path.expanduser("~/tmp/gpt2_trainer"),
    overwrite_output_dir=True,
    num_train_epochs=2,  
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    bf16=True,  # Enable bf16 training only
    logging_dir=os.path.expanduser("~/tmp/gpt2_trainer/logs"),
    logging_steps=200,
    report_to="none"  # Disable logging to WandB
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained(os.path.expanduser("~/tmp/gpt2_trainer/final_model"))
tokenizer.save_pretrained(os.path.expanduser("~/tmp/gpt2_trainer/final_model"))
