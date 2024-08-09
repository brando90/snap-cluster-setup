"""

TODO: perhaps later have accelerate ddp, fsdp for this (to see if num_proc interferes with procs in ddp)

ref: https://chatgpt.com/c/6e4de8fb-272f-41ae-a468-7162fe03aaad
"""
import time
from datasets import Dataset
from transformers import GPT2Tokenizer
import multiprocessing

# Sample data
num_examples: int = 120_000
data = {"text": ["This is a sample sentence because we want to test if num_proc actually speeds up or not the .map process that seems slow, this is assuming we have not preprocessed the huggingface data set and saved it in disk and then loading it later."] * num_examples}

# Create a Dataset
dataset = Dataset.from_dict(data)

# Initialize GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

# Define a preprocessing function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Set number of processes
# num_proc = multiprocessing.cpu_count() // 2  # Use half of the available CPU cores
num_proc = multiprocessing.cpu_count()
# num_proc = None
print(f'{num_proc=}, {multiprocessing.cpu_count()=}')

# Apply the tokenize function with multiple processes
start_time = time.time()
tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=num_proc)
end_time = time.time()

time_taken = end_time - start_time
print(f"---> Time taken with num_proc={num_proc}: {time_taken} seconds")

# Display the tokenized dataset
print(tokenized_dataset)

# Convert tokenized dataset to a DataFrame for display
import pandas as pd
tokenized_df = pd.DataFrame(tokenized_dataset)

# Display the tokenized dataset
print(tokenized_df.head())