from datasets import load_dataset, DatasetDict

# Load a dataset from Hugging Face
dataset = load_dataset('squad', split='train')

# Split the dataset into training and validation sets
# Specify the fraction for the test set (validation set)
train_val_split = dataset.train_test_split(test_size=0.1)

# Extract the training and validation datasets
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']

# Print the size of the datasets
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

# Save the datasets if needed
# train_dataset.save_to_disk('path/to/train_dataset')
# val_dataset.save_to_disk('path/to/val_dataset')
