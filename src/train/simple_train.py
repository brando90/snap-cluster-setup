import os
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
from typing import Dict, Tuple, Optional
from pathlib import Path

# Clear CUDA cache to free up memory
torch.cuda.empty_cache()

# Load the accuracy metric from the datasets library
metric = load_metric('accuracy')

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compute the accuracy of the model.

    Args:
    eval_pred: A tuple containing the model predictions and labels.

    Returns:
    A dictionary with the accuracy score.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def preprocess_function_proofnet(examples: Dict[str, list], tokenizer: GPT2Tokenizer) -> Dict[str, torch.Tensor]:
    """
    Preprocess the input data for the proofnet dataset.

    Args:
    examples: The examples to preprocess.
    tokenizer: The tokenizer for encoding the texts.

    Returns:
    The processed model inputs.
    """
    inputs = [f"{examples['nl_statement'][i]}{tokenizer.eos_token}{examples['formal_statement'][i]}" for i in range(len(examples['nl_statement']))]
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    labels = model_inputs.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

def setup_and_train_proofnet(pretrained_model_name_or_path: str = "gpt2", 
                            path: str = "hoskinson-center/proofnet",
                            output_dir_val: str = '~/tmp/proofnet/validation',
                            output_dir_test: str = '~/tmp/proofnet/test',
                            path_to_save_model: Optional[str] = None,  # suggested path: '~/tmp/proofnet/model' then expanduser in py code
                            num_train_epochs: int = 3,
                            per_device_train_batch_size: Optional[int] = 2,
                            per_device_eval_batch_size: Optional[int] = 2,
                            save_total_limit: Optional[int] = None,
                            learning_rate: float = 5e-5,
                            weight_decay: float = 0.01,
                            max_grad_norm: float = 1.0, 
                            optim='paged_adamw_32bit',
                            gradient_accumulation_steps = 2, # see: based on alpaca https://github.com/tatsu-lab/stanford_alpaca, allows to process effective_batch_size = gradient_accumulation_steps * batch_size, num its to accumulate before opt update step
                            gradient_checkpointing: Optional[bool] = False,
                            # lr_scheduler_type='cosine',  # TODO: https://discord.com/channels/879548962464493619/1227708244697284724/1227708244697284724
                            # warmup_ratio=0.01,   # TODO: https://discord.com/channels/879548962464493619/1227708244697284724/1227708244697284724
                    ) -> None:
    """
    Set up the environment, preprocess the dataset, and train the model.

    Args:
    tokenizer_name: The name of the tokenizer.
    model_name: The name of the model.
    dataset_path: The path to the dataset.
    """
    # Load tokenizer and model
    if pretrained_model_name_or_path == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path, max_length=1024)
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f'{tokenizer.pad_token=}')
        model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
        # model.resize_token_embeddings(len(tokenizer))  # leaving for reference, not needed since pad = eos for us
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        block_size: int = tokenizer.model_max_length
        print(f'{block_size=}')

    # Load the dataset
    dataset_val = load_dataset(path, split='validation')
    dataset_test = load_dataset(path, split='test')

    # Preprocess the dataset
    if path == "hoskinson-center/proofnet":
        preprocess_function = preprocess_function_proofnet
        val_dataset = dataset_val.map(lambda examples: preprocess_function(examples, tokenizer), batched=True, remove_columns=["nl_statement", "formal_statement"])
        test_dataset = dataset_test.map(lambda examples: preprocess_function(examples, tokenizer), batched=True, remove_columns=["nl_statement", "formal_statement"])

    # Training arguments
    output_dir_val: Path = Path(output_dir_val).expanduser()
    output_dir_val.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir_val,
        evaluation_strategy='no',  # "no"`: No evaluation is done during training. no can be good to avoid memory issues.
        gradient_accumulation_steps=gradient_accumulation_steps,  # based on alpaca https://github.com/tatsu-lab/stanford_alpaca, allows to process effective_batch_size = gradient_accumulation_steps * batch_size, num its to accumulate before opt update step
        gradient_checkpointing = gradient_checkpointing,  # TODO depending on hardware set to true?
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=weight_decay,
        save_total_limit=save_total_limit,
        num_train_epochs=num_train_epochs,
        max_grad_norm=max_grad_norm,
        optim=optim,
        # lr_scheduler_type=lr_scheduler_type  # TODO: https://discord.com/channels/879548962464493619/1227708244697284724/1227708244697284724
        # warmup_ratio=warmup_ratio,
        fp16=False,  # never ever set to True
        bf16=torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8,  # if >= 8 ==> brain float 16 available or set to True if you always want fp32
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=val_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    # Train the model
    trainer.train()

    # Evaluate the model
    output_dir_test: Path = Path(output_dir_test).expanduser()
    output_dir_test.mkdir(parents=True, exist_ok=True)
    # Later, you decide to change the evaluation strategy
    training_args.evaluation_strategy = 'epoch'  # "epoch"`: Evaluation is done at the end of each epoch.
    results = trainer.evaluate(test_dataset)
    print(results)

    # Save the trained model
    if path_to_save_model is not None:
        path_to_save_model: Path = Path(path_to_save_model).expanduser()
        output_dir_test.mkdir(parents=True, exist_ok=True)
        # Later, you decide to change the evaluation strategy
        training_args.evaluation_strategy = 'epoch'  # "epoch"`: Evaluation is done at the end of each epoch.
        model.save_pretrained(path_to_save_model)

def main() -> None:
    """
    Main function to execute the model training and evaluation.
    """
    setup_and_train_proofnet()

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print(f"Time taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")
