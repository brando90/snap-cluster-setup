import os
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
import evaluate

from datasets import load_dataset, load_metric
from typing import Dict, Tuple, Optional
from pathlib import Path

from utils import eval_hf, get_ai4m_v0, get_data_set_args, load_dataset_block_size

from utils import load_model_block_size

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray],
                    path: str = 'accuracy',
                    ) -> Dict[str, float]:
    """
    Compute the accuracy of the model.

    Args:
    eval_pred: A tuple containing the model predictions and labels.

    Returns:
    A dictionary with the accuracy score.
    
    TODO: document properly what accuracy is. Is it tfa, ara, exact string match, avg acc (wrt length etc.) ref: https://huggingface.co/spaces/evaluate-metric/accuracy
    """
    metric = evaluate.load(path=path)   # load metric from file or hf
    predictions, references = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=references)

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

def setup_and_train_big_model(
                            # pretrained_model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.2",
                            pretrained_model_name_or_path: str = "gpt2",
                            path: str = "all_ai4m_datasets_v0",
                            path_test: str = "hoskinson-center/proofnet",
                            output_dir_train: str = '~/tmp/all_ai4m_datasets_v0/train',
                            output_dir_test: str = '~/tmp/all_ai4m_datasets_v0/eval/proofnet/test',
                            path_to_save_model: Optional[str] = '~/tmp/all_ai4m_datasets_v0/model',  # suggested path: '~/tmp/proofnet/model' then expanduser in py code
                            max_steps: int = 2,
                            per_device_train_batch_size: Optional[int] = 2,
                            per_device_eval_batch_size: Optional[int] = 1,
                            save_total_limit: Optional[int] = None,
                            learning_rate: float = 5e-5,
                            weight_decay: float = 0.01,
                            max_grad_norm: float = 1.0, 
                            optim='paged_adamw_32bit',
                            gradient_accumulation_steps = 8, # see: based on alpaca https://github.com/tatsu-lab/stanford_alpaca, allows to process effective_batch_size = gradient_accumulation_steps * batch_size, num its to accumulate before opt update step
                            gradient_checkpointing: Optional[bool] = True,  # If True, use gradient checkpointing to save memory at the expense of slower backward pass.
                            lr_scheduler_type='cosine',
                            warmup_ratio=0.01,
                            evaluation_strategy='no',
                            eval_steps = None,  # TODO
                            report_to: str = 'none',
                            block_size: int = 4096,  # TODO we need to move away from block size training to respecting sentences
                    ) -> None:
    # Clear CUDA cache to free up memory
    torch.cuda.empty_cache()

    # Load tokenizer and model
    model, tokenizer = load_model_block_size(pretrained_model_name_or_path, verbose=True)

    # Load the dataset
    path, name, data_files, split, streaming = get_data_set_args(path)
    # from train.simple_train import preprocess_function_proofnet_simple
    # train_dataset = load_dataset(path="hoskinson-center/proofnet", split='validation').map(lambda examples: preprocess_function_proofnet_simple(examples, tokenizer), batched=True, remove_columns=["nl_statement", "formal_statement"])
    train_dataset = load_dataset_block_size(tokenizer, block_size, path, name, data_files, split, streaming)
    test_dataset = load_dataset(path_test, split='test')  #TODO block size vs non, matters?

    # Preprocess the dataset
    if path == "hoskinson-center/proofnet":
        preprocess_function = preprocess_function_proofnet
        # note: text field is usually more common!
        # val_dataset = val_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True, remove_columns=["nl_statement", "formal_statement"])
        test_dataset = test_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True, remove_columns=["nl_statement", "formal_statement"])

    # Training arguments
    output_dir_train: Path = Path(output_dir_train).expanduser()
    output_dir_train.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir_train,
        max_steps=max_steps,
        evaluation_strategy=evaluation_strategy,  # "no"`: No evaluation is done during training. no can be good to avoid memory issues.
        gradient_accumulation_steps=gradient_accumulation_steps,  # based on alpaca https://github.com/tatsu-lab/stanford_alpaca, allows to process effective_batch_size = gradient_accumulation_steps * batch_size, num its to accumulate before opt update step
        gradient_checkpointing = gradient_checkpointing,  # TODO depending on hardware set to true?
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=weight_decay,
        save_steps=max_steps//3,  # alpaca does 2000, other defaults were 500
        save_total_limit=save_total_limit,
        max_grad_norm=max_grad_norm,
        optim=optim,
        logging_dir=output_dir_train / 'logs',
        logging_first_step=True,
        logging_strategy='steps',
        eval_steps = eval_steps,
        remove_unused_columns=False,  # TODO don't get why https://stackoverflow.com/questions/76879872/how-to-use-huggingface-hf-trainer-train-with-custom-collate-function/76929999#76929999 , https://claude.ai/chat/475a4638-cee3-4ce0-af64-c8b8d1dc0d90
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        report_to = report_to,  # options I recommend: 'none' or 'wandb'
        fp16=False,  # never ever set to True
        bf16=torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8,  # if >= 8 ==> brain float 16 available or set to True if you always want fp32
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,  # None or eval strategy says 'no' is training args
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    # Train the model
    trainer.train()

    # Evaluate the model
    if output_dir_test is not None:
        output_dir_test: Path = Path(output_dir_test).expanduser()
        output_dir_test.mkdir(parents=True, exist_ok=True)
        eval_args = TrainingArguments(output_dir=output_dir_test, fp16=False, bf16=torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8, report_to=report_to)
        trainer = Trainer(model=model, args=eval_args, train_dataset=None, eval_dataset=test_dataset)
        # results: dict[str, float] = trainer.evaluate(test_dataset)
        results: dict[str, float] = eval_hf(trainer, name='', path=path, split='test', eval_dataset=test_dataset)
        print(f'{path=} split=test {results=}')

    # Save the trained model
    if path_to_save_model is not None:
        model.save_pretrained(path_to_save_model)

def main() -> None:
    """
    Main function to execute the model training and evaluation.
    """
    setup_and_train_big_model()

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print(f"Time taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")
