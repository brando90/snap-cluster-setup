# import math
# from typing import Optional, Any, Dict

# # import torch

# # import datasets
# # from datasets import load_dataset, interleave_datasets

# # from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoConfig
# from transformers import Trainer 

# def eval_hf(trainer: Trainer, 
#             path: str, 
#             name: str, 
#             split: str, 
#             eval_dataset = None,  # provide eval data set to overwrite it
#             max_eval_samples: Any = 'Unknown_Eval_Max_Samples',
#             ) -> Dict[str, float]:
#     metrics: Dict[str, float] = trainer.evaluate(eval_dataset)
#     try:
#         perplexity = math.exp(metrics["eval_loss"])
#     except OverflowError:
#         perplexity = float("inf")
#     metrics["perplexity"] = perplexity
#     path = path.replace('/', '_')  # needed only when saving results
#     print(f'Eval metrics {path} {name} {split} {max_eval_samples}: {metrics=}')
#     trainer.log_metrics(f"eval_{path}_{name}_{split}_{max_eval_samples}", metrics)  # display metrics
#     trainer.save_metrics(f"eval_{path}_{name}_{split}_{max_eval_samples}", metrics)
#     return metrics