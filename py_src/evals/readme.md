# Running Check Final Answer Evaluations for Language Models (LMs)

This code is a small adaptation from the Meta-Math original evaluation. 
We have verified that it runs within 1-2% accuracy difference with Mistral7B-base, therefore giving us confidence this code is correct and reliable to use. 
<!-- Note mistral ins 13.1% ref: https://mistral.ai/news/announcing-mistral-7b/ us on MATH TODO, lost value sadly -->

We also did a check with Claude 3.5 Sonnet and [the original Anthropic blog](https://www.anthropic.com/news/claude-3-5-sonnet) reports `71.1%` with `0-shot CoT` on Hendryck's MATH eval benchmark. 
Claude 3 Opus reports `60.1%` with `0-shot Cot` on Hendryck's MATH eval benchmark.. 
Using our own `0-shot Cot` (note: it's impossible to know exactly their prompt and setting) we got `X` result using our own eval code on Hendryck's MATH eval benchmark. 
To verify Claude accuracy run:
```bash
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model claude-3-5-sonnet-20240620 --path_2_eval_dataset ~/snap-cluster-setup/data/MATH/test --mode dryrun
```
Partial output:
```bash
TODO
```

## Quickstart

Create a venv or conda env for this project, instructions here or use these simplified set of instructions:
```bash
# - Create conda env (note: vllm has issues with 3.10 so we are using 3.9, ref: https://gist.github.com/brando90/c55c74e840d42c952d4aec7b74e0be6c)
# conda create -n snap_cluster_setup_py3_9 python=3.9
conda create -n snap_cluster_setup python=3.11
# - Activate your conda env
conda activate snap_cluster_setup
# - Pip install snap-cluster-setup repo in editable mode with pip
pip install --upgrade pip
pip install -e ~/snap-cluster-setup
```

Verify the data has the right number of points (200 August 14 2024):
```bash
jq -c '.[]' /~/putnam-math/data/Putnam_MATH_original_static_final/Putnam_MATH_boxed_problems.json | wc -l
```
Sample output:
```bash
200
```

### Quickstart - Open Source Model Putnam Evaluations
The instructions here are for reproducing our Open Source model evaluations based on [this early version of the manuscript](https://openreview.net/forum?id=1720vDqiBK#discussion) 
and [the results form this table](py_src/evals/eval_images/first_results_putnam_math.png).

Select a GPU:
```bash
export CUDA_VISIBLE_DEVICES=1 
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
```

Now give the model and path to the data set you want to evaluate [in the format of Hendryck's MATH data set](https://github.com/hendrycks/math):
```bash
# - Mistral 7B Base (https://huggingface.co/mistralai/Mistral-7B-v0.1)
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model mistralai/Mistral-7B-v0.1 --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test
# --> Uncomment for newer versions of the data set
# python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model mistralai/Mistral-7B-v0.1 --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static_final

# - LLama 3 8B Base (https://huggingface.co/meta-llama/Meta-Llama-3-8B)
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model meta-llama/Meta-Llama-3-8B --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test

# - Gemma 2B
python boxed_acc_eval.py --model google/gemma-2b --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test 

# - Deep-Seek-Math 7B Base (https://huggingface.co/collections/deepseek-ai/deepseek-math-65f2962739da11599e441681)
python boxed_acc_eval.py --model deepseek-ai/deepseek-math-7b-base --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test --end 348 --batch_size 1 --mode online 

# - Deep-Seek-Math 7B Instruct (https://huggingface.co/collections/deepseek-ai/deepseek-math-65f2962739da11599e441681)
python boxed_acc_eval.py --model deepseek-ai/deepseek-math-7b-instruct --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test --end 348 --batch_size 1 --mode online 

# - Deep-Seek-Math 7B RL (https://huggingface.co/collections/deepseek-ai/deepseek-math-65f2962739da11599e441681)
python boxed_acc_eval.py --model deepseek-ai/deepseek-math-7b-rl --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test --end 348 --batch_size 1 --mode online 

```

### Quickstart - Prioprietry Model Putnam Evaluations
The instructions here are for reproducing our Prioprietry Source model evaluations based on [this version of the early manuscript](https://openreview.net/forum?id=1720vDqiBK#discussion) 
and [the results form this table](py_src/evals/eval_images/first_results_putnam_math.png).

#### OpenAI GPT Evaluations
The following are the commands to run GPT evaluations. 
Tip: use GPT3.5 (or the chepaer version when you read this) to **quickly** and **cheaply** verify everything is working for you before running the larger evaluations (~200 data points as of this writing):
```bash
# - GPT 3.5
python boxed_acc_eval.py --model gpt-3.5-turbo --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test --end 348 --batch_size 348 
# python boxed_acc_eval.py --model gpt-4o-mini --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test --end 348 --batch_size 348 

# - GPT 4 Turbo
python boxed_acc_eval.py --model gpt-4-turbo --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test --end 348 --batch_size 348

# - GPT 4o
python boxed_acc_eval.py --model gpt-4o --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test --end 348 --batch_size 348 
```

#### Anthropic Claude Evaluations
The following are the commands to run [Anthropic's Claude](https://docs.anthropic.com/en/docs/about-claude/models) evaluations. 
```bash
# - Claude 3 Opus

# - Claude 3.5 Sonnet 
# python boxed_acc_eval.py --model  --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test --end 348 --batch_size 348 --mode dryrun
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model claude-3-5-sonnet-20240620 --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static_final --end 348 --batch_size 348 --mode dryrun
# python boxed_acc_eval.py --model claude-3-5-sonnet-20240620 --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static_final/Putnam_MATH_boxed_problems.json --end 348 --batch_size 348 --mode dryrun
```

#### Gemini Evaluations
TODO:

## Evaluations with the Variations Benchmarks

### Generation of Benchmark Datasets with our Python Scripts 
TODO

### Open Source Model Evluations on the Variation Benchmarks
```bash
# python boxed_acc_eval.py --model meta-llama/Meta-Llama-3-8B-Instruct --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_variations_static2/test --end 348 --batch_size 348 --mode online 
# python boxed_acc_eval.py --model deepseek-ai/deepseek-math-7b-base --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_variations_static2/test --end 348 --batch_size 348 --mode online 
```

### Prioprietrg Model Evluations on the Variation Benchmarks
```bash
# python boxed_acc_eval.py --model gpt-3.5-turbo --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_variations_static2/test --end 348 --batch_size 348 --mode online 
# python boxed_acc_eval.py --model gpt-4-turbo --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_variations_static2/test --end 348 --batch_size 348 --mode online 
# python boxed_acc_eval.py --model gpt-4o --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_variations_static2/test --end 348 --batch_size 348 --mode online 
```

## Other Features

### Saving Mode Responses
TODO

Motivation: debugging, human evaluations, and automatic proof evaluations e.g., Teacher Forced Accuracy (tfa).
