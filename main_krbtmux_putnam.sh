#!/bin/bash
# - snap: https://ilwiki.stanford.edu/doku.php?id=snap-servers:snap-servers and support il-action@cs.stanford.edu
# - live server stats: https://ilwiki.stanford.edu/doku.php?id=snap-servers:snap-gpu-servers-stats

krbtmux
reauth

source $AFS/.bashrc
conda activate gold_ai_olympiad
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader | awk '{print NR-1 " " $1}' | sort -nk2 | head -n1 | cut -d' ' -f1)
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=3; 
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES

# -- Run
source $AFS/.bashrc
conda activate gold_ai_olympiad
cd  ~/gold-ai-olympiad/py_src/evals/
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader | awk '{print NR-1 " " $1}' | sort -nk2 | head -n1 | cut -d' ' -f1)
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
# - Debug run
# python boxed_acc_eval.py --model mistralai/Mistral-7B-Instruct-v0.1 --path_2_eval_dataset $path2eval_dataset--end 348 --batch_size 348 --mode dryrun
# python boxed_acc_eval.py --model deepseek-ai/deepseek-math-7b-rl --path_2_eval_dataset $path2eval_dataset --end 348 --batch_size 100 --mode dryrun 
# python boxed_acc_eval.py --model deepseek-ai/deepseek-math-7b-instruct --path_2_eval_dataset $path2eval_dataset--end 348 --batch_size 25 --mode dryrun 

# - Putnam Original
# path2eval_dataset=~/putnam-math/data/Putnam_MATH_original_static3/test
path2eval_dataset=~/putnam-math/data/Putnam_MATH_variations_static3/variations/test

python boxed_acc_eval.py --model mistralai/Mistral-7B-v0.1 --path_2_eval_dataset $path2eval_dataset--end 348 --batch_size 348 --mode online 
python boxed_acc_eval.py --model mistralai/Mistral-7B-Instruct-v0.1 --path_2_eval_dataset $path2eval_dataset--end 348 --batch_size 348 --mode online 

python boxed_acc_eval.py --model meta-llama/Meta-Llama-3-8B --path_2_eval_dataset $path2eval_dataset--end 348 --batch_size 348 --mode online 
python boxed_acc_eval.py --model meta-llama/Meta-Llama-3-8B-Instruct --path_2_eval_dataset $path2eval_dataset--end 348 --batch_size 348 --mode online 

python boxed_acc_eval.py --model google/gemma-2b $path2eval_dataset--end 348 --batch_size 50 --mode online 
python boxed_acc_eval.py --model google/gemma-2b-it $path2eval_dataset--end 348 --batch_size 50 --mode online 

python boxed_acc_eval.py --model deepseek-ai/deepseek-math-7b-base --path_2_eval_dataset $path2eval_dataset--end 348 --batch_size 25 --mode online 
python boxed_acc_eval.py --model deepseek-ai/deepseek-math-7b-instruct --path_2_eval_dataset $path2eval_dataset--end 348 --batch_size 25 --mode online 
python boxed_acc_eval.py --model deepseek-ai/deepseek-math-7b-rl --path_2_eval_dataset $path2eval_dataset --end 348 --batch_size 25 --mode online 

# python boxed_acc_eval.py --model EleutherAI/llemma_7b --hf_gen_type=pipeline --path_2_eval_dataset $path2eval_dataset--end 348 --batch_size 348 --mode online 
# python boxed_acc_eval.py --model morph-labs/morph-prover-v0-7b --hf_gen_type pipeline --path_2_eval_dataset $path2eval_dataset--end 348 --batch_size 348 --mode online 

python boxed_acc_eval.py --model gpt-3.5-turbo --path_2_eval_dataset $path2eval_dataset--end 348 --batch_size 348 --mode online 
python boxed_acc_eval.py --model gpt-4-turbo --path_2_eval_dataset $path2eval_dataset--end 348 --batch_size 348 --mode online 
python boxed_acc_eval.py --model gpt-4o --path_2_eval_dataset $path2eval_dataset--end 348 --batch_size 348 --mode online 


# - Putnam Original vs variation
# path2eval_dataset=~/putnam-math/data/Putnam_MATH_variations_static3/original/test
# path2eval_dataset=~/putnam-math/data/Putnam_MATH_variations_static3/variations/test

python boxed_acc_eval.py --model meta-llama/Meta-Llama-3-8B --path_2_eval_dataset $path2eval_dataset --end 348 --batch_size 348 --mode online 

python boxed_acc_eval.py --model deepseek-ai/deepseek-math-7b-base --path_2_eval_dataset $path2eval_dataset --end 348 --batch_size 25 --mode online 

# python boxed_acc_eval.py --model gpt-3.5-turbo --path_2_eval_dataset $path2eval_dataset --end 348 --batch_size 348 --mode online 
# python boxed_acc_eval.py --model gpt-4-turbo --path_2_eval_dataset $path2eval_dataset --end 348 --batch_size 348 --mode online 
python boxed_acc_eval.py --model gpt-4o --path_2_eval_dataset $path2eval_dataset --end 348 --batch_size 348 --mode online 

