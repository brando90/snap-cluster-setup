# ref: https://chatgpt.com/c/4a5ea292-2bc0-44a8-9174-c23f8741be17
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import os
from pathlib import Path
import fire
import glob

def is_path(path: str) -> bool:
    """True if path str is a path."""
    path = os.path.expanduser(path)
    return path[0] == '/'

def get_sequence_lengths(dataset: list[str], tokenizer, truncation: bool = False):
    lengths = []
    for example in dataset:
        tokens = tokenizer(example, truncation=truncation)
        lengths.append(len(tokens['input_ids']))
    return lengths

def main(
        pretrained_model_name_or_path: str = 'meta-llama/Meta-Llama-3-8B',
        # path_2_dataset: str = '~/snap-cluster-setup/data/MATH/test',
        # path_2_dataset: str = '~/putnam-math/data/OlympiadBench_Dataset/data_math_boxed_21_08_2024_v2',
        # path_2_dataset: str = '~/putnam-math/data/Putnam_MATH_original_static_final_21_08_2024/Putnam_MATH_boxed_problems_full.json',
        # path_2_dataset: str = '~/putnam-math/data/Putnam_MATH_variations_static_constant/test.json',
        path_2_dataset: str = 'AI4M/leandojo-informalized',
        # path_2_dataset: str = '~/data/sft_agi_data_prompt4.2_08_05_2024/**/*.jsonl',
        split: str = 'train',
        desired_col_name: str = 'informalization',
        truncation: bool = False,
):
    # Step 1: Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    # Step 2: Load the dataset
    path_2_dataset = os.path.expanduser(path_2_dataset)
    print(f'{path_2_dataset=}')
    if '**/*.jsonl' in str(path_2_dataset):
        # mostly for data we have ~/data/dir/**/*.jsonl
        jsonl_files: list[str] = glob.glob(os.path.expanduser(path_2_dataset), recursive=True)
        ds = load_dataset('json', data_files=jsonl_files, split='train')
    elif is_path(path_2_dataset):
        from evals.data_eval_utils import get_iter_for_eval_data_set
        path_2_dataset: Path = Path(path_2_dataset).expanduser()
        ds: list[dict] = list(get_iter_for_eval_data_set(path_2_dataset))
        ds: list[str] = [f"Problem: {dp['problem']}\nSolution: {dp['solution']}\n" for dp in ds]
        # math_gold_probs_solns: list[dict] = math_gold_probs_solns[start:end]
        # random.shuffle(math_gold_probs_solns) if shuffle else None
    else:
        ds = load_dataset(path=path_2_dataset, split=split)
        ds: list[str] = [row[desired_col_name] for row in ds]
    print(f'{len(ds)=}')

    # Step 3: Tokenize and calculate sequence lengths
    ds_lengths = get_sequence_lengths(ds, tokenizer, truncation)

    # Calculate mean and standard deviation
    seq_length_mean = np.mean(ds_lengths)
    seq_length_std = np.std(ds_lengths)

    # Print mean and standard deviation
    print(f"Mean Seq Length: {seq_length_mean:.2f}, Std Sequence Length: {seq_length_std:.2f}")

    # Step 4: Plot histograms
    import seaborn as sns
    tips = sns.load_dataset("tips")
    sns.histplot(ds_lengths, bins=50, stat="probability", kde=True)  
    plt.axvline(seq_length_mean, linestyle='--', linewidth=2, label=f'Mean: {seq_length_mean:.2f}')
    # plt.title(f'Sequence Lengths of **MATH test** with **tokenizer Llama3-8B**')
    # plt.title(f'Sequence Lengths of **IMO (OlympiadBench)** with **tokenizer Llama3-8B**')
    # plt.title(f'Sequence Lengths of **Putnam Original** with **tokenizer Llama3-8B**')
    # plt.title(f'Sequence Lengths of **Putnam Variations** with **tokenizer Llama3-8B**')
    plt.title(f'Sequence Lengths of **LeanDojo AIF DeepSeekMath** with **tokenizer Llama3-8B**')
    plt.xlabel('Sequence Length')
    plt.tight_layout()
    plt.grid(True)
    plt.legend()

    # Save the plot to the Desktop
    # desktop_path = os.path.expanduser('~/Desktop/sequence_lengths_plot_math_test.png')
    # desktop_path = os.path.expanduser('~/Desktop/sequence_lengths_plot_imo_ob.png')
    # desktop_path = os.path.expanduser('~/Desktop/sequence_lengths_plot_putnam_original.png')
    # desktop_path = os.path.expanduser('~/Desktop/sequence_lengths_plot_putnam_.png')
    desktop_path = os.path.expanduser('~/Desktop/sequence_lengths_plot_leandojo_informalized.png')
    plt.savefig(desktop_path)
    plt.show()


if __name__ == "__main__":
    import time
    start = time.time()
    fire.Fire(main)
    print("Done!\a")
    print(f"Done!\a Time: {time.time()-start:.2f} sec, {(time.time()-start)/60:.2f} min, {(time.time()-start)/3600:.2f} hr\a")
