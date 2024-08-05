#%%
from transformers import pipeline

# Load the text-generation pipeline with your chosen model (e.g., GPT-2)
generator = pipeline('text-generation', model='gpt2')

# Define a list of prompts for batch generation
prompts = ["The weather today is", "Data science is", "What is the Lean programming language?"]

# Generate text based on the prompts 
n: int = 2
# List[Prompt] <-> List[Completions] <-> List[List[Completion]] since one prompt generate multiple completions Prompt -> List[Completion]
results_for_all_prompts: list[dict] = generator(prompts, max_length=50, num_return_sequences=n)
print(f'\n{results_for_all_prompts=}')
print(f'{len(results_for_all_prompts)=}')

# Print the outputs for each prompt
print()
result_per_prompt: list[dict]  # Completions == list[Completion]
for result_per_prompt in results_for_all_prompts:
    print(f'{result_per_prompt=}')
    print(f'{len(result_per_prompt)=}')
    print(result_per_prompt['generated_text'])