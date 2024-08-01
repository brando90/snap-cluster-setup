# """
# FL := Formal Language
# NL := Natural Language
# """

# def main():
#     # -- Load HF data sets to train, one FL and one NL
#     ...


# if __name__ == '__main__':
#     import fire
#     import time
#     start = time.time()
#     fire.Fire(main)
#     print(f"Done!\a Time: {time.time()-start:.2f} sec, {(time.time()-start)/60:.2f} min, {(time.time()-start)/3600:.2f} hr\a")


from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Step 1: Load and Save the Model using Hugging Face
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.to('cuda')

# # Save the model and tokenizer
# model_save_path = './model/gpt2'
# tokenizer.save_pretrained(model_save_path)
# model.save_pretrained(model_save_path)

# Step 2: Load the model in vLLM using the saved model path
from vllm import LLM, SamplingParams

# Assuming vLLM can load models from a path
vllm_model = LLM(model=model)

# Define sampling parameters and prompts
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
prompts = ["Hello, my name is", "The future of AI is"]

# Generate text using the specified model and parameters in vLLM
outputs = vllm_model.generate(prompts, sampling_params)

# Print the generated texts
for output in outputs:
    print(f"Prompt: {output.prompt}, Generated text: {output.outputs[0].text}")