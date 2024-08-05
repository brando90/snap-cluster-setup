"""
Mental model is:
RequestOutput per Prompt. So Outputs == "One output object per prompt (one of batch of prompts)."
    list[RequestOutput] <--> List[Prompts]
Completions per Prompt. So Completions == "One list of Completions per Prompt (one list of completions per prompt)"
    list[Completion] <--> Prompt
"""
# copy pasted from https://docs.vllm.ai/en/latest/getting_started/quickstart.html

# do export VLLM_USE_MODELSCOPE=True
import vllm
from vllm import LLM, SamplingParams, RequestOutput, CompletionOutput

def test_vllm():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, n=1)
    # sampling_params = SamplingParams(temperature=0.8, top_p=0.95, n=2)


    # llm = LLM(model="facebook/opt-125m")
    llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1")

    # List[Prompt] --> List[RequestOutput] --> List[Completions] --> List[List[Completion]]
    outputs_for_batch_prompts: list[RequestOutput] = llm.generate(prompts, sampling_params)
    print(f'{type(outputs_for_batch_prompts)=}')
    print(f'{type(outputs_for_batch_prompts[0])=}')

    # -- Print the outputs for each prompt
    request_output_per_prompt: RequestOutput
    for request_output_per_prompt in outputs_for_batch_prompts:
        # - Get completions per single prompt
        # Prompt --> list[Completion]
        prompt: str = request_output_per_prompt.prompt
        completions: list[CompletionOutput] = request_output_per_prompt.outputs
        generated_completions_text: list[str] = [completion.text for completion in completions]
        # generated_text: str = request_output_per_prompt.outputs[0].text  # for a single completion per prompt
        print(f"Prompt: {prompt!r}, Generated text: {generated_completions_text!r}")

if __name__ == "__main__":
    import time
    start_time = time.time()
    test_vllm()
    print(f"Time taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")