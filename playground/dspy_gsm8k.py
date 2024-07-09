import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
from pathlib import Path

# Define the module
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)

def main():
    # Load key string from path to file, one line
    path = Path('~/keys/openai_api_key_brandos_koyejolab.txt').expanduser()
    key = open("key.txt").readline().strip()
    print(f'{key=}')

    # Set up the L8ikkuM
    lm = dspy.OpenAI(model='gpt-3.5-turbo-instruct', max_tokens=250)
    lm = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
    dspy.settings.configure(lm=lm)

    # Load math questions from the GSM8K dataset
    gsm8k = GSM8K()
    gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]

    # Compile and evaluate the model
    config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)
    teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
    optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset, valset=gsm8k_devset)

    # Evaluate
    evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)
    results = evaluate(optimized_cot)

    # Inspect the model's history
    history = turbo.inspect_history(n=1)
    print("Last interaction history:", history)

    # Test run with a custom question
    test_question = "What is the square root of 144?"
    test_response = optimized_cot(test_question)
    print("Test question:", test_question)
    print("Test response:", test_response.answer)

if __name__ == "__main__":
    main()
