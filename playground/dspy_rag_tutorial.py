import dspy
from dspy.datasets import HotPotQA
from dspy.teleprompt import BootstrapFewShot

def main():
    # Setting up the language model (LM) and retrieval model (RM)
    lm = dspy.OpenAI(model='gpt-3.5-turbo')
    # lm = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
    dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)

    # Loading the HotPotQA dataset
    dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)
    trainset = [x.with_inputs('question') for x in dataset.train]
    devset = [x.with_inputs('question') for x in dataset.dev]

    # Building the RAG module with signature
    class GenerateAnswer(dspy.Signature):
        context = dspy.InputField(desc="may contain relevant facts")
        question = dspy.InputField()
        answer = dspy.OutputField(desc="often between 1 and 5 words")

    class RAG(dspy.Module):
        def __init__(self, num_passages=3):
            super().__init__()
            self.retrieve = dspy.Retrieve(k=num_passages)
            self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

        def forward(self, question):
            context = self.retrieve(question).passages
            prediction = self.generate_answer(context=context, question=question)
            return dspy.Prediction(context=context, answer=prediction.answer)

    # Compiling the RAG module
    def validate_context_and_answer(example, pred, trace=None):
        answer_EM = dspy.evaluate.answer_exact_match(example, pred)
        answer_PM = dspy.evaluate.answer_passage_match(example, pred)
        return answer_EM and answer_PM

    teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
    compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

    # Running the RAG module
    my_question = "What castle did David Gregory inherit?"
    pred = compiled_rag(my_question)

    # Outputting results
    print(f"Question: {my_question}")
    print(f"Predicted Answer: {pred.answer}")
    print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")

if __name__ == "__main__":
    main()
