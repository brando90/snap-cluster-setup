from typing import List, Set, Tuple
from typing import List

import json
from pathlib import Path

from dataclasses import dataclass, field, asdict

import requests
import httpx
import torch
import torch.nn.functional as F

@dataclass
class MorphEmbedding:
    _url: str = "https://1fd5-129-153-96-190.ngrok-free.app/create_embedding"
    client: httpx.Client = field(default_factory=httpx.Client)
    aclient: httpx.AsyncClient = field(default_factory=httpx.AsyncClient)

    def create(self, prompt: str) -> List[float]:
        response = self.client.post(url=self._url, json=dict(prompt=prompt))
        response.raise_for_status()
        emb = response.json()["embedding"]
        return [x.item() for x in F.normalize(torch.tensor(emb), p=2, dim=0)]

    async def acreate(self, prompt: str) -> List[float]:
        response = await self.aclient.post(url=self._url, json=dict(prompt=prompt))
        response.raise_for_status()
        emb = response.json()["embedding"]
        return [x.item() for x in F.normalize(torch.tensor(emb), p=2, dim=0)]

    def query(self, prompt: str) -> str:
        response = self.client.post(url=self._url, json=dict(prompt=prompt))
        response.raise_for_status()
        response: dict = response.json()
        emb = response.json()["embedding"]
        return [x.item() for x in F.normalize(torch.tensor(emb), p=2, dim=0)]

# url = input("Enter the NgrokTunnel url: ")
# url = "https://9c31-140-83-63-105.ngrok-free.app"
url = "http://44.206.250.24:5000"
token = "dx1vuxJdCENMwF0lpItqFIxuFLA84Yr"
# Endpoint URLs
TOKEN_URL =  url + "/token/"
SEARCH_URL = url + "/search/"

# # Obtain a token from the token endpoint
# def get_token():
#     response = requests.get(TOKEN_URL)
#     if response.status_code == 200:
#         return response.json().get("token")
#     else:
#         raise Exception("Failed to get token")

# Use the token to post a search query
def search_with_token(query: str, token: str) -> str:
    headers = {
        "Authorization": f"Bearer {token}"
    }
    data = {
        "query": query
    }
    response = requests.post(SEARCH_URL, json=data, headers=headers)
    if response.status_code == 200:
        return response.json().get("data")
    else:
        raise Exception(f"Failed to search. Status Code: {response.status_code}. Message: {response.text}")

def get_theorem_and_proof_query_response(query_response: dict) -> Tuple[str, str]:
    """
    Note: proof actually means the entire proof source code for that theorem so it includes the theorem (name) too. 

    Format of text response:

    theorem: {theorem name}

    proof:
    {declaration docstring}
    {declaration code}
    """
    text: str = query_response['text']
    _theorem, proof = text.split('proof:\n')
    # - clean theorem by removing the prefix the embedding model is taking
    # theorem = _theorm.split('theorem: ') 
    # theorem = _theorem.split('declaration: ')
    theorem = 'TODO, implement later'
    # - clean the proof by removing the ```lean\n at the beginning and ``` at the end
    proof = proof[8:-4]
    return theorem, proof

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Compute the Precision at k (P@k) for ranked retrieval.
    This computes how many of the top k retrieved documents are relevant.

        P@k := |R /\ Sk| / |Sk|

    Sk := top k retrieved items.
    R := relevant items.
    |Sk| := k

    Of the documents the system retrieved, how many were actually relevant when considering only the top k results?
    
    :param retrieved: List of document IDs in the order they were retrieved.
    :param relevant: Set of document IDs that are relevant.
    :param k: Cut-off point to compute the precision.
    :return: Precision at k value.
    """
    
    # Get the top k retrieved documents
    top_k_retrieved = retrieved[:k]
    
    # Count how many of the top k retrieved documents are relevant
    relevant_count = sum(1 for doc in top_k_retrieved if doc in relevant)
    # relevant_count: int = 0
    # for doc in top_k_retrieved:
    #     if doc in relevant:
    #         relevant_count += 1
    
    # Compute the precision
    assert len(top_k_retrieved) == k
    return relevant_count / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Compute the Recall at k (R@k) for ranked retrieval.

        R@k := |R /\ Sk| / |R|

    Sk := top k retrieved items.
    R := relevant items.

    Of all the relevant documents in the system, how many did the system successfully retrieve when considering only the top k results?
    
    :param retrieved: List of document IDs in the order they were retrieved.
    :param relevant: Set of document IDs that are relevant.
    :param k: Cut-off point to compute the recall.
    :return: Recall at k value.
    """
    
    # Get the top k retrieved documents
    top_k_retrieved = retrieved[:k]
    
    # Count how many of the top k retrieved documents are relevant
    relevant_count = sum(1 for doc in top_k_retrieved if doc in relevant)
    
    # Compute the recall
    assert len(top_k_retrieved) == k
    return relevant_count / len(relevant)

# -- Tests ---

def test_get_theorem_parser():
#     text: str = """theorem: {theorem name}

# proof:
# {declaration docstring}
# {declaration code}"""
#     theorem, proof = 
#     print(f'')
    pass

def test1():
    """
    """
    print('-- Test when we have 2 documents but 1 is not relevant out of a total of 10 documents.')
    k_value = 2  # only evluate top k results from retrieved docs, so only consider the first k when evaluating
    retrieved_docs: list[str] = ['doc1', 'doc11', 'doc3']
    relevant_docs: Set[str] = {'doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6', 'doc7', 'doc8', 'doc9', 'doc10'}

    prec_k: float = precision_at_k(retrieved_docs, relevant_docs, k_value)
    print(f"P@{k_value}:", precision_at_k(retrieved_docs, relevant_docs, k_value))
    # note the accuracy is 0.5 and not 2/3 because we are only looking at the top 2 results.
    assert prec_k == 0.5

    recall_k: float = recall_at_k(retrieved_docs, relevant_docs, k_value)
    print(f"R@{k_value}:", recall_at_k(retrieved_docs, relevant_docs, k_value))
    # note the recall is not 0.2 because we are only considering the top 2 results, so the 3rd correct result is being discarded in the evaluation.
    assert recall_k == 0.1

    k_value = 3
    prec_k: float = precision_at_k(retrieved_docs, relevant_docs, k_value)
    print(f"P@{k_value}:", precision_at_k(retrieved_docs, relevant_docs, k_value))
    # note the accuracy is 2/3 this time because we are considering the entire window of relevant results.
    assert prec_k == 2/3

    recall_k: float = recall_at_k(retrieved_docs, relevant_docs, k_value)
    print(f"R@{k_value}:", recall_at_k(retrieved_docs, relevant_docs, k_value))
    # note the result is 0.2 this time becuase included one more example, which increase the recall. 
    assert recall_k == 0.2
    # note; recall @ k are a monotonic function at k, it can only improve the results of the model since the numerator can increase but the denominator is finxed

def test2():
    """
    Goal: 

    T := test set :=  {q_i, R_qi}_{i<=N} := {(q_i, d_i)}_{i<=N}
    X := raw thms := {d_i}_{N'}
    N := number of elements in the test set when partitioned by queries
    N' := number of raw elements in the test set e.g., the theorems/formal stmt
    Simplifying Assumption: each query has only 1 relevant result
    """
    # load the test data fron the json file 
    file_path: Path = Path("~/evals-for-autoformalization/src/nlp_eval/test_set.json").expanduser()
    with open(file_path, 'r') as file:
        test_data: list[dict] = json.load(file)
    print(f'Fields for a single data point: {test_data[0].keys}')

    # The query is Cuachy's mean value theorem, and only the first one 
    query = "Cauchy's Mean Value theorem"
    # get the last source code strings
    retrieved_docs: list[str] = [data_pt['source_code'] for data_pt in test_data[-1:]] + ['rand_doc1', 'rand_doc2']
    print(f'{retrieved_docs=}')
    # get all the test data source code as dummy relevant docs
    relevant_docs: Set[str] = {test_data_pt['source_code'] for test_data_pt in test_data}

    k = 1 # p@k=1.0
    prec_at_k: float = precision_at_k(retrieved_docs, relevant_docs, k)
    print(f'{prec_at_k=}')
    assert prec_at_k == 1.0

    k = 2 # p@k=0.5
    prec_at_k: float = precision_at_k(retrieved_docs, relevant_docs, k)
    print(f'{prec_at_k=}')
    assert prec_at_k == 0.5

    k = 3 # p@k=1/3
    prec_at_k: float = precision_at_k(retrieved_docs, relevant_docs, k)
    print(f'{prec_at_k=}')
    assert prec_at_k == 1/3

def test3():
    """
    Now let's test the real retriever model!
response    """
    query: str = "Cauchy's Mean Value theorem"
    response = search_with_token(query, token)
    top_retrieved: dict = response[0]
    print(f'{top_retrieved.keys=}')
    print(f'{top_retrieved=}')
    theorem, proof = get_theorem_and_proof_query_response(top_retrieved)
    print()
    print(f'{theorem=}')
    print()
    print(f'{proof=}')
    print()

def test4_average_precision_of_retrieval_over_test_set():
    # load the test data fron the json file 
    file_path: Path = Path("~/evals-for-autoformalization/src/nlp_eval/test_set.json").expanduser()
    with open(file_path, 'r') as file:
        test_data: list[dict] = json.load(file)
    
    # compute the precision @ k for the retrieval
    data_point: dict = {}
    for data_point in test_data:
        test_query: str = data_point['query']
        ground_truth_theorem_name = data_point['theorem_name']
        # 
        response = search_with_token(test_query, token)
        theorem, proof = get_theorem_and_proof_query_response(top_retrieved)
        prec_k: float = precision_at_k()
        

if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    test4()