"""
This should be useful when giving full documents (e.g., Lean4 full files) to the RAGatouille model.
ref: https://chatgpt.com/c/861cc8d2-5bfb-4a44-a8ec-307592da5927
"""
import requests
from ragatouille import RAGPretrainedModel

# Function to fetch Wikipedia page content
def get_wikipedia_page(title: str) -> str | None:
    URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }
    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1 (example@domain.com)"}
    response = requests.get(URL, params=params, headers=headers)
    data = response.json()
    page = next(iter(data['query']['pages'].values()))
    return page['extract'] if 'extract' in page else None

# Fetch documents from Wikipedia
document_titles: list[str] = ["Hayao_Miyazaki", "Studio_Ghibli"]
documents: list[str | None] = [get_wikipedia_page(title) for title in document_titles]

# Initialize the pre-trained ColBERT model
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Index the documents
index_path = RAG.index(
    collection=documents,  # List of full document texts as list of strings
    document_ids=[f'doc_{i}' for i in range(len(documents))],  # Optional document IDs
    document_metadatas=[{"source": "wikipedia"} for _ in documents],  # Optional metadata for each document
    index_name="Wikipedia_Documents",  # Index name
    max_document_length=256,  # Maximum length of each chunk in tokens
    split_documents=True  # Enable document splitting
)

print(f"Index created at: {index_path}")

# Example query to search the indexed documents
query = "What is Studio Ghibli known for?"
results = RAG.search(query=query, k=3)

# Print the search results
for result in results:
    print(f"Content: {result['content']}, Score: {result['score']}, Rank: {result['rank']}, Document ID: {result['document_id']}")
