"""
This should be useful when giving an already chunked data set (e.g., from a HF dataset with already processed rows, chunked textbooks, formal to formal pairs of rows, etc.) to the RAGatouille model.
ref: https://chatgpt.com/c/861cc8d2-5bfb-4a44-a8ec-307592da5927
"""
import requests
from ragatouille import RAGPretrainedModel
from ragatouille.data import CorpusProcessor, llama_index_sentence_splitter

# Function to fetch Wikipedia page content
def get_wikipedia_page(title: str):
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
document_titles = ["Hayao_Miyazaki", "Studio_Ghibli"]
documents = [get_wikipedia_page(title) for title in document_titles]

# Initialize the CorpusProcessor to split documents into chunks
corpus_processor = CorpusProcessor(document_splitter_fn=llama_index_sentence_splitter)
chunked_documents = corpus_processor.process_corpus(documents, chunk_size=256)

# Extract only the text content from the chunked documents
chunked_texts: list[str] = [chunk['content'] for chunk in chunked_documents]

# Initialize the pre-trained ColBERT model
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Index the chunked documents
index_path = RAG.index(
    collection=chunked_texts,  # List of document chunks as strings
    document_ids=[f'chunk_{i}' for i in range(len(chunked_texts))],  # Optional document IDs
    document_metadatas=[{"source": "wikipedia"} for _ in chunked_texts],  # Optional metadata for each chunk
    index_name="Wikipedia_Chunks",  # Index name
    max_document_length=256,  # Maximum length of each chunk in tokens
    split_documents=False  # Disable additional splitting since chunks are pre-split
)

print(f"Index created at: {index_path}")

# Example query to search the indexed document chunks
query = "What is Studio Ghibli known for?"
results = RAG.search(query=query, k=3)

# Print the search results
for result in results:
    print(f"Content: {result['content']}, Score: {result['score']}, Rank: {result['rank']}, Document ID: {result['document_id']}")
