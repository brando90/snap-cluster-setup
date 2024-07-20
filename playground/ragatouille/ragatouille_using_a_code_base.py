"""
Demo where the list of documents is a code base e.g., Lean4 Mathlib repository.
ref: https://chatgpt.com/c/861cc8d2-5bfb-4a44-a8ec-307592da5927
"""
import os
from pathlib import Path
from ragatouille import RAGPretrainedModel
from ragatouille.data import CorpusProcessor, llama_index_sentence_splitter

# Step 1: Clone the Lean4 Mathlib repository to the home directory
home_dir = Path.home()
mathlib_path = home_dir / "mathlib4"
if not mathlib_path.exists():
    print("Cloning Lean4 Mathlib repository...")
    os.system(f"git clone https://github.com/leanprover-community/mathlib4.git {mathlib_path}")
    print("Cloning completed.")
else:
    print("Lean4 Mathlib repository already exists.")

# Step 2: Traverse the repository to collect .lean files
print("Collecting .lean files from the repository...")
lean_files = []
for root, _, files in os.walk(mathlib_path):
    for file in files:
        if file.endswith(".lean"):
            lean_files.append(os.path.join(root, file))
print(f"Collected {len(lean_files)} .lean files.")

# Step 3: Read the .lean files and prepare documents
print("Reading .lean files...")
documents = []
for lean_file in lean_files:
    with open(lean_file, 'r') as file:
        documents.append(file.read())
print("Reading completed.")

# Step 4: Chunk the documents using CorpusProcessor
print("Chunking documents...")
corpus_processor = CorpusProcessor(document_splitter_fn=llama_index_sentence_splitter)
chunked_documents = corpus_processor.process_corpus(documents, chunk_size=256)
chunked_texts = [chunk['content'] for chunk in chunked_documents]
print(f"Chunked documents into {len(chunked_texts)} chunks.")

# Step 5: Initialize the pre-trained ColBERT model
print("Initializing the pre-trained ColBERT model...")
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
print("Model initialized.")

# Step 6: Index the chunked documents
print("Indexing chunked documents...")
index_path = RAG.index(
    collection=chunked_texts,  # List of document chunks as strings
    document_ids=[f'chunk_{i}' for i in range(len(chunked_texts))],  # Optional document IDs
    document_metadatas=[{"source": "mathlib"} for _ in chunked_texts],  # Optional metadata for each chunk
    index_name="Lean4_Mathlib_Chunks",  # Index name
    max_document_length=256,  # Maximum length of each chunk in tokens
    split_documents=False  # Disable additional splitting since chunks are pre-split
)
print("Indexing completed.")

# Step 7: Save the index
print("Saving the index...")
index_save_path = home_dir / ".ragatouille" / "colbert" / "indexes" / "Lean4_Mathlib_Chunks"
index_save_path.mkdir(parents=True, exist_ok=True)
RAG.save_index(index_path, index_save_path)
print(f"Index saved at: {index_save_path}")

# Step 8: Example query to search the indexed document chunks
query = "What is the definition of a group in mathlib?"
print(f"Searching the index for query: '{query}'")
results = RAG.search(query=query, k=3)

# Step 9: Print the search results
print("Search results:")
for result in results:
    print(f"Content: {result['content']}, Score: {result['score']}, Rank: {result['rank']}, Document ID: {result['document_id']}")
