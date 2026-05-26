from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Search for relevant documents
query = "I want the nutrition of Papaya"

# retrieve the top 3 highest similarity scores chunks to the user query
retriever = db.as_retriever(search_kwargs={"k": 3})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3 # Only return chunks with cosine similarity >= 0.3 (from the range 0 - 1)
#     }
# )

relevant_docs = retriever.invoke(query)

print("--- User Query --- ")
print(query)

#Display results
print("\n--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    page = doc.metadata.get("page", 0) + 1  # get page number from metadata
    source = doc.metadata.get("source", "Unknown source")
    print(f"Similarity {i}: (Source: {source}, Page: {page})")
    print(doc.page_content)
    print("-" * 80)


