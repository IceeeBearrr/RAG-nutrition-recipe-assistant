import os
# Read the document or text
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
# Seperate the document to paragraph, line, word, or character (Chunks)
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Embedding models to converts the chunks to vectors 
from langchain_ollama import OllamaEmbeddings, ChatOllama
# The database to store all the vectors
from langchain_chroma import Chroma
from collections import defaultdict


from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def load_documents(docs_path="docs"):
    """Loads all the pdf from the docs directory"""

    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your files.")

     # Load all .pdf files from the docs directory
    loader = DirectoryLoader(
        path = docs_path,
        glob = "*.pdf",
        loader_cls = PyPDFLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .pdf files found in {docs_path}. Please add your documents.")
    

    # Display total pages from the documents loaded
    print(f"Loaded {len(documents)} pages/documents.")

    # create an empty dictionary
    files_loaded = defaultdict(int)

    # doc will load all pages in each docments, the key is the source, 
    # while the value is the total loop of each document which equavalent to total page of each document
    # The key and value will be stored in files_loaded (exp: "docs\Data Transformations.pdf" : "25")
    for doc in documents:
        source = doc.metadata.get("source", "Unknown source")
        files_loaded[source] += 1

    print("\nFiles loaded:")

    # For each dictionary items store the key as the variable file_path, 
    # and the value stored as the variable store page_count
    for file_path, page_count in files_loaded.items():
        print(f"--- {file_path} ---")
        print(f"page count: {page_count}\n")

    return documents


def split_documents(documents, chunk_size=1200, chunk_overlap=250):
    """Split docments into smaller chunks with overlap"""

    # print(f"Total documents loaded from PDF (before cleanup): {len(documents)}")
    
    # # --- START: Remove exact duplicate pages loaded by the PDF parser ---
    # seen_content = set()
    # unique_documents = []
    
    # for doc in documents:
    #     # Strip whitespace, tab, or line breaks to ensure accurate matching, 
    #     # if accurate matched (from begining until the end of text) then remove it
    #     # for each document's page_content check whether it is uniqe
    #     cleaned_content = doc.page_content.strip()
    #     # used seen_content because it is a set which easier to check 
    #     # as they calculate the exact locker number  to find that word -> lesser memory consumption
    #     if cleaned_content not in seen_content:
    #         seen_content.add(cleaned_content)
    #         unique_documents.append(doc)
            
    # print(f"Unique documents after removing duplicates: {len(unique_documents)}")
    # # -------------------------- FINISH --------------------------


    # --- START: Split the unique documents with RecursiveCharacterTextSplitter ---
    # Create text splitter object, first tries by paraghraph, then lines, words, and characters
    # chunk_size = how many characters per chunk
    # chunk_overlap = how many characters to repeat in the next chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # perform split with the text splitter object to docments
    # Each chunk is a Document object with "page_content" and "metadata"
    # page_content -> the text of the chunk
    # metadata -> PDF info and page number
    chunks = text_splitter.split_documents(documents)
    # -------------------------- FINISH --------------------------


    # --- START: Preview the first five chunks ---
    # Preview first five chunks (enumerate consist i-> index, chunk -> actual chunk object)
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i + 1} ---")
            # Shows which PDF the chunk came from
            print(f"Source: {chunk.metadata['source']}")
            # Show the numbers of characters in this chunk 
            print(f"Length: {len(chunk.page_content)} characters")
            #print the text of the chunk
            print(f"Content: {chunk.page_content}")
            print("-" * 50)
        
        # If still have chunks, print summary of how many more chunks exist
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    # -------------------------- FINISH --------------------------

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """ Create and persist ChromaDB vector store """
    # --- START: Embed the chunks with Ollama ---
    # Creates an embedding model object using Ollama
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    # Create ChromaDB vector store
    print("Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"\nVector store created and saved to {persist_directory}\n")
    # -------------------------- FINISH --------------------------

    return vectorstore

def main():
    # Step 1: Load the Files
    print("\nStep 1: Load the Files") 
    documents = load_documents(docs_path="docs") # the folder to be loaded's file path (docs)
    print("-" * 80)

    # Step 2: Chunking the files
    print("\nStep 2: Chunking the files") 
    chunks = split_documents(documents)
    print("-" * 80)

    # Step 3: Embedding and Storing in Vector DB
    print("\nStep 3: Chunking the files") 
    vectorstore = create_vector_store(chunks)
    print("-" * 80)


if __name__ == "__main__":
    main()
