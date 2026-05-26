import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import hashlib

# Read the document or text
from langchain_community.document_loaders import PyMuPDFLoader
# Seperate the document to paragraph, line, word, or character (Chunks)
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Embedding models to converts the chunks to vectors 
from langchain_ollama import OllamaEmbeddings, ChatOllama
# The database to store all the vectors
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def load_documents(docs_path="docs"):
    """Loads all the pdf from the docs directory using a Layout-Aware strategy.
    Preserves text bounding blocks, column flows, and table structures. """

    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your files.")

    # Loop through all the docs in docs_path and store the pdf type to pdf_files
    pdf_files = [f for f in os.listdir(docs_path) if f.endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError(f"No .pdf files found in {docs_path}.")

    # store all the finalized page structures from every single PDF
    all_documents = []
    

    for file_name in pdf_files:
        # Join the docs and file name (exp: docs\healthy-nutrition-recipe.pdf)
        full_path = os.path.join(docs_path, file_name)
        print(f"Parsing [Layout-Aware PyMuPDF] -> {file_name}...")
        
        # Instantiate PyMuPDFLoader and point it directly at the specific PDF file
        loader = PyMuPDFLoader(file_path=full_path)
        # Runs the parsing computation, read the PDF page structure and 
        # turn them into a temporary list of LangChain Document objects 
        # (each object contains a page's page_content text string and its default metadata dictionary)
        docs = loader.load()
        
        # Loop through every single page extracted from current PDF file
        for doc in docs:
            # Injected a new metadata field to map the file_name (exp: healthy-nutrition-recipe.pdf)
            doc.metadata["source_file"] = file_name
            # Injecting structural flag: defaults to narrative text
            doc.metadata["category"] = "NarrativeText" 
                
        # Take all the newly processed page objects and store all of them into all_documents
        all_documents.extend(docs)

    print(f"\nSuccessfully parsed {len(pdf_files)} files into {len(all_documents)} pages.")
    return all_documents


def split_documents(documents):
    """Split docments into smaller chunks based on semantic meaning transitions.
    Uses vector space distance thresholds rather than rigid character sizes."""

    print(f"Initializing Semantic Chunker using local nomic-embed-text...")
    
    # Initialize the same embedding model used for storage to evaluate semantic distance
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    
    # Using the robust argument names compatible across LangChain versions:
    # 'breakpoint_threshold_type' sets the evaluation metric (percentile)
    # 'breakpoint_threshold_amount' sets the value cutoff (top 5% of meaning shifts)
    semantic_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile", 
        breakpoint_threshold_amount=0.95,  # 95th percentile threshold
        buffer_size=1                       # Compares sentence pairs directly
    )

    print("Executing Semantic Chunking across document layouts...")
    chunks = semantic_splitter.split_documents(documents)
    
    # PRODUCTION GUARDRAIL: Filter out any empty text artifacts or purely blank layout elements
    chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 0]

    # Print telemetry to monitor how the chunking logic behaved
    print(f"Successfully generated {len(chunks)} semantic chunks from your documents.")
    
    # Sample verification layout printout
    if chunks:
        print("\n--- Semantic Chunking Verification Samples ---")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1} (Length: {len(chunk.page_content)} chars | Source: {chunk.metadata.get('source_file')}):")
            print(f"Snippet: ")
            print(f"{chunk.page_content[:120].strip()}...")
            
        if len(chunks) > 3:
            print(f"\n... and {len(chunks) - 3} more semantic nodes initialized.")
            
    return chunks

def enrich_metadata(chunks):
    """
    Enriches semantic chunks by formalizing structural metadata fields 
    and prepending explicit context headers to optimize downstream vector retrieval.
    """
    print(f"\nStep 3: Executing Metadata Enrichment across {len(chunks)} nodes...")
    
    enriched_chunks = []
    
    for idx, chunk in enumerate(chunks):
        # 1. Extract existing metadata structural properties
        source_raw = chunk.metadata.get("source_file", "unknown_source")
        # Standardize readable title names from your files
        doc_title = source_raw.replace(".pdf", "").replace("-", " ").title()
        
        # PyMuPDF extracts page index starting from 0; let's format a 1-based index for humans
        page_num = chunk.metadata.get("page", 0) + 1 
        
        # 2. Construct a Search-Optimized Context Header
        # This explicitly anchors the text block so the embedding engine indexes the context
        context_header = (
            f"[Document Reference: {doc_title} | Source File: {source_raw} | Page: {page_num}]\n"
        )
        
        # 3. Prepend the structural context header to the physical text layout
        original_content = chunk.page_content
        chunk.page_content = context_header + original_content
        
        # 4. Formulate explicit fields inside the metadata dictionary for hard-filtering later
        chunk.metadata["document_title"] = doc_title
        chunk.metadata["page_number"] = page_num
        chunk.metadata["chunk_index"] = idx
        
        enriched_chunks.append(chunk)
        
    print(f"Successfully enriched metadata definitions for all {len(enriched_chunks)} nodes.")
    
    # Validation Sample
    if enriched_chunks:
        print("\n--- Metadata Enrichment Sample ---")
        sample = enriched_chunks[1] # Look at the first valid text block
        print(f"Enriched Metadata Dict: {sample.metadata}")
        print(f"Enriched Page Content Structure:\n{sample.page_content[:250]}...\n")
        
    return enriched_chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """ 
    Creates or updates ChromaDB vector store 
    Implements a deterministic ID hashing strategy to enforce idempotency (upserting).
    """
    # Step 4: Define the Production Embedding Model
    print("Initializing embedding vectorization space via nomic-embed-text...")
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    # Step 5: Enterprise Vector DB Connection with HNSW Cosine Distance Config
    print(f"Connecting to vector store at: {persist_directory}")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={
            "hnsw:space": "cosine",       # Sets distance math formula to Cosine Similarity
            "hnsw:construction_ef": 200,  # Optimizes index graph construction precision
            "hnsw:search_ef": 50          # Balances speed and accuracy during retrieval
        }
    )

    print("\nGenerating unique deterministic hashes for layout chunks...")
    
    # 1. Prepare parallelized arrays for batch upload
    chunk_ids = []
    documents_to_upsert = []
    
    for chunk in chunks:
        # Construct a string using content + document title + index to guarantee a unique fingerprint
        fingerprint_source = f"{chunk.page_content}_{chunk.metadata.get('document_title')}_{chunk.metadata.get('chunk_index')}"
        
        # Convert the fingerprint into a secure, predictable SHA-256 string ID
        deterministic_hash = hashlib.sha256(fingerprint_source.encode('utf-8')).hexdigest()
        
        chunk_ids.append(deterministic_hash)
        documents_to_upsert.append(chunk)

    # 2. Execute Idempotent Batch Processing
    print(f"Upserting {len(documents_to_upsert)} chunks into Chroma Vector DB...")
    
    # .add_documents automatically tracks the 'ids' array to overwrite or skip existing duplicates
    vectorstore.add_documents(
        documents=documents_to_upsert,
        ids=chunk_ids
    )
    
    print("Database sync complete. Index verified.")
    return vectorstore

def main():
    # Step 1: Load the Files
    print("\nStep 1: Load the Files") 
    documents = load_documents(docs_path="docs") # the folder to be loaded's file path (docs)
    print("-" * 80)

    # # Step 2: Chunking the files
    print("\nStep 2: Chunking the files") 
    raw_chunks = split_documents(documents)
    print("-" * 80)

    # Step 3: Metadata Enrichment (Context Injection)
    chunks = enrich_metadata(raw_chunks)
    print("-" * 80)

    # # Step 3: Embedding and Storing in Vector DB
    print("\nStep 3: Chunking the files") 
    vectorstore = create_vector_store(chunks)
    print("-" * 80)


if __name__ == "__main__":
    main()
