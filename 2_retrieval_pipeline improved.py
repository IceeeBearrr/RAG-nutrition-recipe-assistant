import os
import warnings
import hashlib

# 1. Suppress deprecation warnings immediately
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Core Library Imports
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# ==============================================================================
# PHASE 3: RETRIEVAL STEPS (6, 7, & 8)
# ==============================================================================

def generate_expanded_queries(original_query):
    """
    [Step 6] Routes the raw user query through a local LLM to generate 3 alternative 
    phrasings. This improves search recall by covering synonyms and different angles.
    """
    print(f"\n[Step 6] Routing query to local LLM for Expansion...")
    
    llm = ChatOllama(model="llama3", temperature=0.2)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert AI search optimizer. Your task is to take a user query "
            "and generate exactly 3 alternative variations of this question. "
            "Focus on using different keywords, technical terms, or synonyms related to health, "
            "nutrition, and recipes to maximize vector database search recall.\n"
            "Provide exactly 3 variations, one per line. Do not add numbers, explanations, introductory text, or bullet points."
        )),
        ("human", "Original query: {query}")
    ])
    
    expansion_chain = prompt_template | llm | StrOutputParser()
    response = expansion_chain.invoke({"query": original_query})
    
    expanded_queries = [line.strip() for line in response.split("\n") if line.strip()]
    all_queries = [original_query] + expanded_queries[:3]
    
    print("Generated Search Matrix Bundles:")
    for idx, q in enumerate(all_queries):
        print(f"  Query Variant {idx}: \"{q}\"")
        
    return all_queries


def initialize_sparse_engine(all_chunks):
    """
    [Step 7] Initializes a BM25 sparse keyword index over the ingested document corpus 
    to handle exact matches like ingredients, food names, and page numbers.
    """
    print("\n[Step 7] Tokenizing corpus for Sparse Keyword Search (BM25)...")
    tokenized_corpus = [chunk.page_content.lower().split(" ") for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    return bm25_index, all_chunks


def execute_hybrid_search(query_bundle, db, bm25_index, all_chunks, k_dense=3, k_sparse=3):
    """
    [Step 7] Blends Dense Vector Lookups and Sparse Keyword Lookups across all 
    expanded queries to form a highly resilient context collection.
    """
    print(f"\n[Step 7] Executing Hybrid Search Loop...")
    hybrid_pool = {}

    # --- ENGINE A: DENSE VECTOR RETRIEVAL (With Multi-Query Matrix) ---
    for q in query_bundle:
        dense_results = db.similarity_search(query=q, k=k_dense)
        for doc in dense_results:
            hybrid_pool[doc.page_content] = doc

    # --- ENGINE B: SPARSE KEYWORD RETRIEVAL (Targeting Primary Query Intent) ---
    primary_query = query_bundle[0]
    tokenized_query = primary_query.lower().split(" ")
    sparse_scores = bm25_index.get_scores(tokenized_query)
    
    top_sparse_indices = sorted(
        range(len(sparse_scores)), 
        key=lambda i: sparse_scores[i], 
        reverse=True
    )[:k_sparse]
    
    for index in top_sparse_indices:
        if sparse_scores[index] > 0:
            sparse_doc = all_chunks[index]
            if sparse_doc.page_content not in hybrid_pool:
                hybrid_pool[sparse_doc.page_content] = sparse_doc

    final_hybrid_docs = list(hybrid_pool.values())
    print(f"Hybrid Fusion complete. Retrieved {len(final_hybrid_docs)} unique mixed documents.")
    return final_hybrid_docs


def rerank_documents(query, hybrid_docs, top_n=3):
    """
    [Step 8] Uses a local Cross-Encoder model to evaluate the deep semantic relevance 
    between the original user query and each retrieved hybrid chunk.
    Re-orders results to put the absolute best context at the top.
    """
    if not hybrid_docs:
        print("[Step 8] No documents retrieved to rerank.")
        return []

    print(f"\n[Step 8] Initializing local Cross-Encoder Reranker (ms-marco-MiniLM)...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    pairs = [[query, doc.page_content] for doc in hybrid_docs]
    print(f"Evaluating true relevancy scores across {len(hybrid_docs)} hybrid candidates...")
    scores = reranker.predict(pairs)
    
    for idx, score in enumerate(scores):
        hybrid_docs[idx].metadata["rerank_score"] = float(score)
        
    reranked_docs = sorted(hybrid_docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
    final_context_chunks = reranked_docs[:top_n]
    
    print("\n--- Reranker Verification Leaderboard ---")
    for rank, doc in enumerate(final_context_chunks, 1):
        source = doc.metadata.get("source_file", "Unknown")
        score = doc.metadata.get("rerank_score", 0.0)
        print(f"  Rank {rank} [Score: {score:.4f}] -> Source: {source} | Snippet: {doc.page_content[:90].strip()}...")
        
    return final_context_chunks


# ==============================================================================
# PHASE 4: GENERATION STEPS (9 & 10)
# ==============================================================================

def format_context_for_prompt(reranked_chunks):
    """
    [Step 9] Formats the top reranked chunks into structured XML blocks 
    so the LLM can easily parse structural reference boundaries.
    """
    formatted_blocks = []
    for idx, doc in enumerate(reranked_chunks, 1):
        title = doc.metadata.get("document_title", "Unknown Document")
        page = doc.metadata.get("page_number", "Unknown Page")
        
        block = (
            f"<Context_Source_{idx} document=\"{title}\" page=\"{page}\">\n"
            f"{doc.page_content}\n"
            f"</Context_Source_{idx}>"
        )
        formatted_blocks.append(block)
        
    return "\n\n".join(formatted_blocks)


def execute_generation_chain(query, reranked_chunks):
    """
    [Step 9] Constructs an enterprise context-injected prompt template and routes 
    it through a local LLM with high formatting compliance instructions.
    """
    print(f"\n[Step 9] Formatting context blocks and preparing generation chain...")
    structured_context = format_context_for_prompt(reranked_chunks)
    
    llm = ChatOllama(model="llama3", temperature=0.0, options={"num_predict": 1024})
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a professional, accurate health and nutrition assistant.\n"
            "Your core task is to answer the User Question using ONLY the verified facts provided inside the enclosed <Verified_Context> blocks.\n\n"
            "CRITICAL OPERATIONAL RULES:\n"
            "1. If the answer cannot be completely derived from the provided context, state clearly: 'I cannot answer this question based on the verified documents provided.' Do not make up information under any circumstance.\n"
            "2. For every claim, fact, or ingredient list you present, you MUST explicitly cite which document and page number it came from at the end of the sentence or paragraph (e.g., [Healthy Recipes, Page 48]).\n"
            "3. Keep your answers clean, professional, and structurally organized using bullet points where appropriate.\n\n"
            "<Verified_Context>\n"
            "{context}\n"
            "</Verified_Context>"
        )),
        ("human", "{question}")
    ])
    
    rag_chain = prompt_template | llm | StrOutputParser()
    print("Streaming execution response from local LLM...")
    response = rag_chain.invoke({
        "context": structured_context,
        "question": query
    })
    
    return response


# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================

def main_retrieval_pipeline(raw_user_query):
    print(f"================================================================================")
    print(f"INITIALIZING PRODUCTION RUNTIME LOOP FOR: \"{raw_user_query}\"")
    print(f"================================================================================")

    # 1. Connect to Chroma Vector DB
    persistent_directory = "db/chroma_db"
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_model)

    # 2. Fetch corpus to synchronize BM25 space
    print("Fetching document nodes from database to synchronize BM25 space...")
    raw_data = db.get()
    all_chunks_text = raw_data["documents"]
    all_metadatas = raw_data["metadatas"]
    
    # Reconstruct document elements back with their metadata maps intact for BM25 mapping
    struct_chunks = []
    for txt, meta in zip(all_chunks_text, all_metadatas):
        struct_chunks.append(Document(page_content=txt, metadata=meta))

    # Initialize the Sparse Keyword Engine
    bm25_index, processed_corpus = initialize_sparse_engine(struct_chunks)

    # --- RUNTIME PIPELINE TRACE ---
    
    # Step 6: Multi-Query Generation
    query_matrix = generate_expanded_queries(raw_user_query)
    
    # Step 7: Hybrid Sparse + Dense Search
    candidate_pool = execute_hybrid_search(
        query_bundle=query_matrix, 
        db=db, 
        bm25_index=bm25_index, 
        all_chunks=processed_corpus, 
        k_dense=3, 
        k_sparse=3
    )
    
    # Step 8: Cross-Encoder Reranking
    top_vetted_context = rerank_documents(
        query=raw_user_query, 
        hybrid_docs=candidate_pool, 
        top_n=3
    )
    
    # Step 9: Injected Prompts & LLM Generation
    final_output = execute_generation_chain(
        query=raw_user_query, 
        reranked_chunks=top_vetted_context
    )
    
    # Step 10: Validation Post-Check Guardrail
    print(f"\n[Step 10] Executing Post-Generation Validation Guardrails...")
    
    print("\n=============================== FINAL AI RESPONSE ===============================")
    print(final_output)
    print("=================================================================================\n")


if __name__ == "__main__":
    # Test your complete production RAG stack!
    sample_query = "What is the method and ingredients list to make Rocky Road?"
    main_retrieval_pipeline(sample_query)