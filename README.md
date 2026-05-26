# RAG Nutrition Recipe Assistant

A local Retrieval-Augmented Generation (RAG) assistant for answering nutrition and recipe-related questions from PDF documents. The system loads PDF files, splits them into searchable chunks, stores them in ChromaDB, retrieves relevant context, and generates grounded answers using a local Ollama model.

## Features

- PDF document ingestion
- Semantic document chunking
- Metadata enrichment with source file, title, page number, and chunk index
- Local embedding generation using Ollama
- Persistent ChromaDB vector storage
- Hybrid retrieval using Chroma vector search and BM25 keyword search
- Query expansion using a local LLM
- Cross-Encoder reranking
- Context-grounded answer generation
- Citation-style source tracking using document title and page number

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| LangChain | RAG pipeline framework |
| ChromaDB | Vector database |
| Ollama | Local LLM and embedding runtime |
| PyMuPDF | PDF loading |
| BM25 | Keyword-based retrieval |
| Sentence Transformers | Cross-Encoder reranking |
| SHA-256 Hashing | Stable chunk ID generation |

## Repository Structure

```text
RAG-nutrition-recipe-assistant/
├── docs/
├── db/
│   └── chroma_db/
├── 1_ingestion_pipeline.py
├── 1_ingestion_pipeline improved.py
├── 2_retrieval_pipeline.py
├── 2_retrieval_pipeline improved.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Main Files

### `1_ingestion_pipeline improved.py`

Processes PDF documents and stores them in the vector database.

Main responsibilities:

- Loads PDF files from the `docs` folder
- Splits documents into semantic chunks
- Adds metadata such as source file, document title, page number, and chunk index
- Generates embeddings using `nomic-embed-text`
- Stores the embedded chunks in ChromaDB
- Uses SHA-256 hashing to generate stable chunk IDs

### `2_retrieval_pipeline improved.py`

Retrieves relevant document chunks and generates answers.

Main responsibilities:

- Connects to the existing ChromaDB vector database
- Builds a BM25 keyword index from stored documents
- Expands the user query into related search variations
- Retrieves candidate chunks using vector search and BM25 search
- Reranks retrieved chunks using a Cross-Encoder model
- Generates a final answer using only the retrieved context

## RAG Workflow

```text
PDF Documents
      ↓
Load and Split Documents
      ↓
Generate Embeddings
      ↓
Store in ChromaDB
      ↓
User Question
      ↓
Query Expansion
      ↓
Hybrid Retrieval
      ↓
Cross-Encoder Reranking
      ↓
Local LLM Answer Generation
```

## Installation and Setup

### Prerequisites

Make sure you have installed:

- Python
- Git
- Ollama

You also need the required Ollama models:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### Setup Steps

1. Clone the repository:

```bash
git clone https://github.com/IceeeBearrr/RAG-nutrition-recipe-assistant.git
```

2. Go into the project folder:

```bash
cd RAG-nutrition-recipe-assistant
```

3. Create and activate a virtual environment:

```bash
python -m venv venv
```

For Windows:

```bash
venv\Scripts\activate
```

For macOS or Linux:

```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Add your PDF files into the `docs` folder.

```text
docs/
└── your-document.pdf
```

2. Run the ingestion pipeline:

```bash
python "1_ingestion_pipeline improved.py"
```

This creates or updates the local ChromaDB database at:

```text
db/chroma_db
```

3. Run the retrieval pipeline:

```bash
python "2_retrieval_pipeline improved.py"
```

4. Edit the sample query inside the retrieval script:

```python
sample_query = "What is the method and ingredients list to make Rocky Road?"
```

Example replacement:

```python
sample_query = "What are healthy breakfast recipes for weight control?"
```

## Models Used

| Model | Purpose |
|---|---|
| `nomic-embed-text` | Embedding generation |
| `llama3` | Query expansion and answer generation |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking retrieved chunks |
