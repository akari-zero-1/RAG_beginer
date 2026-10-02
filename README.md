# RAG Beginner

>This project is a collection of demos and experiments covering multiple **Retrieval-Augmented Generation (RAG)** techniques with LangChain. It lets users ask questions about documents in the `docs/` directory, >retrieve relevant text chunks through vector search, and use an LLM to generate answers based on the retrieved context.

>The project has two main parts:
>
>- **Main pipeline**: ingests `.txt` documents, creates embeddings, stores them in Chroma, and provides a Streamlit chat interface.
>- **Additional experiments**: demonstrate chunking, retrieval, multi-query retrieval, RRF, hybrid search, reranking, and multimodal RAG through Python scripts and Jupyter notebooks.

## Main Features

- Question answering over documents about Google, Microsoft, Nvidia, SpaceX, and Tesla.
- Multiple retrieval methods selectable from the sidebar:
  - Similarity search.
  - Similarity search with a score threshold.
  - MMR (Maximal Marginal Relevance).
  - Multi-query retrieval combined with Reciprocal Rank Fusion.
- Rewrites questions using conversation history to handle context-dependent follow-up questions.
- Configurable document count `k`, score threshold, MMR lambda, model, and temperature.
- Displays retrieved documents so the input context can be inspected.

## Architecture Overview

```text
Documents in docs/
        |
        v
DirectoryLoader + TextLoader
        |
        v
CharacterTextSplitter
        |
        v
HuggingFaceEmbeddings (all-MiniLM-L6-v2)
        |
        v
Chroma vector store (db/chroma_db)
        |
        v
Retriever / Multi-query / MMR / RRF
        |
        v
ChatGroq (Llama or Mixtral)
        |
        v
Answer in Streamlit
```

## Technologies Used

| Technology                         | Role                                                                                               |
| ---------------------------------- | -------------------------------------------------------------------------------------------------- |
| Python                             | Main programming language.                                                                         |
| LangChain                          | Connects document loaders, text splitters, embeddings, retrievers, and LLMs.                       |
| Chroma                             | Local persistent vector database stored in `db/chroma_db`.                                         |
| Hugging Face Sentence Transformers | Creates embeddings with `all-MiniLM-L6-v2`.                                                        |
| Groq                               | Provides LLM access through `ChatGroq`; the project uses Llama 3.1, Llama 3.3, and Mixtral models. |
| Streamlit                          | Provides the chat interface and RAG configuration controls.                                        |
| Pydantic                           | Enforces the multi-query output schema `queries: List[str]`.                                       |
| python-dotenv                      | Loads environment variables from `.env`.                                                           |
| Cohere Rerank                      | Re-ranks results in `reranker.ipynb`.                                                              |
| BM25                               | Provides keyword/sparse retrieval in `hybrid_search.ipynb`.                                        |
| Unstructured and Tesseract         | Process PDFs, tables, images, and OCR in the multimodal notebook.                                  |

## Techniques Implemented

### 1. Document Ingestion

`ingestion_pipeline.py` uses `DirectoryLoader` and `TextLoader` to read all `.txt` files in `docs/`. The loaded documents retain basic metadata, which can be used to trace retrieved content back to its source.

### 2. Character-Based Chunking

`CharacterTextSplitter` divides documents into chunks with a default size of `1000` characters and `chunk_overlap=0`. This is the chunking strategy used by the main ingestion pipeline to create `db/chroma_db`.

### 3. Recursive Character Chunking

`recursive_character_text_spliter.py` experiments with `RecursiveCharacterTextSplitter`. It prioritizes natural boundaries such as blank lines, lines, sentence endings, spaces, and finally individual characters. This generally preserves document structure better than splitting at a single fixed boundary.

### 4. Semantic Chunking

`semantic_chunking.py` uses `SemanticChunker` and embeddings to find split points based on changes in meaning instead of character count alone. The experiment uses a percentile breakpoint threshold of `70`.

### 5. Agentic Chunking

`agentic_chunking.py` asks an LLM to split text by topic. The LLM is instructed to keep related information in the same chunk, target approximately 200 characters per chunk, and insert the `<<<SPLIT>>>` marker so the program can separate the result.

### 6. Embeddings and Cosine Similarity

Documents and user questions are converted into vectors with `all-MiniLM-L6-v2`. Chroma is configured with `hnsw:space = cosine` and retrieves chunks whose vectors are closest to the query vector.

### 7. Persistent Vector Store

Chroma is persisted in `db/chroma_db`, so the database can be loaded again without recomputing embeddings every time. `dbv1/` and `dbv2/` contain other experimental database versions.

### 8. Similarity Search and Score Threshold

Similarity search returns the top `k` closest chunks. Threshold search keeps only results whose similarity score reaches the configured threshold, reducing irrelevant context. The Streamlit interface lets users change `k` and the threshold.

### 9. MMR Retrieval

MMR balances relevance and diversity. A `lambda_mult` value close to `1` prioritizes relevance, while a value close to `0` prioritizes diversity. The retriever first builds a candidate pool with `fetch_k` and then selects the final `k` chunks.

### 10. Multi-Query Retrieval

The LLM generates three alternative phrasings for the same question. Each query is searched independently, increasing the chance of finding documents when the user's wording differs from the wording in the source documents. The output is constrained with Pydantic structured output.

### 11. Reciprocal Rank Fusion (RRF)

RRF combines multiple ranked result lists based on each chunk's position:

```text
score(chunk) = sum(1 / (k + rank))
```

A chunk that appears in multiple query results and ranks highly receives a stronger final score. The examples use the RRF constant `k=60`.

### 12. History-Aware Query Rewriting

When a user asks a question that depends on previous conversation turns, the LLM rewrites it into a standalone, searchable question. The app passes up to the four most recent messages to the rewriting step.

### 13. Grounded Answer Generation

The LLM receives the user's question and the retrieved chunks in its prompt. The prompt instructs it to use only the provided documents and to state when there is not enough information. This helps reduce hallucination during generation, but it does not replace evaluation of retrieval quality.

### 14. Hybrid Search

`hybrid_search.ipynb` combines:

- **Dense retrieval**: semantic search using embeddings.
- **Sparse retrieval**: keyword search using `BM25Retriever`.

Hybrid search is useful for questions containing proper names, product codes, numbers, or exact keywords that dense retrieval may overlook.

### 15. Reranking

`reranker.ipynb` uses `CohereRerank` to reorder an initial candidate set. The retriever gathers a relatively broad set of candidates, and the reranker evaluates the relationship between the query and each document in more detail.

### 16. Multimodal RAG

`multi_modal_rag .ipynb` experiments with PDF processing through Unstructured, extracting text, tables, and images before generating summaries that include visual content. Tesseract is used for OCR when a PDF contains scanned images. The notebook falls back to fast mode when Tesseract is not available in `PATH`.

## Project Structure

```text
.
|-- app.py                         # Streamlit interface and interactive RAG app
|-- ingestion_pipeline.py          # Load documents, chunk, embed, and create Chroma
|-- answer_generation.py           # Basic retrieval and generation example
|-- history_aware_generation.py    # Chat with history-aware query rewriting
|-- retrieval_methods.py           # Similarity retrieval examples
|-- retrieval_pipeline.py          # Simple retrieval pipeline
|-- multi_query_retrieval.py       # Generate multiple queries with an LLM
|-- reciprocal_rank_fusion.py      # Multi-query retrieval + RRF
|-- recursive_character_text_spliter.py
|-- semantic_chunking.py
|-- agentic_chunking.py
|-- hybrid_search.ipynb            # Dense + BM25 retrieval
|-- reranker.ipynb                 # Cohere reranking
|-- multi_modal_rag .ipynb         # PDF, table, image, and OCR processing
|-- docs/                          # Input .txt documents
|-- db/chroma_db/                  # Persistent vector database
|-- dbv1/, dbv2/                   # Experimental database versions
|-- requirements.txt
```

## Installation

Python 3.10 or later and a virtual environment are recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install streamlit
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
```

`GROQ_API_KEY` is required by `app.py` and the scripts that use Groq. `COHERE_API_KEY` is only required when running the reranking notebook.

## Run the Ingestion Pipeline

If `db/chroma_db` does not exist, run:

```powershell
python ingestion_pipeline.py
```

The script reads documents from `docs/`, splits them into chunks, creates embeddings, and saves the vector store. If the database already exists, the script loads the existing database instead.

## Run the Chat Application

```powershell
streamlit run app.py
```

After Streamlit starts, open the URL shown in the terminal, usually `http://localhost:8501`.

## Run the Examples

```powershell
python answer_generation.py
python retrieval_methods.py
python multi_query_retrieval.py
python reciprocal_rank_fusion.py
python history_aware_generation.py
```

Open the `.ipynb` files in Jupyter or VS Code to explore hybrid search, reranking, and multimodal RAG.

## Current Limitations

- `requirements.txt` does not currently declare `streamlit`, so it must be installed separately for the web interface.
- The application depends on the Groq service and requires a valid `GROQ_API_KEY`.
- Chroma is local storage. When changing source documents or chunking strategies, use a new collection or database directory to avoid mixing old and new data.
- Most scripts and notebooks are standalone demonstrations; they are not all called by `app.py`.
- Multimodal PDF processing may require system dependencies such as Tesseract for full OCR support.
- The chatbot can answer reliably only within the ingested documents and the quality of the retrieved chunks.

## Future Improvements

- Add metadata filters by company, file, and topic.
- Add citations or source links to generated answers.
- Add an evaluation set for Recall@k, MRR, faithfulness, and answer relevancy.
- Move configuration out of `app.py` and share retrieval parameters across the demo scripts.
- Add automated tests for ingestion, retrieval, RRF, and query rewriting.
