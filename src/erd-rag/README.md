# ERD-RAG

Repository: an end-to-end prototype to generate embeddings from Markdown ERD schemas, index them, perform hybrid retrieval, and run RAG prompts with an LLM.

## Features
- Parse ERD from Markdown (table/columns/relations)
- Enrich identifiers (canonicalization)
- Create multi-granular docs (table/column/relation)
- Embed using local SentenceTransformers or OpenAI embeddings
- Vector index using Chroma or FAISS (configurable)
- BM25 sparse index + graph-expansion retriever
- RAG prompt builder
- CLI to build and query
- Dockerfile and GitHub Actions CI

## Quickstart (local)
1. Copy `.env.example` → `.env` and set variables.
2. Create venv & install locally:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
