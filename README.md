# RAG Pipeline with Mistral AI & Supabase

A Retrieval-Augmented Generation (RAG) pipeline that answers questions 
based on your own documents, using Mistral AI for embeddings and generation, 
and Supabase (pgvector) as the vector store.

## Architecture
1. **Ingestion** : Load PDF → Split into chunks → Embed with mistral-embed → Store in Supabase
2. **Query** : Embed question → Retrieve top 5 chunks → Generate answer with mistral-small

## Tech Stack
- Mistral AI (mistral-embed + mistral-small-latest)
- Supabase + pgvector
- LangChain Text Splitters
- pypdf

## Limitations
- Tables and images in PDFs are not extracted (text only)
- For better PDF parsing, consider pdfplumber or LlamaParse

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys (see `.env.example`)
4. Run ingestion: `python ingest.py`
5. Run query: `python query.py`
```
