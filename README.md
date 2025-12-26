# RAG Backend

Minimal FastAPI backend for a Retrieval-Augmented Generation (RAG) system.

## Structure

```
app/
├── chat/          # Chat and conversation endpoints
├── ingest/        # Document ingestion endpoints
├── db/            # Database interfaces
└── main.py        # Main application entry point
```

## Endpoints

- `GET /health` - Health check
- `POST /chat/completion` - Chat completion (to be implemented)
- `POST /ingest/documents` - Document ingestion (to be implemented)

## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```