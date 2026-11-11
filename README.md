# FastAPI Document Chat with MCP Retrieval

A chat API that answers questions from ingested documents. Built with **FastAPI**, **MongoDB** (vector storage), **Model Context Protocol (MCP)** for retrieval, and **Groq** (LLaMA 3.1 8B) for answer generation.

## Architecture

```
┌─────────┐   POST /documents   ┌──────────┐    embed + store    ┌─────────┐
│  Client  │ ──────────────────► │ FastAPI  │ ──────────────────► │ MongoDB │
└─────────┘                      │  (api)   │                     └─────────┘
     │                           └──────────┘                          ▲
     │  POST /chat                    │                                │
     └───────────────────────────►    │  MCP client call               │
                                      ▼                                │
                                ┌────────────┐   vector search    ─────┘
                                │ MCP Server │
                                │ (SSE)      │
                                └────────────┘
                                      │
                                      ▼
                                ┌──────────┐
                                │  Groq    │  (LLaMA 3.1 8B Instant)
                                └──────────┘
```

## Models Used

| Purpose    | Model                              | Provider                |
|------------|------------------------------------|-------------------------|
| Embeddings | `all-MiniLM-L6-v2`                | sentence-transformers (local, free) |
| Chat       | `llama-3.1-8b-instant`             | Groq (free tier)        |

## Setup

### 1. Clone and configure

```bash
cp .env.example .env
# Edit .env and add your Groq API key:
#   GROQ_API_KEY=gsk_your_key_here
```

Get a free Groq API key at: https://console.groq.com/keys

### 2. Start with Docker Compose

```bash
docker compose up --build
```

This starts three services:
- **mongo** – MongoDB 7 on port 27017
- **mcp-server** – MCP retrieval server (SSE) on port 8001
- **api** – FastAPI application on port 8000

### 3. Ingest a sample document

```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@sample_docs/python_overview.txt"
```

Expected response:
```json
{"filename": "python_overview.txt", "chunks_stored": 4}
```

### 4. Ask a grounded question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Who created Python and when?"}'
```

Expected response (grounded answer):
```json
{
  "answer": "According to the document (python_overview.txt), Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands. Its implementation began in December 1989.",
  "sources": ["python_overview.txt"],
  "mcp_tool_called": "search_documents",
  "retrieved_chunks": 5
}
```

### 5. Ask a question the document cannot answer

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

Expected response (no support in documents):
```json
{
  "answer": "The provided documents do not contain enough information to answer this question.",
  "sources": ["python_overview.txt"],
  "mcp_tool_called": "search_documents",
  "retrieved_chunks": 5
}
```

## MCP Tool Call Log

When the `/chat` endpoint is called, the API log shows the MCP tool invocation:

```
2026-11-11 18:00:01 [APP] Connecting to MCP server at http://mcp-server:8001/sse
2026-11-11 18:00:01 [APP] MCP session initialized. Calling search_documents …
2026-11-11 18:00:01 [MCP] search_documents called — question='Who created Python and when?'  top_k=5
2026-11-11 18:00:01 [MCP] Returning 5 chunks
2026-11-11 18:00:01 [APP] MCP tool returned. Processing results …
2026-11-11 18:00:01 [APP] Retrieved 5 chunks via MCP. Calling Groq …
2026-11-11 18:00:02 [APP] Calling Groq model llama-3.1-8b-instant
```

## API Endpoints

| Method | Path         | Description                              |
|--------|--------------|------------------------------------------|
| POST   | `/documents` | Upload a `.txt` file to ingest           |
| POST   | `/chat`      | Ask a question (JSON body: `{"question": "..."}`) |
| GET    | `/health`    | Health check                             |

## Notes

- **MongoDB**: Runs inside Docker Compose (Community Edition). Vector search uses a brute-force cosine similarity fallback since Atlas vector search indexes are not available in Community Edition. For production, use MongoDB Atlas.
- **Embedding model**: Downloaded at Docker build time (~80 MB). Runs inside the container with no GPU required.
- **Groq**: Requires a free API key. The free tier allows 30 requests/minute for `llama-3.1-8b-instant`.
