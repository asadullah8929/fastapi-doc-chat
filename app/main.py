"""FastAPI application — POST /documents and POST /chat."""

import json
import logging
import httpx
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

from app.config import MCP_SERVER_HOST, MCP_SERVER_PORT
from app.embeddings import embed_texts, get_model
from app.text_utils import split_text
from app.db import insert_chunks, ensure_vector_index, get_collection
from app.groq_chat import ask_groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [APP] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up embedding model and ensure DB index on startup."""
    logger.info("Warming up embedding model …")
    get_model()
    ensure_vector_index()
    logger.info("Ready.")
    yield


app = FastAPI(title="Document Chat API", version="1.0.0", lifespan=lifespan)


# ── POST /documents ──────────────────────────────────────────────────────────

@app.post("/documents")
async def ingest_document(file: UploadFile = File(...)):
    """Accept a plain-text file, chunk it, embed it, and store in MongoDB."""
    if not file.filename:
        raise HTTPException(400, "No filename provided.")

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "File must be UTF-8 plain text.")

    chunks = split_text(text)
    if not chunks:
        raise HTTPException(400, "File is empty after splitting.")

    logger.info("Ingesting '%s': %d chunks", file.filename, len(chunks))

    embeddings = embed_texts(chunks)

    records = [
        {"text": chunk, "embedding": emb, "source": file.filename}
        for chunk, emb in zip(chunks, embeddings)
    ]
    count = insert_chunks(records)
    logger.info("Stored %d chunks for '%s'", count, file.filename)

    return {"filename": file.filename, "chunks_stored": count}


# ── POST /chat ───────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    mcp_tool_called: str = "search_documents"
    retrieved_chunks: int = 0


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Answer a question by calling the MCP search_documents tool, then Groq."""
    mcp_url = f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}/sse"
    logger.info("Connecting to MCP server at %s", mcp_url)

    try:
        async with sse_client(mcp_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.info("MCP session initialized. Calling search_documents …")

                result = await session.call_tool(
                    "search_documents",
                    arguments={"question": req.question, "top_k": req.top_k},
                )
                logger.info("MCP tool returned. Processing results …")
    except Exception as exc:
        logger.error("MCP call failed: %s", exc)
        raise HTTPException(502, f"MCP retrieval failed: {exc}")

    # Parse chunks from tool result
    chunks: list[dict] = []
    for content_block in result.content:
        if hasattr(content_block, "text"):
            try:
                chunks = json.loads(content_block.text)
            except json.JSONDecodeError:
                pass

    logger.info("Retrieved %d chunks via MCP. Calling Groq …", len(chunks))

    # Call Groq for answer generation
    try:
        answer_data = ask_groq(req.question, chunks)
    except Exception as exc:
        logger.error("Groq call failed: %s", exc)
        raise HTTPException(502, f"Groq chat failed: {exc}")

    return ChatResponse(
        answer=answer_data["answer"],
        sources=answer_data["sources"],
        mcp_tool_called="search_documents",
        retrieved_chunks=len(chunks),
    )


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}
