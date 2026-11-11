"""MCP server that exposes the search_documents tool over SSE transport.

Run with:  python -m mcp_server.server
"""

import json
import logging
import sys
import os

# Ensure project root is on the path so `app.*` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from app.embeddings import embed_query
from app.db import vector_search, ensure_vector_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MCP] %(message)s")
logger = logging.getLogger(__name__)

mcp = FastMCP("document-retrieval", host="0.0.0.0", port=int(os.getenv("MCP_SERVER_PORT", "8001")))


@mcp.tool()
def search_documents(question: str, top_k: int = 5) -> str:
    """Search ingested documents for chunks relevant to *question*.

    Embeds the question with the same model used at ingestion, runs a
    top-k vector search in MongoDB, and returns matching chunks with
    source filenames as a JSON string.

    Args:
        question: The user's natural-language question.
        top_k: Number of chunks to return (default 5).

    Returns:
        A JSON-encoded list of {text, source, score} objects.
    """
    logger.info("search_documents called — question=%r  top_k=%d", question, top_k)
    query_vec = embed_query(question)
    results = vector_search(query_vec, top_k=top_k)
    logger.info("Returning %d chunks", len(results))
    return json.dumps(results, ensure_ascii=False)


if __name__ == "__main__":
    logger.info("Starting MCP server (SSE) …")
    ensure_vector_index()
    mcp.run(transport="sse")
