"""MongoDB helpers – document storage and vector search."""

import logging
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel
from app.config import MONGO_URI, MONGO_DB, EMBEDDING_DIM

logger = logging.getLogger(__name__)

_client: MongoClient | None = None


def get_collection() -> Collection:
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client[MONGO_DB]["chunks"]


def ensure_vector_index() -> None:
    """Create the Atlas-style vector search index if it does not exist.

    For local MongoDB (without Atlas Search) we fall back to a plain index on
    the 'embedding' field and do brute-force cosine search in Python.  The
    code first tries the Atlas SearchIndexModel; if the server rejects it
    (Community Edition), it silently falls back.
    """
    col = get_collection()
    # Try Atlas vector search index
    try:
        existing = [idx["name"] for idx in col.list_search_indexes()]
        if "vector_index" not in existing:
            idx = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": EMBEDDING_DIM,
                            "similarity": "cosine",
                        }
                    ]
                },
                name="vector_index",
                type="vectorSearch",
            )
            col.create_search_index(idx)
            logger.info("Atlas vector search index created.")
        else:
            logger.info("Atlas vector search index already exists.")
        return
    except Exception as exc:
        logger.warning(
            "Atlas vector search unavailable (%s); using brute-force fallback.", exc
        )
    # Ensure a basic index on source for filtering
    col.create_index("source")


def insert_chunks(records: list[dict]) -> int:
    """Insert chunk records and return the count of inserted documents."""
    col = get_collection()
    result = col.insert_many(records)
    return len(result.inserted_ids)


def vector_search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Return the *top_k* most similar chunks to *query_embedding*.

    Tries Atlas $vectorSearch first; falls back to brute-force in Python.
    """
    col = get_collection()

    # --- Atlas path ---
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "source": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        results = list(col.aggregate(pipeline))
        if results:
            return results
    except Exception:
        pass

    # --- Brute-force fallback for Community Edition ---
    import numpy as np

    query_vec = np.array(query_embedding, dtype=np.float32)
    all_docs = list(col.find({}, {"_id": 0, "text": 1, "source": 1, "embedding": 1}))
    if not all_docs:
        return []
    scored = []
    for doc in all_docs:
        doc_vec = np.array(doc["embedding"], dtype=np.float32)
        cos_sim = float(np.dot(query_vec, doc_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-10
        ))
        scored.append({"text": doc["text"], "source": doc["source"], "score": cos_sim})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
