"""Groq chat completion wrapper."""

import logging
from groq import Groq
from app.config import GROQ_API_KEY, GROQ_CHAT_MODEL

logger = logging.getLogger(__name__)


def ask_groq(question: str, context_chunks: list[dict]) -> dict:
    """Send the question + retrieved context to Groq and return the answer.

    Returns {"answer": str, "sources": list[str]}.
    """
    client = Groq(api_key=GROQ_API_KEY)

    # Build context block
    if not context_chunks:
        context_block = "No relevant documents were found."
    else:
        parts = []
        for i, chunk in enumerate(context_chunks, 1):
            parts.append(f"[Excerpt {i} | source: {chunk['source']}]\n{chunk['text']}")
        context_block = "\n\n".join(parts)

    sources = list({c["source"] for c in context_chunks})

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the "
        "provided document excerpts. If the excerpts do not contain enough "
        "information to answer the question, say: "
        "'The provided documents do not contain enough information to answer "
        "this question.' Do NOT make up information."
    )

    user_prompt = (
        f"## Retrieved Document Excerpts\n\n{context_block}\n\n"
        f"## Question\n{question}\n\n"
        "Answer the question based only on the excerpts above. "
        "Cite the source filename(s) when possible."
    )

    logger.info("Calling Groq model %s", GROQ_CHAT_MODEL)
    response = client.chat.completions.create(
        model=GROQ_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    return {"answer": answer, "sources": sources}
