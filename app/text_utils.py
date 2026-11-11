"""Utilities for splitting plain text into chunks."""


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split *text* into overlapping chunks of roughly *chunk_size* characters.

    The splitter tries to break on sentence boundaries (periods followed by a
    space or newline) so that chunks stay semantically coherent.
    """
    sentences: list[str] = []
    current = ""
    for char in text:
        current += char
        if char in ".!?" and len(current) > 1:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    chunks: list[str] = []
    buf = ""
    for sentence in sentences:
        if len(buf) + len(sentence) + 1 > chunk_size and buf:
            chunks.append(buf.strip())
            # keep overlap from end of previous chunk
            buf = buf[-overlap:] if overlap else ""
        buf += (" " if buf else "") + sentence

    if buf.strip():
        chunks.append(buf.strip())

    return chunks if chunks else [text.strip()] if text.strip() else []
