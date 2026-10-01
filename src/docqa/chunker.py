import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    index: int


def chunk_text(text, max_words=60, overlap=10):
    """Split text into passages of at most `max_words` words.

    Paragraphs are kept together and merged until the limit is reached.
    A paragraph longer than the limit is split into overlapping windows.
    """
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    if not 0 <= overlap < max_words:
        raise ValueError("overlap must be between 0 and max_words - 1")

    paragraphs = [p.split() for p in re.split(r"\n\s*\n", text)]
    paragraphs = [words for words in paragraphs if words]

    chunks = []
    current = []
    for words in paragraphs:
        if len(words) > max_words:
            if current:
                chunks.append(" ".join(current))
                current = []
            step = max_words - overlap
            for start in range(0, len(words), step):
                chunks.append(" ".join(words[start:start + max_words]))
                if start + max_words >= len(words):
                    break
        elif len(current) + len(words) > max_words:
            chunks.append(" ".join(current))
            current = list(words)
        else:
            current.extend(words)

    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(text, source, max_words=60, overlap=10):
    """Chunk text and remember which document each passage came from."""
    return [
        Chunk(text=piece, source=source, index=i)
        for i, piece in enumerate(chunk_text(text, max_words, overlap))
    ]