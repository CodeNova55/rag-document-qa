from .answerer import answer_question
from .chunker import build_chunks
from .loader import load_from_bytes
from .retriever import Retriever


def build_index(documents):
    """Build a retriever from a list of (filename, bytes) pairs.

    Returns (retriever, problems). A file that cannot be read is reported in
    `problems` and skipped, so one bad upload never breaks the others.
    """
    chunks = []
    problems = []
    for filename, data in documents:
        try:
            text = load_from_bytes(data, filename)
        except Exception as error:
            problems.append(f"{filename}: could not be read ({error})")
            continue
        if not text.strip():
            problems.append(
                f"{filename}: no text found (scanned PDFs are not supported)"
            )
            continue
        chunks.extend(build_chunks(text, filename))
    return Retriever(chunks), problems


def ask(retriever, question, client=None, top_k=3):
    """Search the documents and answer. Returns (answer, search_results)."""
    results = retriever.search(question, top_k=top_k)
    return answer_question(question, results, client=client), results
