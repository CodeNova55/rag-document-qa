import io
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_from_bytes(data, filename):
    """Return the text of an uploaded .txt, .md, or .pdf file."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or 'none'}")

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        # utf-8-sig also strips a byte order mark if the file has one
        text = data.decode("utf-8-sig", errors="replace")

    # Windows files use \r\n; use \n everywhere so chunking behaves the same
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load_text(path):
    """Read a file from disk and return its text."""
    path = Path(path)
    return load_from_bytes(path.read_bytes(), path.name)