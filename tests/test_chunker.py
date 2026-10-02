from pathlib import Path

import pytest

from docqa.chunker import build_chunks, chunk_text
from docqa.loader import load_text

SAMPLE_DOC = Path(__file__).resolve().parent.parent / "sample_docs" / "duka_faq.txt"


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", max_words=10, overlap=0) == ["hello world"]


def test_empty_text_returns_no_chunks():
    assert chunk_text("  \n\n  ") == []


def test_paragraphs_merge_until_limit():
    text = "one two three\n\nfour five six\n\nseven eight nine"
    assert chunk_text(text, max_words=6, overlap=0) == [
        "one two three four five six",
        "seven eight nine",
    ]


def test_long_paragraph_is_split_with_overlap():
    text = " ".join(f"w{i}" for i in range(10))
    assert chunk_text(text, max_words=4, overlap=1) == [
        "w0 w1 w2 w3",
        "w3 w4 w5 w6",
        "w6 w7 w8 w9",
    ]


def test_invalid_settings_raise():
    with pytest.raises(ValueError):
        chunk_text("some text", max_words=0)
    with pytest.raises(ValueError):
        chunk_text("some text", max_words=5, overlap=5)


def test_build_chunks_records_source_and_index():
    chunks = build_chunks("a b\n\nc d", "doc.txt", max_words=2, overlap=0)
    assert [c.text for c in chunks] == ["a b", "c d"]
    assert chunks[1].source == "doc.txt"
    assert chunks[1].index == 1


def test_sample_document_splits_by_topic():
    chunks = build_chunks(load_text(SAMPLE_DOC), "duka_faq.txt")
    assert len(chunks) == 4
    assert "Delivery" in chunks[0].text
    assert "Payment" in chunks[1].text
    assert "Returns" in chunks[2].text
    assert "Support" in chunks[3].text