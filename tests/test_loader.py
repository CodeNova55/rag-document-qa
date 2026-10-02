import io
from pathlib import Path

import pytest
from pypdf import PdfWriter

from docqa.loader import load_from_bytes, load_text

SAMPLE_DOC = Path(__file__).resolve().parent.parent / "sample_docs" / "duka_faq.txt"


def test_load_text_file(tmp_path):
    file = tmp_path / "notes.txt"
    file.write_text("Habari za asubuhi", encoding="utf-8")
    assert load_text(file) == "Habari za asubuhi"


def test_load_markdown_file(tmp_path):
    file = tmp_path / "notes.md"
    file.write_text("# Title\n\nSome text", encoding="utf-8")
    assert load_text(file) == "# Title\n\nSome text"


def test_load_from_bytes_strips_byte_order_mark():
    assert load_from_bytes(b"\xef\xbb\xbfHabari", "a.txt") == "Habari"


def test_windows_line_endings_are_normalised():
    assert load_from_bytes(b"line one\r\nline two", "a.txt") == "line one\nline two"


def test_unsupported_extension_raises():
    with pytest.raises(ValueError):
        load_from_bytes(b"a,b,c", "data.csv")


def test_pdf_without_text_returns_empty_string():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    assert load_from_bytes(buffer.getvalue(), "blank.pdf").strip() == ""


def test_loads_sample_document():
    assert "Delivery" in load_text(SAMPLE_DOC)