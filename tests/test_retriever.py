from pathlib import Path

import pytest

from docqa.chunker import Chunk, build_chunks
from docqa.loader import load_text
from docqa.retriever import Retriever

SAMPLE_DOC = Path(__file__).resolve().parent.parent / "sample_docs" / "duka_faq.txt"


@pytest.fixture
def retriever():
    chunks = build_chunks(load_text(SAMPLE_DOC), "duka_faq.txt")
    return Retriever(chunks)


def top_text(retriever, question):
    return retriever.search(question)[0].chunk.text


def test_finds_delivery_info(retriever):
    assert "2pm" in top_text(retriever, "How long does delivery take?")


def test_finds_payment_info(retriever):
    assert "STK push" in top_text(retriever, "Do you accept M-Pesa payment?")


def test_finds_refund_info(retriever):
    assert "Refunds" in top_text(retriever, "When are refunds sent?")


def test_finds_support_info(retriever):
    assert "Monday" in top_text(retriever, "What hours is customer support available?")


def test_unrelated_question_returns_nothing(retriever):
    assert retriever.search("quantum physics black holes") == []


def test_empty_question_returns_nothing(retriever):
    assert retriever.search("   ") == []


def test_no_chunks_returns_nothing():
    assert Retriever([]).search("anything") == []


def test_top_k_limits_results(retriever):
    assert len(retriever.search("delivery", top_k=1)) == 1


def test_results_are_sorted_best_first(retriever):
    scores = [r.score for r in retriever.search("delivery payment returns", top_k=4)]
    assert scores == sorted(scores, reverse=True)


def test_result_keeps_source_and_score(retriever):
    result = retriever.search("When are refunds sent?")[0]
    assert result.chunk.source == "duka_faq.txt"
    assert result.score > 0


def test_only_stop_words_does_not_crash():
    retriever = Retriever([Chunk("the and of", "x.txt", 0)])
    assert retriever.search("the") == []
