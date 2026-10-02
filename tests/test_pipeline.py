from pathlib import Path
from types import SimpleNamespace

from docqa.answerer import NO_ANSWER
from docqa.pipeline import ask, build_index

SAMPLE_DOC = Path(__file__).resolve().parent.parent / "sample_docs" / "duka_faq.txt"


class FakeClient:
    def __init__(self):
        self.messages = self

    def create(self, **kwargs):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Fake answer.")])


def test_answers_a_question_from_the_sample_document():
    retriever, problems = build_index([("duka_faq.txt", SAMPLE_DOC.read_bytes())])
    answer, results = ask(retriever, "How long does delivery take?")
    assert problems == []
    assert "2pm" in answer.text
    assert results[0].score > 0


def test_answers_cite_the_right_document():
    documents = [
        ("cats.txt", b"Cats sleep sixteen hours a day and enjoy warm windowsills."),
        ("dogs.txt", b"Dogs need daily walks and love playing fetch in the park."),
    ]
    retriever, problems = build_index(documents)
    answer, _ = ask(retriever, "How many hours do cats sleep?")
    assert problems == []
    assert answer.sources == ("cats.txt",)


def test_unsupported_file_is_reported_and_others_still_work():
    documents = [("data.csv", b"a,b"), ("note.txt", b"Hello there friendly world")]
    retriever, problems = build_index(documents)
    assert len(problems) == 1
    assert "data.csv" in problems[0]
    answer, _ = ask(retriever, "friendly world")
    assert answer.sources == ("note.txt",)


def test_empty_file_is_reported():
    _, problems = build_index([("empty.txt", b"   \n")])
    assert len(problems) == 1
    assert "empty.txt" in problems[0]


def test_corrupt_pdf_is_reported_without_crashing():
    _, problems = build_index([("bad.pdf", b"this is not a pdf")])
    assert len(problems) == 1
    assert "bad.pdf" in problems[0]


def test_no_documents_gives_no_answer():
    retriever, problems = build_index([])
    answer, results = ask(retriever, "anything at all")
    assert problems == []
    assert answer.text == NO_ANSWER
    assert results == []


def test_client_is_passed_through_to_the_answerer():
    retriever, _ = build_index([("note.txt", b"Orders ship on Monday morning.")])
    answer, _ = ask(retriever, "When do orders ship?", client=FakeClient())
    assert answer.used_llm is True
    assert answer.text == "Fake answer."
