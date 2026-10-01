from pathlib import Path
from types import SimpleNamespace

import pytest

from docqa.answerer import (
    NO_ANSWER,
    answer_question,
    build_prompt,
    default_client,
)
from docqa.chunker import build_chunks
from docqa.loader import load_text
from docqa.retriever import Retriever

SAMPLE_DOC = Path(__file__).resolve().parent.parent / "sample_docs" / "duka_faq.txt"


@pytest.fixture
def retriever():
    return Retriever(build_chunks(load_text(SAMPLE_DOC), "duka_faq.txt"))


class FakeClient:
    """Stands in for the Anthropic client so tests never call the network."""

    def __init__(self, text="Orders before 2pm arrive the same day.", fail=False):
        self.calls = []
        self.messages = self
        self._text = text
        self._fail = fail

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail:
            raise RuntimeError("API down")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._text)])


def test_extractive_answers_delivery_question(retriever):
    question = "How long does delivery take?"
    answer = answer_question(question, retriever.search(question))
    assert "2pm" in answer.text
    assert answer.used_llm is False


def test_extractive_answers_payment_question(retriever):
    question = "Do you accept M-Pesa payment?"
    assert "STK push" in answer_question(question, retriever.search(question)).text


def test_extractive_answers_refund_question(retriever):
    question = "When are refunds sent?"
    assert "3 working days" in answer_question(question, retriever.search(question)).text


def test_no_results_gives_no_answer_message(retriever):
    answer = answer_question("quantum physics", retriever.search("quantum physics"))
    assert answer.text == NO_ANSWER
    assert answer.sources == ()


def test_sources_are_listed_once(retriever):
    question = "delivery payment returns"
    answer = answer_question(question, retriever.search(question, top_k=3))
    assert answer.sources == ("duka_faq.txt",)


def test_claude_answer_is_used_when_client_given(retriever):
    question = "How long does delivery take?"
    client = FakeClient()
    answer = answer_question(question, retriever.search(question), client=client)
    assert answer.used_llm is True
    assert answer.text == "Orders before 2pm arrive the same day."
    assert len(client.calls) == 1
    assert "untrusted" in client.calls[0]["system"]


def test_failed_claude_call_falls_back_to_extractive(retriever):
    question = "How long does delivery take?"
    answer = answer_question(
        question, retriever.search(question), client=FakeClient(fail=True)
    )
    assert answer.used_llm is False
    assert "2pm" in answer.text
    assert answer.note != ""


def test_client_is_not_called_when_nothing_was_found(retriever):
    client = FakeClient()
    answer_question("quantum physics", retriever.search("quantum physics"), client=client)
    assert client.calls == []


def test_prompt_contains_question_and_source(retriever):
    question = "When are refunds sent?"
    prompt = build_prompt(question, retriever.search(question))
    assert question in prompt
    assert "duka_faq.txt" in prompt


def test_model_name_can_be_set_by_environment(retriever, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "my-test-model")
    question = "How long does delivery take?"
    client = FakeClient()
    answer_question(question, retriever.search(question), client=client)
    assert client.calls[0]["model"] == "my-test-model"


def test_no_client_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert default_client() is None
