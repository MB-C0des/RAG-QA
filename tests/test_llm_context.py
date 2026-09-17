import pytest

from langchain_core.documents import Document

from rag import (
    answer_question,
    build_grounded_prompt,
    format_retrieved_documents,
)


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self):
        self.received_prompt = None

    def invoke(self, prompt):
        self.received_prompt = prompt

        return FakeResponse(
            "Test Restaurant is a suitable option."
        )


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content=(
                "Restaurant: Test Restaurant\n"
                "Category: Vegetarian restaurant\n"
                "Address: Croydon\n"
                "Price: ££"
            ),
            metadata={
                "source": "Places",
                "restaurant_name": "Test Restaurant",
                "url": "https://example.com/restaurant",
            },
        ),
        Document(
            page_content=(
                "TikTok caption: Affordable vegetarian "
                "food in Croydon."
            ),
            metadata={
                "source": "TikTok",
                "title": "Vegetarian food in Croydon",
                "url": "https://example.com/video",
            },
        ),
    ]


def test_document_context_contains_metadata(
    sample_documents,
):
    context = format_retrieved_documents(
        sample_documents
    )

    assert "Source: Places" in context
    assert "Test Restaurant" in context
    assert "Vegetarian restaurant" in context
    assert "https://example.com/restaurant" in context

    assert "Source: TikTok" in context
    assert "Affordable vegetarian food" in context


def test_empty_documents_produce_safe_context():
    context = format_retrieved_documents([])

    assert context == "No relevant evidence was retrieved."


def test_grounded_prompt_contains_question_and_context(
    sample_documents,
):
    question = (
        "Recommend affordable vegetarian food in Croydon"
    )

    prompt = build_grounded_prompt(
        question=question,
        documents=sample_documents,
    )

    assert question in prompt
    assert "Test Restaurant" in prompt
    assert "Source: Places" in prompt
    assert "Source: TikTok" in prompt
    assert "Do not invent" in prompt


def test_empty_question_is_rejected(
    sample_documents,
):
    with pytest.raises(
        ValueError,
        match="question cannot be empty",
    ):
        build_grounded_prompt(
            question="   ",
            documents=sample_documents,
        )


def test_llm_receives_exact_retrieved_context(
    sample_documents,
):
    fake_llm = FakeLLM()

    answer = answer_question(
        question=(
            "Recommend affordable vegetarian food "
            "in Croydon"
        ),
        documents=sample_documents,
        llm=fake_llm,
    )

    assert answer == (
        "Test Restaurant is a suitable option."
    )

    assert fake_llm.received_prompt is not None
    assert "Test Restaurant" in fake_llm.received_prompt
    assert "Source: Places" in fake_llm.received_prompt
    assert "Source: TikTok" in fake_llm.received_prompt
    assert "Affordable vegetarian food" in (
        fake_llm.received_prompt
    )