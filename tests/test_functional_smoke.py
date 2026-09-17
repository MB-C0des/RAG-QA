"""Optional end-to-end checks for the real FAISS index and Gemini service."""

import os

import pytest

import rag


def _ensure_vector_index():
    """Create the real index only when a functional test explicitly needs it."""
    if not rag.VECTOR_DB_PATH.exists():
        rag.Create_combined_vector_DB()


@pytest.mark.functional
def test_real_query_smoke():
    """Gemini should return a non-empty string for a real restaurant query."""
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY is required for the Gemini smoke test")

    _ensure_vector_index()
    response = rag.get_QA_Chain().invoke("Recommend a restaurant in Croydon")

    assert isinstance(response, str)
    assert response.strip()


@pytest.mark.functional
def test_real_retrieval_smoke():
    """The real FAISS index should retrieve usable source documents."""
    _ensure_vector_index()
    documents = rag.get_retrieved_documents(
        "Recommend a restaurant in Croydon"
    )

    assert isinstance(documents, list)
    assert 1 <= len(documents) <= rag.TOTAL_RESULTS
    assert all(document.page_content.strip() for document in documents)
    assert all(document.metadata.get("source") for document in documents)

