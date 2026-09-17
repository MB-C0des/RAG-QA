import numpy as np
import pytest

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag import get_embedding_model


@pytest.fixture(scope="module")
def embedding_model():
    return get_embedding_model()


def cosine_similarity(vector_a, vector_b):
    vector_a = np.asarray(vector_a)
    vector_b = np.asarray(vector_b)

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    return float(
        np.dot(vector_a, vector_b) / denominator
    )


def test_embedding_output_dimension(embedding_model):
    embedding = embedding_model.embed_query(
        "Italian restaurant in Croydon"
    )

    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(
        isinstance(value, float)
        for value in embedding
    )


def test_related_text_has_greater_similarity(
    embedding_model,
):
    query = "Italian restaurant serving pizza in Croydon"
    related = "A Croydon restaurant offering pizza and pasta"
    unrelated = "Instructions for repairing a laptop screen"

    query_vector = embedding_model.embed_query(query)
    related_vector = embedding_model.embed_query(related)
    unrelated_vector = embedding_model.embed_query(unrelated)

    related_score = cosine_similarity(
        query_vector,
        related_vector,
    )

    unrelated_score = cosine_similarity(
        query_vector,
        unrelated_vector,
    )

    assert related_score > unrelated_score
    assert related_score >= 0.40


def test_faiss_returns_relevant_document(
    embedding_model,
):
    documents = [
        Document(
            page_content=(
                "Bella Italia is an Italian restaurant "
                "in Croydon serving pizza and pasta."
            ),
            metadata={
                "source": "Places",
                "restaurant_name": "Bella Italia",
            },
        ),
        Document(
            page_content=(
                "Spice Garden serves curry, biryani "
                "and naan."
            ),
            metadata={
                "source": "Places",
                "restaurant_name": "Spice Garden",
            },
        ),
        Document(
            page_content=(
                "Green Cafe serves coffee, breakfast "
                "and cakes."
            ),
            metadata={
                "source": "Places",
                "restaurant_name": "Green Cafe",
            },
        ),
    ]

    vector_store = FAISS.from_documents(
        documents,
        embedding_model,
    )

    results = vector_store.similarity_search(
        "Where can I get pizza and pasta in Croydon?",
        k=2,
    )

    returned_names = [
        document.metadata["restaurant_name"]
        for document in results
    ]

    assert len(results) == 2
    assert "Bella Italia" in returned_names
    assert returned_names[0] == "Bella Italia"