"""Tests for RestoRec's balanced FAISS retrieval behaviour."""

from unittest.mock import Mock, call, patch

from langchain_core.documents import Document

import rag


def _document(source, row_number):
    return Document(
        page_content=f"{source} restaurant {row_number}",
        metadata={"source": source, "row_number": row_number},
    )


def test_blank_query_returns_empty_list_without_loading_faiss():
    with patch.object(rag, "load_vector_DB") as load_vector_db:
        assert rag.get_retrieved_documents("   ") == []

    load_vector_db.assert_not_called()


def test_query_returns_empty_list_when_no_documents_match():
    fake_vector_db = Mock()
    fake_vector_db.similarity_search.return_value = []

    with patch.object(rag, "load_vector_DB", return_value=fake_vector_db):
        documents = rag.get_retrieved_documents(
            "Find a vegan restaurant in Croydon"
        )

    assert documents == []
    assert fake_vector_db.similarity_search.call_count == 4
    assert fake_vector_db.similarity_search.call_args_list[:3] == [
        call(
            "Find a vegan restaurant in Croydon",
            k=3,
            filter={"source": "TikTok"},
            fetch_k=200,
        ),
        call(
            "Find a vegan restaurant in Croydon",
            k=2,
            filter={"source": "Reddit"},
            fetch_k=200,
        ),
        call(
            "Find a vegan restaurant in Croydon",
            k=3,
            filter={"source": "Places"},
            fetch_k=200,
        ),
    ]
    assert fake_vector_db.similarity_search.call_args_list[3] == call(
        "Find a vegan restaurant in Croydon",
        k=rag.TOTAL_RESULTS * 4,
    )


def test_query_returns_balanced_top_8_documents():
    source_results = {
        "TikTok": [_document("TikTok", number) for number in range(1, 4)],
        "Reddit": [_document("Reddit", number) for number in range(1, 3)],
        "Places": [_document("Places", number) for number in range(1, 4)],
    }
    fake_vector_db = Mock()

    def similarity_search(_question, *, k, filter=None, fetch_k=None):
        del fetch_k
        if filter:
            return source_results[filter["source"]][:k]
        return []

    fake_vector_db.similarity_search.side_effect = similarity_search

    with patch.object(rag, "load_vector_DB", return_value=fake_vector_db):
        documents = rag.get_retrieved_documents(
            "Recommend a Croydon restaurant"
        )

    assert len(documents) == 8
    assert [document.metadata["source"] for document in documents] == [
        "TikTok",
        "TikTok",
        "TikTok",
        "Reddit",
        "Reddit",
        "Places",
        "Places",
        "Places",
    ]
    assert fake_vector_db.similarity_search.call_count == 3


def test_retrieval_fills_missing_source_quota_without_duplicates():
    tiktok = [_document("TikTok", 1)]
    reddit = [_document("Reddit", 1), _document("Reddit", 2)]
    places = [_document("Places", number) for number in range(1, 4)]
    extra = [_document("TikTok", number) for number in range(2, 4)]
    fake_vector_db = Mock()

    def similarity_search(_question, *, k, filter=None, fetch_k=None):
        del fetch_k
        if filter == {"source": "TikTok"}:
            return tiktok
        if filter == {"source": "Reddit"}:
            return reddit
        if filter == {"source": "Places"}:
            return places
        return (tiktok + reddit + places + extra)[:k]

    fake_vector_db.similarity_search.side_effect = similarity_search

    with patch.object(rag, "load_vector_DB", return_value=fake_vector_db):
        documents = rag.get_retrieved_documents("Restaurants in Croydon")

    assert len(documents) == 8
    assert len({rag.document_key(document) for document in documents}) == 8

