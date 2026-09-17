"""Unit tests for CSV conversion, FAISS creation and chain validation."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

import rag


def test_resolve_data_file_raises_clear_error_for_missing_file(tmp_path):
    with patch.object(rag, "DATA_DIR", tmp_path):
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            rag.resolve_data_file(("missing.csv",))


def test_load_tiktok_documents_creates_expected_content_and_metadata(tmp_path):
    csv_path = tmp_path / "tiktok.csv"
    csv_path.write_text(
        "id,title,subtitle,poi/poiName,poi/cityName,postPage\n"
        "video-1,Best curry,Very good food,Curry House,Croydon,"
        "https://example.com/video-1\n",
        encoding="utf-8",
    )

    with patch.object(rag, "DATA_DIR", tmp_path):
        documents = rag.load_tiktok_documents()

    assert len(documents) == 1
    assert "TikTok caption: Best curry" in documents[0].page_content
    assert "Place: Curry House" in documents[0].page_content
    assert documents[0].metadata == {
        "source": "TikTok",
        "row_number": 1,
        "video_id": "video-1",
        "title": "Best curry",
        "creator": "",
        "place": "Curry House",
        "url": "https://example.com/video-1",
    }


def test_load_reddit_documents_creates_expected_content_and_metadata(tmp_path):
    csv_path = tmp_path / "reddit.csv"
    csv_path.write_text(
        "data/title,data/selftext,data/subreddit,data/author,data/permalink\n"
        "Croydon food,Try the cafe,croydon,user1,/r/croydon/comments/1\n",
        encoding="utf-8",
    )

    with patch.object(rag, "DATA_DIR", tmp_path):
        documents = rag.load_reddit_documents()

    assert len(documents) == 1
    assert "Reddit title: Croydon food" in documents[0].page_content
    assert documents[0].metadata["source"] == "Reddit"
    assert documents[0].metadata["url"] == (
        "https://www.reddit.com/r/croydon/comments/1"
    )


def test_load_places_documents_skips_rows_without_a_title(tmp_path):
    csv_path = tmp_path / "places.csv"
    csv_path.write_text(
        "title,categoryName,address,totalScore,placeId\n"
        ",Restaurant,Unknown,3.0,missing-title\n"
        "The Green Room,Cafe,1 High Street Croydon,4.6,place-1\n",
        encoding="utf-8",
    )

    with patch.object(rag, "DATA_DIR", tmp_path):
        documents = rag.load_places_documents()

    assert len(documents) == 1
    assert "Restaurant: The Green Room" in documents[0].page_content
    assert documents[0].metadata["source"] == "Places"
    assert documents[0].metadata["restaurant_name"] == "The Green Room"


def test_load_vector_db_requires_existing_faiss_index(tmp_path):
    missing_index = tmp_path / "FAISS_index"
    rag.load_vector_DB.cache_clear()

    with patch.object(rag, "VECTOR_DB_PATH", missing_index):
        with pytest.raises(FileNotFoundError, match="FAISS_index does not exist"):
            rag.load_vector_DB()

    rag.load_vector_DB.cache_clear()


def test_load_vector_db_uses_embedding_model_and_safe_path(tmp_path):
    index_path = tmp_path / "FAISS_index"
    index_path.mkdir()
    fake_embedding = Mock(name="embedding")
    fake_vector_db = Mock(name="vector_db")
    rag.load_vector_DB.cache_clear()

    with patch.object(rag, "VECTOR_DB_PATH", index_path), patch.object(
        rag, "get_embedding_model", return_value=fake_embedding
    ), patch.object(
        rag.FAISS, "load_local", return_value=fake_vector_db
    ) as load_local:
        result = rag.load_vector_DB()

    assert result is fake_vector_db
    load_local.assert_called_once_with(
        str(index_path),
        fake_embedding,
        allow_dangerous_deserialization=True,
    )
    rag.load_vector_DB.cache_clear()


def test_create_combined_vector_db_builds_saves_and_returns_counts(tmp_path):
    groups = {
        "TikTok": [Document(page_content="TikTok", metadata={"source": "TikTok"})],
        "Reddit": [Document(page_content="Reddit", metadata={"source": "Reddit"})],
        "Places": [
            Document(page_content="Place 1", metadata={"source": "Places"}),
            Document(page_content="Place 2", metadata={"source": "Places"}),
        ],
    }
    documents = groups["TikTok"] + groups["Reddit"] + groups["Places"]
    fake_embedding = Mock(name="embedding")
    fake_faiss = Mock(name="faiss")
    index_path = tmp_path / "FAISS_index"

    with patch.object(rag, "load_all_documents", return_value=(groups, documents)), patch.object(
        rag, "get_embedding_model", return_value=fake_embedding
    ), patch.object(rag, "VECTOR_DB_PATH", index_path), patch.object(
        rag.FAISS, "from_documents", return_value=fake_faiss
    ) as from_documents, patch.object(rag.load_vector_DB, "cache_clear") as cache_clear:
        counts = rag.Create_combined_vector_DB()

    assert counts == {"total": 4, "tiktok": 1, "reddit": 1, "places": 2}
    from_documents.assert_called_once_with(documents, fake_embedding)
    fake_faiss.save_local.assert_called_once_with(str(index_path))
    cache_clear.assert_called_once_with()


def test_create_combined_vector_db_rejects_empty_data():
    empty_groups = {"TikTok": [], "Reddit": [], "Places": []}

    with patch.object(rag, "load_all_documents", return_value=(empty_groups, [])):
        with pytest.raises(ValueError, match="No usable documents"):
            rag.Create_combined_vector_DB()


def test_get_qa_chain_requires_google_api_key():
    with patch.object(rag, "GOOGLE_API_KEY", None):
        with pytest.raises(ValueError, match="GOOGLE_API_KEY was not found"):
            rag.get_QA_Chain()

