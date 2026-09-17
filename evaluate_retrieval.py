import csv
import math
import time
from pathlib import Path
from statistics import mean

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag import (
    load_tiktok_documents,
    load_reddit_documents,
    load_places_documents,
)


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

RESULTS_FILE = PROJECT_DIR / "retrieval_evaluation_results.csv"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Preserve the source balance from the current RAG system.
SOURCE_LIMITS = {
    "TikTok": 3,
    "Reddit": 2,
    "Places": 3,
}

TOTAL_RESULTS = sum(SOURCE_LIMITS.values())

# Retrieve more candidates before applying RRF.
CANDIDATES_PER_SOURCE = 10

# RRF settings.
RRF_CONSTANT = 60

# First weight = BM25
# Second weight = dense FAISS
HYBRID_WEIGHTS = {
    "bm25": 0.3,
    "dense": 0.7,
}

# Repeat timing measurements to reduce random variation.
TIMING_REPETITIONS = 3


# =========================================================
# EVALUATION QUESTIONS
# =========================================================
#
# Document IDs follow this format:
#
#     Source:row_number
#
# Examples:
#
#     TikTok:12
#     Reddit:7
#     Places:31
#
# Initially leave relevant_ids empty. The first run prints
# the retrieved document IDs. Inspect the data and add every
# document that is genuinely relevant to each question.
# =========================================================

EVALUATION_QUESTIONS = [
    {
        "question": "Where can I get Nigerian food in Croydon?",
        "relevant_ids": {
            # "TikTok:12",
            # "Places:31",
        },
    },
    {
        "question": "Recommend an affordable restaurant near East Croydon.",
        "relevant_ids": set(),
    },
    {
        "question": "Where can I get burgers in Croydon?",
        "relevant_ids": {
        "TikTok:822",   # Smacks Hamburgers, South Croydon
        "TikTok:574",   # Rio's Piri Piri burgers
        "Places:107",   # Krunk Burgers
        "Places:299",   # Smacks Hamburgers
        "Places:309",   # Rodeos Burgers and Shakes
        "Places:509",   # McDonald's, Wellesley Road
        "Places:717",   # McDonald's, North End
    },
},
    {
        "question": "Recommend somewhere suitable for a date night.",
        "relevant_ids": set(),
    },
    {
        "question": "Which restaurants have vegetarian options?",
        "relevant_ids": set(),
    },
    {
        "question": "Where can I find halal food in Croydon?",
        "relevant_ids": set(),
    },
    {
        "question": "What restaurants are recommended by Reddit users?",
        "relevant_ids": set(),
    },
    {
        "question": "Which restaurants are popular on TikTok?",
        "relevant_ids": set(),
    },
]


# =========================================================
# DOCUMENT HELPERS
# =========================================================

def normalise_source(source):
    """Return the consistent source name used by the project."""

    source_text = str(source).strip().lower()

    if "tiktok" in source_text:
        return "TikTok"

    if "reddit" in source_text:
        return "Reddit"

    if "place" in source_text:
        return "Places"

    return str(source).strip() or "Unknown"


def get_document_id(document):
    """
    Create a stable identifier using the metadata already
    produced by the current rag.py loaders.
    """

    source = normalise_source(
        document.metadata.get("source", "Unknown")
    )

    row_number = document.metadata.get(
        "row_number",
        document.metadata.get("row", "unknown"),
    )

    return f"{source}:{row_number}"


def deduplicate_documents(documents):
    """Remove duplicate documents while preserving ranking."""

    unique_documents = []
    seen = set()

    for document in documents:
        document_id = get_document_id(document)

        if document_id not in seen:
            unique_documents.append(document)
            seen.add(document_id)

    return unique_documents


def preview_text(document, maximum_length=250):
    """Produce a short single-line preview."""

    text = " ".join(document.page_content.split())

    if len(text) > maximum_length:
        return text[:maximum_length] + "..."

    return text


# =========================================================
# LOAD CURRENT PROJECT DOCUMENTS
# =========================================================

def load_evaluation_documents():
    """
    Use the current rag.py loading functions so evaluation
    uses exactly the same document construction as the app.
    """

    print("Loading documents using the current rag.py loaders...")

    tiktok_documents = load_tiktok_documents()
    reddit_documents = load_reddit_documents()
    places_documents = load_places_documents()

    documents_by_source = {
        "TikTok": tiktok_documents,
        "Reddit": reddit_documents,
        "Places": places_documents,
    }

    for source, documents in documents_by_source.items():
        print(f"{source}: {len(documents)} documents")

        # Ensure every document has a consistent source value.
        for document in documents:
            document.metadata["source"] = source

    total = sum(
        len(documents)
        for documents in documents_by_source.values()
    )

    if total == 0:
        raise ValueError(
            "No documents were loaded. Check the files in "
            "the data folder and the loaders in rag.py."
        )

    print(f"Total: {total} documents")

    return documents_by_source


# =========================================================
# BUILD DENSE AND BM25 INDEXES
# =========================================================

def build_retrieval_indexes(documents_by_source):
    """Build one dense and one BM25 index for each source."""

    print("\nLoading MiniLM embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    dense_indexes = {}
    bm25_retrievers = {}

    for source, documents in documents_by_source.items():
        if not documents:
            continue

        print(f"Building indexes for {source}...")

        dense_indexes[source] = FAISS.from_documents(
            documents,
            embeddings,
        )

        bm25_retriever = BM25Retriever.from_documents(
            documents
        )

        bm25_retriever.k = min(
            CANDIDATES_PER_SOURCE,
            len(documents),
        )

        bm25_retrievers[source] = bm25_retriever

    return dense_indexes, bm25_retrievers


# =========================================================
# SOURCE-BALANCED DENSE RETRIEVAL
# =========================================================

def retrieve_dense(query, dense_indexes):
    """Retrieve source-balanced results using FAISS."""

    final_documents = []

    for source, result_limit in SOURCE_LIMITS.items():
        vector_store = dense_indexes.get(source)

        if vector_store is None:
            continue

        documents = vector_store.similarity_search(
            query,
            k=result_limit,
        )

        final_documents.extend(documents)

    return deduplicate_documents(final_documents)[
        :TOTAL_RESULTS
    ]


# =========================================================
# SOURCE-BALANCED BM25 RETRIEVAL
# =========================================================

def retrieve_bm25(query, bm25_retrievers):
    """Retrieve source-balanced results using BM25."""

    final_documents = []

    for source, result_limit in SOURCE_LIMITS.items():
        retriever = bm25_retrievers.get(source)

        if retriever is None:
            continue

        documents = retriever.invoke(query)

        final_documents.extend(documents[:result_limit])

    return deduplicate_documents(final_documents)[
        :TOTAL_RESULTS
    ]


# =========================================================
# RECIPROCAL RANK FUSION
# =========================================================

def reciprocal_rank_fusion(
    bm25_documents,
    dense_documents,
):
    """
    Combine BM25 and dense rankings using weighted RRF.

    RRF score:
        weight / (constant + rank)
    """

    scores = {}
    document_lookup = {}

    ranked_lists = [
        (
            bm25_documents,
            HYBRID_WEIGHTS["bm25"],
        ),
        (
            dense_documents,
            HYBRID_WEIGHTS["dense"],
        ),
    ]

    for documents, weight in ranked_lists:
        for rank, document in enumerate(
            documents,
            start=1,
        ):
            document_id = get_document_id(document)

            document_lookup[document_id] = document

            score = weight / (RRF_CONSTANT + rank)

            scores[document_id] = (
                scores.get(document_id, 0.0) + score
            )

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [
        document_lookup[document_id]
        for document_id in ranked_ids
    ]


# =========================================================
# SOURCE-BALANCED HYBRID RETRIEVAL
# =========================================================

def retrieve_hybrid(
    query,
    dense_indexes,
    bm25_retrievers,
):
    """
    Retrieve candidates independently for each source and
    combine BM25 and FAISS rankings with weighted RRF.
    """

    final_documents = []

    for source, result_limit in SOURCE_LIMITS.items():
        vector_store = dense_indexes.get(source)
        bm25_retriever = bm25_retrievers.get(source)

        if vector_store is None or bm25_retriever is None:
            continue

        dense_candidates = vector_store.similarity_search(
            query,
            k=CANDIDATES_PER_SOURCE,
        )

        bm25_candidates = bm25_retriever.invoke(query)[
            :CANDIDATES_PER_SOURCE
        ]

        fused_documents = reciprocal_rank_fusion(
            bm25_documents=bm25_candidates,
            dense_documents=dense_candidates,
        )

        final_documents.extend(
            fused_documents[:result_limit]
        )

    return deduplicate_documents(final_documents)[
        :TOTAL_RESULTS
    ]


# =========================================================
# METRICS
# =========================================================

def precision_at_k(retrieved_ids, relevant_ids, k):
    """Calculate Precision@K."""

    if k == 0:
        return 0.0

    retrieved_at_k = retrieved_ids[:k]

    relevant_retrieved = sum(
        document_id in relevant_ids
        for document_id in retrieved_at_k
    )

    return relevant_retrieved / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Calculate Recall@K."""

    if not relevant_ids:
        return 0.0

    retrieved_at_k = retrieved_ids[:k]

    relevant_retrieved = sum(
        document_id in relevant_ids
        for document_id in retrieved_at_k
    )

    return relevant_retrieved / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """Calculate reciprocal rank of the first relevant item."""

    for rank, document_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if document_id in relevant_ids:
            return 1 / rank

    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """Calculate binary nDCG@K."""

    if not relevant_ids:
        return 0.0

    dcg = 0.0

    for rank, document_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):
        relevance = int(document_id in relevant_ids)

        dcg += relevance / math.log2(rank + 1)

    ideal_relevant_count = min(
        len(relevant_ids),
        k,
    )

    ideal_dcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_relevant_count + 1,
        )
    )

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


# =========================================================
# TIMING
# =========================================================

def run_with_timing(retrieval_function):
    """
    Run retrieval several times and return the final documents
    plus the average execution time.
    """

    timings = []
    documents = []

    for _ in range(TIMING_REPETITIONS):
        start_time = time.perf_counter()

        documents = retrieval_function()

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        timings.append(elapsed_ms)

    return documents, mean(timings)


# =========================================================
# DISPLAY
# =========================================================

def display_results(
    method,
    question,
    documents,
    elapsed_ms,
):
    """Print retrieved documents for manual assessment."""

    print("\n" + "=" * 80)
    print(f"Method: {method}")
    print(f"Question: {question}")
    print(f"Average retrieval time: {elapsed_ms:.2f} ms")
    print("=" * 80)

    for rank, document in enumerate(
        documents,
        start=1,
    ):
        print(
            f"\nRank {rank}: "
            f"{get_document_id(document)}"
        )
        print(preview_text(document))


# =========================================================
# EVALUATION
# =========================================================

def evaluate(
    dense_indexes,
    bm25_retrievers,
):
    """Evaluate dense, BM25 and hybrid retrieval."""

    metric_results = {
        "Dense": [],
        "BM25": [],
        "Hybrid": [],
    }

    csv_rows = []

    for test_case in EVALUATION_QUESTIONS:
        question = test_case["question"]
        relevant_ids = set(test_case["relevant_ids"])

        retrieval_methods = {
            "Dense": lambda: retrieve_dense(
                question,
                dense_indexes,
            ),
            "BM25": lambda: retrieve_bm25(
                question,
                bm25_retrievers,
            ),
            "Hybrid": lambda: retrieve_hybrid(
                question,
                dense_indexes,
                bm25_retrievers,
            ),
        }

        print("\n\n" + "#" * 80)
        print(f"QUESTION: {question}")
        print(
            "GROUND-TRUTH DOCUMENTS: "
            f"{sorted(relevant_ids)}"
        )
        print("#" * 80)

        for method, retrieval_function in (
            retrieval_methods.items()
        ):
            documents, elapsed_ms = run_with_timing(
                retrieval_function
            )

            retrieved_ids = [
                get_document_id(document)
                for document in documents
            ]

            display_results(
                method,
                question,
                documents,
                elapsed_ms,
            )

            result = {
                "question": question,
                "method": method,
                "precision_at_k": None,
                "recall_at_k": None,
                "reciprocal_rank": None,
                "ndcg_at_k": None,
                "average_time_ms": elapsed_ms,
                "retrieved_ids": ";".join(
                    retrieved_ids
                ),
            }

            if relevant_ids:
                precision = precision_at_k(
                    retrieved_ids,
                    relevant_ids,
                    TOTAL_RESULTS,
                )

                recall = recall_at_k(
                    retrieved_ids,
                    relevant_ids,
                    TOTAL_RESULTS,
                )

                rr = reciprocal_rank(
                    retrieved_ids,
                    relevant_ids,
                )

                ndcg = ndcg_at_k(
                    retrieved_ids,
                    relevant_ids,
                    TOTAL_RESULTS,
                )

                result.update(
                    {
                        "precision_at_k": precision,
                        "recall_at_k": recall,
                        "reciprocal_rank": rr,
                        "ndcg_at_k": ndcg,
                    }
                )

                metric_results[method].append(result)

                print(
                    f"\nPrecision@{TOTAL_RESULTS}: "
                    f"{precision:.3f}"
                )
                print(
                    f"Recall@{TOTAL_RESULTS}:    "
                    f"{recall:.3f}"
                )
                print(f"Reciprocal rank:  {rr:.3f}")
                print(
                    f"nDCG@{TOTAL_RESULTS}:      "
                    f"{ndcg:.3f}"
                )
            else:
                print(
                    "\nNo ground-truth IDs assigned. "
                    "Use the IDs above to label this question."
                )

            csv_rows.append(result)

    return metric_results, csv_rows


# =========================================================
# SAVE DETAILED RESULTS
# =========================================================

def save_results(csv_rows):
    """Save per-question results for dissertation analysis."""

    fieldnames = [
        "question",
        "method",
        "precision_at_k",
        "recall_at_k",
        "reciprocal_rank",
        "ndcg_at_k",
        "average_time_ms",
        "retrieved_ids",
    ]

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nDetailed results saved to: {RESULTS_FILE}")


# =========================================================
# SUMMARY
# =========================================================

def print_summary(metric_results, csv_rows):
    """Print average results across labelled questions."""

    print("\n\n" + "=" * 94)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 94)

    print(
        f"\n{'Method':<12}"
        f"{'Precision@8':>14}"
        f"{'Recall@8':>12}"
        f"{'MRR':>10}"
        f"{'nDCG@8':>12}"
        f"{'Time (ms)':>14}"
    )

    print("-" * 94)

    for method in ["Dense", "BM25", "Hybrid"]:
        labelled_results = metric_results[method]

        method_rows = [
            row
            for row in csv_rows
            if row["method"] == method
        ]

        average_time = mean(
            row["average_time_ms"]
            for row in method_rows
        )

        if labelled_results:
            average_precision = mean(
                row["precision_at_k"]
                for row in labelled_results
            )

            average_recall = mean(
                row["recall_at_k"]
                for row in labelled_results
            )

            mrr = mean(
                row["reciprocal_rank"]
                for row in labelled_results
            )

            average_ndcg = mean(
                row["ndcg_at_k"]
                for row in labelled_results
            )
        else:
            average_precision = 0.0
            average_recall = 0.0
            mrr = 0.0
            average_ndcg = 0.0

        print(
            f"{method:<12}"
            f"{average_precision:>14.3f}"
            f"{average_recall:>12.3f}"
            f"{mrr:>10.3f}"
            f"{average_ndcg:>12.3f}"
            f"{average_time:>14.2f}"
        )

    if not any(metric_results.values()):
        print(
            "\nThe questions are not labelled yet. "
            "Inspect the retrieved IDs, add the relevant IDs "
            "to EVALUATION_QUESTIONS, and run the script again."
        )


# =========================================================
# MAIN
# =========================================================

def main():
    print("RestoRec retrieval evaluation")
    print(f"Total results per question: {TOTAL_RESULTS}")
    print(f"Source limits: {SOURCE_LIMITS}")
    print(f"Hybrid weights: {HYBRID_WEIGHTS}")
    print(f"RRF constant: {RRF_CONSTANT}")

    documents_by_source = load_evaluation_documents()

    dense_indexes, bm25_retrievers = (
        build_retrieval_indexes(
            documents_by_source
        )
    )

    metric_results, csv_rows = evaluate(
        dense_indexes,
        bm25_retrievers,
    )

    save_results(csv_rows)

    print_summary(metric_results, csv_rows)


if __name__ == "__main__":
    main()