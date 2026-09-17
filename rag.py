import csv
import html
import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# Configuration
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
VECTOR_DB_PATH = PROJECT_DIR / "FAISS_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"

TIKTOK_CANDIDATES = ("tiktok.csv", "tiktok(1).csv")
REDDIT_CANDIDATES = ("reddit.csv",)
PLACES_CANDIDATES = ("places.csv",)

# Eight results, with no single source allowed to dominate.
SOURCE_LIMITS = {"TikTok": 3, "Reddit": 2, "Places": 3}
TOTAL_RESULTS = sum(SOURCE_LIMITS.values())


# =========================================================
# CSV helpers
# =========================================================

def resolve_data_file(candidates):
    """Return the first matching file from the data directory."""
    for filename in candidates:
        path = DATA_DIR / filename
        if path.exists():
            return path

    expected = ", ".join(str(DATA_DIR / name) for name in candidates)
    raise FileNotFoundError(f"Data file not found. Expected one of: {expected}")


def clean(value):
    """Clean text and repair common CSV mojibake such as Â£."""
    if value is None:
        return ""

    value = html.unescape(str(value)).strip()
    value = re.sub(r"\s+", " ", value)

    if any(marker in value for marker in ("Â", "Ã", "â€", "ðŸ")):
        try:
            value = value.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    return value


def is_true(value):
    return clean(value).lower() in {"true", "yes", "1"}


def detect_delimiter(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(16384)

    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        return "\t" if "\t" in sample.partition("\n")[0] else ","


def read_rows(path):
    """Read comma- or tab-separated exports using their real headers."""
    delimiter = detect_delimiter(path)

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        for raw_row in reader:
            row = {
                clean(key).lstrip("\ufeff"): clean(value)
                for key, value in raw_row.items()
                if key is not None
            }
            if any(row.values()):
                yield row


def add_part(parts, label, value):
    value = clean(value)
    if value:
        parts.append(f"{label}: {value}")


def unique_join(values):
    seen = set()
    result = []
    for value in values:
        value = clean(value)
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return ", ".join(result)


def collect_numbered_values(row, prefix):
    """Collect fields such as hashtags/0, hashtags/1, ..."""
    values = []
    pattern = re.compile(rf"^{re.escape(prefix)}/\d+$")
    for key, value in row.items():
        if pattern.match(key) and clean(value):
            values.append(value)
    return unique_join(values)


def collect_place_features(row, section):
    """Convert true additionalInfo flags into readable feature names."""
    features = []
    prefix = f"additionalInfo/{section}/"

    for key, value in row.items():
        if key.startswith(prefix) and is_true(value):
            features.append(key.rsplit("/", 1)[-1])

    return unique_join(features)


# =========================================================
# Source-specific document creation
# =========================================================

def load_places_documents():
    """Create one concise document per Google Places row."""
    path = resolve_data_file(PLACES_CANDIDATES)
    documents = []

    for row_number, row in enumerate(read_rows(path), start=1):
        title = row.get("title", "")
        category = row.get("categoryName", "")
        address = row.get("address", "")
        description = row.get("description") or row.get("ownerDescription", "")

        # Ignore records that cannot identify a place.
        if not title:
            continue

        if is_true(row.get("permanentlyClosed")):
            closure = "Permanently closed"
        elif is_true(row.get("temporarilyClosed")):
            closure = "Temporarily closed"
        else:
            closure = ""

        parts = []
        add_part(parts, "Restaurant", title)
        add_part(parts, "Category", category)
        add_part(parts, "Description", description)
        add_part(parts, "Address", address)
        add_part(parts, "Neighbourhood", row.get("neighborhood"))
        add_part(parts, "City", row.get("city"))
        add_part(parts, "Price", row.get("price"))
        add_part(parts, "Google rating", row.get("totalScore"))
        add_part(parts, "Number of reviews", row.get("reviewsCount"))
        add_part(parts, "Menu", row.get("menu"))
        add_part(parts, "Status", closure)

        feature_sections = {
            "Atmosphere": "Atmosphere",
            "Offerings": "Food and drink options",
            "Dining options": "Dining options",
            "Service options": "Service options",
            "Accessibility": "Accessibility",
            "Amenities": "Amenities",
            "Children": "Children",
            "Crowd": "Suitable for",
            "Highlights": "Highlights",
            "Planning": "Booking information",
            "Payments": "Payments",
        }

        for csv_section, output_label in feature_sections.items():
            add_part(
                parts,
                output_label,
                collect_place_features(row, csv_section),
            )

        documents.append(
            Document(
                page_content="\n".join(parts),
                metadata={
                    "source": "Places",
                    "row_number": row_number,
                    "place_id": row.get("placeId", ""),
                    "restaurant_name": title,
                    "category": category,
                    "address": address,
                    "rating": row.get("totalScore", ""),
                    "url": row.get("url") or row.get("website", ""),
                },
            )
        )

    return documents


def load_reddit_documents():
    """Create focused documents from the nested Reddit export."""
    path = resolve_data_file(REDDIT_CANDIDATES)
    documents = []

    for row_number, row in enumerate(read_rows(path), start=1):
        title = row.get("data/title", "")
        body = row.get("data/selftext", "")

        if not title and not body:
            continue

        parts = []
        add_part(parts, "Reddit title", title)
        add_part(parts, "Post text", body)
        add_part(parts, "Subreddit", row.get("data/subreddit"))
        add_part(parts, "Author", row.get("data/author"))
        add_part(parts, "Score", row.get("data/score"))
        add_part(parts, "Number of comments", row.get("data/num_comments"))
        add_part(parts, "Post type", row.get("data/link_flair_text"))

        permalink = row.get("data/permalink", "")
        if permalink.startswith("/"):
            permalink = f"https://www.reddit.com{permalink}"

        documents.append(
            Document(
                page_content="\n".join(parts),
                metadata={
                    "source": "Reddit",
                    "row_number": row_number,
                    "post_id": row.get("data/name", ""),
                    "title": title,
                    "subreddit": row.get("data/subreddit", ""),
                    "author": row.get("data/author", ""),
                    "score": row.get("data/score", ""),
                    "url": permalink or row.get("data/url", ""),
                },
            )
        )

    return documents


def load_tiktok_documents():
    """Create focused documents from the nested TikTok export."""
    path = resolve_data_file(TIKTOK_CANDIDATES)
    documents = []

    for row_number, row in enumerate(read_rows(path), start=1):
        title = row.get("title", "")
        subtitle = row.get("subtitle", "")
        poi_name = row.get("poi/poiName", "") or row.get("poi", "")
        hashtags = collect_numbered_values(row, "hashtags")

        if not any((title, subtitle, poi_name, hashtags)):
            continue

        parts = []
        add_part(parts, "TikTok caption", title)
        add_part(parts, "Subtitle or transcript", subtitle)
        add_part(parts, "Place", poi_name)
        add_part(parts, "Place address", row.get("poi/address"))
        add_part(parts, "City", row.get("poi/cityName"))
        add_part(parts, "Hashtags", hashtags)
        add_part(parts, "Creator", row.get("channel/name"))
        add_part(parts, "Username", row.get("channel/username"))
        add_part(parts, "Views", row.get("views"))
        add_part(parts, "Likes", row.get("likes"))
        add_part(parts, "Comments", row.get("comments"))
        add_part(parts, "Bookmarks", row.get("bookmarks"))
        add_part(parts, "Uploaded", row.get("uploadedAtFormatted"))

        documents.append(
            Document(
                page_content="\n".join(parts),
                metadata={
                    "source": "TikTok",
                    "row_number": row_number,
                    "video_id": row.get("id", ""),
                    "title": title,
                    "creator": row.get("channel/name", ""),
                    "place": poi_name,
                    "url": row.get("postPage", ""),
                },
            )
        )

    return documents


def load_all_documents():
    groups = {
        "TikTok": load_tiktok_documents(),
        "Reddit": load_reddit_documents(),
        "Places": load_places_documents(),
    }
    combined = groups["TikTok"] + groups["Reddit"] + groups["Places"]
    return groups, combined


# =========================================================
# Embeddings and FAISS
# =========================================================

@lru_cache(maxsize=1)
def get_embedding_model():
    # Do not pass show_progress_bar here; some installed versions
    # otherwise pass that argument to SentenceTransformer twice.
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def load_vector_DB():
    if not VECTOR_DB_PATH.exists():
        raise FileNotFoundError(
            "FAISS_index does not exist. Click 'Create Knowledge Base' first."
        )

    return FAISS.load_local(
        str(VECTOR_DB_PATH),
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )


def Create_combined_vector_DB():
    groups, documents = load_all_documents()

    if not documents:
        raise ValueError("No usable documents were found in the three CSV files.")

    vector_db = FAISS.from_documents(documents, get_embedding_model())
    vector_db.save_local(str(VECTOR_DB_PATH))
    load_vector_DB.cache_clear()

    return {
        "total": len(documents),
        "tiktok": len(groups["TikTok"]),
        "reddit": len(groups["Reddit"]),
        "places": len(groups["Places"]),
    }


# =========================================================
# Balanced retrieval
# =========================================================

def document_key(document):
    return (
        document.metadata.get("source", ""),
        document.metadata.get("row_number", ""),
        document.page_content,
    )


def get_retrieved_documents(question):
    question = clean(question)
    if not question:
        return []

    vector_db = load_vector_DB()
    selected = []
    seen = set()

    for source, limit in SOURCE_LIMITS.items():
        try:
            results = vector_db.similarity_search(
                question,
                k=limit,
                filter={"source": source},
                fetch_k=200,
            )
        except TypeError:
            results = vector_db.similarity_search(
                question,
                k=limit,
                filter={"source": source},
            )

        for document in results:
            key = document_key(document)
            if key not in seen:
                selected.append(document)
                seen.add(key)

    # Fill any empty positions if a source has fewer documents than its quota.
    if len(selected) < TOTAL_RESULTS:
        for document in vector_db.similarity_search(question, k=TOTAL_RESULTS * 4):
            key = document_key(document)
            if key not in seen:
                selected.append(document)
                seen.add(key)
            if len(selected) == TOTAL_RESULTS:
                break

    return selected[:TOTAL_RESULTS]


def format_retrieved_documents(documents):
    """Convert retrieved documents into structured evidence."""

    if not documents:
        return "No relevant evidence was retrieved."

    blocks = []

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}

        source = metadata.get("source", "Unknown")
        restaurant_name = metadata.get(
            "restaurant_name",
            metadata.get("title", "Unknown"),
        )
        url = metadata.get("url", "Not available")

        blocks.append(
            f"Evidence {index}\n"
            f"Source: {source}\n"
            f"Restaurant or title: {restaurant_name}\n"
            f"Content:\n{document.page_content}\n"
            f"URL: {url}"
        )

    return "\n\n---\n\n".join(blocks)


RAG_PROMPT = PromptTemplate.from_template(
    """
You are RestoRec, a restaurant recommendation assistant.

Answer the user's question using only the retrieved evidence.
The evidence may come from TikTok, Reddit or Google Places.

Rules:
- Recommend only places supported by the evidence.
- Prefer results that match the requested location, cuisine, price,
  dietary requirement or atmosphere.
- Treat TikTok and Reddit statements as user opinions, not verified facts.
- Treat Places ratings, addresses and features as structured listing data.
- Do not invent names, ratings, prices, dishes, addresses or URLs.
- If useful evidence exists, give a direct answer and briefly explain
  each recommendation.
- If the evidence only partially answers the question, state the limitation.
- Say "I do not know based on the available TikTok, Reddit and Places data"
  only when none of the evidence can answer the question.

Retrieved evidence:
{context}

User question:
{question}

Answer:
""".strip()
)


def build_grounded_prompt(question, documents):
    """Construct the complete grounded prompt sent to Gemini."""

    question = clean(question)

    if not question:
        raise ValueError("The question cannot be empty.")

    context = format_retrieved_documents(documents)

    return RAG_PROMPT.format(
        context=context,
        question=question,
    )


@lru_cache(maxsize=1)
def get_llm():
    """Create and cache the Gemini model."""

    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY was not found. "
            "Add GOOGLE_API_KEY=your_key to the .env file."
        )

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def extract_response_text(response):
    """Extract text from a Gemini response or mocked response."""

    if hasattr(response, "content"):
        return response.content

    return str(response)


def answer_question(question, documents=None, llm=None):
    """
    Answer a question using retrieved evidence.

    Tests can supply documents and a mocked LLM, avoiding Gemini calls.
    """

    question = clean(question)

    if not question:
        raise ValueError("The question cannot be empty.")

    if documents is None:
        documents = get_retrieved_documents(question)

    prompt = build_grounded_prompt(
        question=question,
        documents=documents,
    )

    model = llm if llm is not None else get_llm()
    response = model.invoke(prompt)

    return extract_response_text(response)


def retrieve_context(question):
    documents = get_retrieved_documents(question)
    return format_retrieved_documents(documents)


def get_QA_Chain():
    """
    Return a runnable RestoRec question-answering chain.

    The API key is validated when the chain is created so configuration
    problems are detected before a user submits a question.
    """

    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY was not found. "
            "Add GOOGLE_API_KEY=your_key to the .env file."
        )

    return RunnableLambda(answer_question)

# =========================================================
# Testable prompt and LLM functions
# =========================================================

def format_retrieved_documents(documents):
    """
    Convert retrieved documents into structured evidence.
    """

    if not documents:
        return "No relevant evidence was retrieved."

    blocks = []

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}

        source = metadata.get("source", "Unknown")

        restaurant_name = metadata.get(
            "restaurant_name",
            metadata.get("title", "Unknown"),
        )

        url = metadata.get("url", "Not available")

        blocks.append(
            f"Evidence {index}\n"
            f"Source: {source}\n"
            f"Restaurant or title: {restaurant_name}\n"
            f"Content:\n{document.page_content}\n"
            f"URL: {url}"
        )

    return "\n\n---\n\n".join(blocks)


TESTABLE_RAG_PROMPT = PromptTemplate.from_template(
    """
You are RestoRec, a restaurant recommendation assistant.

Answer the user's question using only the retrieved evidence.
The evidence may come from TikTok, Reddit or Google Places.

Rules:
- Recommend only places supported by the evidence.
- Prefer results that match the requested location, cuisine, price,
  dietary requirement or atmosphere.
- Treat TikTok and Reddit statements as user opinions, not verified facts.
- Treat Places ratings, addresses and features as structured listing data.
- Do not invent names, ratings, prices, dishes, addresses or URLs.
- If useful evidence exists, give a direct answer and briefly explain
  each recommendation.
- If the evidence only partially answers the question, state the limitation.
- Say "I do not know based on the available TikTok, Reddit and Places data"
  only when none of the evidence can answer the question.

Retrieved evidence:
{context}

User question:
{question}

Answer:
""".strip()
)


def build_grounded_prompt(question, documents):
    """
    Construct the complete prompt sent to Gemini.
    """

    question = clean(question)

    if not question:
        raise ValueError("The question cannot be empty.")

    context = format_retrieved_documents(documents)

    return TESTABLE_RAG_PROMPT.format(
        context=context,
        question=question,
    )


def extract_response_text(response):
    """
    Convert Gemini or mocked responses into a plain string.

    Gemini may return either:
    - a plain string;
    - an AIMessage whose content is a string; or
    - an AIMessage whose content is a list of content blocks.
    """

    content = (
        response.content
        if hasattr(response, "content")
        else response
    )

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):
                text = block.get("text")

                if text:
                    text_parts.append(str(text))

            elif hasattr(block, "text"):
                text = block.text

                if text:
                    text_parts.append(str(text))

        return "\n".join(text_parts).strip()

    return str(content)


def answer_question(question, documents=None, llm=None):
    """
    Produce an answer using retrieved documents.

    A mocked LLM can be supplied during testing so that the
    test does not contact Gemini or consume API quota.
    """

    question = clean(question)

    if not question:
        raise ValueError("The question cannot be empty.")

    if documents is None:
        documents = get_retrieved_documents(question)

    prompt = build_grounded_prompt(
        question=question,
        documents=documents,
    )

    if llm is None:
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY was not found. "
                "Add GOOGLE_API_KEY=your_key to the .env file."
            )

        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )

    response = llm.invoke(prompt)

    return extract_response_text(response)