import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import PromptTemplate

from langchain_community.document_loaders import CSVLoader

from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------
# Environment variables
# ---------------------------------------------------------

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:

    raise ValueError(
        "GOOGLE_API_KEY was not found. "
        "Check your .env file contains "
        "GOOGLE_API_KEY=your_key"
    )


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# TikTok / restaurant CSV
TIKTOK_CSV_PATH = os.path.join(
    BASE_DIR,
    "tiktok.csv"
)


# Reddit CSV
REDDIT_CSV_PATH = os.path.join(
    BASE_DIR,
    "reddit.csv"
)


# FAISS vector database
VECTOR_DB_PATH = os.path.join(
    BASE_DIR,
    "FAISS_index"
)


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

instruction_embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# Gemini model
# ---------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=google_api_key,
    temperature=0.1,
)


# ---------------------------------------------------------
# Load a CSV
# ---------------------------------------------------------

def load_csv_documents(
    file_path,
    source_name
):

    print(
        f"\nLoading {source_name} CSV from:"
    )

    print(
        file_path
    )


    # Check file exists
    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"{source_name} CSV file not found: "
            f"{file_path}"
        )


    print(
        f"{source_name} CSV file found."
    )


    # Load CSV
    loader = CSVLoader(
        file_path=file_path,
        encoding="utf-8-sig"
    )


    try:

        documents = loader.load()

    except Exception as e:

        print(
            f"\nERROR WHILE LOADING "
            f"{source_name.upper()} CSV:"
        )

        print(
            type(e).__name__
        )

        print(
            e
        )

        raise


    # Make sure data exists
    if len(documents) == 0:

        raise ValueError(
            f"{source_name} CSV was loaded "
            f"but contains no documents."
        )


    # Add metadata to identify source
    for document in documents:

        document.metadata[
            "knowledge_source"
        ] = source_name


    print(
        f"Successfully loaded "
        f"{len(documents)} "
        f"{source_name} documents."
    )


    return documents


# ---------------------------------------------------------
# Create Combined Knowledge Base
# ---------------------------------------------------------

def Create_combined_vector_DB():

    print(
        "\nCreating combined TikTok + Reddit "
        "knowledge base..."
    )


    # -----------------------------------------------------
    # Load TikTok data
    # -----------------------------------------------------

    tiktok_documents = load_csv_documents(
        TIKTOK_CSV_PATH,
        "tiktok"
    )


    # -----------------------------------------------------
    # Load Reddit data
    # -----------------------------------------------------

    reddit_documents = load_csv_documents(
        REDDIT_CSV_PATH,
        "reddit"
    )


    # -----------------------------------------------------
    # Combine documents
    # -----------------------------------------------------

    combined_documents = (
        tiktok_documents
        +
        reddit_documents
    )


    print(
        "\n----------------------------------"
    )

    print(
        f"TikTok documents: "
        f"{len(tiktok_documents)}"
    )

    print(
        f"Reddit documents: "
        f"{len(reddit_documents)}"
    )

    print(
        f"Combined documents: "
        f"{len(combined_documents)}"
    )

    print(
        "----------------------------------"
    )


    if len(combined_documents) == 0:

        raise ValueError(
            "No documents were loaded."
        )


    # -----------------------------------------------------
    # Test embeddings
    # -----------------------------------------------------

    print(
        "\nCreating embeddings..."
    )


    texts = [
        document.page_content
        for document in combined_documents
    ]


    embeddings = (
        instruction_embedding.embed_documents(
            texts
        )
    )


    print(
        f"Embeddings created successfully: "
        f"{len(embeddings)} vectors."
    )


    # -----------------------------------------------------
    # Create FAISS vector database
    # -----------------------------------------------------

    print(
        "\nCreating FAISS vector database..."
    )


    vector_DB = FAISS.from_documents(
        documents=combined_documents,
        embedding=instruction_embedding
    )


    print(
        "FAISS vector database created "
        "in memory."
    )


    # -----------------------------------------------------
    # Save FAISS database
    # -----------------------------------------------------

    vector_DB.save_local(
        VECTOR_DB_PATH
    )


    print(
        "FAISS vector database "
        "saved successfully."
    )

    print(
        f"Saved to: {VECTOR_DB_PATH}"
    )


    return len(
        combined_documents
    )


# ---------------------------------------------------------
# Retrieve documents for debugging
# ---------------------------------------------------------

def get_retrieved_documents(
    question,
    k=15
):

    # Check FAISS database exists
    if not os.path.exists(
        VECTOR_DB_PATH
    ):

        raise FileNotFoundError(
            "FAISS index does not exist. "
            "Create the knowledge base first."
        )


    # Load FAISS
    vector_DB = FAISS.load_local(
        VECTOR_DB_PATH,
        instruction_embedding,
        allow_dangerous_deserialization=True
    )


    # Search FAISS
    documents = (
        vector_DB.similarity_search(
            question,
            k=k
        )
    )


    return documents


# ---------------------------------------------------------
# Build QA Chain
# ---------------------------------------------------------

def get_QA_Chain():

    # -----------------------------------------------------
    # Check FAISS exists
    # -----------------------------------------------------

    if not os.path.exists(
        VECTOR_DB_PATH
    ):

        raise FileNotFoundError(
            "FAISS index does not exist. "
            "Create the knowledge base first."
        )


    print(
        "\nLoading FAISS vector database..."
    )


    # -----------------------------------------------------
    # Load FAISS
    # -----------------------------------------------------

    vector_DB = FAISS.load_local(
        VECTOR_DB_PATH,
        instruction_embedding,
        allow_dangerous_deserialization=True
    )


    # -----------------------------------------------------
    # Retriever
    # -----------------------------------------------------

    # FAISS searches the full knowledge base
    # but returns the 15 most relevant documents
    retriever = vector_DB.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 15
        }
    )


    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt_template = """
You are RestoRec, a restaurant recommendation assistant
focused on Croydon and nearby areas.

Answer the user's question using ONLY the information
contained in the retrieved context.

The context contains information from two sources:

1. TikTok / restaurant data
2. Reddit community discussion data

Use both sources when they are relevant.

A restaurant does NOT need to appear in both sources
to be recommended.

When recommending restaurants, consider:

- restaurant name
- cuisine
- location
- area
- price
- average meal price
- atmosphere or vibe
- dietary options
- popular dishes
- restaurant description
- Reddit comments
- Reddit sentiment
- Reddit discussion score
- other useful information in the retrieved documents

If Reddit evidence supports a recommendation,
you can mention that Reddit discussion was positive
or that users discussed the restaurant favourably.

Do NOT present Reddit opinions as objective facts.

If multiple restaurants match the question,
recommend the best matches and explain why.

Do not invent:

- restaurant names
- locations
- ratings
- prices
- dishes
- reviews
- Reddit comments
- facts that are not present in the context

If the context genuinely contains no useful information
for answering the user's question, say:

"I don't know based on the available restaurant and Reddit data."

CONTEXT:

{context}


QUESTION:

{question}


ANSWER:
"""


    prompt = PromptTemplate.from_template(
        prompt_template
    )


    # -----------------------------------------------------
    # Format retrieved documents
    # -----------------------------------------------------

    def format_docs(
        docs
    ):

        formatted_documents = []


        for index, doc in enumerate(
            docs,
            start=1
        ):

            source = doc.metadata.get(
                "knowledge_source",
                "unknown"
            )


            formatted_document = (
                f"\nDOCUMENT {index}\n"
                f"SOURCE: {source.upper()}\n"
                f"{doc.page_content}\n"
            )


            formatted_documents.append(
                formatted_document
            )


        return "\n\n".join(
            formatted_documents
        )


    # -----------------------------------------------------
    # Build RAG chain
    # -----------------------------------------------------

    chain = (

        {
            "context":
                retriever
                |
                format_docs,

            "question":
                RunnablePassthrough()
        }

        |

        prompt

        |

        llm

        |

        StrOutputParser()

    )


    return chain