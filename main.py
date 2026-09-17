import streamlit as st


st.set_page_config(
    page_title="RestoRec",
    page_icon="🍽",
    layout="centered",
)

st.title("RestoRec 🍽")
st.write(
    "Restaurant recommendations using TikTok, Reddit and Places data."
)


try:
    from rag import (
        Create_combined_vector_DB,
        VECTOR_DB_PATH,
        answer_question,
        get_retrieved_documents,
    )
except Exception as error:
    st.error("Failed to load the RAG application.")
    st.exception(error)
    st.stop()


# =========================================================
# Knowledge base
# =========================================================

st.subheader("Knowledge Base")

if VECTOR_DB_PATH.exists():
    st.success("The FAISS knowledge base is available.")
else:
    st.info("Create the knowledge base before asking a question.")


if st.button("Create Knowledge Base", use_container_width=True):
    try:
        with st.spinner(
            "Reading the CSV files and creating embeddings. "
            "The first run can take several minutes..."
        ):
            counts = Create_combined_vector_DB()

        st.cache_resource.clear()

        st.success(
            "Knowledge base created successfully from "
            f"{counts['total']} documents: "
            f"{counts['tiktok']} TikTok, "
            f"{counts['reddit']} Reddit and "
            f"{counts['places']} Places documents."
        )

    except Exception as error:
        st.error("Failed to create the knowledge base.")
        st.exception(error)


# =========================================================
# Question and answer
# =========================================================

st.divider()

with st.form("question_form"):
    question = st.text_input(
        "Question:",
        placeholder=(
            "For example: Recommend an Indian restaurant "
            "in South Croydon"
        ),
    )

    submitted = st.form_submit_button(
        "Ask RestoRec",
        type="primary",
        use_container_width=True,
    )


if submitted:
    question = question.strip()

    if not question:
        st.warning("Enter a question first.")

    elif not VECTOR_DB_PATH.exists():
        st.warning("Create the knowledge base first.")

    else:
        try:
            with st.spinner(
                "Searching TikTok, Reddit and Places data..."
            ):
                # Retrieval is performed once.
                retrieved_documents = get_retrieved_documents(
                    question
                )

                # The exact same documents are sent to Gemini.
                answer = answer_question(
                    question=question,
                    documents=retrieved_documents,
                )

            st.subheader("Answer")
            st.write(answer)

            with st.expander(
                "View documents retrieved from FAISS"
            ):
                st.write(
                    f"FAISS retrieved "
                    f"{len(retrieved_documents)} relevant "
                    "documents from the combined knowledge base."
                )

                if not retrieved_documents:
                    st.info(
                        "No relevant documents were retrieved."
                    )

                for index, document in enumerate(
                    retrieved_documents,
                    start=1,
                ):
                    metadata = document.metadata or {}
                    source = metadata.get(
                        "source",
                        "Unknown",
                    )

                    st.markdown(
                        f"### Retrieved document {index}"
                    )
                    st.markdown(
                        f"**Source:** {source}"
                    )

                    if metadata.get("restaurant_name"):
                        st.markdown(
                            "**Restaurant:** "
                            f"{metadata['restaurant_name']}"
                        )

                    if metadata.get("title"):
                        st.markdown(
                            f"**Title:** {metadata['title']}"
                        )

                    st.text(document.page_content)

                    if metadata.get("url"):
                        st.markdown(
                            f"[Open source]"
                            f"({metadata['url']})"
                        )

                    st.divider()

        except Exception as error:
            st.error(
                "RestoRec could not answer the question."
            )
            st.exception(error)