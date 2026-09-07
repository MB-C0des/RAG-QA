import streamlit as st

st.set_page_config(
    page_title="RestoRec",
    page_icon="🌱"
)

st.title("RestoRec 🌱")


# ---------------------------------------------------------
# Import LangChain helper functions
# ---------------------------------------------------------

try:
    from Langchain_helper import (
        Create_combined_vector_DB,
        get_QA_Chain,
        get_retrieved_documents
    )

except Exception as e:
    st.error("Failed to load LangChain helper.")
    st.exception(e)
    st.stop()


# ---------------------------------------------------------
# Create Knowledge Base
# ---------------------------------------------------------

if st.button("Create Knowledge Base"):

    try:

        with st.spinner(
            "Creating knowledge base from TikTok and Reddit data..."
        ):

            number_of_documents = Create_combined_vector_DB()

        st.success(
            f"Knowledge base created successfully "
            f"from {number_of_documents} documents."
        )

    except Exception as e:

        st.error("Failed to create knowledge base.")

        st.exception(e)


# ---------------------------------------------------------
# Question input
# ---------------------------------------------------------

question = st.text_input(
    "Question:",
    placeholder="e.g. Recommend an Indian restaurant in Croydon"
)


# ---------------------------------------------------------
# Answer question
# ---------------------------------------------------------

if question:

    try:

        with st.spinner("Searching knowledge base..."):

            # Build the RAG chain
            chain = get_QA_Chain()

            # Ask the question
            response = chain.invoke(question)

            # Also retrieve the actual documents
            # so we can inspect what FAISS found
            retrieved_docs = get_retrieved_documents(
                question,
                k=15
            )

        # -------------------------------------------------
        # Display answer
        # -------------------------------------------------

        st.header("Answer")

        st.write(response)


        # -------------------------------------------------
        # Display retrieved evidence
        # -------------------------------------------------

        with st.expander(
            "View documents retrieved from FAISS"
        ):

            st.write(
                f"FAISS retrieved "
                f"{len(retrieved_docs)} documents."
            )

            for number, doc in enumerate(
                retrieved_docs,
                start=1
            ):

                source = doc.metadata.get(
                    "knowledge_source",
                    "unknown"
                )

                st.markdown(
                    f"### Document {number}"
                )

                st.write(
                    f"**Source:** {source}"
                )

                st.text(
                    doc.page_content
                )

                st.divider()


    except Exception as e:

        st.error(
            "An error occurred while answering the question."
        )

        st.exception(e)