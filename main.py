import streamlit as st

st.set_page_config(
    page_title="RestoRec",
    page_icon="🌱"
)

st.title("RestoRec 🌱")

try:
    from Langchain_helper import (
        Create_combined_vector_DB,
        get_QA_Chain
    )

except Exception as e:
    st.error("Failed to load LangChain helper.")
    st.exception(e)
    st.stop()


# ---------------------------------------------------------
# Create knowledge base
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
# Question
# ---------------------------------------------------------

question = st.text_input(
    "Question:",
    placeholder="e.g. Recommend an Indian restaurant near East Croydon"
)


# ---------------------------------------------------------
# Answer question
# ---------------------------------------------------------

if question:

    try:

        with st.spinner("Searching knowledge base..."):

            chain = get_QA_Chain()

            response = chain.invoke(question)

        st.header("Answer")

        st.write(response)

    except Exception as e:

        st.error(
            "An error occurred while answering the question."
        )

        st.exception(e)