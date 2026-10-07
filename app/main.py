import streamlit as st
from pathlib import Path
import requests
import os

from services.parser import parse_document
from services.chunker import chunk_document
from services.embedder import embed_chunks
from services.vector_store import store_chunks, get_collection_count
from services.api_client import ask_question


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)


DOCUMENT_FOLDER = Path("data/documents")
ALLOWED_EXTENSIONS = ["pdf", "docx", "txt"]


st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="📄"
)


st.title("Enterprise AI Knowledge Assistant")

st.write(
    "Upload a PDF, DOCX, or TXT document and ask questions "
    "about the stored knowledge."
)


# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------

st.subheader("Upload Document")

uploaded_file = st.file_uploader(
    "Choose a document",
    type=ALLOWED_EXTENSIONS
)


if uploaded_file is not None:

    DOCUMENT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = DOCUMENT_FOLDER / uploaded_file.name

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    try:

        # --------------------------------------------------
        # PARSE DOCUMENT
        # --------------------------------------------------

        text = parse_document(file_path)

        st.subheader("Extracted Text")

        if text.strip():

            st.text_area(
                "Document content",
                text,
                height=500
            )

        else:

            st.warning(
                "No text could be extracted from this document."
            )


        # --------------------------------------------------
        # CHUNK DOCUMENT
        # --------------------------------------------------

        chunks = chunk_document(
            text,
            uploaded_file.name
        )

        st.subheader("Document Chunks")

        st.write(
            f"Total chunks: {len(chunks)}"
        )

        for chunk in chunks:

            st.write(
                f"Chunk {chunk['metadata']['chunk_id']} "
                f"— {chunk['metadata']['source']}"
            )

            st.text_area(
                "Chunk content",
                chunk["text"],
                height=150,
                key=(
                    f"chunk_"
                    f"{uploaded_file.name}_"
                    f"{chunk['metadata']['chunk_id']}"
                )
            )


        # --------------------------------------------------
        # CREATE EMBEDDINGS
        # --------------------------------------------------

        embeddings = embed_chunks(chunks)

        st.subheader("Embeddings")

        st.write(
            f"Total embeddings: {len(embeddings)}"
        )

        if embeddings:

            st.write(
                f"Vector dimensions: {len(embeddings[0])}"
            )

            st.write("First embedding:")

            st.write(
                embeddings[0]
            )


        # --------------------------------------------------
        # STORE IN CHROMADB
        # --------------------------------------------------

        stored_count = store_chunks(
            chunks,
            embeddings
        )

        total_records = get_collection_count()

        st.success(
            f"Stored {stored_count} chunks in ChromaDB."
        )

        st.write(
            f"Total records in ChromaDB: {total_records}"
        )


        # --------------------------------------------------
        # DOCUMENT SUMMARY
        # --------------------------------------------------

        st.subheader("Document Summary")

        if st.button(
            "Generate Summary",
            key=f"summary_{uploaded_file.name}"
        ):

            try:

                response = requests.post(
                    f"{API_URL}/summarize",
                    json={
                        "text": text
                    }
                )

                response.raise_for_status()

                summary_result = response.json()

                st.write(
                    summary_result["summary"]
                )

            except Exception as error:

                st.error(
                    f"Error generating summary: {error}"
                )


    except ValueError as error:

        st.error(
            str(error)
        )


    except Exception as error:

        st.error(
            f"Error processing document: {error}"
        )


# --------------------------------------------------
# CHAT
# --------------------------------------------------

st.divider()

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(
            message["content"]
        )

        if message["role"] == "assistant" and "sources" in message:

            st.caption("Sources")

            for source in message["sources"]:

                with st.expander(
                    f"{source['source']} — "
                    f"Chunk {source['chunk_id']}"
                ):

                    st.write(
                        source["text"]
                    )


# Chat input

query = st.chat_input(
    "Ask a question about your documents..."
)


if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):

        st.write(
            query
        )

    try:

        result = ask_question(
            query
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"]
            }
        )

        with st.chat_message("assistant"):

            st.write(
                result["answer"]
            )

            st.caption("Sources")

            for source in result["sources"]:

                with st.expander(
                    f"{source['source']} — "
                    f"Chunk {source['chunk_id']}"
                ):

                    st.write(
                        source["text"]
                    )

    except Exception as error:

        st.error(
            f"Error connecting to the API: {error}"
        )