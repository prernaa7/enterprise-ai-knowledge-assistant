import streamlit as st
from pathlib import Path

from services.parser import parse_document
from services.chunker import chunk_document
from services.embedder import embed_chunks
from services.vector_store import store_chunks, get_collection_count
from services.retriever import retrieve_documents


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
# Document Upload
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
        # Step 1: Parse Document
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
        # Step 2: Chunk Document
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
                key=f"chunk_{uploaded_file.name}_{chunk['metadata']['chunk_id']}"
            )


        # --------------------------------------------------
        # Step 3: Create Embeddings
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
        # Step 4: Store in ChromaDB
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


    except ValueError as error:

        st.error(
            str(error)
        )

    except Exception as error:

        st.error(
            f"Error processing document: {error}"
        )


# --------------------------------------------------
# Retrieval
# --------------------------------------------------

st.divider()

st.subheader("Ask the Knowledge Base")

query = st.text_input(
    "Enter your question"
)


if query:

    retrieved_chunks = retrieve_documents(
        query
    )

    st.write(
        f"Retrieved {len(retrieved_chunks)} chunks"
    )

    for chunk in retrieved_chunks:

        st.write(
            f"Source: {chunk['metadata']['source']} "
            f"| Chunk: {chunk['metadata']['chunk_id']}"
        )

        st.text_area(
            "Retrieved content",
            chunk["text"],
            height=150,
            key=f"retrieved_{chunk['metadata']['source']}_{chunk['metadata']['chunk_id']}"
        )