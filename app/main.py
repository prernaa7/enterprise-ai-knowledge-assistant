import streamlit as st
from pathlib import Path

from services.parser import parse_document
from services.chunker import chunk_document
from services.embedder import embed_chunks
from services.vector_store import store_chunks


DOCUMENT_FOLDER = Path("data/documents")
ALLOWED_EXTENSIONS = ["pdf", "docx", "txt"]


st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="📄"
)

st.title("Enterprise AI Knowledge Assistant")

st.write("Upload a PDF, DOCX, or TXT document.")


uploaded_file = st.file_uploader(
    "Choose a document",
    type=ALLOWED_EXTENSIONS
)


if uploaded_file is not None:

    file_path = DOCUMENT_FOLDER / uploaded_file.name

    DOCUMENT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.success(f"Uploaded: {uploaded_file.name}")

    try:
        text = parse_document(file_path)

        st.subheader("Extracted Text")

        if text.strip():
            st.text_area(
                "Document content",
                text,
                height=500
            )

            chunks = chunk_document(text, uploaded_file.name)
            embeddings = embed_chunks(chunks)
            stored_count = store_chunks(chunks, embeddings)
            st.success(f"Stored {stored_count} chunks in ChromaDB.")
            
            st.subheader("Document Chunks")

            st.write(f"Total chunks: {len(chunks)}")

            for chunk in chunks:

                st.write(
                    f"Chunk {chunk['metadata']['chunk_id']} "
                    f"— {chunk['metadata']['source']}"
                )

                st.text_area(
                    "Chunk content",
                    chunk["text"],
                    height=150,
                    key=f"chunk_{chunk['metadata']['chunk_id']}"
                )

            st.subheader("Embeddings")

            st.write(f"Total embeddings: {len(embeddings)}")

            if embeddings:
                st.write(
                    f"Vector dimensions: {len(embeddings[0])}"
                )

                st.write("First embedding:")
                st.write(embeddings[0])

        else:
            st.warning("No text could be extracted from this document.")

    except ValueError as error:
        st.error(str(error))

    except Exception as error:
        st.error(f"Error processing document: {error}")