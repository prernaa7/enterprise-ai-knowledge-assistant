import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings


@st.cache_resource
def load_embedding_model():

    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


def embed_chunks(chunks):

    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.embed_documents(texts)

    return embeddings