import chromadb


CHROMA_PATH = "chroma_db"


client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


collection = client.get_or_create_collection(
    name="enterprise_documents"
)


def store_chunks(chunks, embeddings):

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:

        chunk_id = chunk["metadata"]["chunk_id"]
        source = chunk["metadata"]["source"]

        ids.append(
            f"{source}_{chunk_id}"
        )

        documents.append(
            chunk["text"]
        )

        metadatas.append({
            "source": source,
            "chunk_id": chunk_id
        })

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(ids)


def get_collection_count():

    return collection.count()