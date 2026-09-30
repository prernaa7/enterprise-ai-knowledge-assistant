from app.services.vector_store import collection
from app.services.embedder import model


def retrieve_documents(query, top_k=3):

    query_embedding = model.embed_query(
        query
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_chunks = []

    for index, document in enumerate(
        results["documents"][0]
    ):

        retrieved_chunks.append(
            {
                "text": document,

                "metadata": {
                    "source": (
                        results["metadatas"][0][index]["source"]
                    ),

                    "chunk_id": (
                        results["metadatas"][0][index]["chunk_id"]
                    )
                }
            }
        )

    return retrieved_chunks