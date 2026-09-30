from services.vector_store import collection
from services.embedder import model


def retrieve_documents(query, top_k=3):

    # Convert user's question into an embedding
    query_embedding = model.embed_query(
        query
    )


    # Search ChromaDB for similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )


    retrieved_chunks = []


    # Convert ChromaDB results into our application format
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