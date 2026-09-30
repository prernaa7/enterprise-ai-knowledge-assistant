from fastapi import FastAPI
from pydantic import BaseModel

from app.services.retriever import retrieve_documents
from app.services.llm import generate_answer


app = FastAPI(
    title="Enterprise AI Knowledge Assistant API",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Enterprise AI Knowledge Assistant API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/query")
def query_knowledge_base(request: QuestionRequest):

    retrieved_chunks = retrieve_documents(
        request.question,
        top_k=3
    )

    answer = generate_answer(
        request.question,
        retrieved_chunks
    )

    sources = []

    for chunk in retrieved_chunks:

        sources.append(
            {
                "source": chunk["metadata"]["source"],
                "chunk_id": chunk["metadata"]["chunk_id"]
            }
        )

    return {
        "question": request.question,
        "answer": answer,
        "sources": sources
    }