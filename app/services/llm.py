import os

from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

# Load variables from .env
load_dotenv()


# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")


# Check whether API key exists
if not api_key:

    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Please add it to the .env file."
    )


# Create Gemini client
client = genai.Client(
    api_key=api_key
)


# Gemini model
MODEL_NAME = "gemini-2.5-flash"


# --------------------------------------------------
# DOCUMENT SUMMARIZATION
# --------------------------------------------------

def summarize_document(text):

    prompt = f"""
You are an enterprise document summarization assistant.

Summarize the following document using ONLY the
information provided.

Rules:
1. Do not add information that is not in the document.
2. Keep the summary concise.
3. Highlight the main policies, rules, and important points.
4. Use clear bullet points where appropriate.

Document:
{text}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# --------------------------------------------------
# QUESTION ANSWERING
# --------------------------------------------------

def generate_answer(question, retrieved_chunks):

    # Build context from retrieved chunks
    context = "\n\n".join(
        [
            (
                f"Source: {chunk['metadata']['source']}\n"
                f"Chunk: {chunk['metadata']['chunk_id']}\n"
                f"Content:\n{chunk['text']}"
            )
            for chunk in retrieved_chunks
        ]
    )


    # Prompt for the LLM
    prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the information
provided in the context below.

Rules:

1. Do not use outside knowledge.
2. Do not make up information.
3. If the answer cannot be found in the context, say:
"I couldn't find this information in the provided documents."
4. Keep the answer concise and clear.
5. Mention the relevant source when possible.

Context:
{context}

User question:
{question}
"""


    # Send prompt to Gemini
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )


    # Return generated answer
    return response.text