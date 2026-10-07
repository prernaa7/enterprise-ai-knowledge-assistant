import os
import requests


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)


def ask_question(question):

    response = requests.post(
        f"{API_URL}/query",
        json={
            "question": question
        }
    )

    response.raise_for_status()

    return response.json()