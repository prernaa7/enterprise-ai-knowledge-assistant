from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_document(text, source):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    chunk_data = []

    for index, chunk in enumerate(chunks, start=1):

        chunk_data.append({
            "text": chunk,
            "metadata": {
                "source": source,
                "chunk_id": index
            }
        })

    return chunk_data