import pymupdf
from docx import Document


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def parse_pdf(file_path):
    text = ""

    with pymupdf.open(file_path) as document:
        for page in document:
            text += page.get_text()

    return text


def parse_docx(file_path):
    document = Document(file_path)

    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text


def parse_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def parse_document(file_path):
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return parse_pdf(file_path)

    elif extension == ".docx":
        return parse_docx(file_path)

    elif extension == ".txt":
        return parse_txt(file_path)

    else:
        raise ValueError("Unsupported file type")