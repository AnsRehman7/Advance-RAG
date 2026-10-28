from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import config

SUPPORTED_TYPES = ["pdf", "docx", "txt", "md"]


def save_uploaded_file(uploaded_file) -> Path:
    """Save a file uploaded in Streamlit into data/documents and return its path."""
    file_path = config.DOCUMENT_DIR / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return file_path


def load_document(file_path: Path) -> list[Document]:
    """
    Extract the text from a PDF, DOCX or TXT file.

    A PDF gives one Document per page, the other types give one Document for the whole file.
    """
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif extension == ".docx":
        loader = Docx2txtLoader(str(file_path))
    elif extension in (".txt", ".md"):
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"File type '{extension}' is not supported")

    documents = loader.load()

    # keep just the file name, the full path is not needed in the answer
    for doc in documents:
        doc.metadata["source"] = file_path.name

    # drop empty pages, for example scanned PDF pages that have no text
    return [doc for doc in documents if doc.page_content.strip()]
