import json
import uuid
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from llmClient import get_chat_model, get_embedding_model


SUMMARY_PROMPT = """Write a short overview of this file in 3 to 5 sentences: what kind of document it is,
what it is about and its main topics. Use only the text below, which is the beginning of the file.

File name: {name}

Text:
{text}

Overview:"""


def load_doc_store() -> dict:
    """
    Read doc_store.json.

    It keeps the parent chunks (sent to the LLM), the child chunks (needed for BM25 search),
    the names of the files already added and a short overview of each file.
    """
    if not config.DOC_STORE_PATH.exists():
        return {"files": [], "parents": {}, "children": [], "summaries": {}}

    with open(config.DOC_STORE_PATH, "r", encoding="utf-8") as f:
        doc_store = json.load(f)

    # doc stores saved before overviews were added do not have this key
    doc_store.setdefault("summaries", {})
    return doc_store


def save_doc_store(doc_store: dict):
    with open(config.DOC_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(doc_store, f, ensure_ascii=False, indent=2)


def load_vector_store():
    """Load the FAISS index from disk. Returns None if no document is added yet."""
    if not (config.FAISS_DIR / "index.faiss").exists():
        return None

    # allow_dangerous_deserialization is fine here because we created this index ourselves
    return FAISS.load_local(str(config.FAISS_DIR), get_embedding_model(), allow_dangerous_deserialization=True)


def split_into_chunks(documents: list[Document]):
    """
    Parent-child chunking.

    Every document is cut into big parent chunks, and every parent is cut again into small child chunks.
    Small chunks are better for searching, big chunks give the LLM enough context to answer.
    Each child remembers its parent through "parent_id".
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.PARENT_CHUNK_OVERLAP
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP
    )

    parents = {}
    children = []

    for parent in parent_splitter.split_documents(documents):
        parent_id = str(uuid.uuid4())
        parents[parent_id] = {"text": parent.page_content, "metadata": parent.metadata}

        for child_text in child_splitter.split_text(parent.page_content):
            child_metadata = {**parent.metadata, "parent_id": parent_id}
            children.append(Document(page_content=child_text, metadata=child_metadata))

    return parents, children


def add_documents(documents: list[Document], file_names: list[str]):
    """Chunk the documents, embed the child chunks into FAISS and save everything to disk."""
    parents, children = split_into_chunks(documents)

    vector_store = load_vector_store()
    if vector_store is None:
        vector_store = FAISS.from_documents(children, get_embedding_model())
    else:
        vector_store.add_documents(children)
    vector_store.save_local(str(config.FAISS_DIR))

    doc_store = load_doc_store()
    doc_store["files"] += file_names
    doc_store["parents"].update(parents)
    doc_store["children"] += [{"text": c.page_content, "metadata": c.metadata} for c in children]
    save_doc_store(doc_store)

    return len(parents), len(children)


def get_file_summaries() -> dict:
    """
    Return a short overview of every added file, writing the ones that are missing.

    The LLM writes each overview from the start of the file. It helps with broad questions like
    "what is this book about?", which the few chunks found by search cannot answer well.
    """
    doc_store = load_doc_store()
    summaries = doc_store["summaries"]
    added = 0

    for name in doc_store["files"]:
        if name in summaries:
            continue

        text = ""
        for parent in doc_store["parents"].values():
            if parent["metadata"].get("source") == name:
                text += parent["text"] + "\n"
                if len(text) >= config.SUMMARY_INPUT_CHARS:
                    break

        prompt = SUMMARY_PROMPT.format(name=name, text=text[:config.SUMMARY_INPUT_CHARS])
        try:
            summaries[name] = get_chat_model().invoke(prompt).content.strip()
            added += 1
        except Exception as e:
            # not fatal, it is tried again the next time
            print(f"[ingest] Could not write an overview of {name}: {e}")

    if added:
        save_doc_store(doc_store)
    return summaries


def remove_file(file_name: str):
    """Remove one file: its child chunks in FAISS, its chunks in the doc store and the file itself."""
    doc_store = load_doc_store()
    if doc_store["files"] == [file_name]:
        clear_all()
        return

    vector_store = load_vector_store()
    ids = [
        doc_id for doc_id in vector_store.index_to_docstore_id.values()
        if vector_store.docstore.search(doc_id).metadata.get("source") == file_name
    ]
    if ids:
        vector_store.delete(ids)
    vector_store.save_local(str(config.FAISS_DIR))

    doc_store["files"].remove(file_name)
    doc_store["summaries"].pop(file_name, None)
    doc_store["parents"] = {
        parent_id: parent for parent_id, parent in doc_store["parents"].items()
        if parent["metadata"].get("source") != file_name
    }
    doc_store["children"] = [c for c in doc_store["children"] if c["metadata"].get("source") != file_name]
    save_doc_store(doc_store)

    (config.DOCUMENT_DIR / file_name).unlink(missing_ok=True)


def clear_all():
    """Delete all uploaded files, the FAISS index and the doc store."""
    for folder in (config.DOCUMENT_DIR, config.FAISS_DIR):
        for file in folder.iterdir():
            file.unlink()

    config.DOC_STORE_PATH.unlink(missing_ok=True)
