import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def get_int(name: str, default: int) -> int:
    """
    Read an integer setting from the environment.

    Everything in a `.env` file is text, so "400" arrives as the string "400".
    This helper converts it to a real integer and falls back to the default if the variable is missing or is not a valid number.

    Args:
        name: Environment variable name, for example "CHILD_CHUNK_SIZE".
        default: Value to use when the variable is missing or invalid.

    Returns:
        The setting as an integer.
    """

    raw_value = os.getenv(name)

    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError:
        print(f"[config] '{name} = {raw_value}' is not a number. Using the value -> {default}")
        return default


def get_float(name: str, default: float) -> float:
    """
    Read a decimal setting from the environment, for example "-4.0".

    Works the same way as `get_int`, but converts the value to a float.

    Args:
        name: Environment variable name, for example "MIN_RERANK_SCORE".
        default: Value to use when the variable is missing or invalid.

    Returns:
        The setting as a float.
    """

    raw_value = os.getenv(name)

    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return float(raw_value)
    except ValueError:
        print(f"[config] '{name} = {raw_value}' is not a number. Using the value -> {default}")
        return default


#folders
PROJECT_ROOT = Path(__file__).parent

DOCUMENT_DIR = PROJECT_ROOT / "data" / "documents"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
FAISS_DIR = VECTORSTORE_DIR / "faiss"

DOC_STORE_PATH = VECTORSTORE_DIR / "doc_store.json"

#LLM Models
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_TEMPERATURE = 0.0

# FAISS
VECTOR_DB = "faiss"

# Chunking (parent-child)
# Small child chunks are embedded and searched; the bigger parent chunk is sent to the LLM.
PARENT_CHUNK_SIZE = get_int("PARENT_CHUNK_SIZE", 1600)
PARENT_CHUNK_OVERLAP = get_int("PARENT_CHUNK_OVERLAP", 200)
CHILD_CHUNK_SIZE = get_int("CHILD_CHUNK_SIZE", 400)
CHILD_CHUNK_OVERLAP = get_int("CHILD_CHUNK_OVERLAP", 60)

# Retrieval
NUM_QUERY_VARIANTS = get_int("NUM_QUERY_VARIANTS", 3)
VECTOR_TOP_K = get_int("VECTOR_TOP_K", 6)
BM25_TOP_K = get_int("BM25_TOP_K", 6)
HYBRID_TOP_K = get_int("HYBRID_TOP_K", 10)

# Reranking
RERANKER = os.getenv("RERANKER", "cross-encoder").strip().lower()
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_K = get_int("RERANK_TOP_K", 4)
MIN_RERANK_SCORE = get_float("MIN_RERANK_SCORE", -4.0)

# Generation and memory
MAX_CONTEXT_CHARS = get_int("MAX_CONTEXT_CHARS", 8000)
MEMORY_WINDOW = get_int("MEMORY_WINDOW", 4)

# File overviews: how much text from the start of each file the LLM reads to describe it
SUMMARY_INPUT_CHARS = get_int("SUMMARY_INPUT_CHARS", 8000)


# Make sure the folders exist before anything reads or writes them
for folder in (DOCUMENT_DIR, EVALUATION_DIR, FAISS_DIR):
    folder.mkdir(parents=True, exist_ok=True)


def describe_settings() -> dict:
    return {
        "Chat Model": LLM_MODEL,
        "Embedding Model": EMBEDDING_MODEL,
        "Vector DB": VECTOR_DB,
        "Parent Chunk Size": PARENT_CHUNK_SIZE,
        "Child Chunk Size": CHILD_CHUNK_SIZE
    }