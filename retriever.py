import re
from functools import lru_cache
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import config
from ingest import load_doc_store, load_vector_store
from llmClient import get_chat_model


QUERY_PROMPT = """You help search a collection of documents.
Write {n} different versions of the user's question to find the right information.
The first version must be the question rewritten so it makes sense on its own,
use the chat history to replace words like "it" or "that" with what they refer to.
If the question is not in English (for example Roman Urdu), write the versions in English,
because most documents are in English.
Write one question per line, nothing else.

Chat history:
{history}

Question: {question}"""


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(config.CROSS_ENCODER_MODEL)


@lru_cache(maxsize=1)
def load_search_indexes(version: int):
    """
    Load FAISS, the doc store and the BM25 index once and keep them in memory.
    `version` is the time doc_store.json was last saved, so they are loaded again only after the documents change.
    """
    vector_store = load_vector_store()
    doc_store = load_doc_store()

    children = [Document(page_content=c["text"], metadata=c["metadata"]) for c in doc_store["children"]]
    bm25 = BM25Retriever.from_documents(children, k=config.BM25_TOP_K) if children else None

    return vector_store, doc_store, bm25


def get_search_indexes():
    """Return the FAISS store, the doc store and BM25, or three Nones if no document is added yet."""
    if not config.DOC_STORE_PATH.exists():
        return None, None, None
    return load_search_indexes(config.DOC_STORE_PATH.stat().st_mtime_ns)


def make_queries(question: str, history: str):
    """
    Multi-query: ask the LLM for a few versions of the question.
    A different wording can find chunks that the original question misses.

    Returns the list of queries to search with and a standalone version of the question.
    """
    if config.NUM_QUERY_VARIANTS <= 0:
        return [question], question

    prompt = QUERY_PROMPT.format(n=config.NUM_QUERY_VARIANTS, history=history, question=question)
    response = get_chat_model().invoke(prompt).content

    variants = []
    for line in response.split("\n"):
        # remove numbering like "1." or "-" in case the LLM adds it
        line = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line).strip()
        if line:
            variants.append(line)
    variants = variants[:config.NUM_QUERY_VARIANTS]

    standalone_question = variants[0] if variants else question
    return [question] + variants, standalone_question


def hybrid_search(queries: list[str], vector_store, bm25) -> list[Document]:
    """
    Search every query with FAISS (meaning) and BM25 (exact keywords),
    then merge all result lists with Reciprocal Rank Fusion:
    a chunk gets 1 / (60 + rank) points from each list it appears in, so chunks found many times win.
    """
    scores = {}
    chunks = {}

    for query in queries:
        vector_results = vector_store.similarity_search(query, k=config.VECTOR_TOP_K)
        keyword_results = bm25.invoke(query)

        for results in (vector_results, keyword_results):
            for rank, chunk in enumerate(results):
                key = (chunk.metadata["parent_id"], chunk.page_content)
                scores[key] = scores.get(key, 0) + 1 / (60 + rank)
                chunks[key] = chunk

    best_keys = sorted(scores, key=scores.get, reverse=True)[:config.HYBRID_TOP_K]
    return [chunks[key] for key in best_keys]


def rerank(question: str, chunks: list[Document]) -> list[Document]:
    """
    Score each chunk against the question with a cross-encoder, which reads both together
    and is more accurate than vector search. Low scoring chunks are dropped.
    """
    if config.RERANKER != "cross-encoder" or not chunks:
        return chunks

    scores = get_reranker().predict([(question, chunk.page_content) for chunk in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)

    return [chunk for score, chunk in ranked if score >= config.MIN_RERANK_SCORE]


def get_parent_chunks(child_chunks: list[Document], doc_store: dict) -> list[Document]:
    """Replace each child chunk with its bigger parent chunk, without repeating a parent."""
    parents = []
    used_ids = set()

    for child in child_chunks:
        parent_id = child.metadata["parent_id"]
        if parent_id in used_ids:
            continue
        used_ids.add(parent_id)

        parent = doc_store["parents"][parent_id]
        parents.append(Document(page_content=parent["text"], metadata=parent["metadata"]))

        if len(parents) == config.RERANK_TOP_K:
            break

    return parents


def retrieve(question: str, history: str) -> list[Document]:
    """Full retrieval: multi-query -> hybrid search -> rerank -> parent chunks."""
    vector_store, doc_store, bm25 = get_search_indexes()
    if vector_store is None or bm25 is None:
        return []

    queries, standalone_question = make_queries(question, history)
    chunks = hybrid_search(queries, vector_store, bm25)
    chunks = rerank(standalone_question, chunks)

    return get_parent_chunks(chunks, doc_store)
