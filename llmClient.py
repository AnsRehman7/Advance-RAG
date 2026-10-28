from functools import lru_cache
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
import config

@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """
    Build the chat model used to generate answers.

    Reads LLM_MODEL, LLM_API_KEY and LLM_TEMPERATURE from `config`, so you can switch
    models by editing `.env` instead of the code.

    Returns:
        A LangChain chat model that is ready to call with `.invoke()`.
    """
    return init_chat_model(
        model = config.LLM_MODEL,
        temperature = config.LLM_TEMPERATURE,
        api_key = config.LLM_API_KEY,
        model_provider='groq'
    )

@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """
    Build the embedding model that turns text into vectors for FAISS.

    Runs EMBEDDING_MODEL from `config` locally on the CPU, so it needs no API key.
    Vectors are normalized, which makes FAISS similarity scores behave like cosine similarity.

    Returns:
        A LangChain embedding model with `.embed_documents()` and `.embed_query()`.
    """
    return HuggingFaceEmbeddings(
        model_name = config.EMBEDDING_MODEL,
        model_kwargs = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True}
    )
