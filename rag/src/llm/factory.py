"""Single place to build Ollama clients (embeddings + LLM)."""
from functools import lru_cache

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from src.config import settings


@lru_cache(maxsize=1)
def build_embed_model() -> OllamaEmbedding:
    """Build the Ollama embedding client.

    ``keep_alive`` keeps the model warm between requests -- without it,
    Ollama unloads the model after 5 min of idleness, then the next
    batch takes ~10s to reload.

    ``parallel`` enables concurrent HTTP requests at the library level
    when LlamaIndex fans out (e.g. batch embed calls).
    """
    import os

    keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    parallel = int(os.getenv("OLLAMA_EMBED_PARALLEL", "4")) or 0
    return OllamaEmbedding(
        model_name=settings.embed_model,
        base_url=settings.ollama_base_url,
        keep_alive=keep_alive,
        parallel=parallel,
    )


@lru_cache(maxsize=1)
def build_llm() -> Ollama:
    """Build the Ollama LLM client.

    ``thinking=False`` prevents llama-index's chat wrapper from passing the
    ``think`` kwarg to the underlying ollama-py client. That kwarg was
    silently ignored in older ollama-py versions but is now rejected
    by ollama-py 0.6+ (it expects ``think`` inside ``options``).
    """
    return Ollama(
        model=settings.gen_model,
        base_url=settings.ollama_base_url,
        thinking=False,
    )


def reset_cache() -> None:
    """Useful in tests."""
    build_embed_model.cache_clear()
    build_llm.cache_clear()
