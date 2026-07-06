"""RAGAS judge: wraps our local Ollama LLM (and embeddings) for ragas.

For the embedding side we reuse our existing Ollama embedding model
(settings.embed_model, e.g. nomic-embed-text-v2-moe) so no extra model
downloads sit in our cache.

Compatibility note (read this if ragas fails to import)
=======================================================
ragas == 0.2.7 still does
    from langchain_community.chat_models.vertexai import ChatVertexAI
but langchain-community >= 0.4 removed that path. We register a
``sys.modules`` stub under that exact name so the import succeeds
without installing google-cloud-aiplatform. The stub class is never
actually instantiated because we use our local Ollama judge via
LangchainLLMWrapper.
"""
from __future__ import annotations

import os
import sys
import types


def _install_vertexai_stub() -> None:
    """Register a ``langchain_community.chat_models.vertexai`` stub module.

    If the real module exists (older langchain-community), this is a no-op.
    Otherwise, drop a placeholder ``ChatVertexAI`` class into sys.modules
    so the ragas import chain succeeds. The placeholder raises a clear
    error if anyone tries to instantiate one.
    """
    if "langchain_community.chat_models.vertexai" in sys.modules:
        return
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
        return
    except Exception:
        pass

    stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class _ChatVertexAI:
        """Stub for ragas 0.2.x compatibility.

        ragas references this class by name only. On our Ollama code
        path it is never constructed.
        """

        def __init__(self, *args, **kwargs):  # pragma: no cover
            raise RuntimeError(
                "ChatVertexAI stub hit. The RAGAS Ollama judge does not "
                "need it; investigate which ragas code path is reaching "
                "this construct."
            )

    stub.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = stub


_install_vertexai_stub()

from langchain_ollama import ChatOllama, OllamaEmbeddings  # noqa: E402
from langchain_ollama.chat_models import ChatOllama as _ChatOllamaClass  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402

from src.config import settings  # noqa: E402


# Workaround for langchain-ollama >= 0.4 / ollama-py >= 0.5:
# Both ``ChatOllama._chat_params`` and ``ollama.AsyncClient.chat`` reject
# the legacy ``temperature`` keyword. RAGAS 0.2.x still passes it through.
# We monkey-patch the chat-params builder so that ``temperature`` is
# pulled out of kwargs and folded into the ``options`` dict (where the
# new ollama-py client actually expects it).
_orig_chat_params = _ChatOllamaClass._chat_params


def _patched_chat_params(self, messages, stop=None, **kwargs):
    if "temperature" in kwargs and "options" not in kwargs:
        kwargs["options"] = {"temperature": kwargs.pop("temperature")}
    return _orig_chat_params(self, messages, stop=stop, **kwargs)


_ChatOllamaClass._chat_params = _patched_chat_params


def build_judge_llm() -> LangchainLLMWrapper:
    """Wrap our local Ollama chat model as a RAGAS-compatible judge LLM."""
    os.environ.setdefault("OPENAI_API_KEY", "not-set")  # ragas import path only
    chat = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.judge_model,
        temperature=settings.judge_temperature,
        # RAGAS 0.2.x calls ``generate_prompt(temperature=...)`` directly; in
        # newer langchain-ollama that needs to flow through client kwargs,
        # not the pydantic field. We pre-bind temperature via model config
        # above so it is consistent for sync and async paths.
    )
    # Disable ragas's deprecated ``timeout`` kwarg for ollama-py paths; nothing
    # to do here -- we just keep ChatOllama's defaults.
    return LangchainLLMWrapper(chat)


def build_judge_embeddings():
    """Reuse the existing Ollama embedding model (e.g. nomic-embed-text-v2-moe).

    No new downloads; sends a /api/embed HTTP call per query.
    """
    emb = OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.embed_model,
    )
    return LangchainEmbeddingsWrapper(emb)
