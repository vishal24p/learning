"""Query rewriting: rewrite the user's single query into a clearer search query.

Runs **after** the guard says ``safe`` and **before** any retrieval.
Takes one input query, returns one output query. No multi-query expansion.

The rewritten query is used only for retrieval (dense + sparse + rerank).
The original user query is still passed to the Generator so the LLM
answers the question the user actually asked.

Fail-safe: if the LLM call fails or returns an empty string, the original
query is returned unchanged so the pipeline never breaks.
"""
from __future__ import annotations

import logging
import re
import time

from llama_index.core.llms import ChatMessage, MessageRole

from src.config import settings
from src.llm.factory import build_rewrite_llm

log = logging.getLogger(__name__)

_ROLE_PREFIX_RE = re.compile(r"^(assistant|user|system)\s*:\s*", re.IGNORECASE)


REWRITE_SYSTEM_PROMPT = (
    "You are a query rewriter for a Kubernetes documentation RAG system.\n"
    "Your job: rewrite the user's question into a single, clear,\n"
    "self-contained search query that will retrieve relevant documentation.\n\n"
    "Rules:\n"
    "- Output ONE query only. No explanations, no preamble.\n"
    "- Keep it concise (under 30 words).\n"
    "- Preserve the user's core intent.\n"
    "- Expand abbreviations if they help retrieval.\n"
    "- Do not add information the user did not ask about.\n"
)


def _clean_rewritten_query(text: str) -> str:
    """Normalize model output into one plain query string.

    Some chat wrappers stringify responses as ``assistant: ...``. That prefix
    is not part of the user's search intent and can break ParadeDB BM25 syntax
    because ``:`` is a query operator. Keep this cleanup narrow: remove common
    role labels, trim quote wrappers, and collapse multiline output to spaces.
    """
    cleaned = " ".join(text.strip().split())
    cleaned = cleaned.strip("\"'`")
    while True:
        next_cleaned = _ROLE_PREFIX_RE.sub("", cleaned).strip()
        if next_cleaned == cleaned:
            break
        cleaned = next_cleaned.strip("\"'`")
    return cleaned


class QueryRewriter:
    """Rewrite a single query via an Ollama LLM call."""

    def __init__(self, llm=None) -> None:
        self._llm = llm  # injected for tests

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        return build_rewrite_llm()

    def rewrite(self, query: str) -> str:
        """Return a rewritten query. Falls back to original on any error."""
        if not query.strip():
            return query

        t = time.perf_counter()
        try:
            llm = self._get_llm()
            system_msg = ChatMessage(role=MessageRole.SYSTEM, content=REWRITE_SYSTEM_PROMPT)
            user_msg = ChatMessage(role=MessageRole.USER, content=query)
            resp = llm.chat([system_msg, user_msg])
            rewritten = _clean_rewritten_query(str(resp))

            if not rewritten:
                log.warning("query rewrite returned empty, using original: %r", query[:80])
                return query

            log.info(
                "query rewrite: %r -> %r (%.0f ms)",
                query[:80], rewritten[:80],
                (time.perf_counter() - t) * 1000.0,
            )
            return rewritten
        except Exception as exc:
            log.warning("query rewrite failed, using original: %s", exc)
            return query


def rewrite_query(query: str, *, rewriter: QueryRewriter | None = None) -> str:
    """Rewrite a single query. Returns the (possibly unchanged) query.

    Honors ``settings.query_rewrite_enabled``: when False, the original
    query is returned without any Ollama call.
    """
    if not settings.query_rewrite_enabled:
        log.debug("query rewrite disabled via QUERY_REWRITE_ENABLED=false; passing query through")
        return query

    r = rewriter or QueryRewriter()
    return r.rewrite(query)
