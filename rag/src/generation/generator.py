"""LLM-based answer generator using Ollama.

Uses two message roles (system + user) via LlamaIndex's chat API. Falls back
to a single prompt if the underlying Ollama model does not support chat.
"""
from __future__ import annotations

import logging
import time

from llama_index.core.llms import ChatMessage, MessageRole

from src.config import settings
from src.generation.prompts import QA_TEMPLATE, SYSTEM_PROMPT, format_context
from src.llm.factory import build_llm

log = logging.getLogger(__name__)


class Generator:
    def __init__(self) -> None:
        self.llm = build_llm()

    def generate(self, query: str, contexts: list[str] | list[dict]) -> str:
        """Build the labelled passages and call the chat LLM.

        `contexts` may be a list of strings or a list of dicts with a "content"
        key (as returned by the retrievers).
        """
        chunks: list[dict] = (
            [{"content": c} if isinstance(c, str) else c for c in contexts]
            if contexts
            else []
        )
        context_block = format_context(chunks)
        user_msg = QA_TEMPLATE.format(context=context_block, query=query)
        log.info("generation start model=%s contexts=%d", settings.gen_model, len(chunks))

        system_msg = ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)
        user_chat = ChatMessage(role=MessageRole.USER, content=user_msg)

        t = time.perf_counter()
        try:
            chat_resp = self.llm.chat([system_msg, user_chat])
            text = str(chat_resp)
            log.info("llm chat OK words=%d (%.1f ms)",
                     len(text.split()), (time.perf_counter() - t) * 1000.0)
            return text
        except Exception as exc:
            log.warning("llm chat failed, fallback complete(): %s", exc)
            t = time.perf_counter()
            full_prompt = f"{SYSTEM_PROMPT}\n\n{user_msg}"
            log.debug("complete prompt chars=%d", len(full_prompt))
            text = str(self.llm.complete(
                full_prompt,
                temperature=settings.gen_temperature,
                max_tokens=settings.gen_max_tokens,
            ))
            log.info("llm complete OK words=%d (%.1f ms)",
                     len(text.split()), (time.perf_counter() - t) * 1000.0)
            return text
