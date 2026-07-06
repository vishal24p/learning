"""Llama-Guard3 protocol for the guard pipeline.

The Ollama wrapper for ``llama-guard3:1b`` does NOT support the
``tools`` parameter (verified via a 400 response). Llama Guard's native
format is instead a *one-token classifier response*: the model emits
``safe`` or ``unsafe`` as plain text. When ``unsafe``, the standard
Llama Guard prompt spec also asks the model to add a comma-separated
list of violated categories on the next line (e.g.
``S1, S2``, ``S5``). We extract both lines.

This module is the *only* place that knows about that protocol. The rest
of the codebase (``guard_or_terminate``, ``GuardDecision``) stays
unchanged.
"""
from __future__ import annotations

import logging
import time

from ollama import chat as ollama_chat
from ollama import ChatResponse

from src.config import settings
from src.guard.reason import (
    ACTION_SAFE,
    ACTION_TERMINATE,
    GuardDecision,
)

log = logging.getLogger(__name__)


# Category codes from the Llama Guard 3 spec (kept short for the log).
# https://llama.meta.com/docs/model-cards-and-prompt-formats/llama-guard-3/
CATEGORY_NAMES: dict[str, str] = {
    "S1": "violent-crimes",
    "S2": "non-violent-crimes",
    "S3": "sex-related-crimes",
    "S4": "child-sexual-exploitation",
    "S5": "defamation",
    "S6": "specialized-advice",
    "S7": "privacy",
    "S8": "intellectual-property",
    "S9": "hate",
    "S10": "self-harm",
    "S11": "sexual-content",
    "S12": "election-interference",
    "S13": "code-interpreter-abuse",
}


# Llama Guard 3 system prompt. This is *the* prompt format the model was
# trained on; we send it verbatim.
LLAMA_GUARD3_SYSTEM_PROMPT: str = (
    "You are a classifier for a Kubernetes-documentation Q&A bot. "
    "Your job is to classify whether the user's last user-prompt is "
    "safe to answer with a Kubernetes-documentation RAG response.\n\n"
    "Respond ONLY with one word on the first line: ``safe`` or ``unsafe``.\n"
    "If ``unsafe``, on the second line you MAY list comma-separated "
    "Llama Guard 3 category codes (S1..S13) that the prompt violates.\n"
    "Do not write anything else."
)

DEFAULT_REFUSAL: str = (
    "I can't help with that, but I can answer Kubernetes "
    "documentation questions. Please rephrase your request."
)


class LlamaGuard3:
    """Thin wrapper around Ollama's chat-completions for llama-guard3:1b.

    The model returns ``safe`` or ``unsafe`` (plus optionally one line of
    category codes). We map that onto ``GuardDecision``. Anything that
    is NOT exactly ``safe`` on the first line is treated as ``unsafe``
    (fail-closed).
    """

    def __init__(
        self,
        model: str | None = None,
        client=None,
    ) -> None:
        self.model = model or settings.guard_model
        self._client = client  # injected for tests

    def decide(self, query: str) -> GuardDecision:
        """Run the guard once on `query` and return a structured decision."""
        t = time.perf_counter()
        try:
            response: ChatResponse = self._call_ollama(query)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "guard ollama call failed (%.0f ms): %s",
                (time.perf_counter() - t) * 1000.0,
                exc,
            )
            return self._fail_closed(reason="guard-unreachable")

        decision = self._parse_response(response)
        log.info(
            "guard %s (%.0f ms) reason=%r",
            decision.action,
            (time.perf_counter() - t) * 1000.0,
            decision.reason,
        )
        return decision

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _call_ollama(self, query: str) -> ChatResponse:
        """One HTTP POST /api/chat; *no* tools parameter."""
        messages = [
            {"role": "system", "content": LLAMA_GUARD3_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        if self._client is not None:
            return self._client(model=self.model, messages=messages)
        return ollama_chat(model=self.model, messages=messages)

    @staticmethod
    def _parse_response(response: ChatResponse) -> GuardDecision:
        """Map the raw ChatResponse text onto a GuardDecision.

        Acceptance rule: the *first non-empty line*, when lowercased and
        stripped, must be exactly the word ``safe``. Everything else,
        including empty content, *trailing punctuation* (e.g. ``safe.``),
        additional sentences, or unexpected tokens, is treated as
        ``unsafe`` (fail-closed).
        """
        message = getattr(response, "message", None)
        if message is None:
            return LlamaGuard3._fail_closed(reason="guard-empty-response")

        raw = (getattr(message, "content", None) or "").strip()
        if not raw:
            return LlamaGuard3._fail_closed(reason="guard-empty-response")

        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        head = lines[0].lower()

        # Strip a trailing period (some model builds emit "safe.").
        head_clean = head.rstrip(".").strip()

        if head_clean == "safe":
            return GuardDecision(
                action=ACTION_SAFE,
                reason="llama-guard3:safe",
            )

        # Unsafe path: parse the optional category list on line 2.
        categories = []
        if len(lines) >= 2:
            categories = [
                code.strip().upper()
                for code in lines[1].split(",")
                if code.strip()
            ]

        reason = LlamaGuard3._build_reason(head_clean, categories)
        return GuardDecision(
            action=ACTION_TERMINATE,
            reason=reason,
            refusal=DEFAULT_REFUSAL,
        )

    @staticmethod
    def _fail_closed(reason: str) -> GuardDecision:
        return GuardDecision(
            action=ACTION_TERMINATE,
            reason=reason,
            refusal=DEFAULT_REFUSAL,
        )

    @staticmethod
    def _build_reason(head: str, categories: list[str]) -> str:
        if not categories:
            return f"llama-guard3:unsafe:{head[:32] or 'unclassified'}"
        names = [CATEGORY_NAMES.get(c, c) for c in categories]
        codes = ",".join(categories)
        return f"llama-guard3:unsafe:{codes}:{','.join(names)}"
