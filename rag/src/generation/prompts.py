"""Prompt templates for the Generator.

Two templates:
- SYSTEM_PROMPT: system-role instructions (always present).
- QA_TEMPLATE:   user-role message that interleaves labelled passages and asks
                 for a cited answer. The placeholder for the passages block
                 and the question are written here once; Generator fills them
                 in at call time.
"""
from __future__ import annotations

from llama_index.core import PromptTemplate


SYSTEM_PROMPT = (
    "You are a question-answering assistant.\n"
    "Answer ONLY using the provided context passages.\n"
    "If the answer is not in the passages, reply exactly: I don't know.\n"
    "Do not invent facts. Do not rely on prior knowledge.\n"
    "Cite sources inline with [n] labels that match the passage labels in the"
    " user message."
)


QA_TEMPLATE = PromptTemplate(
    "Passages (each block is labelled; cite sources as [n] where n is the label):\n"
    "{context}\n\n"
    "Question:\n{query}\n\n"
    "Answer in 3-6 sentences with inline citations."
)


def format_context(chunks: list[dict]) -> str:
    """Render passages as labelled blocks: [1] <content>, [2] <content>, ..."""
    lines: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        text = ch["content"].replace("\n", " ").strip()
        lines.append(f"[{i}] {text}")
    return "\n\n".join(lines)
