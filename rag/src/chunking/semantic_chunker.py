"""Semantic chunking via LlamaIndex + Ollama (shared factory).

Defaults are tuned for technical documentation: a larger buffer reduces
false splits, an 80th-percentile threshold catches real semantic shifts
without forcing one. Either can be overridden via env vars
(``BUFFER_SIZE``, ``BREAKPOINT_PERCENTILE``) without code changes.
"""
from __future__ import annotations

import os

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser

from src.llm.factory import build_embed_model


DEFAULT_BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "5"))
DEFAULT_BREAKPOINT_PERCENTILE = int(os.getenv("BREAKPOINT_PERCENTILE", "80"))


class SemanticChunker:
    def __init__(
        self,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        breakpoint_percentile_threshold: int = DEFAULT_BREAKPOINT_PERCENTILE,
    ) -> None:
        self.embed_model = build_embed_model()
        self.parser = SemanticSplitterNodeParser(
            embed_model=self.embed_model,
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        )

    def chunk_text(self, text: str) -> list:
        return self.parser.get_nodes_from_documents([Document(text=text)])
