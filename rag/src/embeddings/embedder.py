"""Embed chunks using Ollama via the shared factory."""
from __future__ import annotations

from src.llm.factory import build_embed_model


class Embedder:
    def __init__(self) -> None:
        self.embed_model = build_embed_model()

    def embed_chunk(self, chunk_text: str) -> list[float]:
        result = self.embed_model.get_text_embedding(chunk_text)
        return result[0] if isinstance(result, tuple) else result

    def embed_chunks(self, chunks) -> list[dict]:
        """Embed a list of chunks. Accepts objects with `.text` OR plain strings.

        Uses Ollama's native batch embedding (``get_text_embedding_batch``)
        which makes a single HTTP POST containing all inputs; the server
        then pipelines them. Falls back to per-chunk calls if the batch
        API is unavailable on this version.
        """
        texts = [chunk.text if hasattr(chunk, "text") else str(chunk) for chunk in chunks]
        vectors = self._embed_texts(texts)
        return [{"text": t, "embedding": v} for t, v in zip(texts, vectors)]

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        batch_method = getattr(self.embed_model, "get_text_embedding_batch", None)
        if callable(batch_method):
            result = batch_method(texts)
            # Some client versions return a list directly; some return a tuple per item.
            if result and isinstance(result[0], (list, tuple)):
                return list(result)
        # Fall back: call get_text_embedding one at a time.
        return [self.embed_chunk(t) for t in texts]
