"""Cross-encoder reranker over retrieved chunks.

Wraps sentence-transformers CrossEncoder. Lazy-loads the model on first call.
If the model can't be imported (no internet / fails), `rerank()` falls back to
identity ordering so the pipeline still completes during local debugging.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Callable

from src.config import settings

log = logging.getLogger(__name__)


def _softmax(scores: list[float]) -> list[float]:
    """Return numerically stable softmax probabilities for ``scores``."""
    if not scores:
        return []
    max_score = max(scores)
    exponentials = [math.exp(score - max_score) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str | None = None,
        score_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self.model_name = model_name or settings.rerank_model
        self._model = None
        self._score_fn = score_fn  # allows tests to inject a stub

    def _ensure_model(self) -> None:
        if self._model is not None or self._score_fn is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            t = time.perf_counter()
            log.info("loading CrossEncoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
            log.info("cross-encoder loaded: %s (%.1f ms)",
                     self.model_name, (time.perf_counter() - t) * 1000.0)
        except Exception as exc:  # noqa: BLE001
            log.warning("cross-encoder unavailable, identity fallback: %s", exc)
            self._model = None

    def rerank(self, query: str, candidates: list[dict], top_n: int | None = None) -> list[dict]:
        if not candidates:
            return []

        t = time.perf_counter()
        self._ensure_model()
        keep = top_n if top_n is not None else settings.rerank_top_n

        # Compute scores
        if self._score_fn is not None:
            scores = [float(self._score_fn(query, c["content"])) for c in candidates]
            log.info("rerank via injected score_fn (fallback/test)")
        elif self._model is not None:
            t_pred = time.perf_counter()
            pairs = [(query, c["content"]) for c in candidates]
            log.info("rerank pairs=%d (truncated text per pair below)",
                     len(pairs))
            for i, (q, c) in enumerate(pairs):
                log.debug("  pair[%d] q=%r c=%r", i, q[:60], c[:60])
            raw = self._model.predict(pairs, convert_to_numpy=True)
            try:
                scores = [float(x) for x in raw.tolist()]
            except AttributeError:
                scores = [float(x) for x in raw]
            log.info("cross-encoder predict scored=%d pairs (%.1f ms)",
                     len(scores), (time.perf_counter() - t_pred) * 1000.0)
        else:
            log.info("rerank kept in RRF order (identity fallback) in=%d out=%d",
                     len(candidates), keep)
            return candidates[:keep]

        if settings.rerank_use_softmax:
            scores = _softmax(scores)
            log.info("rerank applied softmax score normalization")

        scored = [
            {"chunk_id": c["chunk_id"], "content": c["content"], "score": float(s)}
            for c, s in zip(candidates, scores)
        ]
        scored.sort(key=lambda r: r["score"], reverse=True)
        out = scored[:keep]
        top3 = ", ".join(f"{r['score']:.2f}" for r in out)
        log.info(
            "rerank in=%d out=%d top=%s (total %.1f ms)",
            len(candidates), len(out), top3,
            (time.perf_counter() - t) * 1000.0,
        )
        log.debug("rerank scores (full): %s", [round(s, 4) for s in scores])
        return out
