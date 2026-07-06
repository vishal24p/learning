r"""Full RAG demo: RRF pool -> reranker -> Generator.

Reads the query from argv if given, otherwise prompts on stdin.
Prints the RRF pool, the reranked top-N, then the generator's answer.

Usage:
    python scripts\run_rag.py                # interactive prompt
    python scripts\run_rag.py "your question?"  # one-shot
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.generation.generator import Generator  # noqa: E402
from src.guard.pipeline import guard_or_terminate  # noqa: E402
from src.guard.reason import PipelineTerminated  # noqa: E402
from src.logging_setup import setup_logging  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.retrieval.reranked_retriever import RerankedRetriever  # noqa: E402
from src.retrieval.reranker import filter_positive_chunks  # noqa: E402

setup_logging()
log = logging.getLogger("run_rag")

DEFAULT_QUERY = "How do H1, H2, H3 heading levels work in markdown?"
PROMPT_LABEL = "[you]"


def resolve_query(argv: list[str]) -> str:
    if len(argv) > 1 and argv[1].strip():
        return " ".join(argv[1:]).strip()
    try:
        return input(f"{PROMPT_LABEL} ").strip()
    except EOFError:
        return ""


def show(name: str, hits: list[dict]) -> None:
    print(f"\n== {name} ({len(hits)} hits) ==")
    for i, h in enumerate(hits, 1):
        preview = h["content"].replace("\n", " ")[:160]
        print(f"  {i}. id={h['chunk_id']:<5}  score={h['score']:.4f}  | {preview}...")


def main() -> int:
    query = resolve_query(sys.argv) or DEFAULT_QUERY
    print(f"\n[query] {query}")
    print(f"[cfg] model={settings.gen_model} temp={settings.gen_temperature} "
          f"max_tokens={settings.gen_max_tokens} "
          f"guard={'on' if settings.guard_enabled else 'off'} "
          f"guard_model={settings.guard_model}")

    # Pre-pipeline guard. If it terminates, we never embed, never query
    # Postgres, never call the generator. Just print the refusal and quit.
    try:
        query = guard_or_terminate(query)
    except PipelineTerminated as exc:
        log.warning("pipeline terminated by guard: %s", exc.decision.reason)
        print("\n== GUARD: TERMINATE ==")
        print(f"reason : {exc.decision.reason}")
        print(f"refusal: {exc.decision.refusal}")
        return 0

    hybrid = HybridRetriever(settings.db_url)
    rr = RerankedRetriever(settings.db_url)
    gen = Generator()

    log.info("running pipeline...")
    rrf_pool = hybrid.retrieve(
        query,
        top_k_dense=10,
        top_k_sparse=10,
        top_k=settings.rerank_candidates,
    )
    show("RRF pool", rrf_pool)

    rrf_again, reranked = rr.retrieve(
        query,
        top_k_dense=10,
        top_k_sparse=10,
        candidates=settings.rerank_candidates,
        top_n=settings.rerank_top_n,
    )
    show("RRF (top 5)", rrf_again[: settings.rerank_top_n])
    show("RERANKED (final top 5)", reranked)

    # Only feed chunks the reranker is *positive* about. If all five were
    # non-positive, we still answer with the single best-ranked chunk so the
    # LLM has something to ground on instead of refusing silently.
    filtered = filter_positive_chunks(reranked, fallback_top_n=1)
    show("RERANKED -> POSITIVE ONLY (fed to LLM)", filtered)

    answer = gen.generate(query, filtered)
    print("\n== ANSWER ==")
    print(answer)
    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
