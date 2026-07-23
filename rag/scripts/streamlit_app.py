r"""Streamlit UI for the RAG pipeline (no chat memory).

Layout:
- Title and short subtitle.
- Free-form query input. On Enter / Send:
    * runs Hybrid -> Rerank -> Generator fresh per submission
    * renders the user's question
    * shows a live "Pipeline status" panel + an in-flight log feed on the
      right as the pipeline runs
    * renders the answer
    * expandable panels for RRF pool and reranked top-5
    * citations panel with the [n] labels used by the answer
    * logs panel (kept open by default) with INFO lines from the pipeline

Run from PowerShell with:
    $env:PYTHONPATH = r"C:\Users\visha\rag"
    python -m streamlit run scripts\run_ui.py
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import settings  # noqa: E402
from src.generation.generator import Generator  # noqa: E402
from src.generation.prompts import QA_TEMPLATE, SYSTEM_PROMPT, format_context  # noqa: E402
from src.guard.pipeline import guard_or_terminate  # noqa: E402
from src.guard.reason import PipelineTerminated  # noqa: E402
from src.logging_setup import setup_logging  # noqa: E402
from src.query_rewrite import REWRITE_SYSTEM_PROMPT, rewrite_query  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.retrieval.reranked_retriever import RerankedRetriever  # noqa: E402
from src.ui.keys import derive_ragas_button_key as _derive_ragas_button_key

setup_logging()
_CITE_RE = re.compile(r"\[(\d+)\]")


class _MemoryHandler(logging.Handler):
    """One-shot handler that collects formatted records into a list.

    Captures DEBUG and above; we render the full timeline in the UI.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[str] = []
        self.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(self.format(record))
        except Exception:
            pass


class _StreamingHandler(_MemoryHandler):
    """Forward each record to both records[] AND a Streamlit container."""

    def __init__(self, sink) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            self._sink.text += "\n\n" + self.format(record)
        except Exception:
            pass


def collect_logs_with_status(callable_, status_box, log_box, prefix=""):
    """Run `callable_` while streaming log lines into `log_box`.

    Returns (result, captured_records_list).
    """
    handler = _StreamingHandler(log_box)
    root = logging.getLogger()
    # If the user set LOG_LEVEL=ERROR or higher in env, that's deliberate.
    # We do NOT bypass it; we only use the root logger as an event source.
    root.addHandler(handler)
    handler.setLevel(logging.DEBUG)
    try:
        result = callable_()
        if status_box is not None:
            status_box.update(label=f"{prefix}done", state="complete")
        return result, handler.records
    except Exception as exc:
        if status_box is not None:
            status_box.update(label=f"{prefix}failed: {exc}", state="error")
        raise
    finally:
        root.removeHandler(handler)


@st.cache_resource
def load_components() -> tuple[HybridRetriever, RerankedRetriever, Generator]:
    """Build the pipeline once per Streamlit session.

    Cached so the dense + sparse Postgres connections and the cross-encoder
    aren't reloaded on every keystroke.
    """
    hybrid = HybridRetriever(settings.db_url)
    reranked = RerankedRetriever(settings.db_url)
    generator = Generator()
    return hybrid, reranked, generator


def _chunk_preview(content: str, limit: int = 280) -> str:
    preview = content.replace("\n", " ").strip()
    if len(preview) <= limit:
        return preview
    return f"{preview[:limit]}..."


def render_chunks(name: str, hits: list[dict]) -> None:
    if not hits:
        st.write(f"_{name}_: no results")
        return
    for i, h in enumerate(hits, 1):
        content = h.get("content", "")
        chunk_id = h.get("chunk_id", "<missing>")
        score = h.get("score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        with st.expander(
            f"[{i}] id={chunk_id} score={score_text} - {_chunk_preview(content, 110)}",
            expanded=False,
        ):
            st.markdown(
                f"**rank:** `{i}`  \n"
                f"**chunk_id:** `{chunk_id}`  \n"
                f"**score:** `{score_text}`"
            )
            st.code(content, language="text")


def _render_generator_inspection(
    query: str,
    fed_chunks: list[dict],
    answer: str,
) -> None:
    context_block = format_context(fed_chunks)
    user_prompt = QA_TEMPLATE.format(context=context_block, query=query)

    with st.expander("Generator inspection", expanded=False):
        st.markdown(
            f"**generator model:** `{settings.gen_model}`  \n"
            f"**fed chunks:** `{len(fed_chunks)}`"
        )
        with st.expander("SYSTEM_PROMPT", expanded=False):
            st.code(SYSTEM_PROMPT, language="text")
        with st.expander("Rendered user prompt", expanded=False):
            st.code(user_prompt, language="text")
        with st.expander("Generated answer/output", expanded=False):
            st.code(answer, language="text")


def _render_query_rewrite_inspection(
    query: str,
    rewritten_query: str | None,
) -> None:
    rewritten = rewritten_query or query
    with st.expander("Query rewrite inspection", expanded=False):
        st.markdown(
            f"**enabled:** `{settings.query_rewrite_enabled}`  \n"
            f"**rewrite model:** `{settings.query_rewrite_model}`"
        )
        with st.expander("REWRITE_SYSTEM_PROMPT", expanded=False):
            st.code(REWRITE_SYSTEM_PROMPT, language="text")
        st.markdown("**Original user query**")
        st.code(query, language="text")
        st.markdown("**Rewritten retrieval query**")
        st.code(rewritten, language="text")


def _ragas_input_bundle(
    query: str,
    contexts: list[str],
    answer: str,
) -> dict:
    return {
        "user_input": query,
        "response": answer,
        "retrieved_contexts": contexts,
        "reference": "",
        "reference_contexts": [],
        "judge_model": settings.judge_model,
        "embedding_model": settings.embed_model,
        "metrics": ["faithfulness", "answer_relevancy", "context_precision"],
    }


def extract_used_citations(answer: str) -> list[int]:
    seen = []
    for m in _CITE_RE.finditer(answer):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def main() -> None:
    st.set_page_config(page_title="RAG (Hybrid + Rerank)", layout="wide")
    st.title("RAG with BM25 (ParadeDB) + Dense (pgvector) + Cross-Encoder Rerank")
    st.caption(
        "Each question runs a fresh Hybrid -> Rerank -> Generator pipeline; "
        "no chat memory is kept across submissions."
    )

    # Pipeline config snapshot (read-only chip block)
    with st.expander("Pipeline settings", expanded=False):
        cfg_pairs = [
            ("Embedding model", settings.embed_model),
            ("Reranker model", settings.rerank_model),
            ("Rerank candidates", str(settings.rerank_candidates)),
            ("Rerank top-N", str(settings.rerank_top_n)),
            ("model", settings.gen_model),
            ("temp", str(settings.gen_temperature)),
            ("max tokens", str(settings.gen_max_tokens)),
            ("Query rewrite", "on" if settings.query_rewrite_enabled else "off"),
            ("Query rewrite model", settings.query_rewrite_model),
            ("Guard", "on" if settings.guard_enabled else "off"),
            ("Guard model", settings.guard_model),
        ]
        st.table({k: v for k, v in cfg_pairs})

    query = st.text_input(
        "Question",
        placeholder="e.g. How do H1, H2, H3 heading levels work in markdown?",
        label_visibility="collapsed",
    )
    send = st.button("Ask", type="primary")

    if send and query.strip():
        # Pre-pipeline safety guard. The guard is always the first
        # call that touches Ollama: no embedding of the query, no
        # Postgres query for dense/sparse, no rerank, no generation.
        # On terminate, we render the refusal inline and NOT cache
        # anything to session_state, so the RAGAS button never has
        # a "last_answer" to score against.
        status = st.status("Running guard...", expanded=True)
        live_log = st.empty()

        def do_guard() -> str:
            return guard_or_terminate(query)

        try:
            _, guard_logs = collect_logs_with_status(
                do_guard, None, live_log, prefix="Guard "
            )
        except PipelineTerminated as exc:
            status.update(
                label=f"Guard terminated: {exc.decision.reason}",
                state="error",
            )
            # Keep the refusal on its own -- no retrieval details to
            # show and no citations. Audit log already written inside
            # guard_or_terminate (one INFO line per decision).
            st.markdown("---")
            st.markdown(f"**You:** {query}")
            st.error(f"Refused: {exc.decision.refusal}")
            st.caption(f"Audit reason: `{exc.decision.reason}`")
            return

        # Single-query rewrite for retrieval. Runs after guard => safe,
        # before retrieval. Returns one rewritten query. Original query
        # is preserved for the Generator and the answer display.
        def do_rewrite() -> str:
            return rewrite_query(query)

        status.update(label="Rewriting query...", state="running")
        live_log_rw = st.empty()
        original_query = query
        rewritten_query, rw_logs = collect_logs_with_status(
            do_rewrite, None, live_log_rw, prefix="Rewrite "
        )

        status.update(label="Retrieving...", expanded=True)
        live_log = st.empty()

        hybrid, reranked, generator = load_components()

        log_chunks: list[str] = []
        log_chunks.append("--- guard: safe_to_continue ---")
        log_chunks.extend(rw_logs)

        def do_retrieve() -> tuple[list, list]:
            # Retrieval uses the rewritten query for better recall.
            rrf_pool = hybrid.retrieve(
                rewritten_query,
                top_k_dense=10,
                top_k_sparse=10,
                top_k=settings.rerank_candidates,
            )
            _, top5 = reranked.retrieve(
                rewritten_query,
                top_k_dense=10,
                top_k_sparse=10,
                candidates=settings.rerank_candidates,
                top_n=settings.rerank_top_n,
            )
            return rrf_pool, top5

        (rrf_pool, top5), recv_logs = collect_logs_with_status(
            do_retrieve, None, live_log, prefix="Retrieval "
        )
        log_chunks.extend(recv_logs)

        status.update(label="Generating answer...", state="running")
        # Generator answers the user's *original* query, not the rewritten one.
        answer, recv_logs = collect_logs_with_status(
            lambda: generator.generate(original_query, top5),
            None, live_log, prefix="Generation ",
        )
        log_chunks.extend(recv_logs)
        status.update(label="Answer ready", state="complete")

        # Persist last run so it survives the rerun that follows any RAGAS click.
        st.session_state["last_query"] = original_query
        st.session_state["last_rewritten_query"] = rewritten_query
        st.session_state["last_top5"] = top5
        st.session_state["last_answer"] = answer
        st.session_state["last_rrf_pool"] = rrf_pool
        st.session_state["last_log_chunks"] = log_chunks

    # Re-render last result + RAGAS button on every rerun. Single render site.
    last = _read_last_result()
    if last:
        _render_last_result(**last)
        _render_ragas_eval_block(last["query"], last["top5"])


def _read_last_result() -> dict | None:
    """Pull the most recent run from session_state; None if absent."""
    if "last_query" not in st.session_state:
        return None
    return {
        "query": st.session_state["last_query"],
        "rewritten_query": st.session_state.get("last_rewritten_query"),
        "top5": st.session_state["last_top5"],
        "rrf_pool": st.session_state["last_rrf_pool"],
        "answer": st.session_state["last_answer"],
        "log_chunks": st.session_state["last_log_chunks"],
    }


def _render_last_result(query: str, rewritten_query: str | None,
                        top5: list[dict],
                        rrf_pool: list[dict], answer: str,
                        log_chunks: list[str]) -> None:
    """Render a cached query result. Reusable on every rerun."""
    st.markdown("---")
    st.markdown(f"**You:** {query}")
    if rewritten_query and rewritten_query != query:
        st.caption(f"rewritten for retrieval: {rewritten_query}")

    st.markdown("**Answer:**")
    st.write(answer)
    _render_query_rewrite_inspection(query, rewritten_query)
    _render_generator_inspection(query, top5, answer)

    used = extract_used_citations(answer)
    if used:
        st.markdown("---")
        st.markdown(f"**Citations ({len(used)}):**")
        for n in used:
            hit = top5[n - 1] if 1 <= n <= len(top5) else None
            if hit:
                preview = hit["content"].replace("\n", " ")[:320]
                st.markdown(
                    f"- **[{n}]** id={hit['chunk_id']} (rerank score "
                    f"{hit['score']:.4f}) - {preview}..."
                )

    with st.expander("Retrieval details", expanded=False):
        st.markdown("##### RRF pool (top 20 fused candidates)")
        render_chunks("RRF pool", rrf_pool)
        st.markdown("##### Reranked top 5")
        render_chunks("Reranked", top5)

    with st.expander("Logs (last run)", expanded=True):
        if log_chunks:
            st.code("\n".join(log_chunks), language="text")
        else:
            st.write("No log lines emitted for this query.")


def _render_ragas_eval_block(query: str, top5: list[dict]) -> None:
    """Below the answer: a button that runs RAGAS on the same Q + retrieval.

    Faithfulness, answer_relevancy, context_precision. No labelled doc set.
    Passes the (q, contexts, answer) we already produced, so the metric
    measures the answer the user just saw -- no extra pipeline run.

    RAGAS scores the same full reranked top-5 context list that the LLM saw.
    """
    btn_key = _derive_ragas_button_key(query, top5)
    with st.expander("RAGAS eval (last answer)", expanded=False):
        answer = st.session_state.get("last_answer", "")
        contexts = [
            ch["content"] if isinstance(ch, dict) and "content" in ch else str(ch)
            for ch in top5
        ]
        input_bundle = _ragas_input_bundle(query, contexts, answer)
        st.caption(
            f"Runs RAGAS metrics on the question above using the "
            f"{len(top5)} reranked chunk(s) the LLM actually saw "
            f"as `retrieved_contexts`. Judge = {settings.judge_model}."
        )
        with st.expander("RAGAS input bundle", expanded=False):
            st.json(input_bundle)
        run = st.button("Run RAGAS on this answer", key=btn_key)
        if run:
            with st.spinner("Scoring with RAGAS..."):
                try:
                    from src.evaluation.judge import build_judge_llm, build_judge_embeddings
                    from src.evaluation.ragas_runner import score_with_ragas

                    judge = build_judge_llm()
                    emb = build_judge_embeddings()
                    ragas_log_box = st.empty()
                    score, ragas_logs = collect_logs_with_status(
                        lambda: score_with_ragas(query, contexts, answer, judge, emb),
                        None,
                        ragas_log_box,
                        prefix="RAGAS ",
                    )

                    mean = score["mean"]
                    cols = st.columns(3)
                    cols[0].metric("faithfulness", f"{mean.get('faithfulness', 0):.3f}")
                    cols[1].metric("answer_relevancy", f"{mean.get('answer_relevancy', 0):.3f}")
                    cols[2].metric("context_precision", f"{mean.get('context_precision', 0):.3f}")
                    with st.expander("Raw RAGAS scores", expanded=False):
                        st.json(score.get("scores", {}))
                    with st.expander("RAGAS logs", expanded=False):
                        if ragas_logs:
                            st.code("\n".join(ragas_logs), language="text")
                        else:
                            st.write("No log lines emitted during RAGAS scoring.")
                    st.caption("elapsed (not measured for single-row runs)")
                except Exception as exc:
                    st.error(f"RAGAS run failed: {exc}")


if __name__ == "__main__":
    main()
