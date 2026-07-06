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
from src.guard.pipeline import guard_or_terminate  # noqa: E402
from src.guard.reason import PipelineTerminated  # noqa: E402
from src.logging_setup import setup_logging  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.retrieval.reranked_retriever import RerankedRetriever  # noqa: E402
from src.retrieval.reranker import filter_positive_chunks  # noqa: E402
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


def render_chunks(name: str, hits: list[dict]) -> None:
    if not hits:
        st.write(f"_{name}_: no results")
        return
    for i, h in enumerate(hits, 1):
        preview = h["content"].replace("\n", " ")[:280]
        st.markdown(
            f"**[{i}] id={h['chunk_id']}  score={h['score']:.4f}**  \n"
            f"<span style='color:#888'>{preview}...</span>",
            unsafe_allow_html=True,
        )


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
        status = st.status("Running guard…", expanded=True)
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

        status.update(label="Retrieving…", expanded=True)
        live_log = st.empty()

        hybrid, reranked, generator = load_components()
        st.markdown("---")
        st.markdown(f"**You:** {query}")

        log_chunks: list[str] = []
        log_chunks.append("--- guard: safe_to_continue ---")

        def do_retrieve() -> tuple[list, list]:
            rrf_pool = hybrid.retrieve(
                query,
                top_k_dense=10,
                top_k_sparse=10,
                top_k=settings.rerank_candidates,
            )
            _, top5 = reranked.retrieve(
                query,
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

        status.update(label="Generating answer…", state="running")
        # Only feed chunks the reranker is *positive* about. If all five
        # were non-positive, fall back to the single top-ranked chunk so the
        # LLM has something to ground on instead of refusing silently.
        fed_chunks = filter_positive_chunks(top5, fallback_top_n=1)
        if fed_chunks and len(fed_chunks) < len(top5):
            status.update(
                label=(
                    f"Generating answer… (filtered: {len(fed_chunks)}/{len(top5)} positive)"
                ),
                state="running",
            )
        answer, recv_logs = collect_logs_with_status(
            lambda: generator.generate(query, fed_chunks),
            None, live_log, prefix="Generation ",
        )
        log_chunks.extend(recv_logs)
        status.update(label="Answer ready", state="complete")

        # Persist last run so it survives the rerun that follows any RAGAS click.
        st.session_state["last_query"] = query
        st.session_state["last_top5"] = top5
        st.session_state["last_fed_chunks"] = fed_chunks
        st.session_state["last_answer"] = answer
        st.session_state["last_rrf_pool"] = rrf_pool
        st.session_state["last_log_chunks"] = log_chunks

    # Re-render last result + RAGAS button on every rerun. Single render site.
    last = _read_last_result()
    if last:
        _render_last_result(**last)
        _render_ragas_eval_block(last["query"], last["top5"], last["fed_chunks"])


def _read_last_result() -> dict | None:
    """Pull the most recent run from session_state; None if absent."""
    if "last_query" not in st.session_state:
        return None
    return {
        "query": st.session_state["last_query"],
        "top5": st.session_state["last_top5"],
        "fed_chunks": st.session_state.get("last_fed_chunks", st.session_state["last_top5"]),
        "rrf_pool": st.session_state["last_rrf_pool"],
        "answer": st.session_state["last_answer"],
        "log_chunks": st.session_state["last_log_chunks"],
    }


def _render_last_result(query: str, top5: list[dict], fed_chunks: list[dict],
                        rrf_pool: list[dict], answer: str,
                        log_chunks: list[str]) -> None:
    """Render a cached query result. Reusable on every rerun."""
    st.markdown("---")
    st.markdown(f"**You:** {query}")

    st.markdown("**Answer:**")
    st.write(answer)

    used = extract_used_citations(answer)
    if used:
        st.markdown("---")
        st.markdown(f"**Citations ({len(used)}):**")
        for n in used:
            hit = fed_chunks[n - 1] if 1 <= n <= len(fed_chunks) else (
                top5[n - 1] if 1 <= n <= len(top5) else None
            )
            if hit:
                preview = hit["content"].replace("\n", " ")[:320]
                st.markdown(
                    f"- **[{n}]** id={hit['chunk_id']} (rerank score "
                    f"{hit['score']:.4f}) — {preview}..."
                )

    with st.expander("Retrieval details", expanded=False):
        st.markdown("##### RRF pool (top 20 fused candidates)")
        render_chunks("RRF pool", rrf_pool)
        st.markdown("##### Reranked (top 5 kept for the LLM)")
        render_chunks("Reranked", top5)
        dropped = [c for c in top5 if c not in fed_chunks]
        st.markdown(
            f"##### Positive-only fed to LLM: "
            f"**{len(fed_chunks)}/{len(top5)}** kept"
            + (f" — dropped {len(dropped)} non-positive chunk(s)" if dropped else "")
        )
        render_chunks("Positive-only", fed_chunks)

    with st.expander("Logs (last run)", expanded=True):
        if log_chunks:
            st.code("\n".join(log_chunks), language="text")
        else:
            st.write("No log lines emitted for this query.")


def _render_ragas_eval_block(query: str, top5: list[dict],
                              fed_chunks: list[dict]) -> None:
    """Below the answer: a button that runs RAGAS on the same Q + retrieval.

    Faithfulness, answer_relevancy, context_precision. No labelled doc set.
    Passes the (q, contexts, answer) we already produced, so the metric
    measures the answer the user just saw -- no extra pipeline run.

    `fed_chunks` controls which contexts RAGAS scores against -- i.e. the
    positive-only filtered list that the LLM actually saw, NOT the raw
    reranked top-5.
    """
    btn_key = _derive_ragas_button_key(query, top5)
    with st.expander("RAGAS eval (last answer)", expanded=False):
        st.caption(
            f"Runs RAGAS metrics on the question above using the "
            f"{len(fed_chunks)} positive-only chunk(s) the LLM actually saw "
            "as `retrieved_contexts`. Judge = same Ollama model."
        )
        run = st.button("Run RAGAS on this answer", key=btn_key)
        if run:
            with st.spinner("Scoring with RAGAS..."):
                try:
                    from src.evaluation.judge import build_judge_llm, build_judge_embeddings
                    from src.evaluation.ragas_runner import score_with_ragas

                    contexts = [
                        ch["content"] if isinstance(ch, dict) and "content" in ch else str(ch)
                        for ch in fed_chunks
                    ]
                    answer = st.session_state.get("last_answer", "")

                    judge = build_judge_llm()
                    emb = build_judge_embeddings()
                    score = score_with_ragas(query, contexts, answer, judge, emb)

                    mean = score["mean"]
                    cols = st.columns(3)
                    cols[0].metric("faithfulness", f"{mean.get('faithfulness', 0):.3f}")
                    cols[1].metric("answer_relevancy", f"{mean.get('answer_relevancy', 0):.3f}")
                    cols[2].metric("context_precision", f"{mean.get('context_precision', 0):.3f}")
                    st.caption("elapsed (not measured for single-row runs)")
                except Exception as exc:
                    st.error(f"RAGAS run failed: {exc}")


if __name__ == "__main__":
    main()
