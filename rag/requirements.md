# Python requirements

This document lists every Python package the RAG pipeline needs. Inventory is
derived directly from the `import` statements in `src/` and `scripts/` (no
guessing). The same set is pinned in `requirements.txt` for `pip install`.

## Prerequisites you must already have on the box

These are **not** Python packages and **not** installed by `requirements.txt`:

- **Ollama** running locally on `http://localhost:11434`, with these models
  pulled (`ollama pull <name>`). You said you already have these:
  - `nomic-embed-text-v2-moe` (embeddings)
  - `llama-guard3:1b` (pre-pipeline safety guard)
  - the generator model declared in `GEN_MODEL`
- **ParadeDB** (a Postgres container) reachable on `localhost:56432` with
  the `rag` (and optionally `rag_v2`) database. See `scripts/setup_paradedb.py`.

Everything below this line is pip-installable into a virtualenv.

---

## How `requirements.md` organizes packages

Each section groups packages by **why the pipeline needs them**, not by
alphabetical order. Version pins reflect the floor used when the code was
written; loosen them if your env already has working newer releases.

### 1. LLM + embedding clients (Ollama talking)

- `ollama` -- the official Ollama Python SDK. Used by the pre-pipeline
  guard (`src/guard/guard.py`: `from ollama import chat as ollama_chat,
  from ollama import ChatResponse`).
- `llama-index-core` -- Document / PromptTemplate / ChatMessage /
  MessageRole / SemanticSplitterNodeParser. Used by `src/chunking/`,
  `src/generation/generator.py`, `src/generation/prompts.py`.
- `llama-index-llms-ollama` -- `from llama_index.llms.ollama import Ollama`
  in `src/llm/factory.py`. Required for the generator chat path.
- `llama-index-embeddings-ollama` -- `OllamaEmbedding` in
  `src/llm/factory.py`. Required for the embed path.
- `llama-index` -- aggregate meta-package; installing it pulls in
  `llama-index-core` + community integrations.

### 2. Postgres / vector store driver

- `psycopg[binary]` -- `from psycopg import connect` shows up in
  `src/storage/postgres_store.py`, `src/retrieval/dense_retriever.py`,
  `src/retrieval/sparse_retriever.py`, and most of the `scripts/*.py`
  migration tools. (psycopg 3.x, with the `binary` extra so you don't
  need a local libpq build.)
- `pgvector` -- ParadedDB side, not strictly needed on the client; the
  `CREATE EXTENSION` lives server-side.

### 3. Cross-encoder reranker

- `sentence-transformers` -- `from sentence_transformers import CrossEncoder`
  in `src/retrieval/reranker.py`. Used only the first time
  `CrossEncoderReranker._ensure_model()` runs; lazy-loaded so the rest of
  the pipeline can skip the download if you A/B-test with the identity
  fallback.
- `torch` -- transitive dep of `sentence-transformers`. `transformers` is
  also pulled in transitively (the logging_setup silences its logger to
  ERROR on purpose, see point 5).

### 4. RAGAS evaluation stack

- `ragas` -- `from ragas.dataset_schema import EvaluationDataset,
  SingleTurnSample`, `from ragas.evaluation import evaluate`,
  `from ragas.llms import LangchainLLMWrapper`,
  `from ragas.metrics import answer_relevancy, context_precision,
  faithfulness`. Targets `ragas==0.2.x`.
- `langchain-ollama` -- `from langchain_ollama import ChatOllama,
  OllamaEmbeddings` in `src/evaluation/judge.py`.
- `langchain-community` -- ragas 0.2.x still imports
  `langchain_community.chat_models.vertexai`; we register a
  `sys.modules` stub, but the package itself must be importable so the
  `from langchain_community.chat_models.vertexai` lookup inside the ragas
  import chain succeeds.

### 5. UI

- `streamlit` -- `import streamlit as st` in `scripts/streamlit_app.py`.
  Run with `--server.fileWatcherType=none` (already wired in
  `scripts/run_ui.py`).

### 6. Tests (dev-only)

Pulled in for running `pytest tests/`:
- `pytest`

---

## Optional / infra

- `mermaid-cli` (`mmdc`) -- only if you run `python scripts/render_diagrams.py --render`.
  Not Python; install with `npm i -g @mermaid-js/mermaid-cli` if you want
  PNGs. The `.mmd` sources render natively on GitHub anyway.
- `python-dotenv` -- **NOT required.** `src/config.py` reads the `.env`
  file manually with `pathlib.Path.read_text()`, so no dotenv dep is
  shipped. `from dotenv import ...` would be dead weight.

---

## Minimum versions observed during development

These are lower bounds, not exact pins:

```
python>=3.10
ollama>=0.3
llama-index-core>=0.12
llama-index-llms-ollama>=0.3
llama-index-embeddings-ollama>=0.5
llama-index>=0.12
psycopg[binary]>=3.1
sentence-transformers>=3.0
torch>=2.0
ragas==0.2.7
langchain-ollama>=0.2
langchain-community>=0.3
streamlit>=1.30
pytest>=8.0
```

If a `pip install -r requirements.txt` fails on one of these, loosen the
pin to `>=` only and try again -- new Ollama-side API changes are the most
common source of breakage (`thinking=`, `temperature` kwarg migration,
`tools=` rejection on the small models, etc.). spot-fixes are documented
inline in `src/llm/factory.py` and `src/evaluation/judge.py`.

---

## Sanity-check after install

```bash
python -c "
import ollama, psycopg, streamlit, sentence_transformers, torch
import llama_index.core, llama_index.llms.ollama, llama_index.embeddings.ollama
from ragas.evaluation import evaluate
from langchain_ollama import ChatOllama
print('ok')
"
```

That should print `ok` with no exceptions. If any import errors, the most
likely culprits, in order:

1. `ragas==0.2.x` <-> `langchain-community>=0.4` mismatch (the codebase
   patches it via `src/evaluation/judge.py:_install_vertexai_stub`, but
   if you upgrade either package, check the stub still installs).
2. `ollama>=0.6` + `thinking=` kwarg pass-through handled in
   `src/llm/factory.py:build_llm()`.
3. `pytest-asyncio` not needed -- tests are synchronous.
