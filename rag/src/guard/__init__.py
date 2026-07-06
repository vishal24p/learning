"""LLM guard pipeline.

A separate, small-guard-model call that runs *before* any retrieval
happens. The guard uses Ollama's ``tools`` parameter to force the model
to choose between exactly two actions:

  - ``safe_to_continue(query, reason)``  ->  proceed to retrieval
  - ``terminate(query, reason, refusal)``  ->  short-circuit, do NOT
     run the embedder, do NOT query Postgres, do NOT call the
     generator, log the refusal for audit, and return the refusal text
     to the caller.

The guard package stays out of the way of the rest of the codebase:
callers invoke ``guard_or_terminate(query)`` and either get the original
query back (safe) or a ``PipelineTerminated`` exception carrying the
refusal. There is no global state, no Streamlit coupling, and no
Postgres coupling.

The default guard model is ``llama-guard3:1b`` (a dedicated safety
classifier from Meta, served through Ollama). Pull it yourself with:

    ollama pull llama-guard3:1b
"""
