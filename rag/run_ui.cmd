@echo off
REM Convenience launcher for the RAG Streamlit UI.
REM
REM What this does:
REM   1. Sets PYTHONPATH so `src.*` imports resolve from the repo root.
REM   2. Forwards to scripts/run_ui.py, which itself runs streamlit with
REM      --server.fileWatcherType=none (so transformers/torchvision don't
REM      try to import on every rerun and crash).
REM
REM Pre-reqs (run these once yourself):
REM   ollama serve                       (already running on port 11434)
REM   ollama pull llama-guard3:1b        (the pre-pipeline guard model)
REM   ParadeDB running on the DB_HOST/DB_PORT from .env
REM   python scripts/apply_migration.py  (creates extensions + indexes)
REM   a populated CHUNKS_TABLE in DB_NAME (local reindex helpers are ignored)

setlocal
set PYTHONPATH=%~dp0
cd /d "%~dp0"
..\.venv\Scripts\python.exe scripts\run_ui.py
endlocal
