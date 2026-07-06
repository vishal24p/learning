"""Convenience launcher for the Streamlit UI.

Usage:
    python scripts/run_ui.py
        # opens http://localhost:8501 and sets PYTHONPATH so the app can
        # import src.*

We pass --server.fileWatcherType=none to silence Streamlit's hot-reload
watcher, which otherwise imports every transitively-loaded package
(including `transformers`, whose multimodal image processors fail to
import torchvision in this environment).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = Path(__file__).parent / "streamlit_app.py"

# Streamlit flags; suppresses file-watcher chatter (it tries to import
# torchvision via transformers' multimodal image processors).
STREAMLIT_FLAGS = (
    "--server.fileWatcherType=none",
    "--browser.gatherUsageStats=false",
)


def main() -> int:
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH), *STREAMLIT_FLAGS]
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
