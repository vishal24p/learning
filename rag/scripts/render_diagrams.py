"""Render the bundled mermaid diagrams.

Output options:
  --print       print the .mmd source for each diagram to stdout
                (fast, paste-friendly, no deps required)
  --render      if `mmdc` (mermaid-cli) is on PATH, emit PNGs alongside
                each .mmd. Otherwise prints an install hint and falls
                back to --print behavior.

This script is intentionally side-effect-light; the .mmd sources
themselves are the source of truth and render natively on GitHub.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIAGRAMS_DIR = HERE / "diagrams"

DIAGRAMS: tuple[str, ...] = (
    "pipeline.mmd",   # end-to-end pipeline with guard
    "guard.mmd",      # guard decision tree
    "classes.mmd",    # class wiring
)


def _read(name: str) -> str:
    return (DIAGRAMS_DIR / name).read_text(encoding="utf-8")


def _render_png(name: str) -> int:
    """Try `mmdc -i <mmd> -o <png>`. Returns the exit code."""
    src = DIAGRAMS_DIR / name
    dst = DIAGRAMS_DIR / (Path(name).stem + ".png")
    cmd = ["mmdc", "-i", str(src), "-o", str(dst), "-b", "transparent"]
    print(f"[render] {name} -> {dst.name}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    return proc.returncode


def cmd_print(_: argparse.Namespace) -> int:
    for name in DIAGRAMS:
        print(f"\n========== {name} ==========")
        print(_read(name))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    if shutil.which("mmdc") is None:
        print(
            "[render] mmdc (mermaid-cli) not found on PATH.\n"
            "         Install with:  npm i -g @mermaid-js/mermaid-cli\n"
            "         Falling back to --print so the sources are still visible."
        )
        return cmd_print(args)
    rc = 0
    for name in DIAGRAMS:
        rc |= _render_png(name)
    if rc == 0:
        print("[render] done; PNGs live in scripts/diagrams/")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="Render bundled mermaid diagrams")
    ap.add_argument("--print", action="store_true",
                    help="print each .mmd source to stdout (default if neither flag given)")
    ap.add_argument("--render", action="store_true",
                    help="render PNGs via `mmdc` if installed")
    args = ap.parse_args()

    # Default to --print when neither flag is passed; this keeps the
    # script useful in any environment without needing npx.
    if not args.print and not args.render:
        args.print = True

    rc = 0
    if args.print:
        rc = cmd_print(args)
    if args.render:
        rc |= cmd_render(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
