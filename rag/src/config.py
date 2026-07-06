"""Centralized environment-driven config."""
import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = _int_env("DB_PORT", 5432)
    db_name: str = os.getenv("DB_NAME", "rag")
    db_user: str = os.getenv("DB_USER", "rag")
    db_password: str = os.getenv("DB_PASSWORD", "")
    chunks_table: str = os.getenv("CHUNKS_TABLE", "chunks")

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text-v2-moe")
    gen_model: str = os.getenv("GEN_MODEL", "minimax-m3:cloud")

    # Pre-pipeline safety guard. Pull the model yourself before enabling:
    #     ollama pull llama-guard3:1b
    # When guard_enabled is False the whole guard step is skipped
    # (no Ollama call) so A/B comparison with an unguarded baseline
    # is a single env toggle.
    guard_model: str = os.getenv("GUARD_MODEL", "llama-guard3:1b")
    guard_enabled: bool = os.getenv("GUARD_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }

    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_candidates: int = _int_env("RERANK_CANDIDATES", 20)
    rerank_top_n: int = _int_env("RERANK_TOP_N", 5)

    gen_temperature: float = float(os.getenv("GEN_TEMPERATURE", "0.2"))
    gen_max_tokens: int = _int_env("GEN_MAX_TOKENS", 400)

    judge_model: str = os.getenv("JUDGE_MODEL", "gemma4:e2b")
    judge_temperature: float = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))

    @property
    def db_url(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


settings = Settings()
