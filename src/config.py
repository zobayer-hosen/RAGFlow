"""All project settings in one place (CLAUDE.md rule 1).

Secrets come only from `.env`; everything tunable lives here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = folder that contains main.py
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

# --- Paths ---
DATA_DIRECTORY = str(BASE_DIR / "documents")  # input PDF/TXT files
PERSIST_DIRECTORY = str(BASE_DIR / "chroma_db")  # generated vector DB
COLLECTION_NAME = "documents"  # Chroma collection name

# --- Models ---
LLM_MODEL = "gpt-4o-mini"  # cheap, good-enough chat model
LLM_TEMPERATURE = 0.0  # deterministic answers for Q&A
EMBEDDING_MODEL = "text-embedding-3-small"  # cheap embedding model

# --- Splitting ---
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # characters shared between neighbouring chunks

# --- Retrieval ---
# One of: "simple", "compression", "multi_query", "hybrid"
# NOTE: "compression" and "multi_query" make extra LLM calls (= extra cost) per question.
RETRIEVER_TYPE = "simple"
TOP_K = 4  # number of chunks sent to the LLM
HYBRID_WEIGHTS = [0.5, 0.5]  # [BM25 keyword, semantic] weights for "hybrid"

# --- Optimisation ---
ENABLE_CACHE = True  # in-memory LLM cache: repeat questions in one session are free
BATCH_MAX_CONCURRENCY = 5  # parallel requests in run_batch()

# --- Conversation ---
DEFAULT_SESSION_ID = "default"

# --- Monitoring (LangSmith) ---
LANGSMITH_PROJECT_DEFAULT = "rag-project"  # used if LANGSMITH_PROJECT is not set in .env

# --- CLI ---
EXIT_COMMANDS = {"exit", "quit", "q"}


class ConfigError(RuntimeError):
    """Raised for setup mistakes the user can fix (shown as a clear message)."""


def check_api_key() -> None:
    """Stop early with a clear message instead of a cryptic OpenAI error later."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ConfigError(
            "OPENAI_API_KEY not found. Copy .env.example to a file named .env in the "
            "project folder and add OPENAI_API_KEY=sk-... to it."
        )
