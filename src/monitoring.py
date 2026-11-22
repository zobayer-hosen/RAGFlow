"""Optional LangSmith tracing (FR19). Silently does nothing without a key."""

import os

from src import config


def setup_monitoring() -> bool:
    """Enable tracing via env vars; LangChain picks them up automatically.

    Accepts the newer LANGSMITH_* names and the older LANGCHAIN_* names.
    Returns True if tracing was enabled. Never prints the key.
    """
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        return False

    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    project = (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or config.LANGSMITH_PROJECT_DEFAULT
    )
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_PROJECT"] = project
    print(f"📊 LangSmith tracing enabled (project: {project})")
    return True
