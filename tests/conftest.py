"""Keep tests fully offline: no LangSmith tracing, no real API keys."""

import pytest

_NETWORK_ENV = (
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in _NETWORK_ENV:
        monkeypatch.delenv(name, raising=False)
