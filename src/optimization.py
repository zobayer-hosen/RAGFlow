"""Cost/speed helpers: LLM cache and concurrent batch queries (FR17, FR18)."""

import asyncio
from typing import Any, Dict, List

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.runnables import Runnable

from src import config
from src.rag_chain import format_result


def enable_cache() -> None:
    """Identical prompts within one run are answered from memory — no second API charge."""
    set_llm_cache(InMemoryCache())


async def async_batch_query(chain: Runnable, questions: List[str]) -> List[Dict[str, Any]]:
    """Run questions concurrently; a semaphore keeps us under OpenAI rate limits."""
    semaphore = asyncio.Semaphore(config.BATCH_MAX_CONCURRENCY)

    async def _one(question: str) -> Dict[str, Any]:
        async with semaphore:
            return format_result(await chain.ainvoke({"input": question}))

    return await asyncio.gather(*(_one(q) for q in questions))


def run_batch(chain: Runnable, questions: List[str]) -> List[Dict[str, Any]]:
    """Sync wrapper so callers (CLI, scripts) don't need to manage an event loop."""
    return asyncio.run(async_batch_query(chain, questions))
