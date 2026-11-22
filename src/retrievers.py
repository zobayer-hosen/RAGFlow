"""Retrieval strategies selectable via config.RETRIEVER_TYPE (FR7, FR8, §6.2)."""

from typing import List, Optional

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from src import config
from src.config import ConfigError

RETRIEVER_TYPES = ("simple", "compression", "multi_query", "hybrid")


def build_simple_retriever(vectorstore: VectorStore, k: int = config.TOP_K) -> BaseRetriever:
    """Plain top-k similarity search: 0 extra LLM calls, the cheap default."""
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_compression_retriever(
    vectorstore: VectorStore, llm: BaseChatModel, k: int = config.TOP_K
) -> BaseRetriever:
    """LLM trims each retrieved chunk to the relevant part (~k extra LLM calls)."""
    return ContextualCompressionRetriever(
        base_compressor=LLMChainExtractor.from_llm(llm),
        base_retriever=build_simple_retriever(vectorstore, k),
    )


def build_multi_query_retriever(
    vectorstore: VectorStore, llm: BaseChatModel, k: int = config.TOP_K
) -> BaseRetriever:
    """LLM rewrites the question several ways to catch different wordings (1 extra call)."""
    return MultiQueryRetriever.from_llm(retriever=build_simple_retriever(vectorstore, k), llm=llm)


def build_hybrid_retriever(
    vectorstore: VectorStore, all_documents: List[Document], k: int = config.TOP_K
) -> BaseRetriever:
    """BM25 keywords + semantic search; helps with exact names/terms embeddings miss."""
    if not all_documents:
        raise ConfigError("No chunks found in the DB for the hybrid retriever. Run with --rebuild.")
    bm25 = BM25Retriever.from_documents(all_documents)
    bm25.k = k
    return EnsembleRetriever(
        retrievers=[bm25, build_simple_retriever(vectorstore, k)],
        weights=config.HYBRID_WEIGHTS,
    )


def build_retriever(
    retriever_type: str,
    vectorstore: VectorStore,
    llm: Optional[BaseChatModel] = None,
    all_documents: Optional[List[Document]] = None,
    k: int = config.TOP_K,
) -> BaseRetriever:
    """Single entry point so callers only need the config string."""
    if retriever_type == "simple":
        return build_simple_retriever(vectorstore, k)
    if retriever_type in ("compression", "multi_query") and llm is None:
        raise ValueError(f"The '{retriever_type}' retriever needs an llm.")
    if retriever_type == "compression":
        return build_compression_retriever(vectorstore, llm, k)
    if retriever_type == "multi_query":
        return build_multi_query_retriever(vectorstore, llm, k)
    if retriever_type == "hybrid":
        return build_hybrid_retriever(vectorstore, all_documents or [], k)
    raise ConfigError(
        f"Unknown RETRIEVER_TYPE: '{retriever_type}'. "
        f"Use one of: {', '.join(RETRIEVER_TYPES)}"
    )
