"""Facade that wires everything together; main.py (and a future UI) only talk to this."""

from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src import config
from src.config import ConfigError
from src.conversational import ConversationalRAG
from src.document_processor import DocumentProcessor
from src.monitoring import setup_monitoring
from src.optimization import enable_cache, run_batch
from src.rag_chain import RAGChain
from src.retrievers import build_retriever
from src.vector_store import VectorStoreManager


class ProductionRAG:
    """Load-or-build the index, pick a retriever, and answer questions."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        embeddings: Optional[Embeddings] = None,
        data_directory: str = config.DATA_DIRECTORY,
        persist_directory: str = config.PERSIST_DIRECTORY,
    ) -> None:
        # llm/embeddings are injectable for tests; real runs need an API key
        if llm is None or embeddings is None:
            config.check_api_key()
        self.llm = llm or ChatOpenAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
        self.processor = DocumentProcessor(data_directory)
        self.store = VectorStoreManager(persist_directory, embeddings=embeddings)
        self.rag_chain: Optional[RAGChain] = None
        self.chat: Optional[ConversationalRAG] = None

    def initialize(
        self, rebuild: bool = False, retriever_type: str = config.RETRIEVER_TYPE
    ) -> None:
        """Reuse the DB when possible so normal startup costs zero embedding calls (FR5)."""
        setup_monitoring()
        if config.ENABLE_CACHE:
            enable_cache()

        if rebuild or not self.store.exists():
            print("🔨 Indexing documents (this makes embedding API calls)...")
            chunks = self.processor.split_documents(self.processor.load_documents())
            vectorstore = self.store.create_vectorstore(chunks)
        else:
            vectorstore = self.store.load_vectorstore()

        all_docs = self.store.get_all_documents() if retriever_type == "hybrid" else None
        retriever = build_retriever(retriever_type, vectorstore, self.llm, all_docs)
        self.rag_chain = RAGChain(retriever, self.llm)
        self.chat = ConversationalRAG(retriever, self.llm)
        print(f"🔍 Retriever: {retriever_type}")

    def _require_ready(self) -> None:
        if self.rag_chain is None or self.chat is None:
            raise ConfigError("RAG is not initialised. Call initialize() first.")

    def query(self, question: str) -> Dict[str, Any]:
        """One-shot question -> {"answer": str, "sources": [metadata, ...]} (FR10)."""
        self._require_ready()
        return self.rag_chain.ask(question)

    def chat_query(self, question: str, session_id: str = config.DEFAULT_SESSION_ID) -> Dict[str, Any]:
        """Like query() but remembers earlier turns in this session (FR12)."""
        self._require_ready()
        return self.chat.ask(question, session_id)

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Many questions at once, run concurrently (FR11, FR18)."""
        self._require_ready()
        return run_batch(self.rag_chain.chain, questions)
