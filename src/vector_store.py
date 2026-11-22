"""Create, load and inspect the Chroma vector store (FR4, FR5, FR8)."""

import shutil
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src import config
from src.config import ConfigError


class VectorStoreManager:
    """Owns the on-disk Chroma DB so the rest of the app never touches paths directly."""

    def __init__(
        self,
        persist_directory: str = config.PERSIST_DIRECTORY,
        embeddings: Optional[Embeddings] = None,
        collection_name: str = config.COLLECTION_NAME,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        # Embeddings are injectable so tests can use FakeEmbeddings (no network, no cost)
        self.embeddings = embeddings or OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
        self.vectorstore: Optional[Chroma] = None

    def exists(self) -> bool:
        """True only if a DB with at least one chunk exists — lets startup skip paid re-embedding.

        Checking the chunk count (not just the folder) matters: a build that failed half-way
        (e.g. wrong API key) leaves an empty collection behind, which must be rebuilt.
        No embedding calls are made here.
        """
        if not (self.persist_directory.is_dir() and any(self.persist_directory.iterdir())):
            return False
        try:
            store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory),
            )
            return len(store.get(limit=1, include=[])["ids"]) > 0
        except Exception:
            return False

    def create_vectorstore(self, chunks: List[Document]) -> Chroma:
        """Wipe and rebuild the DB. Intentionally destructive: only used on --rebuild or first run."""
        if not chunks:
            raise ConfigError("No text found to index.")
        if self.persist_directory.exists():
            self._wipe_collection()

        # langchain-chroma persists automatically when persist_directory is set (no .persist())
        try:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=str(self.persist_directory),
                collection_metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            # Don't leave a half-built DB that the next start would treat as valid
            self._wipe_collection()
            raise
        print(f"💾 Created a new vector store ({len(chunks)} chunks).")
        return self.vectorstore

    def _wipe_collection(self) -> None:
        """Delete the old collection. Done through Chroma rather than deleting the folder,
        because on Windows the open SQLite file can't be removed while the process holds it."""
        try:
            Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory),
            ).delete_collection()
        except Exception:
            shutil.rmtree(self.persist_directory, ignore_errors=True)

    def load_vectorstore(self) -> Chroma:
        """Open the existing DB without any embedding calls."""
        if not self.exists():
            raise ConfigError("Vector store has not been created yet. Run `python main.py --rebuild`.")
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
            collection_metadata={"hnsw:space": "cosine"},
        )
        print("✅ Loaded existing vector store")
        return self.vectorstore

    def get_all_documents(self) -> List[Document]:
        """Return every stored chunk — needed to build BM25 for the hybrid retriever."""
        if self.vectorstore is None:
            raise ConfigError("Vector store is not initialised.")
        data = self.vectorstore.get(include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(data["documents"], data["metadatas"])
        ]
