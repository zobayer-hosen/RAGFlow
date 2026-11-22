"""Offline tests: FakeEmbeddings + FakeListChatModel, so no network and no cost."""

from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from main import format_sources
from src.config import ConfigError
from src.document_processor import DocumentProcessor
from src.production_rag import ProductionRAG
from src.retrievers import build_retriever
from src.vector_store import VectorStoreManager

SAMPLE = "RAG means Retrieval-Augmented Generation. " * 60


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "documents"
    (d / "sub").mkdir(parents=True)
    (d / "a.txt").write_text(SAMPLE, encoding="utf-8")
    (d / "sub" / "b.txt").write_text("বাংলা টেক্সট। Chroma is a vector database.", encoding="utf-8")
    return d


def fake_embeddings():
    return DeterministicFakeEmbedding(size=32)


def test_load_and_split_includes_subfolders(docs_dir):
    proc = DocumentProcessor(str(docs_dir), chunk_size=200, chunk_overlap=20)
    docs = proc.load_documents()
    assert len(docs) == 2
    chunks = proc.split_documents(docs)
    assert len(chunks) > 2
    assert all("source" in c.metadata for c in chunks)


def test_empty_folder_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ConfigError):
        DocumentProcessor(str(tmp_path / "empty")).load_documents()


def test_missing_folder_raises(tmp_path):
    with pytest.raises(ConfigError):
        DocumentProcessor(str(tmp_path / "nope")).load_documents()


def test_create_then_load(docs_dir, tmp_path):
    proc = DocumentProcessor(str(docs_dir), chunk_size=200, chunk_overlap=20)
    chunks = proc.split_documents(proc.load_documents())
    db = tmp_path / "db"
    mgr = VectorStoreManager(str(db), embeddings=fake_embeddings())
    assert not mgr.exists()
    mgr.create_vectorstore(chunks)
    assert mgr.exists()

    mgr2 = VectorStoreManager(str(db), embeddings=fake_embeddings())
    mgr2.load_vectorstore()
    assert len(mgr2.get_all_documents()) == len(chunks)


@pytest.mark.parametrize("kind", ["simple", "compression", "multi_query", "hybrid"])
def test_retriever_selection(docs_dir, tmp_path, kind):
    proc = DocumentProcessor(str(docs_dir), chunk_size=200, chunk_overlap=20)
    mgr = VectorStoreManager(str(tmp_path / "db"), embeddings=fake_embeddings())
    vs = mgr.create_vectorstore(proc.split_documents(proc.load_documents()))
    retriever = build_retriever(
        kind, vs, llm=FakeListChatModel(responses=["x"]), all_documents=mgr.get_all_documents()
    )
    assert retriever is not None


def test_unknown_retriever_raises(docs_dir, tmp_path):
    proc = DocumentProcessor(str(docs_dir))
    vs = VectorStoreManager(str(tmp_path / "db"), embeddings=fake_embeddings()).create_vectorstore(
        proc.split_documents(proc.load_documents())
    )
    with pytest.raises(ConfigError):
        build_retriever("magic", vs)


def test_query_return_shape(docs_dir, tmp_path):
    rag = ProductionRAG(
        llm=FakeListChatModel(responses=["RAG is retrieval plus generation."]),
        embeddings=fake_embeddings(),
        data_directory=str(docs_dir),
        persist_directory=str(tmp_path / "db"),
    )
    rag.initialize(retriever_type="simple")
    result = rag.query("What is RAG?")
    assert result["answer"] == "RAG is retrieval plus generation."
    assert isinstance(result["sources"], list) and result["sources"]
    assert "source" in result["sources"][0]


def test_chat_query_return_shape(docs_dir, tmp_path):
    # Responses: 1st turn answer; 2nd turn = rewrite + answer
    rag = ProductionRAG(
        llm=FakeListChatModel(responses=["first", "standalone question", "second"]),
        embeddings=fake_embeddings(),
        data_directory=str(docs_dir),
        persist_directory=str(tmp_path / "db"),
    )
    rag.initialize()
    assert rag.chat_query("What is RAG?")["answer"] == "first"
    assert rag.chat_query("explain that more")["answer"] == "second"


def test_batch_query(docs_dir, tmp_path):
    rag = ProductionRAG(
        llm=FakeListChatModel(responses=["ok"]),
        embeddings=fake_embeddings(),
        data_directory=str(docs_dir),
        persist_directory=str(tmp_path / "db"),
    )
    rag.initialize()
    results = rag.batch_query(["q1", "q2"])
    assert len(results) == 2 and all(r["answer"] == "ok" for r in results)


def test_format_sources_dedup_and_one_based():
    lines = format_sources(
        [{"source": "a.pdf", "page": 0}, {"source": "a.pdf", "page": 0}, {"source": "b.txt"}]
    )
    assert lines == ["a.pdf (page 1)", "b.txt"]


class _FailingEmbeddings(DeterministicFakeEmbedding):
    """Simulates a 401 from OpenAI during the first indexing."""

    def embed_documents(self, texts):
        raise RuntimeError("401 invalid api key")


def test_failed_build_is_not_treated_as_existing_db(docs_dir, tmp_path):
    proc = DocumentProcessor(str(docs_dir))
    chunks = proc.split_documents(proc.load_documents())
    db = tmp_path / "db"
    with pytest.raises(RuntimeError):
        VectorStoreManager(str(db), embeddings=_FailingEmbeddings(size=32)).create_vectorstore(chunks)
    assert not VectorStoreManager(str(db), embeddings=fake_embeddings()).exists()


def test_rebuild_over_existing_db(docs_dir, tmp_path):
    proc = DocumentProcessor(str(docs_dir), chunk_size=200, chunk_overlap=20)
    chunks = proc.split_documents(proc.load_documents())
    mgr = VectorStoreManager(str(tmp_path / "db"), embeddings=fake_embeddings())
    mgr.create_vectorstore(chunks)
    mgr.create_vectorstore(chunks[:2])  # rebuild must replace, not append
    assert len(mgr.get_all_documents()) == 2
