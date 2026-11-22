# CLAUDE.md

Instructions for AI coding agents (Claude Code or similar CLI tools) working in this repository.
Read this file fully before making changes. Product requirements and the roadmap live in `PRD.md`.

## 1. Project in one paragraph

A command-line RAG (Retrieval-Augmented Generation) app. The user drops PDF/TXT files into
`documents/`, the app splits them into chunks, embeds them with OpenAI, stores them in a local
Chroma database (`chroma_db/`), and answers questions in the terminal using only the retrieved
chunks, printing the source files/pages with each answer. Built on **LangChain v0.3**.

## 2. Communicating with the user

- The project owner is a Bangla speaker and a beginner-to-intermediate Python developer.
- **Explain plans, changes and errors to the user in Bangla** (simple language, technical terms
  may stay in English). Code, identifiers, commit messages and docstrings stay in English or
  Bangla as already used in the file you are editing — be consistent within a file.
- User-facing CLI messages (prints, error messages) are in **English** (owner's decision,
  2026-11-22). Keep them in English.
- Before any large change (new dependency, refactor across several files, deleting files,
  upgrading LangChain), state the plan in a few lines and wait for approval.
- After a change, summarise in 2–5 lines: which files changed, why, and how to test it.

## 3. Tech stack (pinned — do not change without asking)

| Area                  | Choice                                                             |
| --------------------- | ------------------------------------------------------------------ |
| Language              | Python 3.10+                                                       |
| Framework             | `langchain` / `langchain-core` / `langchain-community` **0.3.x**   |
| LLM + embeddings      | `langchain-openai` 0.3.x — `gpt-4o-mini`, `text-embedding-3-small` |
| Vector store          | `langchain-chroma` 0.2.x + `chromadb` (local, on disk)             |
| Text splitting        | `langchain-text-splitters` 0.3.x                                   |
| Keyword search        | `rank_bm25` (via `BM25Retriever`)                                  |
| Config/secrets        | `python-dotenv`, `.env` file                                       |
| Monitoring (optional) | `langsmith`                                                        |

**Do not upgrade to LangChain 1.x.** In 1.x, `langchain.chains` and `langchain.retrievers`
moved to `langchain-classic`, which breaks `rag_chain.py`, `retrievers.py` and
`conversational.py`. If an upgrade is requested, propose a migration plan first.

## 4. Repository layout

```
rag-project/
├── CLAUDE.md                 # this file
├── PRD.md                    # product requirements + roadmap
├── README.md                 # user-facing setup guide (Bangla)
├── requirements.txt
├── .env.example              # template; real keys go in .env (git-ignored)
├── main.py                   # CLI entry point
├── documents/                # input PDF/TXT files (sample.txt included)
├── chroma_db/                # generated vector DB — never edit by hand, git-ignored
└── src/
    ├── config.py             # ALL settings + check_api_key()
    ├── document_processor.py # DocumentProcessor: load_documents(), split_documents()
    ├── vector_store.py       # VectorStoreManager: create/load vectorstore, exists(), get_all_documents()
    ├── retrievers.py         # build_simple / compression / multi_query / hybrid retriever
    ├── rag_chain.py          # RAGChain + SYSTEM_PROMPT (create_retrieval_chain)
    ├── conversational.py     # ConversationalRAG: history-aware retriever + RunnableWithMessageHistory
    ├── optimization.py       # enable_cache(), async_batch_query(), run_batch()
    ├── monitoring.py         # setup_monitoring() — LangSmith via env vars, no-op without key
    └── production_rag.py     # ProductionRAG: initialize(rebuild, retriever_type), query(), batch_query()
```

Data flow: `main.py` → `ProductionRAG.initialize()` → (load existing DB **or** load → split →
embed → save) → `RAGChain` with the retriever chosen in `config.RETRIEVER_TYPE` → `query()`
returns `{"answer": str, "sources": [metadata dict, ...]}`.

## 5. Commands

```bash
# setup (once)
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # then put the real OPENAI_API_KEY in .env

# run
python main.py                      # Q&A loop; reuses chroma_db/ if it exists
python main.py --rebuild            # re-index after adding/removing files in documents/
python main.py --chat               # conversational mode with memory

# quick checks (no API calls)
python -m py_compile main.py src/*.py
python -c "import src.production_rag"   # verifies imports resolve
```

When tests exist (see PRD milestone M2): `pytest -q`.

## 6. Coding rules

1. **Settings live in `src/config.py`.** No hard-coded model names, paths, chunk sizes or `k`
   values anywhere else. Add new settings there with a short comment.
2. **Import paths** — use these, not the deprecated ones from the original tutorial:
   - `from langchain_text_splitters import RecursiveCharacterTextSplitter` (not `langchain.text_splitter`)
   - `from langchain_chroma import Chroma` (not `langchain_community.vectorstores`)
   - `from langchain_core.documents import Document` (not `langchain.schema`)
   - `from langchain_core.prompts import ...`, `from langchain_core.caches import InMemoryCache`,
     `from langchain_core.globals import set_llm_cache`
3. **Never call `vectorstore.persist()`** — `langchain-chroma` persists automatically when
   `persist_directory` is set.
4. **Do not use** `ConversationSummaryBufferMemory`, `ConversationalRetrievalChain` or other
   `langchain.memory` classes (deprecated). Memory is done with `RunnableWithMessageHistory`.
5. Keep the return shape of `query()` / `ask()` stable: `{"answer": ..., "sources": [...]}`.
   `main.py` and any future UI depend on it.
6. Type hints on public functions; short docstrings explaining _why_, not _what_.
7. Fail with clear, actionable messages (in English) for user mistakes: missing API key, empty
   `documents/`, DB not initialised. Don't swallow exceptions silently.
8. Keep modules small and single-purpose; follow the existing one-responsibility-per-file layout.
9. Prefer editing existing files over creating new ones. New modules go in `src/`.

## 7. Cost and safety rules (important)

- Every embedding and LLM call costs the user money. **Do not run `python main.py`,
  `--rebuild`, or any code that calls OpenAI without telling the user first**, unless they
  asked you to run it.
- Prefer verification that makes no API calls: `py_compile`, import checks, unit tests with
  fakes (`langchain_core.embeddings.FakeEmbeddings`, `langchain_core.language_models.FakeListChatModel`).
- `RETRIEVER_TYPE="compression"` and `"multi_query"` add extra LLM calls per question —
  mention this when suggesting them.
- **Never** print, log, commit or hard-code API keys. `.env` must stay in `.gitignore`.
- Don't delete `chroma_db/` or anything in `documents/` unless the user asks.
  `VectorStoreManager.create_vectorstore()` intentionally wipes and rebuilds `chroma_db/`;
  only `--rebuild` or a missing DB should trigger it.

## 8. Definition of done for a change

- [ ] `python -m py_compile main.py src/*.py` passes
- [ ] Imports resolve (`python -c "import src.production_rag"`)
- [ ] Tests pass if they exist; new logic has a test where feasible (with fakes, no network)
- [ ] No new hard-coded settings outside `config.py`
- [ ] `README.md` (Bangla) updated if setup steps, commands or config options changed
- [ ] `PRD.md` milestone checkbox ticked if a roadmap item was completed
- [ ] Bangla summary given to the user: what changed + how to test

## 9. Common errors and fixes

| Error                                                         | Cause                     | Fix                                                          |
| ------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------ |
| `OPENAI_API_KEY not found`                                    | No `.env` or empty key    | Copy `.env.example` → `.env`, add key                        |
| `ModuleNotFoundError: langchain_classic` / `langchain.chains` | LangChain 1.x installed   | `pip install -r requirements.txt` (pins 0.3.x)               |
| `ModuleNotFoundError: src`                                    | Running from wrong folder | Run from project root: `python main.py`                      |
| Answers ignore new files                                      | Old DB reused             | `python main.py --rebuild`                                   |
| `RateLimitError` / `insufficient_quota`                       | OpenAI credit/limit       | Add billing credit or wait                                   |
| `UnicodeDecodeError` on .txt                                  | Encoding                  | Already handled via `autodetect_encoding`; check the file    |
| Empty/irrelevant answers                                      | Poor retrieval            | Tune `CHUNK_SIZE`, `TOP_K`, or try `RETRIEVER_TYPE="hybrid"` |

## 10. Working style for the agent

1. Read `PRD.md` to see which milestone the task belongs to.
2. Read the files you'll touch before editing; keep diffs minimal and focused.
3. One logical change at a time; don't mix refactors with features.
4. If a request conflicts with this file (e.g. upgrading LangChain, adding a paid service),
   explain the trade-off in Bangla and ask before proceeding.
