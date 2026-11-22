# PRD: Document Q&A CLI (RAG with LangChain v0.3)

|              |                                                                     |
| ------------ | ------------------------------------------------------------------- |
| Status       | MVP code written + offline tests pass; not yet run end-to-end       |
| Owner        | Project owner (you)                                                 |
| Last updated | 2026-11-22                                                          |
| Related      | `CLAUDE.md` (agent instructions), `README.md` (setup guide, Bangla) |

## 1. Problem

General-purpose chatbots don't know the contents of a person's own files (notes, books,
reports, PDFs). Reading long documents to find one answer is slow. The owner wants a tool that
answers questions about their own documents, shows where each answer came from, and is simple
enough to run and understand as a learning project.

## 2. Goals

1. Answer natural-language questions using only the content of files in `documents/`.
2. Show the source file (and page for PDFs) for every answer so it can be verified.
3. Say "I don't know" when the documents don't contain the answer, instead of making things up.
4. Run entirely from the command line with a single command.
5. Keep running costs low (reuse the index, cheap default models, optional caching).
6. Stay readable and modular so the owner can learn RAG concepts from the code.

## 3. Non-goals (for now)

- Web UI, REST API, or multi-user hosting (see M5, optional later).
- Real-time sync with cloud drives.
- Fine-tuning models or embeddings.
- Running fully offline / local LLMs (possible later, not planned).
- Support for formats other than PDF and TXT in the MVP.

## 4. Users

- **Primary:** the owner — a Bangla-speaking Python learner using a terminal on Windows or
  Mac/Linux, with an OpenAI API key.
- **Secondary:** classmates/colleagues who clone the repo and follow `README.md`.

## 5. User stories

| ID  | As a user I want to…                               | So that…                                 |
| --- | -------------------------------------------------- | ---------------------------------------- |
| US1 | put PDF/TXT files in a folder and run one command  | I don't need to write code to index them |
| US2 | type a question and get a short answer             | I find information quickly               |
| US3 | see the file and page each answer came from        | I can verify the answer                  |
| US4 | get "I don't know" for out-of-scope questions      | I can trust the tool                     |
| US5 | not pay for re-indexing every time I start the app | it stays cheap                           |
| US6 | re-index after adding new files                    | new content becomes searchable           |
| US7 | ask follow-up questions ("explain that more")      | it feels like a conversation             |
| US8 | switch retrieval strategy in one config line       | I can experiment and learn               |
| US9 | get clear error messages                           | I can fix setup problems myself          |

## 6. Functional requirements

### 6.1 Ingestion (`document_processor.py`, `vector_store.py`)

- FR1. Load all `**/*.pdf` (PyPDFLoader) and `**/*.txt` (TextLoader, auto-detect encoding)
  from `DATA_DIRECTORY`, including subfolders.
- FR2. Fail with a clear message if the folder is missing or contains no supported files.
- FR3. Split with `RecursiveCharacterTextSplitter` using `CHUNK_SIZE` (default 1000 chars) and
  `CHUNK_OVERLAP` (default 200).
- FR4. Embed with `EMBEDDING_MODEL` and store in Chroma at `PERSIST_DIRECTORY`, cosine distance.
- FR5. On start, if the DB exists, load it without re-embedding. `--rebuild` wipes and rebuilds.
- FR6. Every chunk keeps metadata: `source` (file path) and `page` (PDFs, 0-based).

### 6.2 Retrieval (`retrievers.py`)

Selectable via `RETRIEVER_TYPE` in `config.py`:

| Value              | Behaviour                                               | Extra LLM calls per question |
| ------------------ | ------------------------------------------------------- | ---------------------------- |
| `simple` (default) | top-`TOP_K` similarity search                           | 0                            |
| `compression`      | similarity search, then LLM extracts only relevant text | ~`TOP_K`                     |
| `multi_query`      | LLM rewrites the question 3 ways, merges results        | 1                            |
| `hybrid`           | BM25 keyword + semantic, weights 0.5/0.5                | 0                            |

- FR7. An unknown `RETRIEVER_TYPE` raises a clear error.
- FR8. `hybrid` builds BM25 from all chunks stored in the DB (works on a loaded DB too).

### 6.3 Answering (`rag_chain.py`, `production_rag.py`)

- FR9. Use `create_retrieval_chain` + `create_stuff_documents_chain` with the system prompt in
  `rag_chain.SYSTEM_PROMPT` (answer from context only; say "don't know" otherwise; be concise).
- FR10. `ProductionRAG.query(question)` returns `{"answer": str, "sources": [metadata, ...]}`.
- FR11. `batch_query(questions)` returns a list of the same shape.

### 6.4 Conversation (`conversational.py`)

- FR12. `--chat` mode rewrites follow-up questions into standalone questions using chat history
  (`create_history_aware_retriever`), then answers as in FR9.
- FR13. History is in memory per `session_id`; it is lost when the program exits.

### 6.5 CLI (`main.py`)

- FR14. `python main.py` starts an interactive loop: `Question:` prompt → answer → de-duplicated
  source list (file + page number, 1-based).
- FR15. `exit`, `quit` or `q` ends the loop; empty input is ignored.
- FR16. Flags: `--rebuild`, `--chat` (can be combined).

### 6.6 Optimisation & monitoring

- FR17. In-memory LLM cache is enabled at start (repeat questions in one session are free).
- FR18. `optimization.run_batch(chain, questions)` runs questions concurrently.
- FR19. If `LANGSMITH_API_KEY` is set, tracing is enabled; otherwise it is silently skipped.

### 6.7 Configuration & errors

- FR20. All tunables live in `src/config.py`; secrets only in `.env`.
- FR21. Missing `OPENAI_API_KEY` stops the app at start with a clear message telling the user
  what to do.

## 7. Non-functional requirements

| Area            | Requirement                                                                                                         |
| --------------- | ------------------------------------------------------------------------------------------------------------------- |
| Cost            | Starting the app with an existing DB makes **zero** embedding calls. Default config uses one LLM call per question. |
| Latency         | Target: answer in under ~5 s for `simple` retriever on a normal connection (own target, to be measured in M2).      |
| Portability     | Works on Windows, macOS and Linux with Python 3.10+.                                                                |
| Security        | API keys never committed or printed; `.env` and `chroma_db/` are git-ignored.                                       |
| Maintainability | One responsibility per module; no hard-coded settings outside `config.py`.                                          |
| Language        | CLI messages in English; README in Bangla; questions/answers work in English and Bangla.                            |
| Dependencies    | LangChain pinned to 0.3.x (see `CLAUDE.md` §3).                                                                     |

## 8. Architecture

```
documents/*.pdf|*.txt
        │  DocumentProcessor.load_documents()
        ▼
   Document list ── split_documents() ──► chunks
                                            │ VectorStoreManager.create_vectorstore()
                                            ▼
                                   chroma_db/ (embeddings)
                                            │
question ─► retriever (simple | compression | multi_query | hybrid)
                                            │ top-k chunks
                                            ▼
                     SYSTEM_PROMPT + context + question ─► gpt-4o-mini
                                            │
                                            ▼
                          {"answer": ..., "sources": [...]}  ─► main.py prints
```

## 9. Milestones & roadmap

Tick items as they are completed.

### M1 — MVP runs end-to-end

- [x] Install requirements in a fresh venv without errors
- [ ] `python main.py` indexes `documents/sample.txt` and answers "What is RAG?"
- [ ] Second start loads the DB without re-embedding
- [ ] `--rebuild` picks up a newly added PDF; sources show correct page numbers
- [ ] Out-of-scope question returns "don't know"
- [ ] `--chat` handles a follow-up question correctly
- [ ] Each `RETRIEVER_TYPE` works at least once
- [ ] Fix any bugs found; record them in a "Known issues" section below

### M2 — Tests & quality

- [x] Add `pytest` + `tests/` using `FakeEmbeddings` / `FakeListChatModel` (no network, no cost)
- [x] Tests: loading/splitting, empty folder error, DB create vs load, retriever selection,
      `query()` return shape
- [ ] Add `logging` instead of bare `print` for internal messages
- [ ] Measure latency and cost per question; update §7 targets

### M3 — Better indexing

- [ ] Incremental indexing: only embed new/changed files (track file hash in metadata)
- [ ] De-duplicate identical chunks
- [ ] More formats: `.docx`, `.md`, `.csv`
- [ ] Test retrieval quality on Bangla documents; adjust separators/chunk size if needed

### M4 — Better answers & UX

- [ ] Stream answers token by token in the terminal
- [ ] `--show-context` flag to print the retrieved chunks (for learning/debugging)
- [ ] Persist chat history to disk between sessions (optional)
- [ ] CLI argument parsing with `argparse` (`--retriever hybrid`, `--k 8`, `--docs path`)

### M5 — Optional extensions (only if needed)

- [ ] Simple web UI (Streamlit or Gradio) or FastAPI endpoint reusing `ProductionRAG`
- [ ] Agent-based RAG (decide when to retrieve / web search)
- [ ] Knowledge graph retrieval (Neo4j)
- [ ] Migration plan to LangChain 1.x

## 10. Acceptance criteria for MVP (M1)

1. A new user can go from clone to first answer by following `README.md` only.
2. Answers to questions covered by `sample.txt` are correct and cite `documents/sample.txt`.
3. A question not covered by any document gets a "don't know"-style answer.
4. Restarting the app does not trigger re-embedding (verified by log message
   "Loaded existing vector store").
5. No API key appears in any committed file or console output.

## 11. Risks

| Risk                                   | Impact                 | Mitigation                                                                     |
| -------------------------------------- | ---------------------- | ------------------------------------------------------------------------------ |
| LangChain API changes / deprecations   | Code breaks on install | Pinned versions; migration only as planned work (M5)                           |
| OpenAI costs grow with large documents | Unexpected bills       | Reuse DB, `simple` retriever default, cache, monitor usage in OpenAI dashboard |
| Scanned PDFs have no text layer        | Nothing gets indexed   | Detect empty pages and warn; OCR later if needed                               |
| Poor retrieval for Bangla text         | Wrong/empty answers    | Evaluate in M3; try `hybrid`, tune chunk size                                  |
| Model name retired by OpenAI           | Errors at runtime      | Model name in `config.py`, one-line change                                     |

## 12. Open questions

- Which documents will be the main real-world use case (study notes, books, work reports)?
  This decides chunk size and whether Bangla support needs priority.
- Is a web UI needed eventually, or is the CLI enough?
- Should answers be in the question's language automatically (Bangla question → Bangla answer)?

## Known issues

- **Fixed (2026-11-22):** `TextLoader(autodetect_encoding=True)` needs `chardet`, which was not
  in `requirements.txt`. Added `chardet<6`: chardet 6+ returns a `mime_type` key that
  langchain-community 0.3 rejects (`TypeError: FileEncoding.__new__()`).
- **Fixed (2026-11-22):** tests clear LangSmith/OpenAI env vars (`tests/conftest.py`) so a
  real `.env` never makes the test suite send traces over the network.
- **Fixed (2026-11-22):** a first indexing that failed (e.g. `401` invalid API key) left an empty
  `chroma_db/`, which the next start loaded as if valid. `exists()` now requires ≥1 chunk and a
  failed build cleans up. OpenAI auth/rate-limit/connection errors now show clear messages.
