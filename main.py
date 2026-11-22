"""CLI entry point: python main.py [--rebuild] [--chat]  (FR14–FR16)."""

import argparse
import sys
from typing import Any, Dict, List

from src import config
from src.config import ConfigError


def format_sources(sources: List[Dict[str, Any]]) -> List[str]:
    """De-duplicated 'file (page N)' lines; PDF pages shown 1-based for humans."""
    seen, lines = set(), []
    for meta in sources:
        source = meta.get("source", "unknown")
        page = meta.get("page")
        line = f"{source} (page {int(page) + 1})" if page is not None else source
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return lines


def explain_openai_error(exc: Exception) -> str:
    """Turn common OpenAI errors into actionable messages (CLAUDE.md rule 7)."""
    name = type(exc).__name__
    if name == "AuthenticationError":
        return (
            "Your OpenAI API key is invalid or has been revoked (401). "
            "Create a new key at https://platform.openai.com/api-keys, put it on the "
            "OPENAI_API_KEY=... line in .env, then run again."
        )
    if name == "RateLimitError":
        return (
            "OpenAI rate limit reached or credit used up (429). Add billing credit "
            "to your account, or wait a moment and try again."
        )
    if name in ("APIConnectionError", "APITimeoutError"):
        return "Could not connect to OpenAI. Check your internet connection and try again."
    return f"{name}: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer questions from your own documents (RAG)")
    parser.add_argument("--rebuild", action="store_true", help="re-index the files in documents/")
    parser.add_argument("--chat", action="store_true", help="conversational mode with memory")
    return parser.parse_args()


def main() -> int:
    # Windows terminals often default to a non-UTF-8 code page; Bangla would crash print()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    args = parse_args()
    try:
        from src.production_rag import ProductionRAG

        rag = ProductionRAG()
        rag.initialize(rebuild=args.rebuild)
    except ConfigError as exc:
        print(f"❌ {exc}")
        return 1
    except Exception as exc:  # e.g. invalid API key during first indexing
        print(f"❌ {explain_openai_error(exc)}")
        return 1

    mode = "Chat mode (remembers earlier questions)" if args.chat else "Q&A mode"
    print(f"\n💬 {mode}. Type exit / quit / q to leave.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in config.EXIT_COMMANDS:
            break

        try:
            result = rag.chat_query(question) if args.chat else rag.query(question)
        except ConfigError as exc:
            print(f"❌ {exc}\n")
            continue
        except Exception as exc:  # show API/network errors, keep the loop alive
            print(f"❌ Could not generate an answer: {explain_openai_error(exc)}\n")
            continue

        print(f"\nAnswer: {result['answer']}")
        sources = format_sources(result["sources"])
        if sources:
            print("\nSources:")
            for line in sources:
                print(f"  - {line}")
        print()

    print("👋 Goodbye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
