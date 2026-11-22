"""Question -> retrieved context -> answer (FR9)."""

from typing import Any, Dict, List

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

SYSTEM_PROMPT = (
    "You are an assistant that answers questions using ONLY the context below.\n"
    "If the answer is not in the context, say you don't know — do not make anything up.\n"
    "Answer in the same language as the question. Keep the answer concise "
    "(at most a few sentences).\n\n"
    "Context:\n{context}"
)


def format_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise chain output to the stable {"answer", "sources"} shape (CLAUDE.md rule 5)."""
    docs: List[Document] = result.get("context", [])
    return {"answer": result.get("answer", ""), "sources": [d.metadata for d in docs]}


class RAGChain:
    """Stuff the top-k chunks into one prompt and ask the LLM once."""

    def __init__(self, retriever: BaseRetriever, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{input}")]
        )
        qa_chain = create_stuff_documents_chain(llm, prompt)
        self.chain = create_retrieval_chain(retriever, qa_chain)

    def ask(self, question: str) -> Dict[str, Any]:
        return format_result(self.chain.invoke({"input": question}))
