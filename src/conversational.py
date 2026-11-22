"""Chat mode with memory of earlier turns (FR12, FR13)."""

from typing import Any, Dict

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory

from src import config
from src.rag_chain import SYSTEM_PROMPT, format_result

CONTEXTUALIZE_PROMPT = (
    "Given the chat history and the latest user question, which may refer to the history, "
    "rewrite it as a standalone question that can be understood without the history. "
    "Do NOT answer it; only rewrite it if needed, otherwise return it as is."
)


class ConversationalRAG:
    """Rewrites follow-ups ("explain that more") into standalone questions before retrieval."""

    def __init__(self, retriever: BaseRetriever, llm: BaseChatModel) -> None:
        # History lives in memory only; lost on exit (FR13)
        self._store: Dict[str, InMemoryChatMessageHistory] = {}

        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        rag_chain = create_retrieval_chain(
            history_aware_retriever, create_stuff_documents_chain(llm, qa_prompt)
        )

        self.chain = RunnableWithMessageHistory(
            rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        return self._store[session_id]

    def ask(self, question: str, session_id: str = config.DEFAULT_SESSION_ID) -> Dict[str, Any]:
        result = self.chain.invoke(
            {"input": question}, config={"configurable": {"session_id": session_id}}
        )
        return format_result(result)

    def clear(self, session_id: str = config.DEFAULT_SESSION_ID) -> None:
        self._store.pop(session_id, None)
