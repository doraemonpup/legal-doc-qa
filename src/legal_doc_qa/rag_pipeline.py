from __future__ import annotations

from typing import Generator, List, Optional

from langchain_core.documents import Document

from legal_doc_qa.config import get_llm, Settings
from legal_doc_qa.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from legal_doc_qa.vector_store import get_retriever

# Bound the history sent to the condense prompt: last 4 exchanges.
MAX_HISTORY_MESSAGES = 8


def format_docs(docs: List[Document], max_chars: int = 8000) -> str:
    """Format documents with configurable truncation to prevent prompt overflow."""
    texts = []
    total = 0
    for doc in docs:
        text = doc.page_content
        remaining = max_chars - total
        if remaining <= 100:
            break
        if len(text) > remaining:
            texts.append(text[:remaining] + "...")
            break
        texts.append(text)
        total += len(text)
    return "\n\n".join(texts)


def format_chat_history(chat_history: List[dict]) -> str:
    """Render recent messages as 'Human:/AI:' lines for the condense prompt."""
    msgs = [m for m in chat_history if m.get("role") in ("user", "assistant")]
    msgs = msgs[-MAX_HISTORY_MESSAGES:]
    return "\n".join(
        f"{'Human' if m['role'] == 'user' else 'AI'}: {m['content']}" for m in msgs
    )


def retrieve_context(
    vectorstore,
    settings: Settings,
    question: str,
    chat_history: Optional[List[dict]] = None,
) -> List[Document]:
    """Retrieve chunks for a question, condensing it with chat history if present."""
    query = question
    if chat_history:
        llm = get_llm(settings)
        messages = CONDENSE_QUESTION_PROMPT.format_messages(
            chat_history=format_chat_history(chat_history),
            question=question,
        )
        condensed = llm.invoke(messages).content.strip()
        if condensed:
            query = condensed

    retriever = get_retriever(vectorstore, k=settings.k_retrieval)
    return retriever.invoke(query)


def stream_answer(
    settings: Settings,
    question: str,
    source_docs: List[Document],
) -> Generator[str, None, None]:
    """Yield answer tokens for a question given already-retrieved context."""
    llm = get_llm(settings)
    messages = QA_PROMPT.format_messages(
        input=question, context=format_docs(source_docs)
    )
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content