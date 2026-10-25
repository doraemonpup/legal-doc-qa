from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from legal_doc_qa.config import get_llm, Settings
from legal_doc_qa.prompts import SUMMARY_PROMPT


def generate_summary(docs: List[Document], settings: Settings) -> str:
    """Generate key findings summary from document chunks."""
    if not docs:
        return "No documents to summarize."

    llm = get_llm(settings)

    # Rough cap on prompt size (~4000 tokens, approx 3.5 chars per token).
    max_chars = 14_000

    context_parts = []
    current_len = 0

    for doc in docs:
        text = doc.page_content
        if current_len + len(text) > max_chars:
            break
        context_parts.append(text)
        current_len += len(text)

    context = "\n\n".join(context_parts)

    messages = SUMMARY_PROMPT.format_messages(context=context)

    response = llm.invoke(messages)
    return response.content
