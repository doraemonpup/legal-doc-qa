from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from utils.config import get_llm, Settings
from utils.prompts import SUMMARY_PROMPT


def generate_summary(docs: List[Document], settings: Settings) -> str:
    llm = get_llm(settings)

    context_parts = []
    char_limit = 6000
    current_len = 0

    for doc in docs:
        text = doc.page_content
        if current_len + len(text) > char_limit:
            remaining = char_limit - current_len
            if remaining > 200:
                context_parts.append(text[:remaining])
            break
        context_parts.append(text)
        current_len += len(text)

    context = "\n\n".join(context_parts)

    messages = SUMMARY_PROMPT.format_messages(context=context)
    response = llm.invoke(messages)
    return response.content
