from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from legal_doc_qa.config import get_llm, Settings
from legal_doc_qa.prompts import SUMMARY_PROMPT, SUMMARY_REDUCE_PROMPT

# Rough cap on prompt size per batch (~4000 tokens, approx 3.5 chars per token).
MAX_BATCH_CHARS = 14_000


def _page_tag(doc: Document) -> str:
    page = doc.metadata.get("page", "?")
    return f"[Page {page}]"


def _batch_chunks(docs: List[Document]) -> List[List[Document]]:
    """Group chunks into batches that fit the per-call context budget."""
    batches: List[List[Document]] = []
    current: List[Document] = []
    size = 0
    for doc in docs:
        text = doc.page_content
        if current and size + len(text) > MAX_BATCH_CHARS:
            batches.append(current)
            current, size = [], 0
        current.append(doc)
        size += len(text)
    if current:
        batches.append(current)
    return batches


def _summarize_batch(llm, batch: List[Document]) -> str:
    parts = [f"{_page_tag(doc)}\n{doc.page_content}" for doc in batch]
    context = "\n\n".join(parts)
    messages = SUMMARY_PROMPT.format_messages(context=context)
    return llm.invoke(messages).content


def _reduce(llm, partials: List[str]) -> str:
    joined = "\n\n---\n\n".join(
        f"### Section {i + 1}\n{p}" for i, p in enumerate(partials)
    )
    messages = SUMMARY_REDUCE_PROMPT.format_messages(partials=joined)
    return llm.invoke(messages).content


def generate_summary(docs: List[Document], settings: Settings) -> str:
    """Generate key findings via map-reduce so every page is covered."""
    if not docs:
        return "No documents to summarize."

    llm = get_llm(settings)
    batches = _batch_chunks(docs)
    partials = [_summarize_batch(llm, batch) for batch in batches]

    if len(partials) == 1:
        return partials[0]
    return _reduce(llm, partials)