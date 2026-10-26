from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Generator, List, Optional

from langchain_core.documents import Document

from legal_doc_qa.config import get_llm, Settings
from legal_doc_qa.prompts import (
    SUMMARY_PROMPT,
    SUMMARY_REDUCE_PROMPT,
    SUGGESTED_QUESTIONS_PROMPT,
)

# Rough cap on prompt size per batch (~4000 tokens, approx 3.5 chars per token).
MAX_BATCH_CHARS = 14_000

# Map steps run concurrently to cut wall-time on long contracts.
MAP_WORKERS = 3


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


def _batch_context(batch: List[Document]) -> str:
    return "\n\n".join(f"{_page_tag(doc)}\n{doc.page_content}" for doc in batch)


def _summarize_batch(llm, batch: List[Document]) -> str:
    messages = SUMMARY_PROMPT.format_messages(context=_batch_context(batch))
    return llm.invoke(messages).content


def _map_partials(llm, batches: List[List[Document]], progress_cb=None) -> List[str]:
    """Summarize all batches concurrently, preserving section order."""
    partials: List[Optional[str]] = [None] * len(batches)
    done = 0
    with ThreadPoolExecutor(max_workers=min(MAP_WORKERS, len(batches))) as pool:
        futures = {
            pool.submit(_summarize_batch, llm, batch): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            partials[futures[future]] = future.result()
            done += 1
            if progress_cb:
                progress_cb(done, len(batches))
    return partials


def _stream_reduce(llm, partials: List[str]) -> Generator[str, None, None]:
    joined = "\n\n---\n\n".join(
        f"### Section {i + 1}\n{p}" for i, p in enumerate(partials)
    )
    messages = SUMMARY_REDUCE_PROMPT.format_messages(partials=joined)
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def generate_summary_progressive(
    docs: List[Document],
    settings: Settings,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Generator[str, None, None]:
    """Yield the final summary token-by-token.

    Map steps run as blocking calls (progress_cb reports each), the single-batch
    fast path streams the one SUMMARY_PROMPT call directly, and the multi-batch
    path streams the reduce call. No LLM call is made twice.
    """
    if not docs:
        yield "No documents to summarize."
        return

    llm = get_llm(settings)
    batches = _batch_chunks(docs)

    def report(i: int) -> None:
        if progress_cb:
            progress_cb(i, len(batches))

    if len(batches) == 1:
        report(1)
        messages = SUMMARY_PROMPT.format_messages(context=_batch_context(batches[0]))
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content
        return

    partials = _map_partials(llm, batches, progress_cb)
    yield from _stream_reduce(llm, partials)


def generate_summary(docs: List[Document], settings: Settings) -> str:
    """Non-streaming key findings via map-reduce (every page is covered)."""
    if not docs:
        return "No documents to summarize."

    llm = get_llm(settings)
    partials = _map_partials(llm, _batch_chunks(docs))

    if len(partials) == 1:
        return partials[0]

    joined = "\n\n---\n\n".join(
        f"### Section {i + 1}\n{p}" for i, p in enumerate(partials)
    )
    messages = SUMMARY_REDUCE_PROMPT.format_messages(partials=joined)
    return llm.invoke(messages).content


def generate_suggested_questions(summary: str, settings: Settings) -> List[str]:
    """Generate 3-5 clickable follow-up questions from the final summary."""
    if not summary or summary == "No documents to summarize.":
        return []

    llm = get_llm(settings)
    messages = SUGGESTED_QUESTIONS_PROMPT.format_messages(summary=summary)
    text = llm.invoke(messages).content

    questions = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-•0123456789. ").strip()
        if cleaned:
            questions.append(cleaned)
    return questions[:5]