from __future__ import annotations

from typing import Generator, List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from legal_doc_qa.config import get_llm, Settings
from legal_doc_qa.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from legal_doc_qa.vector_store import get_retriever

# Bound the history sent to the condense prompt: last 4 exchanges.
MAX_HISTORY_MESSAGES = 8

# Reciprocal Rank Fusion constant (standard value from the RRF paper).
RRF_K = 60


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


def _condense_query(settings: Settings, question: str, chat_history: List[dict]) -> str:
    llm = get_llm(settings)
    messages = CONDENSE_QUESTION_PROMPT.format_messages(
        chat_history=format_chat_history(chat_history),
        question=question,
    )
    condensed = llm.invoke(messages).content.strip()
    return condensed or question


def retrieve_context(
    vectorstore,
    settings: Settings,
    question: str,
    chat_history: Optional[List[dict]] = None,
    chunks: Optional[List[Document]] = None,
) -> List[Document]:
    """Hybrid retrieval for a question, condensing it with chat history if present."""
    query = question
    if chat_history:
        query = _condense_query(settings, question, chat_history)
    return hybrid_retrieve(vectorstore, chunks, settings, query)


def hybrid_retrieve(
    vectorstore,
    chunks: Optional[List[Document]],
    settings: Settings,
    query: str,
) -> List[Document]:
    """Fuse BM25 keyword hits with vector hits via Reciprocal Rank Fusion.

    The vector relevance threshold is the primary out-of-scope guard; BM25 adds
    exact-identifier matching (clause numbers, amounts, dates). Falls back to
    vector-only when no chunks are available.
    """
    k = settings.k_retrieval
    vector_hits = _vector_candidates(
        vectorstore, query, k, settings.retrieval_score_threshold
    )
    keyword_hits = _bm25_candidates(chunks or [], query, k)

    if not vector_hits and not keyword_hits:
        return []

    fused = _rrf_fuse([doc for doc, _ in vector_hits], keyword_hits)
    return fused[:k]


def _vector_candidates(vectorstore, query: str, k: int, threshold: float):
    """Vector hits with relevance scores, filtered by the threshold."""
    try:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        try:
            results = [(d, 1.0) for d in vectorstore.similarity_search(query, k=k)]
        except Exception:
            return []
    return [(doc, score) for doc, score in results if score >= threshold]


def _tokenize(text: str) -> List[str]:
    # Whitespace tokenization: fine for European languages, weak for CJK —
    # semantic search covers those cases.
    return text.lower().split()


def _bm25_candidates(chunks: List[Document], query: str, k: int) -> List[Document]:
    """Top-k chunks by BM25 keyword score, keeping only positive scores."""
    if not chunks:
        return []
    corpus = [_tokenize(doc.page_content) for doc in chunks]
    if not any(corpus):
        return []
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(scores, range(len(chunks))), key=lambda pair: -pair[0])
    return [chunks[i] for score, i in ranked[:k] if score > 0]


def _rrf_fuse(*ranked_lists: List[Document], k_rrf: int = RRF_K) -> List[Document]:
    """Reciprocal Rank Fusion across ranked candidate lists, deduped by content."""
    scores: dict = {}
    docs_by_key: dict = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.page_content
            docs_by_key.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank + 1)
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    return [docs_by_key[key] for key, _ in ordered]


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