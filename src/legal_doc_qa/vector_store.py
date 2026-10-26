from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma

from legal_doc_qa.config import Settings, get_embeddings
from legal_doc_qa.summary_cache import clear_summary

# Chunks embedded per add_documents call so progress can be reported.
EMBED_BATCH_SIZE = 32


def create_store(docs: List[Document], settings: Settings, progress_cb=None) -> Chroma:
    """Create a new Chroma vector store for the uploaded document.

    Embeds in batches so a progress callback can report incremental progress.
    """
    persist_dir = Path(settings.chroma_persist_dir)

    # Reset the store on every upload so chunks from a previous document
    # don't leak into retrieval for the new one.
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings(settings)

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    total = len(docs)
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = docs[start:start + EMBED_BATCH_SIZE]
        vectorstore.add_documents(batch)
        if progress_cb:
            progress_cb(min(start + EMBED_BATCH_SIZE, total), total)
    return vectorstore


def load_store(settings: Settings) -> Optional[Chroma]:
    """Load an existing Chroma vector store if it exists and contains data."""
    persist_dir = settings.chroma_persist_dir
    if not Path(persist_dir).is_dir():
        return None

    try:
        embeddings = get_embeddings(settings)
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        if vectorstore._collection.count() == 0:
            return None
        return vectorstore
    except Exception:
        return None


def docs_from_store(vectorstore: Chroma) -> List[Document]:
    """Rebuild Document objects from everything currently in the store."""
    data = vectorstore._collection.get(include=["documents", "metadatas"])
    docs = []
    for text, meta in zip(data.get("documents") or [], data.get("metadatas") or []):
        docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


def reset_persisted_state(settings: Settings) -> None:
    """Delete the persisted vector store and summary cache."""
    persist_dir = Path(settings.chroma_persist_dir)
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    clear_summary(settings)


def get_retriever(vectorstore: Chroma, k: int = 5) -> BaseRetriever:
    """Get a retriever configured with the specified number of results."""
    return vectorstore.as_retriever(search_kwargs={"k": k})
