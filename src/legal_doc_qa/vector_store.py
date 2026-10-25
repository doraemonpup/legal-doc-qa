from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma

from legal_doc_qa.config import Settings, get_embeddings


def create_store(docs: List[Document], settings: Settings) -> Chroma:
    """Create a new Chroma vector store for the uploaded document."""
    persist_dir = Path(settings.chroma_persist_dir)

    # Reset the store on every upload so chunks from a previous document
    # don't leak into retrieval for the new one.
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings(settings)
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
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


def get_retriever(vectorstore: Chroma, k: int = 5) -> BaseRetriever:
    """Get a retriever configured with the specified number of results."""
    return vectorstore.as_retriever(search_kwargs={"k": k})
