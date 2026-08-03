from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from legal_doc_qa.config import Settings, get_embeddings


def create_store(docs: List[Document], settings: Settings) -> Chroma:
    embeddings = get_embeddings(settings)
    persist_dir = settings.chroma_persist_dir

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def load_store(settings: Settings) -> Optional[Chroma]:
    persist_dir = settings.chroma_persist_dir
    if not os.path.isdir(persist_dir):
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


def get_retriever(vectorstore: Chroma, k: int = 5):
    return vectorstore.as_retriever(search_kwargs={"k": k})
