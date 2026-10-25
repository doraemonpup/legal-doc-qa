from __future__ import annotations

from typing import List, Union
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document

from legal_doc_qa.config import get_llm, Settings
from legal_doc_qa.prompts import QA_PROMPT
from legal_doc_qa.vector_store import get_retriever


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


def build_qa_chain(vectorstore, settings: Settings):
    llm = get_llm(settings)
    retriever = get_retriever(
        vectorstore,
        k=settings.k_retrieval
    )

    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(x["context"])
        )
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    return RunnableParallel(
        {"context": retriever, "input": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)


def answer_question(chain, question: str) -> dict[str, Union[str, List[Document]]]:
    result = chain.invoke(question)
    return {
        "answer": result["answer"],
        "source_docs": result["context"],
    }
