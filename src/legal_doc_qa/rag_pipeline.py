from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from legal_doc_qa.config import get_llm, Settings
from legal_doc_qa.prompts import QA_PROMPT
from legal_doc_qa.vector_store import get_retriever


def build_qa_chain(vectorstore, settings: Settings):
    llm = get_llm(settings)
    retriever = get_retriever(vectorstore, k=settings.k_retrieval)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "input": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source


def answer_question(chain, question: str):
    result = chain.invoke(question)
    return {
        "answer": result["answer"],
        "source_docs": result["context"],
    }
