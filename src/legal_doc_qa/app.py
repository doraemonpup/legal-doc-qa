from __future__ import annotations

import streamlit as st

from legal_doc_qa.config import Settings
from legal_doc_qa.document_processor import process_pdf
from legal_doc_qa.vector_store import (
    create_store,
    docs_from_store,
    load_store,
    reset_persisted_state,
)
from legal_doc_qa.summary_cache import load_summary, save_summary
from legal_doc_qa.summarizer import generate_summary
from legal_doc_qa.rag_pipeline import build_qa_chain, answer_question

WELCOME_TEXT = """
# 📜 Legal Document Analyzer

Upload a PDF contract to get:
- **Key findings summary** (critical clauses, obligations, risks)
- **Q&A chat session** (ask questions in any language)

The analyzer will help you understand your contract's terms clearly.
"""

def init_session_state():
    if "settings" not in st.session_state:
        st.session_state.settings = Settings()
    if "doc_processed" not in st.session_state:
        st.session_state.doc_processed = False
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "filename" not in st.session_state:
        st.session_state.filename = None
    if "page_warnings" not in st.session_state:
        st.session_state.page_warnings = []
    if "welcomed" not in st.session_state:
        st.session_state.welcomed = False

def clear_session():
    keys = ["doc_processed", "vectorstore", "chain", "summary", "messages", "filename", "page_warnings"]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    if "welcomed" in st.session_state:
        del st.session_state["welcomed"]
    reset_persisted_state(st.session_state.settings)
    st.rerun()

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")

        settings = st.session_state.settings
        st.info(f"**LLM Provider:** `{settings.llm_provider}`")
        if settings.llm_provider == "ollama":
            st.caption(f"Model: {settings.ollama_model}")
        elif settings.llm_provider == "openrouter":
            st.caption(f"Model: {settings.openrouter_model}")
        else:
            st.caption(f"Model: {settings.openai_model}")

        if st.session_state.doc_processed:
            if st.button("🔄 New Document", use_container_width=True):
                clear_session()

        st.divider()
        st.caption("Legal Document Analyzer v1.1")

def handle_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name
    settings = st.session_state.settings

    with st.spinner("📖 Extracting text from PDF..."):
        try:
            docs, page_warnings = process_pdf(file_bytes, settings, filename)
        except ValueError as e:
            st.error(str(e))
            return

    with st.spinner("🧠 Embedding and storing in vector database..."):
        try:
            vectorstore = create_store(docs, settings)
        except Exception as e:
            st.error(f"Failed to build the vector store: {e}")
            return

    with st.spinner("📋 Generating key findings summary..."):
        try:
            summary = generate_summary(docs, settings)
        except Exception as e:
            st.error(f"Failed to generate the key findings summary: {e}")
            return

    chain = build_qa_chain(vectorstore, settings)

    st.session_state.vectorstore = vectorstore
    st.session_state.chain = chain
    st.session_state.summary = summary
    st.session_state.doc_processed = True
    st.session_state.filename = filename
    st.session_state.messages = []
    st.session_state.page_warnings = page_warnings

    save_summary(settings, filename, summary)

    st.rerun()

def try_restore_session():
    """Reload a previous document (vector store + cached summary) after a refresh."""
    if st.session_state.doc_processed:
        return
    settings = st.session_state.settings
    store = load_store(settings)
    if store is None:
        return

    cached = load_summary(settings)
    if cached is None:
        return

    try:
        chain = build_qa_chain(store, settings)
    except Exception:
        return

    st.session_state.vectorstore = store
    st.session_state.chain = chain
    st.session_state.summary = cached["summary"]
    st.session_state.filename = cached.get("filename") or "restored document"
    st.session_state.doc_processed = True

def render_summary():
    st.subheader("🔑 Key Findings Summary", divider="blue")
    st.markdown(st.session_state.summary or "_Summary unavailable — re-upload the document to regenerate it._")

def render_chat():
    st.subheader("💬 Ask Questions About Your Document", divider="blue")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "source_docs" in message:
                with st.expander("📄 Source References"):
                    for i, doc in enumerate(message["source_docs"]):
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Excerpt {i+1}** (Page {page}):")
                        st.markdown(f"> {doc.page_content[:500]}...")
                        st.divider()

    if prompt := st.chat_input("Ask a question in any language..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = answer_question(
                        st.session_state.chain,
                        prompt,
                    )
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.session_state.messages.pop()
                    return

            answer_text = result["answer"]
            source_docs = result["source_docs"]

            st.markdown(answer_text)

            if source_docs:
                with st.expander("📄 Source References"):
                    for i, doc in enumerate(source_docs):
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Excerpt {i+1}** (Page {page}):")
                        st.markdown(f"> {doc.page_content[:500]}...")
                        st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "source_docs": source_docs,
            })

def render_welcome():
    if not st.session_state.welcomed and not st.session_state.doc_processed:
        st.markdown(WELCOME_TEXT)
        st.session_state.welcomed = True

def render_upload():
    container = st.container()
    with container:
        if not st.session_state.doc_processed:
            st.subheader("📤 Upload Contract")
            uploaded_file = st.file_uploader(
                "Drag and drop your PDF here",
                type="pdf",
                label_visibility="collapsed",
                accept_multiple_files=False,
            )

            if uploaded_file is not None:
                handle_upload(uploaded_file)
        else:
            st.success(f"✅ **{st.session_state.filename}** processed successfully")
            if st.session_state.page_warnings:
                st.warning(
                    f"⚠️ {len(st.session_state.page_warnings)} page(s) could not be read and were skipped:\n\n"
                    + "\n".join(f"- {w}" for w in st.session_state.page_warnings)
                )
            if st.button("📤 Upload Different Document", type="secondary"):
                clear_session()

def main():
    init_session_state()
    try_restore_session()
    render_sidebar()
    render_welcome()
    render_upload()

    if st.session_state.doc_processed:
        render_summary()
        render_chat()

if __name__ == "__main__":
    main()
