from __future__ import annotations

import streamlit as st

from legal_doc_qa.config import Settings
from legal_doc_qa.document_processor import process_pdf
from legal_doc_qa.vector_store import (
    create_store,
    load_store,
    reset_persisted_state,
)
from legal_doc_qa.summary_cache import load_summary, save_summary
from legal_doc_qa.summarizer import (
    generate_suggested_questions,
    generate_summary_progressive,
)
from legal_doc_qa.rag_pipeline import retrieve_context, stream_answer

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
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "filename" not in st.session_state:
        st.session_state.filename = None
    if "page_warnings" not in st.session_state:
        st.session_state.page_warnings = []
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "welcomed" not in st.session_state:
        st.session_state.welcomed = False

def clear_session():
    keys = [
        "doc_processed", "vectorstore", "summary", "messages", "filename",
        "page_warnings", "suggested_questions", "pending_question",
    ]
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
        st.caption("Legal Document Analyzer v1.2")

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

    summary = ""
    with st.status("📋 Generating key findings summary...", expanded=True) as status:
        try:
            def progress(i, total):
                status.update(label=f"📋 Analyzing section {i}/{total}...")

            summary = st.write_stream(
                generate_summary_progressive(docs, settings, progress)
            ) or ""
            status.update(label="🔑 Key findings ready", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Summary failed", state="error")
            st.error(f"Failed to generate the key findings summary: {e}")
            return

    suggested = []
    with st.spinner("💡 Preparing suggested questions..."):
        try:
            suggested = generate_suggested_questions(summary, settings)
        except Exception:
            suggested = []

    st.session_state.vectorstore = vectorstore
    st.session_state.summary = summary
    st.session_state.doc_processed = True
    st.session_state.filename = filename
    st.session_state.messages = []
    st.session_state.page_warnings = page_warnings
    st.session_state.suggested_questions = suggested

    save_summary(settings, filename, summary, suggested)

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

    st.session_state.vectorstore = store
    st.session_state.summary = cached["summary"]
    st.session_state.filename = cached.get("filename") or "restored document"
    st.session_state.suggested_questions = cached.get("questions", [])
    st.session_state.doc_processed = True

def render_summary():
    st.subheader("🔑 Key Findings Summary", divider="blue")
    st.markdown(st.session_state.summary or "_Summary unavailable — re-upload the document to regenerate it._")

def render_suggestions():
    if not st.session_state.doc_processed:
        return
    questions = st.session_state.suggested_questions or []
    if not questions:
        return

    st.caption("💡 Suggested questions — click one to ask")
    cols = st.columns(min(3, len(questions)))
    for idx, question in enumerate(questions):
        with cols[idx % len(cols)]:
            if st.button(question, key=f"suggested_{idx}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()

def _render_sources(source_docs):
    if not source_docs:
        return
    with st.expander("📄 Source References"):
        for i, doc in enumerate(source_docs):
            page = doc.metadata.get("page", "?")
            st.markdown(f"**Excerpt {i + 1}** (Page {page}):")
            st.markdown(f"> {doc.page_content[:500]}...")
            st.divider()

def render_chat():
    st.subheader("💬 Ask Questions About Your Document", divider="blue")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "source_docs" in message:
                _render_sources(message["source_docs"])

    prompt = st.chat_input("Ask a question in any language...")
    if not prompt and st.session_state.pending_question:
        prompt = st.session_state.pending_question
        st.session_state.pending_question = None

    if not prompt:
        return

    settings = st.session_state.settings
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("🔍 Searching the document..."):
                # History excludes the question we just appended.
                history = st.session_state.messages[:-1]
                source_docs = retrieve_context(
                    st.session_state.vectorstore, settings, prompt, history
                )

            answer_text = st.write_stream(
                stream_answer(settings, prompt, source_docs)
            ) or ""
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages.pop()
            return

        _render_sources(source_docs)

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
        render_suggestions()
        render_chat()

if __name__ == "__main__":
    main()