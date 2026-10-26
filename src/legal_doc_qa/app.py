from __future__ import annotations

import time

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

# Anything inside a blockquote is original contract text (styled distinctly);
# everything outside is the analyst's voice.
EVIDENCE_CSS = """
<style>
.stMarkdown blockquote {
    background: rgba(76, 126, 242, 0.08);
    border-left: 4px solid #4c7ef3;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
}
</style>
"""

NOT_FOUND_MESSAGE = (
    "🔍 I couldn't find anything about that in this document. "
    "Try rephrasing your question — it may not be covered by this contract."
)

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
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "processed_sig" not in st.session_state:
        st.session_state.processed_sig = None
    if "failed_sig" not in st.session_state:
        st.session_state.failed_sig = None
    if "welcomed" not in st.session_state:
        st.session_state.welcomed = False

def clear_session():
    keys = [
        "doc_processed", "vectorstore", "summary", "messages", "filename",
        "page_warnings", "suggested_questions", "pending_question", "chunks",
        "processed_sig", "failed_sig",
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
            model_name = settings.ollama_model
        elif settings.llm_provider == "openrouter":
            model_name = settings.openrouter_model
        else:
            model_name = settings.openai_model
        st.caption(f"Model: {model_name}")
        if "free" in model_name.lower():
            st.caption("ℹ️ Free-tier models are rate-limited and can be slow.")

        if st.session_state.doc_processed:
            if st.button("🔄 New Document", use_container_width=True):
                clear_session()

        st.divider()
        st.caption("Legal Document Analyzer v1.4")

def handle_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name
    sig = (filename, getattr(uploaded_file, "size", None))
    settings = st.session_state.settings

    t0 = time.perf_counter()
    with st.spinner("📖 Extracting text from PDF..."):
        try:
            docs, page_warnings = process_pdf(file_bytes, settings, filename)
        except ValueError as e:
            st.session_state.failed_sig = sig
            st.error(str(e))
            return
    st.caption(f"📖 Extracted {len(docs)} chunks in {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    embed_bar = st.progress(0, text="🧠 Embedding chunks…")
    try:
        def embed_progress(done, total):
            embed_bar.progress(min(done / total, 1.0), text=f"🧠 Embedding chunks… {done}/{total}")

        vectorstore = create_store(docs, settings, progress_cb=embed_progress)
    except Exception as e:
        st.session_state.failed_sig = sig
        st.error(f"Failed to build the vector store: {e}")
        return
    st.caption(f"🧠 Embedded in {time.perf_counter() - t1:.1f}s")

    summary = ""
    t2 = time.perf_counter()
    with st.status("📋 Generating key findings summary...", expanded=True) as status:
        try:
            def progress(done, total):
                elapsed = time.perf_counter() - t2
                status.update(label=f"📋 Analyzed {done}/{total} sections… ({elapsed:.0f}s)")

            summary = st.write_stream(
                generate_summary_progressive(docs, settings, progress)
            ) or ""
            status.update(label="🔑 Key findings ready", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Summary failed", state="error")
            st.session_state.failed_sig = sig
            st.error(f"Failed to generate the key findings summary: {e}")
            return
    st.caption(f"📋 Summary generated in {time.perf_counter() - t2:.1f}s")

    suggested = []
    with st.spinner("💡 Preparing suggested questions..."):
        try:
            suggested = generate_suggested_questions(summary, settings)
        except Exception:
            suggested = []

    st.session_state.vectorstore = vectorstore
    st.session_state.chunks = docs
    st.session_state.summary = summary
    st.session_state.doc_processed = True
    st.session_state.filename = filename
    st.session_state.messages = []
    st.session_state.page_warnings = page_warnings
    st.session_state.suggested_questions = suggested
    st.session_state.processed_sig = sig
    st.session_state.failed_sig = None

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
    st.session_state.chunks = docs_from_store(store)
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
                    st.session_state.vectorstore,
                    settings,
                    prompt,
                    history,
                    chunks=st.session_state.chunks,
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages.pop()
            return

        if not source_docs:
            # Nothing above threshold — answer deterministically, skip the LLM.
            st.markdown(NOT_FOUND_MESSAGE)
            st.session_state.messages.append({
                "role": "assistant",
                "content": NOT_FOUND_MESSAGE,
            })
            return

        try:
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
                sig = (uploaded_file.name, getattr(uploaded_file, "size", None))
                if sig == st.session_state.failed_sig:
                    st.warning(f"⚠️ '{uploaded_file.name}' failed to process earlier.")
                    if st.button("🔁 Try again"):
                        st.session_state.failed_sig = None
                        st.rerun()
                else:
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
    st.markdown(EVIDENCE_CSS, unsafe_allow_html=True)
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