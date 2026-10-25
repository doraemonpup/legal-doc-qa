# legal-doc-qa

Legal Document Analyzer — upload a contract PDF (in any language) and get a key-findings summary plus a cross-lingual Q&A chat with verbatim source quotes.

## Features

- 📄 **PDF upload** — pdfplumber extracts text and tables; one corrupt page won't fail the whole document (up to 50MB).
- 🔑 **Key findings summary** — critical clauses, obligations, hidden risks, governing law.
- 💬 **Cross-lingual Q&A** — ask in your language; answers quote the original source text.
- 🔒 **Local embeddings** — multilingual e5 model runs in-process; no embedding service needed.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

Copy `.env.example` to `.env` and configure:

| Variable | Values | Notes |
|---|---|---|
| `LLM_PROVIDER` | `openrouter` \| `openai` \| `ollama` | API key required for openrouter/openai; Ollama must be running for ollama |
| `EMBEDDING_PROVIDER` | `huggingface` \| `ollama` | Default `huggingface` — runs in-process, no external service |
| `EMBEDDING_MODEL` | HuggingFace model id | Default `intfloat/multilingual-e5-base` (multilingual, retrieval-tuned; `query:`/`passage:` prefixes applied automatically). Downloads on first run (~1.1GB). Changing it re-embeds on next upload. |

If `EMBEDDING_PROVIDER=ollama`, `EMBEDDING_MODEL` should be an Ollama model such as `qwen3-embedding:0.6b`.

## Run

```sh
uv run streamlit run src/legal_doc_qa/app.py
```

## Notes

- Vector state persists in `./chroma_db` (gitignored). The store is reset on every upload, so only the current document is searchable. Delete the directory to reset everything.
- Retrieved context is capped (8k chars for Q&A, ~14k for the summary) to avoid prompt overflow.
