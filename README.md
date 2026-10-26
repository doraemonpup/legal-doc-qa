# legal-doc-qa

Legal Document Analyzer — upload a contract PDF (in any language) and get a key-findings summary plus a cross-lingual Q&A chat with verbatim source quotes.

## Features

- 📄 **PDF upload** — pdfplumber extracts text and tables; one corrupt page won't fail the whole document (up to 50MB).
- 🔑 **Key findings summary** — critical clauses, obligations, hidden risks, governing law.
  Map-reduce over the whole document, so long contracts are fully covered; findings cite
  `[Page N]` references.
- 💬 **Cross-lingual Q&A** — ask in your language; answers quote the original source text.
  Hybrid retrieval (BM25 + vector) catches exact clause numbers, amounts, and dates;
  follow-ups are rewritten with chat history; answers stream in as they're generated.
- 📜 **Evidence styling** — verbatim contract quotes render as tinted blockquotes
  (`📜 Original` / `🌐 Translation`), clearly separated from the analyst's own text.
- 💡 **Suggested questions** — 3–5 clickable follow-ups generated from the summary.
- 🔄 **Session restore** — a page refresh reloads the document, summary, suggestions, and chat from disk.
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

- Vector state persists in `./chroma_db` (gitignored) with the summary cached in
  `./summary_cache.json`. The store is reset on every upload, so only the current
  document is searchable. On startup the app restores the previous session from disk;
  "New Document" wipes both. Delete the files to reset everything.
- Retrieved Q&A context is capped at 8k chars; the summary runs map-reduce in 14k-char
  batches so full contracts are covered.
- Retrieval is hybrid (BM25 + vector via reciprocal rank fusion). `RETRIEVAL_SCORE_THRESHOLD`
  (default 0.2) filters weak vector matches; when nothing survives, the app answers with a
  fixed not-found message without calling the LLM.
- Uploads run in a dedicated processing step (extraction → embedding → analysis) with live
  progress and per-stage timings; the intro text stays on screen while it works. Failures
  return to the upload screen with a 🔁 Try again gate. LLM calls time out after 90s —
  free-tier routers such as `openrouter/free` are rate-limited and can be slow.
