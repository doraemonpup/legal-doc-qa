# legal-doc-qa

Cross-Lingual Contract Analyzer — upload a contract PDF and get key findings plus a multilingual Q&A chat.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

Copy `.env.example` to `.env` and fill in your keys if using OpenAI/OpenRouter.

## Run

```sh
uv run streamlit run src/legal_doc_qa/app.py
```
