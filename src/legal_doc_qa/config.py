from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# Load .env only once
from dotenv import load_dotenv
load_dotenv()


class E5PrefixEmbeddings:
    """Wrap an HF embeddings model, adding the 'query:'/'passage:' prefixes
    required by the E5 family (intfloat/multilingual-e5-*) for best retrieval
    accuracy. Other models are passed through unchanged."""

    def __init__(self, model_name: str):
        from langchain_huggingface import HuggingFaceEmbeddings

        self._use_prefixes = "e5" in model_name.lower()
        self._inner = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._use_prefixes:
            texts = [f"passage: {t}" for t in texts]
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self._use_prefixes:
            text = f"query: {text}"
        return self._inner.embed_query(text)


@dataclass
class Settings:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5"))
    openrouter_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    openrouter_model: str = field(default_factory=lambda: os.getenv("OPENROUTER_MODEL", "openrouter/free"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3.5:9b"))
    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "huggingface"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"))
    chroma_persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")))
    k_retrieval: int = field(default_factory=lambda: int(os.getenv("K_RETRIEVAL", "5")))
    retrieval_score_threshold: float = field(default_factory=lambda: float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.2")))

    def __post_init__(self):
        self.llm_provider = self.llm_provider.lower()
        self.embedding_provider = self.embedding_provider.lower()
        self.validate()

    def validate(self) -> None:
        if self.llm_provider not in ("ollama", "openai", "openrouter"):
            raise ValueError(f"Invalid LLM_PROVIDER: {self.llm_provider}")

        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for openai provider")
        if self.llm_provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY required for openrouter provider")

        if self.embedding_provider not in ("huggingface", "ollama"):
            raise ValueError(f"Invalid EMBEDDING_PROVIDER: {self.embedding_provider}")


LLM_TIMEOUT_SECONDS = 90
LLM_MAX_RETRIES = 2


def get_llm(settings: Settings):
    provider = settings.llm_provider
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            api_key=settings.openai_api_key,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            default_headers={"HTTP-Referer": "http://localhost:8501"},
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=settings.ollama_model,
        temperature=0,
        client_kwargs={"timeout": LLM_TIMEOUT_SECONDS},
    )


def get_embeddings(settings: Settings):
    if settings.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=settings.embedding_model,
            dimensions=768,
        )
    return E5PrefixEmbeddings(settings.embedding_model)
