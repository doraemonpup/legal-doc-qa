from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5"))
    openrouter_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    openrouter_model: str = field(default_factory=lambda: os.getenv("OPENROUTER_MODEL", "openrouter/owl-alpha"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"))
    chroma_persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")))
    k_retrieval: int = field(default_factory=lambda: int(os.getenv("K_RETRIEVAL", "5")))

    def __post_init__(self):
        self.llm_provider = self.llm_provider.lower()


def get_llm(settings: Settings):
    provider = settings.llm_provider
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            default_headers={"HTTP-Referer": "http://localhost:8501"},
        )
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=settings.ollama_model,
        temperature=0,
    )


def get_embeddings(settings: Settings):
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
    )
