from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from legal_doc_qa.config import Settings


def _cache_path(settings: Settings) -> Path:
    return Path(settings.chroma_persist_dir).with_name("summary_cache.json")


def save_summary(settings: Settings, filename: str, summary: str, questions: Optional[List[str]] = None) -> None:
    """Persist the key-findings summary and suggested questions next to the vector store."""
    path = _cache_path(settings)
    path.write_text(
        json.dumps(
            {"filename": filename, "summary": summary, "questions": questions or []},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_summary(settings: Settings) -> Optional[dict]:
    """Return {"filename", "summary"} if a valid cache exists, else None."""
    path = _cache_path(settings)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict) and data.get("summary"):
        return data
    return None


def clear_summary(settings: Settings) -> None:
    _cache_path(settings).unlink(missing_ok=True)