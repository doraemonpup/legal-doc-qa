from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import List, Tuple

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from legal_doc_qa.config import Settings


def extract_text_and_tables(file_bytes: bytes) -> List[Tuple[int, str]]:
    pages_content = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        with pdfplumber.open(tmp_path) as pdf:
            if pdf.metadata and pdf.metadata.get("encrypted"):
                raise ValueError("PDF is encrypted and cannot be processed.")

            for page in pdf.pages:
                text = (page.extract_text() or "").strip()
                tables = page.extract_tables()

                table_md = ""
                if tables:
                    for table in tables:
                        if not table:
                            continue
                        header = table[0] if table else []
                        rows = table[1:] if len(table) > 1 else []
                        if header:
                            table_md += "| " + " | ".join(str(c or "") for c in header) + " |\n"
                            table_md += "| " + " | ".join("---" for _ in header) + " |\n"
                            for row in rows:
                                table_md += "| " + " | ".join(str(c or "") for c in row) + " |\n"
                            table_md += "\n"

                combined = text
                if table_md:
                    combined += "\n\n[Table Data]\n" + table_md

                if combined.strip():
                    pages_content.append((page.page_number, combined.strip()))

    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not pages_content:
        raise ValueError(
            "Could not extract any text from this PDF. "
            "It may be a scanned/image-based document. "
            "Try a document with selectable text."
        )

    return pages_content


def chunk_documents(
    pages_content: List[Tuple[int, str]],
    settings: Settings,
    filename: str = "document.pdf",
) -> List[Document]:
    docs = []
    for page_num, text in pages_content:
        doc = Document(
            page_content=text,
            metadata={
                "page": page_num,
                "source": filename,
            },
        )
        docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )

    chunked_docs = splitter.split_documents(docs)
    for i, doc in enumerate(chunked_docs):
        doc.metadata["chunk_index"] = i

    return chunked_docs


def process_pdf(file_bytes: bytes, settings: Settings, filename: str = "document.pdf") -> List[Document]:
    pages_content = extract_text_and_tables(file_bytes)
    chunked_docs = chunk_documents(pages_content, settings, filename)
    return chunked_docs
