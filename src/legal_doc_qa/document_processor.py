from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from legal_doc_qa.config import Settings

MAX_PDF_SIZE_MB = 50

def extract_text_and_tables(file_bytes: bytes, filename: str = "document.pdf") -> List[Tuple[int, str]]:
    """Extract text and tables from PDF with robust error handling."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        raise ValueError(f"PDF too large ({size_mb:.1f}MB). Max: {MAX_PDF_SIZE_MB}MB")

    pages_content = []
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix=f"legalqa_{filename}_") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        with pdfplumber.open(tmp_path) as pdf:
            if not pdf.pages:
                raise ValueError("PDF has no pages")

            for page in pdf.pages:
                try:
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

                except Exception as page_error:
                    print(f"⚠️ Warning: Skipped page {page.page_number} - {str(page_error)}")

    except pdfplumber.pdfminer.pdfdocument.PDFPasswordIncorrect:
        raise ValueError("PDF is encrypted with a password. Remove password protection or provide it.")
    except Exception as e:
        raise ValueError(f"Failed to process PDF: {str(e)}")
    finally:
        if tmp_path:
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
    """Convert page content to Document objects and split into chunks."""
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
    """Main entry point: process PDF bytes into Document objects."""
    pages_content = extract_text_and_tables(file_bytes, filename)
    chunked_docs = chunk_documents(pages_content, settings, filename)
    return chunked_docs
