"""PDF 載入、Metadata 正規化與中文文字切分。"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings

SOURCE_FILENAME_KEY = "source_filename"
PAGE_NUMBER_KEY = "page_number"
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


class PDFProcessingError(RuntimeError):
    """代表使用者可理解的 PDF 處理錯誤。"""


@dataclass(frozen=True, slots=True)
class DocumentBatch:
    document_count: int
    page_count: int
    chunks: list[Document]


def _read_upload(uploaded_file: BinaryIO) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()  # type: ignore[no-any-return]
    content = uploaded_file.read()
    return content if isinstance(content, bytes) else bytes(content)


def _load_pdf_pages(uploaded_file: BinaryIO) -> list[Document]:
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(_read_upload(uploaded_file))
            temp_path = temp_file.name
        return PyPDFLoader(temp_path).load()
    except Exception as exc:
        filename = getattr(uploaded_file, "name", "未命名.pdf")
        raise PDFProcessingError(f"無法解析 PDF「{filename}」，請確認檔案未損毀或加密。") from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def normalize_page_metadata(pages: Sequence[Document], filename: str) -> list[Document]:
    """將 PyPDFLoader 的零起算頁碼轉成適合顯示的一起算頁碼。"""

    normalized: list[Document] = []
    for index, page in enumerate(pages):
        metadata = dict(page.metadata)
        raw_page = metadata.get("page", index)
        try:
            page_number = int(raw_page) + 1
        except (TypeError, ValueError):
            page_number = index + 1
        metadata.update(
            {
                "source": filename,
                SOURCE_FILENAME_KEY: filename,
                PAGE_NUMBER_KEY: page_number,
            }
        )
        normalized.append(Document(page_content=page.page_content, metadata=metadata))
    return normalized


def split_pdf_pages(
    pages: Sequence[Document], filename: str, settings: Settings
) -> list[Document]:
    normalized_pages = normalize_page_metadata(pages, filename)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=CHINESE_SEPARATORS,
        add_start_index=True,
    )
    return splitter.split_documents(normalized_pages)


def process_uploaded_pdfs(
    uploaded_files: Sequence[BinaryIO], settings: Settings
) -> DocumentBatch:
    """解析多份 PDF 並回傳統計資料與全部文字區塊。"""

    all_chunks: list[Document] = []
    total_pages = 0
    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", "未命名.pdf")
        pages = _load_pdf_pages(uploaded_file)
        total_pages += len(pages)
        if not any(page.page_content.strip() for page in pages):
            raise PDFProcessingError(
                f"PDF「{filename}」沒有可解析的文字；掃描型 PDF 可能需要先進行 OCR。"
            )
        chunks = split_pdf_pages(pages, filename, settings)
        all_chunks.extend(chunk for chunk in chunks if chunk.page_content.strip())

    if not all_chunks:
        raise PDFProcessingError("PDF 沒有可建立索引的文字；掃描型 PDF 可能需要先進行 OCR。")
    return DocumentBatch(len(uploaded_files), total_pages, all_chunks)
