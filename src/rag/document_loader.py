"""PDF 載入、Metadata 正規化與中文文字切分。"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import BinaryIO, Sequence

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings

SOURCE_FILENAME_KEY = "source_filename"
PAGE_NUMBER_KEY = "page_number"
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
PDF_PROCESSING_CACHE_VERSION = 1
SPLITTER_ADD_START_INDEX = True


class PDFProcessingError(RuntimeError):
    """代表使用者可理解的 PDF 處理錯誤。"""


@dataclass(frozen=True, slots=True)
class DocumentBatch:
    document_count: int
    page_count: int
    chunks: list[Document]
    parsed_text_by_source: dict[str, str] = field(default_factory=dict)


def _package_version(distribution_name: str) -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "unknown"


def pdf_processing_cache_settings(settings: Settings) -> dict[str, object]:
    """回傳所有會影響 PDF 解析、Metadata 或 Chunk 內容的快取設定。"""

    return {
        "cache_version": PDF_PROCESSING_CACHE_VERSION,
        "pdf_loader": "langchain_community.document_loaders.PyPDFLoader",
        "text_splitter": (
            "langchain_text_splitters.RecursiveCharacterTextSplitter"
        ),
        "pypdf_version": _package_version("pypdf"),
        "langchain_community_version": _package_version("langchain-community"),
        "langchain_text_splitters_version": _package_version(
            "langchain-text-splitters"
        ),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "separators": CHINESE_SEPARATORS,
        "add_start_index": SPLITTER_ADD_START_INDEX,
        "source_filename_key": SOURCE_FILENAME_KEY,
        "page_number_key": PAGE_NUMBER_KEY,
        "page_number_base": 1,
    }


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
        add_start_index=SPLITTER_ADD_START_INDEX,
    )
    return splitter.split_documents(normalized_pages)


def process_uploaded_pdfs(
    uploaded_files: Sequence[BinaryIO], settings: Settings
) -> DocumentBatch:
    """解析多份 PDF 並回傳統計資料與全部文字區塊。"""

    all_chunks: list[Document] = []
    parsed_text_by_source: dict[str, str] = {}
    total_pages = 0
    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", "未命名.pdf")
        pages = _load_pdf_pages(uploaded_file)
        total_pages += len(pages)
        if not any(page.page_content.strip() for page in pages):
            raise PDFProcessingError(
                f"PDF「{filename}」沒有可解析的文字；掃描型 PDF 可能需要先進行 OCR。"
            )
        parsed_text = "\n".join(page.page_content for page in pages)
        if filename in parsed_text_by_source:
            parsed_text = f"{parsed_text_by_source[filename]}\n{parsed_text}"
        parsed_text_by_source[filename] = parsed_text
        chunks = split_pdf_pages(pages, filename, settings)
        all_chunks.extend(chunk for chunk in chunks if chunk.page_content.strip())

    if not all_chunks:
        raise PDFProcessingError("PDF 沒有可建立索引的文字；掃描型 PDF 可能需要先進行 OCR。")
    return DocumentBatch(
        len(uploaded_files),
        total_pages,
        all_chunks,
        parsed_text_by_source,
    )
