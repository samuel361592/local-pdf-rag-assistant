"""PDF 載入、Metadata 正規化與中文文字切分。"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
import logging
from pathlib import Path
from typing import BinaryIO, Sequence

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .ocr_service import (
    OCR_ENGINE_NAME,
    OCRError,
    OCRService,
    SUPPORTED_IMAGE_SUFFIXES,
    package_version,
)

SOURCE_FILENAME_KEY = "source_filename"
PAGE_NUMBER_KEY = "page_number"
SOURCE_TYPE_KEY = "source_type"
TEXT_EXTRACTION_METHOD_KEY = "text_extraction_method"
OCR_CONFIDENCE_KEY = "ocr_confidence"
OCR_ENGINE_KEY = "ocr_engine"
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
PDF_PROCESSING_CACHE_VERSION = 2
SPLITTER_ADD_START_INDEX = True
PDF_SOURCE_TYPE = "pdf"
IMAGE_SOURCE_TYPE = "image"
NATIVE_EXTRACTION_METHOD = "native"
OCR_EXTRACTION_METHOD = "ocr"
LOGGER = logging.getLogger(__name__)


class PDFProcessingError(RuntimeError):
    """代表使用者可理解的 PDF 處理錯誤。"""


def _ocr_user_message(filename: str) -> str:
    return (
        f"無法完成「{filename}」的 OCR 辨識。請確認 OCR 相依套件已安裝、"
        "模型已下載完成，或改用較清晰的圖片/PDF 後再試。"
    )


@dataclass(frozen=True, slots=True)
class DocumentExtractionStats:
    native_page_count: int = 0
    ocr_page_count: int = 0
    image_page_count: int = 0
    ocr_confidences: tuple[float, ...] = ()

    @property
    def average_ocr_confidence(self) -> float | None:
        if not self.ocr_confidences:
            return None
        return sum(self.ocr_confidences) / len(self.ocr_confidences)


@dataclass(frozen=True, slots=True)
class DocumentBatch:
    document_count: int
    page_count: int
    chunks: list[Document]
    parsed_text_by_source: dict[str, str] = field(default_factory=dict)
    ocr_text_by_source: dict[str, str] = field(default_factory=dict)
    extraction_stats: DocumentExtractionStats = field(
        default_factory=DocumentExtractionStats
    )


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
        "pymupdf_version": package_version("pymupdf"),
        "paddleocr_version": package_version("paddleocr"),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "separators": CHINESE_SEPARATORS,
        "add_start_index": SPLITTER_ADD_START_INDEX,
        "source_filename_key": SOURCE_FILENAME_KEY,
        "page_number_key": PAGE_NUMBER_KEY,
        "page_number_base": 1,
        "ocr_mode": settings.ocr_mode,
        "ocr_lang": settings.ocr_lang,
        "ocr_min_text_chars": settings.ocr_min_text_chars,
        "ocr_dpi": settings.ocr_dpi,
        "ocr_enable_images": settings.ocr_enable_images,
        "ocr_enable_mkldnn": settings.ocr_enable_mkldnn,
        "ocr_engine": OCR_ENGINE_NAME,
    }


def _read_upload(uploaded_file: BinaryIO) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()  # type: ignore[no-any-return]
    content = uploaded_file.read()
    return content if isinstance(content, bytes) else bytes(content)


def _write_temp_upload(uploaded_file: BinaryIO, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(_read_upload(uploaded_file))
        return Path(temp_file.name)


def _needs_ocr(page: Document, settings: Settings) -> bool:
    if settings.ocr_mode == "disabled":
        return False
    if settings.ocr_mode == "force":
        return True
    return len(page.page_content.strip()) < settings.ocr_min_text_chars


def _load_pdf_pages(
    uploaded_file: BinaryIO,
    settings: Settings,
    ocr_service: OCRService | None = None,
) -> tuple[list[Document], DocumentExtractionStats, dict[str, str]]:
    temp_path: str | None = None
    try:
        temp_path = str(_write_temp_upload(uploaded_file, ".pdf"))
        pages = PyPDFLoader(temp_path).load()
        native_count = 0
        ocr_count = 0
        ocr_confidences: list[float] = []
        ocr_text_by_source: dict[str, str] = {}
        filename = getattr(uploaded_file, "name", "未命名.pdf")
        service = ocr_service or OCRService(settings)

        processed_pages: list[Document] = []
        for index, page in enumerate(pages):
            metadata = dict(page.metadata)
            metadata[SOURCE_TYPE_KEY] = PDF_SOURCE_TYPE
            if _needs_ocr(page, settings):
                ocr_result = service.pdf_page_to_text(Path(temp_path), index)
                metadata[TEXT_EXTRACTION_METHOD_KEY] = OCR_EXTRACTION_METHOD
                metadata[OCR_ENGINE_KEY] = OCR_ENGINE_NAME
                if ocr_result.confidence is not None:
                    metadata[OCR_CONFIDENCE_KEY] = ocr_result.confidence
                    ocr_confidences.append(ocr_result.confidence)
                ocr_text = ocr_result.text.strip()
                if ocr_text:
                    ocr_text_by_source[f"{filename}#page={index + 1}"] = ocr_text
                page_content = ocr_text
                ocr_count += 1
            else:
                metadata[TEXT_EXTRACTION_METHOD_KEY] = NATIVE_EXTRACTION_METHOD
                page_content = page.page_content
                native_count += 1
            processed_pages.append(
                Document(page_content=page_content, metadata=metadata)
            )

        return (
            processed_pages,
            DocumentExtractionStats(
                native_page_count=native_count,
                ocr_page_count=ocr_count,
                ocr_confidences=tuple(ocr_confidences),
            ),
            ocr_text_by_source,
        )
    except Exception as exc:
        filename = getattr(uploaded_file, "name", "未命名.pdf")
        if isinstance(exc, OCRError):
            LOGGER.warning("OCR failed while parsing PDF %r.", filename, exc_info=True)
            raise PDFProcessingError(_ocr_user_message(filename)) from exc
        raise PDFProcessingError(f"無法解析 PDF「{filename}」，請確認檔案未損毀或加密。") from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def _load_image_page(
    uploaded_file: BinaryIO,
    settings: Settings,
    ocr_service: OCRService | None = None,
) -> tuple[list[Document], DocumentExtractionStats, dict[str, str]]:
    filename = getattr(uploaded_file, "name", "未命名圖片")
    if not settings.ocr_enable_images:
        raise PDFProcessingError("圖片檔支援已停用，請上傳 PDF 或啟用 OCR 圖片支援。")
    if settings.ocr_mode == "disabled":
        raise PDFProcessingError("圖片檔需要 OCR；請將 OCR 模式改為自動或強制。")

    temp_path: Path | None = None
    try:
        suffix = Path(filename).suffix.lower() or ".png"
        temp_path = _write_temp_upload(uploaded_file, suffix)
        service = ocr_service or OCRService(settings)
        ocr_result = service.image_file_to_text(temp_path)
    except Exception as exc:
        if isinstance(exc, OCRError):
            LOGGER.warning("OCR failed while parsing image %r.", filename, exc_info=True)
            raise PDFProcessingError(_ocr_user_message(filename)) from exc
        raise PDFProcessingError(f"無法解析圖片「{filename}」。") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    metadata: dict[str, object] = {
        "source": filename,
        SOURCE_FILENAME_KEY: filename,
        PAGE_NUMBER_KEY: 1,
        SOURCE_TYPE_KEY: IMAGE_SOURCE_TYPE,
        TEXT_EXTRACTION_METHOD_KEY: OCR_EXTRACTION_METHOD,
        OCR_ENGINE_KEY: OCR_ENGINE_NAME,
    }
    confidences: tuple[float, ...] = ()
    if ocr_result.confidence is not None:
        metadata[OCR_CONFIDENCE_KEY] = ocr_result.confidence
        confidences = (ocr_result.confidence,)
    text = ocr_result.text.strip()
    return (
        [Document(page_content=text, metadata=metadata)],
        DocumentExtractionStats(
            ocr_page_count=1,
            image_page_count=1,
            ocr_confidences=confidences,
        ),
        {filename: text} if text else {},
    )


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
    """解析多份 PDF/圖片並回傳統計資料與全部文字區塊。"""

    all_chunks: list[Document] = []
    parsed_text_by_source: dict[str, str] = {}
    ocr_text_by_source: dict[str, str] = {}
    total_pages = 0
    native_pages = 0
    ocr_pages = 0
    image_pages = 0
    ocr_confidences: list[float] = []
    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", "未命名.pdf")
        suffix = Path(filename).suffix.lower()
        if suffix in SUPPORTED_IMAGE_SUFFIXES:
            pages, stats, source_ocr_text = _load_image_page(uploaded_file, settings)
        else:
            pages, stats, source_ocr_text = _load_pdf_pages(uploaded_file, settings)
        total_pages += len(pages)
        native_pages += stats.native_page_count
        ocr_pages += stats.ocr_page_count
        image_pages += stats.image_page_count
        ocr_confidences.extend(stats.ocr_confidences)
        if not any(page.page_content.strip() for page in pages):
            raise PDFProcessingError(
                f"檔案「{filename}」沒有可建立索引的文字；請確認 OCR 結果或檔案品質。"
            )
        parsed_text = "\n".join(page.page_content for page in pages)
        if filename in parsed_text_by_source:
            parsed_text = f"{parsed_text_by_source[filename]}\n{parsed_text}"
        parsed_text_by_source[filename] = parsed_text
        ocr_text_by_source.update(source_ocr_text)
        chunks = split_pdf_pages(pages, filename, settings)
        all_chunks.extend(chunk for chunk in chunks if chunk.page_content.strip())

    if not all_chunks:
        raise PDFProcessingError("檔案沒有可建立索引的文字；請確認 OCR 結果或檔案品質。")
    return DocumentBatch(
        len(uploaded_files),
        total_pages,
        all_chunks,
        parsed_text_by_source,
        ocr_text_by_source,
        DocumentExtractionStats(
            native_page_count=native_pages,
            ocr_page_count=ocr_pages,
            image_page_count=image_pages,
            ocr_confidences=tuple(ocr_confidences),
        ),
    )
