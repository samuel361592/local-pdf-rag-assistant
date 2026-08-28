"""PDF/image loading, OCR/VLM enrichment, metadata normalization and chunking."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import BinaryIO, Sequence

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .image_renderer import (
    ImageRenderError,
    normalize_image,
    pdf_page_has_images,
    render_pdf_page_to_image,
)
from .ocr_service import (
    OCR_ENGINE_NAME,
    OCRError,
    OCRResult,
    OCRService,
    SUPPORTED_IMAGE_SUFFIXES,
    package_version,
)
from .visual_prompt import VISUAL_PROMPT_VERSION
from .vlm_service import (
    VLMError,
    VLMService,
    format_visual_analysis,
    has_substantive_visual_content,
)

SOURCE_FILENAME_KEY = "source_filename"
PAGE_NUMBER_KEY = "page_number"
SOURCE_TYPE_KEY = "source_type"
CONTENT_TYPE_KEY = "content_type"
TEXT_EXTRACTION_METHOD_KEY = "text_extraction_method"
OCR_CONFIDENCE_KEY = "ocr_confidence"
OCR_ENGINE_KEY = "ocr_engine"
VLM_MODEL_KEY = "vlm_model"
VLM_PROMPT_VERSION_KEY = "vlm_prompt_version"
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
PDF_PROCESSING_CACHE_VERSION = 3
SPLITTER_ADD_START_INDEX = True
PDF_SOURCE_TYPE = "pdf"
IMAGE_SOURCE_TYPE = "image"
TEXT_CONTENT_TYPE = "text"
VISUAL_CONTENT_TYPE = "visual"
NATIVE_EXTRACTION_METHOD = "native"
OCR_EXTRACTION_METHOD = "ocr"
VLM_EXTRACTION_METHOD = "vlm"
LOGGER = logging.getLogger(__name__)


class PDFProcessingError(RuntimeError):
    """代表使用者可理解的文件處理錯誤。"""


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
    vlm_page_count: int = 0
    vlm_failure_count: int = 0
    visual_chunk_count: int = 0

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
    visual_text_by_source: dict[str, str] = field(default_factory=dict)
    vlm_failures_by_source: dict[str, str] = field(default_factory=dict)


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
        "text_splitter": "langchain_text_splitters.RecursiveCharacterTextSplitter",
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
        "vlm_enabled": settings.vlm_enabled,
        "vlm_mode": settings.vlm_mode,
        "vlm_model": settings.vlm_model,
        "vlm_max_image_edge": settings.vlm_max_image_edge,
        "vlm_num_predict": settings.vlm_num_predict,
        "vlm_prompt_version": VISUAL_PROMPT_VERSION,
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


def _vlm_active(settings: Settings) -> bool:
    return settings.vlm_enabled and settings.vlm_mode != "disabled"


def _pdf_needs_vlm(
    pdf_path: Path,
    page_index: int,
    needs_ocr: bool,
    settings: Settings,
) -> bool:
    if not _vlm_active(settings):
        return False
    if settings.vlm_mode == "all" or needs_ocr:
        return True
    try:
        return pdf_page_has_images(pdf_path, page_index)
    except ImageRenderError:
        LOGGER.warning("Unable to inspect PDF page images; skipping auto VLM page.")
        return False


def _ocr_from_rendered_page(
    service: OCRService,
    image_bytes: bytes,
    pdf_path: Path,
    page_index: int,
) -> OCRResult:
    bytes_method = getattr(service, "image_bytes_to_text", None)
    if callable(bytes_method):
        return bytes_method(image_bytes, suffix=".png")
    return service.pdf_page_to_text(pdf_path, page_index)


def _visual_metadata(
    filename: str,
    page_number: int,
    source_type: str,
    settings: Settings,
) -> dict[str, object]:
    return {
        "source": filename,
        SOURCE_FILENAME_KEY: filename,
        PAGE_NUMBER_KEY: page_number,
        SOURCE_TYPE_KEY: source_type,
        CONTENT_TYPE_KEY: VISUAL_CONTENT_TYPE,
        TEXT_EXTRACTION_METHOD_KEY: VLM_EXTRACTION_METHOD,
        VLM_MODEL_KEY: settings.vlm_model,
        VLM_PROMPT_VERSION_KEY: VISUAL_PROMPT_VERSION,
    }


def _safe_vlm_failure_message(error: Exception) -> str:
    if isinstance(error, VLMError):
        return str(error)
    return "VLM 視覺分析失敗，已略過此頁的視覺描述。"


def _load_pdf_pages(
    uploaded_file: BinaryIO,
    settings: Settings,
    ocr_service: OCRService | None = None,
    vlm_service: VLMService | None = None,
) -> tuple[
    list[Document], list[Document], DocumentExtractionStats,
    dict[str, str], dict[str, str], dict[str, str],
]:
    temp_path: Path | None = None
    filename = getattr(uploaded_file, "name", "未命名.pdf")
    try:
        temp_path = _write_temp_upload(uploaded_file, ".pdf")
        pages = PyPDFLoader(str(temp_path)).load()
        native_count = ocr_count = vlm_count = vlm_failures = 0
        ocr_confidences: list[float] = []
        ocr_text_by_source: dict[str, str] = {}
        visual_text_by_source: dict[str, str] = {}
        vlm_failures_by_source: dict[str, str] = {}
        ocr = ocr_service
        visual_pages: list[Document] = []
        processed_pages: list[Document] = []

        for index, page in enumerate(pages):
            page_number = index + 1
            source_key = f"{filename}#page={page_number}"
            metadata = dict(page.metadata)
            metadata.update({SOURCE_TYPE_KEY: PDF_SOURCE_TYPE, CONTENT_TYPE_KEY: TEXT_CONTENT_TYPE})
            needs_ocr = _needs_ocr(page, settings)
            needs_vlm = _pdf_needs_vlm(temp_path, index, needs_ocr, settings)
            if needs_ocr:
                ocr = ocr or OCRService(settings)
            shared_ocr_render = needs_ocr and callable(
                getattr(ocr, "image_bytes_to_text", None)
            )
            rendered_image: bytes | None = None
            if shared_ocr_render or needs_vlm:
                try:
                    rendered_image = render_pdf_page_to_image(temp_path, index, dpi=settings.ocr_dpi)
                except ImageRenderError as exc:
                    if needs_ocr:
                        raise OCRError(str(exc)) from exc
                    vlm_failures += 1
                    vlm_failures_by_source[source_key] = _safe_vlm_failure_message(exc)

            if needs_ocr:
                assert ocr is not None
                if rendered_image is None:
                    ocr_result = ocr.pdf_page_to_text(temp_path, index)
                else:
                    ocr_result = _ocr_from_rendered_page(
                        ocr, rendered_image, temp_path, index
                    )
                metadata[TEXT_EXTRACTION_METHOD_KEY] = OCR_EXTRACTION_METHOD
                metadata[OCR_ENGINE_KEY] = OCR_ENGINE_NAME
                if ocr_result.confidence is not None:
                    metadata[OCR_CONFIDENCE_KEY] = ocr_result.confidence
                    ocr_confidences.append(ocr_result.confidence)
                page_content = ocr_result.text.strip()
                if page_content:
                    ocr_text_by_source[source_key] = page_content
                ocr_count += 1
            else:
                metadata[TEXT_EXTRACTION_METHOD_KEY] = NATIVE_EXTRACTION_METHOD
                page_content = page.page_content
                native_count += 1
            processed_pages.append(Document(page_content=page_content, metadata=metadata))

            if needs_vlm and rendered_image is not None and vlm_service is not None:
                try:
                    normalized, mime_type = normalize_image(rendered_image, max_edge=settings.vlm_max_image_edge)
                    result = vlm_service.analyze_image(
                        normalized, mime_type=mime_type, filename=filename, page_number=page_number
                    )
                    vlm_count += 1
                    if has_substantive_visual_content(result):
                        visual_text = format_visual_analysis(result)
                        visual_text_by_source[source_key] = visual_text
                        visual_pages.append(Document(
                            page_content=visual_text,
                            metadata=_visual_metadata(filename, page_number, PDF_SOURCE_TYPE, settings),
                        ))
                except Exception as exc:
                    LOGGER.warning("VLM failed for PDF %r page %d.", filename, page_number, exc_info=True)
                    vlm_failures += 1
                    vlm_failures_by_source[source_key] = _safe_vlm_failure_message(exc)

        return (
            processed_pages,
            visual_pages,
            DocumentExtractionStats(
                native_page_count=native_count,
                ocr_page_count=ocr_count,
                ocr_confidences=tuple(ocr_confidences),
                vlm_page_count=vlm_count,
                vlm_failure_count=vlm_failures,
            ),
            ocr_text_by_source,
            visual_text_by_source,
            vlm_failures_by_source,
        )
    except Exception as exc:
        if isinstance(exc, OCRError):
            LOGGER.warning("OCR failed while parsing PDF %r.", filename, exc_info=True)
            raise PDFProcessingError(_ocr_user_message(filename)) from exc
        if isinstance(exc, PDFProcessingError):
            raise
        raise PDFProcessingError(f"無法解析 PDF「{filename}」，請確認檔案未損毀或加密。") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Unable to remove temporary PDF %s.", temp_path)


def _ocr_uploaded_image(service: OCRService, image_bytes: bytes, suffix: str) -> OCRResult:
    bytes_method = getattr(service, "image_bytes_to_text", None)
    if callable(bytes_method):
        return bytes_method(image_bytes, suffix=suffix)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as file:
            file.write(image_bytes)
            temp_path = Path(file.name)
        return service.image_file_to_text(temp_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _load_image_page(
    uploaded_file: BinaryIO,
    settings: Settings,
    ocr_service: OCRService | None = None,
    vlm_service: VLMService | None = None,
) -> tuple[
    list[Document], list[Document], DocumentExtractionStats,
    dict[str, str], dict[str, str], dict[str, str],
]:
    filename = getattr(uploaded_file, "name", "未命名圖片")
    suffix = Path(filename).suffix.lower() or ".png"
    image_bytes = _read_upload(uploaded_file)
    text = ""
    confidence: float | None = None
    ocr_count = 0
    ocr_error: Exception | None = None
    if settings.ocr_enable_images and settings.ocr_mode != "disabled":
        try:
            ocr = ocr_service or OCRService(settings)
            ocr_result = _ocr_uploaded_image(ocr, image_bytes, suffix)
            text = ocr_result.text.strip()
            confidence = ocr_result.confidence
            ocr_count = 1
        except Exception as exc:
            ocr_error = exc
            LOGGER.warning("OCR failed while parsing image %r.", filename, exc_info=True)

    visual_pages: list[Document] = []
    visual_text_by_source: dict[str, str] = {}
    vlm_failures_by_source: dict[str, str] = {}
    vlm_count = vlm_failure_count = 0
    if _vlm_active(settings) and vlm_service is not None:
        try:
            normalized, mime_type = normalize_image(image_bytes, max_edge=settings.vlm_max_image_edge)
            result = vlm_service.analyze_image(
                normalized, mime_type=mime_type, filename=filename, page_number=1
            )
            vlm_count = 1
            if has_substantive_visual_content(result):
                visual_text = format_visual_analysis(result)
                visual_text_by_source[filename] = visual_text
                visual_pages.append(Document(
                    page_content=visual_text,
                    metadata=_visual_metadata(filename, 1, IMAGE_SOURCE_TYPE, settings),
                ))
        except Exception as exc:
            LOGGER.warning("VLM failed for image %r.", filename, exc_info=True)
            vlm_failure_count = 1
            vlm_failures_by_source[filename] = _safe_vlm_failure_message(exc)

    if not text and not visual_pages:
        if isinstance(ocr_error, OCRError):
            raise PDFProcessingError(_ocr_user_message(filename)) from ocr_error
        if not settings.ocr_enable_images and not _vlm_active(settings):
            raise PDFProcessingError("圖片檔支援已停用；請啟用 OCR 圖片支援或 VLM。")
        if settings.ocr_mode == "disabled" and not _vlm_active(settings):
            raise PDFProcessingError("圖片檔需要 OCR 或 VLM；請啟用其中一項功能。")
        raise PDFProcessingError(f"檔案「{filename}」的 OCR 與 VLM 都沒有產生可建立索引的內容。")

    metadata: dict[str, object] = {
        "source": filename,
        SOURCE_FILENAME_KEY: filename,
        PAGE_NUMBER_KEY: 1,
        SOURCE_TYPE_KEY: IMAGE_SOURCE_TYPE,
        CONTENT_TYPE_KEY: TEXT_CONTENT_TYPE,
        TEXT_EXTRACTION_METHOD_KEY: OCR_EXTRACTION_METHOD,
        OCR_ENGINE_KEY: OCR_ENGINE_NAME,
    }
    confidences: tuple[float, ...] = ()
    if confidence is not None:
        metadata[OCR_CONFIDENCE_KEY] = confidence
        confidences = (confidence,)
    text_pages = [Document(page_content=text, metadata=metadata)] if text else []
    return (
        text_pages,
        visual_pages,
        DocumentExtractionStats(
            ocr_page_count=ocr_count,
            image_page_count=1,
            ocr_confidences=confidences,
            vlm_page_count=vlm_count,
            vlm_failure_count=vlm_failure_count,
        ),
        {filename: text} if text else {},
        visual_text_by_source,
        vlm_failures_by_source,
    )


def normalize_page_metadata(pages: Sequence[Document], filename: str) -> list[Document]:
    """正規化檔名與一起算頁碼，同時保留已設定的 Visual 頁碼。"""

    normalized: list[Document] = []
    for index, page in enumerate(pages):
        metadata = dict(page.metadata)
        if PAGE_NUMBER_KEY in metadata:
            page_number = metadata[PAGE_NUMBER_KEY]
        else:
            raw_page = metadata.get("page", index)
            try:
                page_number = int(raw_page) + 1
            except (TypeError, ValueError):
                page_number = index + 1
        metadata.update({
            "source": filename,
            SOURCE_FILENAME_KEY: filename,
            PAGE_NUMBER_KEY: page_number,
            CONTENT_TYPE_KEY: metadata.get(CONTENT_TYPE_KEY, TEXT_CONTENT_TYPE),
        })
        normalized.append(Document(page_content=page.page_content, metadata=metadata))
    return normalized


def split_pdf_pages(pages: Sequence[Document], filename: str, settings: Settings) -> list[Document]:
    normalized_pages = normalize_page_metadata(pages, filename)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=CHINESE_SEPARATORS,
        add_start_index=SPLITTER_ADD_START_INDEX,
    )
    return splitter.split_documents(normalized_pages)


def process_uploaded_pdfs(
    uploaded_files: Sequence[BinaryIO],
    settings: Settings,
    *,
    ocr_service: OCRService | None = None,
    vlm_service: VLMService | None = None,
) -> DocumentBatch:
    """解析多份 PDF/圖片，建立文字與 Visual Chunks 並回傳統計。"""

    if _vlm_active(settings) and vlm_service is None:
        try:
            vlm_service = VLMService(settings)
        except VLMError as exc:
            raise PDFProcessingError(str(exc)) from exc

    all_chunks: list[Document] = []
    parsed_text_by_source: dict[str, str] = {}
    ocr_text_by_source: dict[str, str] = {}
    visual_text_by_source: dict[str, str] = {}
    vlm_failures_by_source: dict[str, str] = {}
    total_pages = native_pages = ocr_pages = image_pages = 0
    vlm_pages = vlm_failures = 0
    ocr_confidences: list[float] = []

    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", "未命名.pdf")
        suffix = Path(filename).suffix.lower()
        if suffix in SUPPORTED_IMAGE_SUFFIXES:
            loaded = _load_image_page(uploaded_file, settings, ocr_service, vlm_service)
        else:
            loaded = _load_pdf_pages(uploaded_file, settings, ocr_service, vlm_service)
        text_pages, visual_pages, stats, source_ocr, source_visual, source_failures = loaded
        total_pages += 1 if suffix in SUPPORTED_IMAGE_SUFFIXES else len(text_pages)
        native_pages += stats.native_page_count
        ocr_pages += stats.ocr_page_count
        image_pages += stats.image_page_count
        vlm_pages += stats.vlm_page_count
        vlm_failures += stats.vlm_failure_count
        ocr_confidences.extend(stats.ocr_confidences)

        if text_pages:
            parsed_text = "\n".join(page.page_content for page in text_pages)
            if filename in parsed_text_by_source:
                parsed_text = f"{parsed_text_by_source[filename]}\n{parsed_text}"
            parsed_text_by_source[filename] = parsed_text
        ocr_text_by_source.update(source_ocr)
        visual_text_by_source.update(source_visual)
        vlm_failures_by_source.update(source_failures)
        chunks = split_pdf_pages([*text_pages, *visual_pages], filename, settings)
        usable_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
        if not usable_chunks:
            raise PDFProcessingError(
                f"檔案「{filename}」沒有可建立索引的文字或視覺描述；"
                "請確認 OCR/VLM 結果與檔案品質。"
            )
        all_chunks.extend(usable_chunks)

    if not all_chunks:
        raise PDFProcessingError("檔案沒有可建立索引的文字或視覺描述；請確認 OCR/VLM 結果與檔案品質。")
    visual_chunk_count = sum(
        chunk.metadata.get(CONTENT_TYPE_KEY) == VISUAL_CONTENT_TYPE for chunk in all_chunks
    )
    return DocumentBatch(
        document_count=len(uploaded_files),
        page_count=total_pages,
        chunks=all_chunks,
        parsed_text_by_source=parsed_text_by_source,
        ocr_text_by_source=ocr_text_by_source,
        extraction_stats=DocumentExtractionStats(
            native_page_count=native_pages,
            ocr_page_count=ocr_pages,
            image_page_count=image_pages,
            ocr_confidences=tuple(ocr_confidences),
            vlm_page_count=vlm_pages,
            vlm_failure_count=vlm_failures,
            visual_chunk_count=visual_chunk_count,
        ),
        visual_text_by_source=visual_text_by_source,
        vlm_failures_by_source=vlm_failures_by_source,
    )
