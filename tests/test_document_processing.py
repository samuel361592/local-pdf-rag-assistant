from __future__ import annotations

from io import BytesIO

import pytest
from langchain_core.documents import Document

from src.rag.config import Settings, load_settings
from src.rag import document_loader
from src.rag.document_loader import (
    CHINESE_SEPARATORS,
    OCR_EXTRACTION_METHOD,
    PAGE_NUMBER_KEY,
    PDFProcessingError,
    SOURCE_FILENAME_KEY,
    TEXT_EXTRACTION_METHOD_KEY,
    NATIVE_EXTRACTION_METHOD,
    process_uploaded_pdfs,
    split_pdf_pages,
)
from src.rag.ocr_service import OCRError, OCRResult


class NamedBytesIO(BytesIO):
    def __init__(self, content: bytes, name: str) -> None:
        super().__init__(content)
        self.name = name


def test_chinese_separators_are_configured() -> None:
    for separator in ["\n\n", "\n", "。", "！", "？", "；", "，"]:
        assert separator in CHINESE_SEPARATORS


def test_chunk_metadata_keeps_filename_and_page_number() -> None:
    pages = [Document(page_content="這是一段治理文件內容。" * 10, metadata={"page": 2})]
    chunks = split_pdf_pages(
        pages,
        "治理指引.pdf",
        Settings(chunk_size=40, chunk_overlap=5),
    )

    assert chunks
    assert all(chunk.metadata[SOURCE_FILENAME_KEY] == "治理指引.pdf" for chunk in chunks)
    assert all(chunk.metadata[PAGE_NUMBER_KEY] == 3 for chunk in chunks)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=100, chunk_overlap=100)


def test_load_settings_reads_ocr_options(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "OCR_MODE=force",
                "OCR_LANG=ch",
                "OCR_MIN_TEXT_CHARS=12",
                "OCR_DPI=250",
                "OCR_ENABLE_IMAGES=false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("OCR_MODE", raising=False)
    monkeypatch.delenv("OCR_ENABLE_IMAGES", raising=False)

    settings = load_settings(env_path=env_path, override_env=True)

    assert settings.ocr_mode == "force"
    assert settings.ocr_min_text_chars == 12
    assert settings.ocr_dpi == 250
    assert not settings.ocr_enable_images


def test_pdf_blank_page_uses_ocr_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLoader:
        def __init__(self, _path: str) -> None:
            pass

        def load(self) -> list[Document]:
            return [Document(page_content="", metadata={"page": 0})]

    class FakeOCRService:
        def __init__(self, _settings: Settings) -> None:
            pass

        def pdf_page_to_text(self, _path, page_index: int) -> OCRResult:
            assert page_index == 0
            return OCRResult("掃描頁文字", 0.9)

    monkeypatch.setattr(document_loader, "PyPDFLoader", FakeLoader)
    monkeypatch.setattr(document_loader, "OCRService", FakeOCRService)

    batch = process_uploaded_pdfs(
        [NamedBytesIO(b"%PDF fake", "scan.pdf")],
        Settings(chunk_size=50, chunk_overlap=0),
    )

    assert batch.extraction_stats.ocr_page_count == 1
    assert batch.extraction_stats.native_page_count == 0
    assert batch.parsed_text_by_source["scan.pdf"] == "掃描頁文字"
    assert batch.chunks[0].metadata[TEXT_EXTRACTION_METHOD_KEY] == OCR_EXTRACTION_METHOD
    assert batch.ocr_text_by_source["scan.pdf#page=1"] == "掃描頁文字"


def test_pdf_text_page_skips_ocr_in_auto_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLoader:
        def __init__(self, _path: str) -> None:
            pass

        def load(self) -> list[Document]:
            return [Document(page_content="這頁已經有足夠文字內容。", metadata={"page": 0})]

    class FailingOCRService:
        def __init__(self, _settings: Settings) -> None:
            pass

        def pdf_page_to_text(self, *_args) -> OCRResult:
            raise AssertionError("一般文字頁不應執行 OCR")

    monkeypatch.setattr(document_loader, "PyPDFLoader", FakeLoader)
    monkeypatch.setattr(document_loader, "OCRService", FailingOCRService)

    batch = process_uploaded_pdfs(
        [NamedBytesIO(b"%PDF fake", "text.pdf")],
        Settings(chunk_size=50, chunk_overlap=0, ocr_min_text_chars=5),
    )

    assert batch.extraction_stats.native_page_count == 1
    assert batch.extraction_stats.ocr_page_count == 0
    assert batch.chunks[0].metadata[TEXT_EXTRACTION_METHOD_KEY] == NATIVE_EXTRACTION_METHOD


def test_image_upload_is_converted_to_ocr_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOCRService:
        def __init__(self, _settings: Settings) -> None:
            pass

        def image_file_to_text(self, _path) -> OCRResult:
            return OCRResult("圖片中的文字", 0.8)

    monkeypatch.setattr(document_loader, "OCRService", FakeOCRService)

    batch = process_uploaded_pdfs(
        [NamedBytesIO(b"fake image", "receipt.jpg")],
        Settings(chunk_size=50, chunk_overlap=0),
    )

    assert batch.page_count == 1
    assert batch.extraction_stats.image_page_count == 1
    assert batch.extraction_stats.ocr_page_count == 1
    assert batch.chunks[0].metadata[SOURCE_FILENAME_KEY] == "receipt.jpg"
    assert batch.chunks[0].metadata[PAGE_NUMBER_KEY] == 1
    assert batch.chunks[0].metadata[TEXT_EXTRACTION_METHOD_KEY] == OCR_EXTRACTION_METHOD


def test_ocr_runtime_details_are_not_exposed_to_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingOCRService:
        def __init__(self, _settings: Settings) -> None:
            pass

        def image_file_to_text(self, _path) -> OCRResult:
            raise OCRError(
                "OCR 辨識圖片失敗：tmpg0c_ad81.png；"
                "(Unimplemented) ConvertPirAttribute2RuntimeAttribute "
                "not support onednn_instruction.cc:118"
            )

    monkeypatch.setattr(document_loader, "OCRService", FailingOCRService)

    with pytest.raises(PDFProcessingError) as raised:
        process_uploaded_pdfs(
            [NamedBytesIO(b"fake image", "scan.png")],
            Settings(chunk_size=50, chunk_overlap=0),
        )

    message = str(raised.value)
    assert "scan.png" in message
    assert "tmpg0c_ad81" not in message
    assert "Unimplemented" not in message
    assert "onednn" not in message
