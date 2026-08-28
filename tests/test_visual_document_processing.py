from __future__ import annotations

from io import BytesIO

import pytest
from langchain_core.documents import Document
from PIL import Image

from src.rag import document_loader
from src.rag.config import Settings
from src.rag.document_loader import (
    CONTENT_TYPE_KEY,
    PAGE_NUMBER_KEY,
    PDFProcessingError,
    SOURCE_FILENAME_KEY,
    TEXT_CONTENT_TYPE,
    VISUAL_CONTENT_TYPE,
    process_uploaded_pdfs,
)
from src.rag.ocr_service import OCRResult
from src.rag.vlm_service import VLMError, VisualAnalysisResult


class NamedBytesIO(BytesIO):
    def __init__(self, content: bytes, name: str) -> None:
        super().__init__(content)
        self.name = name


def image_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (20, 10), "white").save(output, format="PNG")
    return output.getvalue()


class FakeOCR:
    def __init__(self, text: str = "OCR 文字") -> None:
        self.text = text
        self.calls = 0

    def image_bytes_to_text(self, _data: bytes, *, suffix: str = ".png") -> OCRResult:
        self.calls += 1
        return OCRResult(self.text, 0.9)


class FakeVLM:
    def __init__(self, *, fail: bool = False, summary: str = "看到白色齒輪。") -> None:
        self.fail = fail
        self.summary = summary
        self.calls: list[tuple[str, int, str]] = []

    def analyze_image(
        self,
        _data: bytes,
        *,
        mime_type: str,
        filename: str,
        page_number: int,
    ) -> VisualAnalysisResult:
        self.calls.append((filename, page_number, mime_type))
        if self.fail:
            raise VLMError("VLM 測試失敗。")
        return VisualAnalysisResult(page_type="設備照片", summary=self.summary)


def install_fake_pdf_loader(
    monkeypatch: pytest.MonkeyPatch,
    pages: list[Document],
) -> None:
    class FakeLoader:
        def __init__(self, _path: str) -> None:
            pass

        def load(self) -> list[Document]:
            return pages

    monkeypatch.setattr(document_loader, "PyPDFLoader", FakeLoader)


def test_vlm_disabled_does_not_call_model() -> None:
    vlm = FakeVLM()
    batch = process_uploaded_pdfs(
        [NamedBytesIO(image_bytes(), "photo.png")],
        Settings(vlm_enabled=False),
        ocr_service=FakeOCR(),
        vlm_service=vlm,
    )

    assert not vlm.calls
    assert all(chunk.metadata[CONTENT_TYPE_KEY] == TEXT_CONTENT_TYPE for chunk in batch.chunks)


def test_auto_image_creates_visual_chunk_with_linked_metadata() -> None:
    vlm = FakeVLM()
    batch = process_uploaded_pdfs(
        [NamedBytesIO(image_bytes(), "photo.png")],
        Settings(vlm_enabled=True, vlm_mode="auto", chunk_size=200, chunk_overlap=0),
        ocr_service=FakeOCR(),
        vlm_service=vlm,
    )

    assert len(vlm.calls) == 1
    text_chunk = next(c for c in batch.chunks if c.metadata[CONTENT_TYPE_KEY] == TEXT_CONTENT_TYPE)
    visual_chunk = next(c for c in batch.chunks if c.metadata[CONTENT_TYPE_KEY] == VISUAL_CONTENT_TYPE)
    assert text_chunk.metadata[SOURCE_FILENAME_KEY] == visual_chunk.metadata[SOURCE_FILENAME_KEY]
    assert text_chunk.metadata[PAGE_NUMBER_KEY] == visual_chunk.metadata[PAGE_NUMBER_KEY] == 1
    assert batch.extraction_stats.visual_chunk_count == 1
    assert batch.visual_text_by_source["photo.png"]


def test_auto_ocr_pdf_renders_once_for_ocr_and_vlm(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_pdf_loader(monkeypatch, [Document(page_content="", metadata={"page": 0})])
    render_calls: list[int] = []

    def fake_render(_path, page_index: int, *, dpi: int) -> bytes:
        render_calls.append(page_index)
        return image_bytes()

    monkeypatch.setattr(document_loader, "render_pdf_page_to_image", fake_render)
    vlm = FakeVLM()
    batch = process_uploaded_pdfs(
        [NamedBytesIO(b"fake pdf", "scan.pdf")],
        Settings(vlm_enabled=True, vlm_mode="auto", chunk_size=200, chunk_overlap=0),
        ocr_service=FakeOCR(),
        vlm_service=vlm,
    )

    assert render_calls == [0]
    assert len(vlm.calls) == 1
    assert batch.extraction_stats.ocr_page_count == 1
    assert batch.extraction_stats.vlm_page_count == 1


def test_auto_skips_text_pdf_without_embedded_images(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_pdf_loader(
        monkeypatch,
        [Document(page_content="這是一頁足量的原生文字。", metadata={"page": 0})],
    )
    monkeypatch.setattr(document_loader, "pdf_page_has_images", lambda *_args: False)
    monkeypatch.setattr(
        document_loader,
        "render_pdf_page_to_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("不應渲染")),
    )
    vlm = FakeVLM()

    process_uploaded_pdfs(
        [NamedBytesIO(b"fake pdf", "text.pdf")],
        Settings(vlm_enabled=True, vlm_mode="auto", ocr_min_text_chars=5),
        vlm_service=vlm,
    )

    assert not vlm.calls


def test_all_mode_analyzes_text_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_pdf_loader(
        monkeypatch,
        [Document(page_content="原生文字內容", metadata={"page": 0})],
    )
    monkeypatch.setattr(
        document_loader, "render_pdf_page_to_image", lambda *_args, **_kwargs: image_bytes()
    )
    vlm = FakeVLM()
    batch = process_uploaded_pdfs(
        [NamedBytesIO(b"fake pdf", "text.pdf")],
        Settings(vlm_enabled=True, vlm_mode="all", ocr_min_text_chars=2),
        vlm_service=vlm,
    )

    assert len(vlm.calls) == 1
    assert any(c.metadata[CONTENT_TYPE_KEY] == VISUAL_CONTENT_TYPE for c in batch.chunks)


def test_vlm_failure_keeps_existing_text(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_pdf_loader(
        monkeypatch,
        [Document(page_content="仍可建立索引的原生文字", metadata={"page": 0})],
    )
    monkeypatch.setattr(
        document_loader, "render_pdf_page_to_image", lambda *_args, **_kwargs: image_bytes()
    )
    batch = process_uploaded_pdfs(
        [NamedBytesIO(b"fake pdf", "manual.pdf")],
        Settings(vlm_enabled=True, vlm_mode="all", ocr_min_text_chars=2),
        vlm_service=FakeVLM(fail=True),
    )

    assert batch.chunks[0].page_content == "仍可建立索引的原生文字"
    assert batch.extraction_stats.vlm_failure_count == 1
    assert "manual.pdf#page=1" in batch.vlm_failures_by_source


def test_blank_ocr_image_can_build_index_from_visual_description() -> None:
    batch = process_uploaded_pdfs(
        [NamedBytesIO(image_bytes(), "blank.png")],
        Settings(vlm_enabled=True, vlm_mode="auto"),
        ocr_service=FakeOCR(text=""),
        vlm_service=FakeVLM(),
    )

    assert batch.chunks
    assert all(c.metadata[CONTENT_TYPE_KEY] == VISUAL_CONTENT_TYPE for c in batch.chunks)


def test_blank_ocr_and_non_substantive_vlm_returns_clear_error() -> None:
    with pytest.raises(PDFProcessingError, match="OCR 與 VLM"):
        process_uploaded_pdfs(
            [NamedBytesIO(image_bytes(), "blank.png")],
            Settings(vlm_enabled=True, vlm_mode="auto"),
            ocr_service=FakeOCR(text=""),
            vlm_service=FakeVLM(summary=""),
        )
