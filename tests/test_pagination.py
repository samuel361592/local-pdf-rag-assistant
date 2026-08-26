from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app import (
    filter_chunks,
    friendly_error,
    get_page_window,
    ocr_preview_text_key,
    shorten_filename,
)
from src.rag.document_loader import PAGE_NUMBER_KEY, SOURCE_FILENAME_KEY
from src.rag.reranker import (
    RerankerComputeError,
    RerankerDependencyError,
    RerankerDownloadError,
    RerankerLoadError,
    RerankerMemoryError,
    RerankerModelNotFoundError,
)


def test_page_window_returns_only_requested_page() -> None:
    assert get_page_window(total_items=25, current_page=2, page_size=10) == (
        2,
        3,
        10,
        20,
    )


def test_page_window_clamps_page_and_last_page_end() -> None:
    assert get_page_window(total_items=25, current_page=99, page_size=10) == (
        3,
        3,
        20,
        25,
    )


def test_page_window_handles_empty_collection() -> None:
    assert get_page_window(total_items=0, current_page=1, page_size=10) == (
        1,
        1,
        0,
        0,
    )


@pytest.mark.parametrize(
    ("total_items", "page_size"),
    [(-1, 10), (10, 0)],
)
def test_page_window_rejects_invalid_values(
    total_items: int, page_size: int
) -> None:
    with pytest.raises(ValueError):
        get_page_window(total_items, current_page=1, page_size=page_size)


def test_filter_chunks_combines_file_page_and_keyword_filters() -> None:
    chunks = [
        Document(
            page_content="AI 治理與風險管理",
            metadata={SOURCE_FILENAME_KEY: "report.pdf", PAGE_NUMBER_KEY: 1},
        ),
        Document(
            page_content="個人資料保護",
            metadata={SOURCE_FILENAME_KEY: "report.pdf", PAGE_NUMBER_KEY: 2},
        ),
        Document(
            page_content="其他文件的 AI 說明",
            metadata={SOURCE_FILENAME_KEY: "guide.pdf", PAGE_NUMBER_KEY: 1},
        ),
    ]

    matches = filter_chunks(
        chunks,
        filename="report.pdf",
        page_number=1,
        keyword="ai",
    )

    assert matches == [(1, chunks[0])]


def test_filter_chunks_preserves_original_chunk_numbers() -> None:
    chunks = [
        Document(page_content="不符合", metadata={}),
        Document(page_content="符合關鍵字", metadata={}),
    ]

    assert filter_chunks(chunks, keyword="關鍵字") == [(2, chunks[1])]


def test_shorten_filename_preserves_both_ends() -> None:
    filename = "行政院及所屬機關使用生成式AI參考指引總說明及規定.pdf"

    shortened = shorten_filename(filename, max_length=24)

    assert len(shortened) == 24
    assert shortened.startswith("行政院")
    assert shortened.endswith("規定.pdf")
    assert "…" in shortened


def test_shorten_filename_keeps_short_names_unchanged() -> None:
    assert shorten_filename("report.pdf") == "report.pdf"


def test_friendly_error_hides_unknown_internal_details() -> None:
    message = friendly_error(RuntimeError("internal stack detail at engine.cc:118"))

    assert "internal stack detail" not in message
    assert "engine.cc" not in message
    assert "未預期錯誤" in message


@pytest.mark.parametrize(
    ("error", "expected_text"),
    [
        (RerankerDependencyError("missing"), "FlagEmbedding"),
        (RerankerDownloadError("connection failed"), "無法下載重排模型"),
        (RerankerModelNotFoundError("404"), "找不到指定的重排模型"),
        (RerankerLoadError("load failed"), "無法載入重排模型"),
        (RerankerComputeError("compute failed"), "重排模型計算失敗"),
        (RerankerMemoryError("out of memory"), "記憶體不足"),
    ],
)
def test_friendly_error_distinguishes_reranker_failures(
    error: Exception,
    expected_text: str,
) -> None:
    message = friendly_error(error)

    assert expected_text in message
    assert "Ollama 模型" not in message


def test_ocr_preview_text_key_changes_with_source() -> None:
    assert ocr_preview_text_key("page-1.png") != ocr_preview_text_key("page-2.png")
