from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.rag.config import Settings
from src.rag.document_loader import (
    CHINESE_SEPARATORS,
    PAGE_NUMBER_KEY,
    SOURCE_FILENAME_KEY,
    split_pdf_pages,
)


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
