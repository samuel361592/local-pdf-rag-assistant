from __future__ import annotations

from langchain_core.documents import Document

from src.rag.config import Settings
from src.rag.prompt import INSUFFICIENT_INFORMATION_MESSAGE
from src.rag.rag_service import RagService, format_context


class EmptyVectorStore:
    def similarity_search(self, _question: str, k: int) -> list[Document]:
        assert k == 4
        return []


class FailingChatModel:
    def invoke(self, _messages: object) -> object:
        raise AssertionError("沒有檢索結果時不應呼叫模型")


def test_context_contains_source_number_filename_and_page() -> None:
    document = Document(
        page_content="應注意資料治理。",
        metadata={"source_filename": "參考指引.pdf", "page_number": 7},
    )

    context, sources = format_context([document])

    assert "[來源 1]" in context
    assert "參考指引.pdf" in context
    assert "頁碼：7" in context
    assert sources[0].filename == "參考指引.pdf"


def test_no_results_returns_fixed_insufficient_information_message() -> None:
    service = RagService(EmptyVectorStore(), Settings(), chat_model=FailingChatModel())

    result = service.answer("文件有說明採購哪個產品嗎？")

    assert result.answer == INSUFFICIENT_INFORMATION_MESSAGE
    assert result.sources == []
    assert result.chunks == []
