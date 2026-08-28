from __future__ import annotations

from threading import Event

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from src.rag.config import Settings
from src.rag.prompt import INSUFFICIENT_INFORMATION_MESSAGE
from src.rag.rag_service import (
    RagAnswerCancelled,
    RagProgressEvent,
    RagService,
    format_context,
)


class EmptyVectorStore:
    def similarity_search(self, _question: str, k: int) -> list[Document]:
        assert k == 4
        return []


class FailingChatModel:
    def invoke(self, _messages: object) -> object:
        raise AssertionError("沒有檢索結果時不應呼叫模型")


class SingleDocumentVectorStore:
    def __init__(self) -> None:
        self.calls = 0

    def similarity_search(self, _question: str, k: int) -> list[Document]:
        assert k == 4
        self.calls += 1
        return [
            Document(
                page_content="應注意資料治理。",
                metadata={"source_filename": "參考指引.pdf", "page_number": 7},
            )
        ]


class StreamingChatModel:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks

    def stream(self, _messages: object):
        for chunk in self.chunks:
            yield AIMessageChunk(content=chunk)


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


def test_context_and_source_mark_visual_content_type() -> None:
    document = Document(
        page_content="[可見異常]\n- 齒輪邊緣疑似缺損。",
        metadata={
            "source_filename": "維修手冊.pdf",
            "page_number": 3,
            "content_type": "visual",
        },
    )

    context, sources = format_context([document])

    assert "內容類型：VLM 視覺分析" in context
    assert sources[0].content_type_label == "視覺分析"


def test_no_results_returns_fixed_insufficient_information_message() -> None:
    service = RagService(
        EmptyVectorStore(),
        Settings(reranker_enabled=False),
        chat_model=FailingChatModel(),
    )

    result = service.answer("文件有說明採購哪個產品嗎？")

    assert result.answer == INSUFFICIENT_INFORMATION_MESSAGE
    assert result.sources == []
    assert result.chunks == []


def test_answer_reports_progress_events_when_no_results() -> None:
    service = RagService(
        EmptyVectorStore(),
        Settings(reranker_enabled=False),
        chat_model=FailingChatModel(),
    )
    events: list[RagProgressEvent] = []

    service.answer("文件有說明採購哪個產品嗎？", progress_callback=events.append)

    assert [event.stage for event in events] == [
        "received",
        "retrieve",
        "retrieved",
        "no_results",
    ]


def test_stream_answer_preserves_spaces_between_chunks() -> None:
    vector_store = SingleDocumentVectorStore()
    service = RagService(
        vector_store,
        Settings(reranker_enabled=False),
        chat_model=StreamingChatModel(["第一段", " 第二段", "。"]),
    )
    tokens: list[str] = []

    result = service.answer_stream(
        "文件說了什麼？",
        cancel_event=Event(),
        token_callback=tokens.append,
    )

    assert result.answer == "第一段 第二段。"
    assert tokens == ["第一段", " 第二段", "。"]
    assert result.sources[0].filename == "參考指引.pdf"


def test_stream_answer_can_cancel_and_keep_partial_result() -> None:
    cancel_event = Event()
    events: list[RagProgressEvent] = []
    tokens: list[str] = []
    service = RagService(
        SingleDocumentVectorStore(),
        Settings(reranker_enabled=False),
        chat_model=StreamingChatModel(["保留這段", " 不應處理這段"]),
    )

    def on_token(token: str) -> None:
        tokens.append(token)
        cancel_event.set()

    with pytest.raises(RagAnswerCancelled) as raised:
        service.answer_stream(
            "文件說了什麼？",
            cancel_event=cancel_event,
            progress_callback=events.append,
            token_callback=on_token,
        )

    assert tokens == ["保留這段"]
    assert raised.value.partial_result.answer == "保留這段"
    assert raised.value.partial_result.sources[0].filename == "參考指引.pdf"
    assert events[-1].stage == "cancelled"


def test_stream_answer_cancelled_before_retrieval_does_not_query_vector_store() -> None:
    vector_store = SingleDocumentVectorStore()
    cancel_event = Event()
    cancel_event.set()
    service = RagService(
        vector_store,
        Settings(reranker_enabled=False),
        chat_model=StreamingChatModel([]),
    )

    with pytest.raises(RagAnswerCancelled):
        service.answer_stream("文件說了什麼？", cancel_event=cancel_event)

    assert vector_store.calls == 0
