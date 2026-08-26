from __future__ import annotations

import sys
from threading import Event
from types import ModuleType

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from src.rag import rag_service
from src.rag.config import Settings, load_settings
from src.rag.rag_service import RagAnswerCancelled, RagProgressEvent, RagService
from src.rag.reranker import create_reranker, rerank_documents


def make_document(name: str) -> Document:
    return Document(
        page_content=name,
        metadata={
            "source_filename": f"{name}.pdf",
            "page_number": len(name),
            "extraction_method": "native",
            "custom": {"name": name},
        },
    )


class FakeReranker:
    def __init__(self, scores: object) -> None:
        self.scores = scores
        self.calls: list[tuple[list[list[str]], bool]] = []

    def compute_score(self, pairs: list[list[str]], normalize: bool) -> object:
        self.calls.append((pairs, normalize))
        return self.scores


class FakeVectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.requested_k: list[int] = []

    def similarity_search(self, _question: str, k: int) -> list[Document]:
        self.requested_k.append(k)
        return self.documents[:k]


class FakeChatModel:
    def __init__(self) -> None:
        self.invocations: list[object] = []
        self.streams: list[object] = []

    def invoke(self, messages: object) -> str:
        self.invocations.append(messages)
        return "完成"

    def stream(self, messages: object):
        self.streams.append(messages)
        yield AIMessageChunk(content="完")
        yield AIMessageChunk(content="成")


def test_rerank_orders_by_descending_score() -> None:
    documents = [make_document("A"), make_document("B"), make_document("C")]
    reranker = FakeReranker([0.2, 0.9, 0.5])

    result = rerank_documents("問題", documents, reranker, 3)

    assert [document.page_content for document in result] == ["B", "C", "A"]
    assert reranker.calls == [
        ([
            ["問題", "A"],
            ["問題", "B"],
            ["問題", "C"],
        ], True)
    ]


def test_rerank_returns_at_most_top_k_and_preserves_metadata() -> None:
    documents = [make_document(name) for name in "ABCDE"]
    original_metadata = [document.metadata.copy() for document in documents]

    result = rerank_documents(
        "問題",
        documents,
        FakeReranker([0.1, 0.5, 0.4, 0.3, 0.2]),
        2,
    )

    assert len(result) == 2
    assert result == [documents[1], documents[2]]
    assert [document.metadata for document in documents] == original_metadata


def test_equal_scores_keep_faiss_order() -> None:
    documents = [make_document("A"), make_document("B")]

    result = rerank_documents("問題", documents, FakeReranker([0.5, 0.5]), 2)

    assert result == documents


def test_empty_candidates_do_not_call_model() -> None:
    reranker = FakeReranker([1.0])

    assert rerank_documents("問題", [], reranker, 4) == []
    assert reranker.calls == []


def test_single_candidate_accepts_scalar_score() -> None:
    document = make_document("A")

    assert rerank_documents("問題", [document], FakeReranker(0.7), 1) == [document]


def test_score_count_must_match_document_count() -> None:
    documents = [make_document("A"), make_document("B"), make_document("C")]

    with pytest.raises(ValueError, match="分數數量與候選文件數量不同"):
        rerank_documents("問題", documents, FakeReranker([0.2, 0.9]), 3)


@pytest.mark.parametrize(
    ("question", "top_k", "message"),
    [("  ", 1, "問題不可為空白"), ("問題", 0, "top_k 必須大於 0")],
)
def test_rerank_rejects_invalid_arguments(
    question: str,
    top_k: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        rerank_documents(question, [make_document("A")], FakeReranker([1.0]), top_k)


def test_create_reranker_lazily_imports_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[tuple[str, bool]] = []
    fake_module = ModuleType("FlagEmbedding")

    class FakeFlagReranker:
        def __init__(self, model_name: str, use_fp16: bool) -> None:
            created.append((model_name, use_fp16))

    fake_module.FlagReranker = FakeFlagReranker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_module)
    create_reranker.cache_clear()

    first = create_reranker("test/model", False)
    second = create_reranker("test/model", False)

    assert first is second
    assert created == [("test/model", False)]
    create_reranker.cache_clear()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"retrieval_top_k": 0}, "RETRIEVAL_TOP_K"),
        ({"top_k": 0}, "TOP_K"),
        ({"retrieval_top_k": 3, "top_k": 4}, "大於或等於 TOP_K"),
        ({"reranker_model": "   "}, "RERANKER_MODEL"),
    ],
)
def test_settings_reject_invalid_reranker_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Settings(**kwargs)


@pytest.mark.parametrize(("raw_value", "expected"), [("yes", True), ("OFF", False)])
def test_load_settings_reads_reranker_boolean(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
    expected: bool,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(f"RERANKER_ENABLED={raw_value}\n", encoding="utf-8")
    monkeypatch.delenv("RERANKER_ENABLED", raising=False)

    settings = load_settings(env_path=env_path, override_env=True)

    assert settings.reranker_enabled is expected
    assert settings.retrieval_top_k == 20
    assert settings.reranker_model == "BAAI/bge-reranker-v2-m3"


def test_load_settings_rejects_invalid_reranker_boolean(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("RERANKER_ENABLED=maybe\n", encoding="utf-8")
    monkeypatch.delenv("RERANKER_ENABLED", raising=False)

    with pytest.raises(ValueError, match="RERANKER_ENABLED 必須是布林值"):
        load_settings(env_path=env_path, override_env=True)


def test_disabled_service_uses_top_k_without_creating_reranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = [make_document(name) for name in "ABCDE"]
    vector_store = FakeVectorStore(documents)
    chat_model = FakeChatModel()

    def fail_create(*_args: object) -> object:
        raise AssertionError("停用時不得建立 Reranker")

    monkeypatch.setattr(rag_service, "create_reranker", fail_create)
    service = RagService(
        vector_store,
        Settings(reranker_enabled=False, top_k=4),
        chat_model=chat_model,
    )

    result = service.answer("問題")

    assert vector_store.requested_k == [4]
    assert result.chunks == documents[:4]


def test_enabled_service_with_no_candidates_does_not_create_reranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector_store = FakeVectorStore([])

    def fail_create(*_args: object) -> object:
        raise AssertionError("沒有候選文件時不得建立 Reranker")

    monkeypatch.setattr(rag_service, "create_reranker", fail_create)
    service = RagService(
        vector_store,
        Settings(),
        chat_model=FakeChatModel(),
    )

    result = service.answer("問題")

    assert vector_store.requested_k == [20]
    assert result.chunks == []


def test_enabled_service_reranks_all_candidates_and_only_sends_top_k() -> None:
    documents = [make_document(name) for name in "ABCDE"]
    vector_store = FakeVectorStore(documents)
    reranker = FakeReranker([0.1, 0.9, 0.8, 0.3, 0.2])
    chat_model = FakeChatModel()
    events: list[RagProgressEvent] = []
    service = RagService(
        vector_store,
        Settings(retrieval_top_k=5, top_k=2),
        chat_model=chat_model,
        reranker=reranker,
    )

    result = service.answer("問題", progress_callback=events.append)

    assert vector_store.requested_k == [5]
    assert [pair[1] for pair in reranker.calls[0][0]] == list("ABCDE")
    assert [document.page_content for document in result.chunks] == ["B", "C"]
    assert [source.filename for source in result.sources] == ["B.pdf", "C.pdf"]
    human_prompt = chat_model.invocations[0][1][1]
    assert "內容：B" in human_prompt
    assert "內容：C" in human_prompt
    assert "內容：A" not in human_prompt
    assert [event.stage for event in events] == [
        "received",
        "retrieve",
        "retrieved",
        "rerank",
        "reranked",
        "context",
        "generate",
        "complete",
    ]


def test_answer_and_stream_return_same_reranked_chunk_order() -> None:
    documents = [make_document(name) for name in "ABC"]
    vector_store = FakeVectorStore(documents)
    reranker = FakeReranker([0.2, 0.9, 0.5])
    chat_model = FakeChatModel()
    service = RagService(
        vector_store,
        Settings(retrieval_top_k=3, top_k=2),
        chat_model=chat_model,
        reranker=reranker,
    )

    regular = service.answer("問題")
    streamed = service.answer_stream("問題", cancel_event=Event())

    assert [document.page_content for document in regular.chunks] == ["B", "C"]
    assert [document.page_content for document in streamed.chunks] == ["B", "C"]
    assert [source.filename for source in regular.sources] == ["B.pdf", "C.pdf"]
    assert [source.filename for source in streamed.sources] == ["B.pdf", "C.pdf"]


def test_stream_checks_cancellation_immediately_after_rerank() -> None:
    cancel_event = Event()

    class CancellingReranker(FakeReranker):
        def compute_score(self, pairs: list[list[str]], normalize: bool) -> object:
            result = super().compute_score(pairs, normalize)
            cancel_event.set()
            return result

    service = RagService(
        FakeVectorStore([make_document("A")]),
        Settings(retrieval_top_k=1, top_k=1),
        chat_model=FakeChatModel(),
        reranker=CancellingReranker(0.9),
    )

    with pytest.raises(RagAnswerCancelled):
        service.answer_stream("問題", cancel_event=cancel_event)
