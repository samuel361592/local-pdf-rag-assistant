"""檢索、Context 格式化與模型回答協調。"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Any, Callable, Sequence

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from .config import Settings
from .document_loader import (
    CONTENT_TYPE_KEY,
    PAGE_NUMBER_KEY,
    SOURCE_FILENAME_KEY,
    VISUAL_CONTENT_TYPE,
)
from .prompt import INSUFFICIENT_INFORMATION_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from .reranker import create_reranker, rerank_documents
from .vector_store import similarity_search


@dataclass(frozen=True, slots=True)
class SourceReference:
    number: int
    filename: str
    page_number: int | str
    content_type: str = "text"

    @property
    def content_type_label(self) -> str:
        return "視覺分析" if self.content_type == VISUAL_CONTENT_TYPE else "文件文字"

    @property
    def context_content_type_label(self) -> str:
        return (
            "VLM 視覺分析"
            if self.content_type == VISUAL_CONTENT_TYPE
            else "文件文字"
        )


@dataclass(frozen=True, slots=True)
class RagResult:
    answer: str
    sources: list[SourceReference]
    chunks: list[Document]


@dataclass(frozen=True, slots=True)
class RagProgressEvent:
    stage: str
    message: str
    detail: str = ""


ProgressCallback = Callable[[RagProgressEvent], None]
TokenCallback = Callable[[str], None]
CancellationCallback = Callable[[], None]


class RagAnswerCancelled(Exception):
    """使用者取消回答，並保留取消前已產生的內容。"""

    def __init__(self, partial_result: RagResult) -> None:
        super().__init__("回答已由使用者停止。")
        self.partial_result = partial_result


def _source_details(document: Document, number: int) -> SourceReference:
    filename = str(
        document.metadata.get(SOURCE_FILENAME_KEY)
        or document.metadata.get("source")
        or "未知檔案"
    )
    page_number = document.metadata.get(PAGE_NUMBER_KEY, "未知")
    content_type = str(document.metadata.get(CONTENT_TYPE_KEY, "text"))
    return SourceReference(number, filename, page_number, content_type)


def format_context(documents: Sequence[Document]) -> tuple[str, list[SourceReference]]:
    sections: list[str] = []
    sources: list[SourceReference] = []
    for number, document in enumerate(documents, start=1):
        source = _source_details(document, number)
        sources.append(source)
        sections.append(
            f"[來源 {number}]\n"
            f"檔名：{source.filename}\n"
            f"頁碼：{source.page_number}\n"
            f"內容類型：{source.context_content_type_label}\n"
            f"內容：{document.page_content.strip()}"
        )
    return "\n\n---\n\n".join(sections), sources


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [item.get("text", "") if isinstance(item, dict) else str(item) for item in content]
        return "".join(parts).strip()
    return str(content).strip()


def _chunk_text(response: Any) -> str:
    """取出串流片段文字，保留片段前後空白以免拼接後文字黏在一起。"""

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _emit_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    message: str,
    detail: str = "",
) -> None:
    if progress_callback is not None:
        progress_callback(RagProgressEvent(stage, message, detail))


class RagService:
    def __init__(
        self,
        vector_store: Any,
        settings: Settings,
        chat_model: Any | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.settings = settings
        self.reranker = reranker
        self.chat_model = chat_model or ChatOllama(
            model=settings.chat_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    def _retrieve_chunks(
        self,
        question: str,
        progress_callback: ProgressCallback | None = None,
        cancellation_callback: CancellationCallback | None = None,
    ) -> list[Document]:
        """執行一次 FAISS 搜尋，並視設定套用第二階段重排。"""

        def check_cancelled() -> None:
            if cancellation_callback is not None:
                cancellation_callback()

        candidate_k = (
            self.settings.retrieval_top_k
            if self.settings.reranker_enabled
            else self.settings.top_k
        )
        check_cancelled()
        _emit_progress(
            progress_callback,
            "retrieve",
            f"正在使用 FAISS 檢索前 {candidate_k} 個候選文字區塊。",
        )
        candidates = similarity_search(self.vector_store, question, candidate_k)
        check_cancelled()
        _emit_progress(
            progress_callback,
            "retrieved",
            f"已找到 {len(candidates)} 個候選文字區塊。",
        )
        if not candidates or not self.settings.reranker_enabled:
            return candidates

        check_cancelled()
        _emit_progress(
            progress_callback,
            "rerank",
            f"正在使用 {self.settings.reranker_model} 重新排序。",
        )
        if self.reranker is None:
            self.reranker = create_reranker(
                self.settings.reranker_model,
                self.settings.reranker_use_fp16,
            )
        chunks = rerank_documents(
            question,
            candidates,
            self.reranker,
            self.settings.top_k,
        )
        check_cancelled()
        sources = [
            _source_details(chunk, number)
            for number, chunk in enumerate(chunks, start=1)
        ]
        detail = "\n".join(
            f"[來源 {source.number}｜{source.content_type_label}] "
            f"{source.filename}，第 {source.page_number} 頁"
            for source in sources
        )
        _emit_progress(
            progress_callback,
            "reranked",
            f"重排完成，選出前 {len(chunks)} 個文字區塊。",
            detail,
        )
        return chunks

    def answer(
        self,
        question: str,
        progress_callback: ProgressCallback | None = None,
    ) -> RagResult:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("請先輸入問題。")

        _emit_progress(progress_callback, "received", "已收到問題。", clean_question)
        chunks = self._retrieve_chunks(clean_question, progress_callback)
        if not chunks:
            _emit_progress(
                progress_callback,
                "no_results",
                "沒有檢索到可用文字區塊。",
            )
            return RagResult(INSUFFICIENT_INFORMATION_MESSAGE, [], [])

        _emit_progress(
            progress_callback,
            "context",
            "正在組合帶來源編號的 Context。",
        )
        context, sources = format_context(chunks)
        _emit_progress(
            progress_callback,
            "generate",
            f"正在呼叫 {self.settings.chat_model} 產生回答。",
        )
        response = self.chat_model.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", build_user_prompt(clean_question, context)),
            ]
        )
        answer = _message_text(response) or INSUFFICIENT_INFORMATION_MESSAGE
        _emit_progress(progress_callback, "complete", "回答產生完成。")
        return RagResult(answer, sources, chunks)

    def answer_stream(
        self,
        question: str,
        *,
        cancel_event: Event,
        progress_callback: ProgressCallback | None = None,
        token_callback: TokenCallback | None = None,
    ) -> RagResult:
        """以串流方式回答問題，並在安全檢查點回應取消要求。"""

        clean_question = question.strip()
        if not clean_question:
            raise ValueError("請先輸入問題。")

        chunks: list[Document] = []
        sources: list[SourceReference] = []
        answer_parts: list[str] = []

        def raise_if_cancelled() -> None:
            if not cancel_event.is_set():
                return
            partial_answer = "".join(answer_parts).strip()
            _emit_progress(progress_callback, "cancelled", "回答已由使用者停止。")
            raise RagAnswerCancelled(
                RagResult(partial_answer, list(sources), list(chunks))
            )

        raise_if_cancelled()
        _emit_progress(progress_callback, "received", "已收到問題。", clean_question)
        chunks = self._retrieve_chunks(
            clean_question,
            progress_callback,
            raise_if_cancelled,
        )
        if not chunks:
            _emit_progress(
                progress_callback,
                "no_results",
                "沒有檢索到可用文字區塊。",
            )
            return RagResult(INSUFFICIENT_INFORMATION_MESSAGE, [], [])

        raise_if_cancelled()
        _emit_progress(
            progress_callback,
            "context",
            "正在組合帶來源編號的 Context。",
        )
        context, sources = format_context(chunks)
        raise_if_cancelled()
        _emit_progress(
            progress_callback,
            "generate",
            f"正在呼叫 {self.settings.chat_model} 產生回答。",
        )

        response_stream = self.chat_model.stream(
            [
                ("system", SYSTEM_PROMPT),
                ("human", build_user_prompt(clean_question, context)),
            ]
        )
        try:
            for response_chunk in response_stream:
                raise_if_cancelled()
                text = _chunk_text(response_chunk)
                if not text:
                    continue
                answer_parts.append(text)
                if token_callback is not None:
                    token_callback(text)
            raise_if_cancelled()
        finally:
            close_stream = getattr(response_stream, "close", None)
            if callable(close_stream):
                close_stream()

        answer = "".join(answer_parts).strip() or INSUFFICIENT_INFORMATION_MESSAGE
        _emit_progress(progress_callback, "complete", "回答產生完成。")
        return RagResult(answer, sources, chunks)
