"""檢索、Context 格式化與模型回答協調。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from .config import Settings
from .document_loader import PAGE_NUMBER_KEY, SOURCE_FILENAME_KEY
from .prompt import INSUFFICIENT_INFORMATION_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from .vector_store import similarity_search


@dataclass(frozen=True, slots=True)
class SourceReference:
    number: int
    filename: str
    page_number: int | str


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


def _source_details(document: Document, number: int) -> SourceReference:
    filename = str(
        document.metadata.get(SOURCE_FILENAME_KEY)
        or document.metadata.get("source")
        or "未知檔案"
    )
    page_number = document.metadata.get(PAGE_NUMBER_KEY, "未知")
    return SourceReference(number, filename, page_number)


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
    ) -> None:
        self.vector_store = vector_store
        self.settings = settings
        self.chat_model = chat_model or ChatOllama(
            model=settings.chat_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    def answer(
        self,
        question: str,
        progress_callback: ProgressCallback | None = None,
    ) -> RagResult:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("請先輸入問題。")

        _emit_progress(progress_callback, "received", "已收到問題。", clean_question)
        _emit_progress(
            progress_callback,
            "retrieve",
            f"正在檢索最相關的 {self.settings.top_k} 個文字區塊。",
        )
        chunks = similarity_search(self.vector_store, clean_question, self.settings.top_k)
        if not chunks:
            _emit_progress(
                progress_callback,
                "no_results",
                "沒有檢索到可用文字區塊。",
            )
            return RagResult(INSUFFICIENT_INFORMATION_MESSAGE, [], [])

        retrieved_sources = [
            _source_details(chunk, number)
            for number, chunk in enumerate(chunks, start=1)
        ]
        retrieved_detail = "\n".join(
            f"[來源 {source.number}] {source.filename}，第 {source.page_number} 頁"
            for source in retrieved_sources
        )
        _emit_progress(
            progress_callback,
            "retrieved",
            f"已找到 {len(chunks)} 個相關文字區塊。",
            retrieved_detail,
        )

        context, sources = format_context(chunks)
        _emit_progress(
            progress_callback,
            "context",
            "正在組合帶來源編號的 Context。",
            f"Context 長度約 {len(context):,} 個字元。",
        )
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
