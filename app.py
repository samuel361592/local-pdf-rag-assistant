"""本機 RAG 問答助手的 Streamlit 入口。"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, replace
from math import ceil
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Sequence
from uuid import uuid4

import streamlit as st
from langchain_core.documents import Document

from src.rag.config import Settings, load_settings
from src.rag.document_loader import (
    CONTENT_TYPE_KEY,
    OCR_CONFIDENCE_KEY,
    OCR_EXTRACTION_METHOD,
    PAGE_NUMBER_KEY,
    PDFProcessingError,
    SOURCE_FILENAME_KEY,
    TEXT_EXTRACTION_METHOD_KEY,
    VISUAL_CONTENT_TYPE,
    VLM_EXTRACTION_METHOD,
    process_uploaded_pdfs,
)
from src.rag.rag_service import (
    RagAnswerCancelled,
    RagProgressEvent,
    RagResult,
    RagService,
)
from src.rag.reranker import (
    RerankerComputeError,
    RerankerDependencyError,
    RerankerDownloadError,
    RerankerLoadError,
    RerankerMemoryError,
    RerankerModelNotFoundError,
)
from src.rag.vector_store import create_vector_store


ACTIVE_JOB_STATUSES = frozenset({"running", "cancelling"})
LOGGER = logging.getLogger(__name__)
OCR_MODE_LABELS = {
    "auto": "自動",
    "force": "強制",
    "disabled": "停用",
}
OCR_MODE_VALUES = {label: value for value, label in OCR_MODE_LABELS.items()}
VLM_MODE_LABELS = {
    "auto": "自動",
    "all": "全部頁面",
    "disabled": "停用",
}
VLM_MODE_VALUES = {label: value for value, label in VLM_MODE_LABELS.items()}


@dataclass(frozen=True, slots=True)
class RagJobUpdate:
    kind: str
    payload: Any = None


@dataclass(slots=True)
class RagAnswerJob:
    job_id: str
    question: str
    status: str = "running"
    cancel_event: Event = field(default_factory=Event)
    updates: Queue[RagJobUpdate] = field(default_factory=Queue)
    progress_events: list[RagProgressEvent] = field(default_factory=list)
    partial_answer: str = ""
    result: RagResult | None = None
    error: Exception | None = None
    thread: Thread | None = None


def is_job_active(job: RagAnswerJob | None) -> bool:
    return job is not None and job.status in ACTIVE_JOB_STATUSES


def _run_rag_job(
    job: RagAnswerJob,
    vector_store: Any,
    settings: Settings,
) -> None:
    """在背景執行問答；此函式不得呼叫任何 Streamlit API。"""

    try:
        service = RagService(vector_store, settings)
        result = service.answer_stream(
            job.question,
            cancel_event=job.cancel_event,
            progress_callback=lambda event: job.updates.put(
                RagJobUpdate("progress", event)
            ),
            token_callback=lambda token: job.updates.put(RagJobUpdate("token", token)),
        )
        job.updates.put(RagJobUpdate("completed", result))
    except RagAnswerCancelled as exc:
        job.updates.put(RagJobUpdate("cancelled", exc.partial_result))
    except Exception as exc:
        LOGGER.exception("RAG answer job failed.")
        job.updates.put(RagJobUpdate("failed", exc))


def start_rag_job(
    question: str,
    vector_store: Any,
    settings: Settings,
) -> RagAnswerJob:
    job = RagAnswerJob(job_id=uuid4().hex, question=question.strip())
    job.thread = Thread(
        target=_run_rag_job,
        args=(job, vector_store, settings),
        daemon=True,
        name=f"rag-answer-{job.job_id[:8]}",
    )
    job.thread.start()
    return job


def drain_rag_job_updates(job: RagAnswerJob) -> bool:
    """將背景工作訊息套用至 UI 狀態，回傳本次是否進入終止狀態。"""

    was_active = is_job_active(job)
    while True:
        try:
            update = job.updates.get_nowait()
        except Empty:
            break

        if update.kind == "progress":
            job.progress_events.append(update.payload)
        elif update.kind == "token":
            job.partial_answer += str(update.payload)
        elif update.kind == "completed":
            job.result = update.payload
            job.partial_answer = job.result.answer
            job.status = "completed"
        elif update.kind == "cancelled":
            job.result = update.payload
            job.partial_answer = job.result.answer
            job.status = "cancelled"
        elif update.kind == "failed":
            job.error = update.payload
            job.status = "failed"

    return was_active and not is_job_active(job)


def friendly_error(error: Exception) -> str:
    message = str(error)
    lowered = message.lower()
    if isinstance(error, PDFProcessingError):
        return message
    if isinstance(error, RerankerDependencyError):
        return "未安裝 FlagEmbedding。請先執行 pip install -r requirements.txt。"
    if isinstance(error, RerankerMemoryError):
        return "載入或執行重排模型時記憶體不足。請關閉其他程式後再試，或停用 Reranker。"
    if isinstance(error, RerankerDownloadError):
        return "無法下載重排模型。請檢查網路連線、Hugging Face 存取狀態或本機模型快取。"
    if isinstance(error, RerankerModelNotFoundError):
        return "找不到指定的重排模型。請確認 RERANKER_MODEL 名稱與 Hugging Face 存取權限。"
    if isinstance(error, RerankerLoadError):
        return "無法載入重排模型。請確認已安裝 FlagEmbedding，並檢查網路連線或模型快取。"
    if isinstance(error, RerankerComputeError):
        return "重排模型計算失敗。請檢查模型快取與可用記憶體後再試。"
    if "paddleocr" in lowered or "paddlepaddle" in lowered:
        return "需要 OCR 時找不到 PaddleOCR 或 PaddlePaddle。請先安裝 OCR 相依套件。"
    if "ocr" in lowered:
        return "OCR 辨識失敗。請確認 OCR 模型已下載完成，或改用較清晰的圖片/PDF 後再試。"
    if "connection" in lowered or "connect" in lowered or "refused" in lowered:
        return "無法連線到 Ollama。請確認 Ollama 已啟動，且 OLLAMA_BASE_URL 設定正確。"
    if "not found" in lowered or "pull model" in lowered or "404" in lowered:
        return "找不到指定的 Ollama 模型。請先下載 bge-m3 與 qwen3:4b。"
    return "處理時發生未預期錯誤，請檢查檔案、OCR 與 Ollama 設定。"


def get_page_window(
    total_items: int, current_page: int, page_size: int
) -> tuple[int, int, int, int]:
    """回傳校正後頁碼、總頁數，以及當頁的起訖索引。"""

    if total_items < 0:
        raise ValueError("total_items 不可小於 0")
    if page_size <= 0:
        raise ValueError("page_size 必須大於 0")

    total_pages = max(1, ceil(total_items / page_size))
    safe_page = min(max(current_page, 1), total_pages)
    start = (safe_page - 1) * page_size
    end = min(start + page_size, total_items)
    return safe_page, total_pages, start, end


def change_chunk_preview_page(offset: int, total_pages: int) -> None:
    """將預覽頁碼往前或往後移動，並限制在有效範圍內。"""

    current_page = int(st.session_state.get("chunk_preview_page", 1))
    st.session_state["chunk_preview_page"] = min(
        max(current_page + offset, 1), total_pages
    )


def reset_chunk_preview_page() -> None:
    """篩選或每頁筆數變更時回到第一頁。"""

    st.session_state["chunk_preview_page"] = 1


def filter_chunks(
    chunks: Sequence[Document],
    filename: str | None = None,
    page_number: object | None = None,
    keyword: str = "",
) -> list[tuple[int, Document]]:
    """依來源、頁碼及關鍵字篩選，並保留 Chunk 的原始編號。"""

    clean_keyword = keyword.strip().casefold()
    matches: list[tuple[int, Document]] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata
        if filename is not None and metadata.get(SOURCE_FILENAME_KEY) != filename:
            continue
        if page_number is not None and metadata.get(PAGE_NUMBER_KEY) != page_number:
            continue
        if clean_keyword and clean_keyword not in chunk.page_content.casefold():
            continue
        matches.append((index, chunk))
    return matches


def shorten_filename(filename: object, max_length: int = 44) -> str:
    """縮短過長檔名供列表標題使用，完整檔名仍顯示在展開內容中。"""

    text = str(filename)
    if max_length < 8:
        raise ValueError("max_length 不可小於 8")
    if len(text) <= max_length:
        return text

    tail_length = min(12, max_length // 3)
    head_length = max_length - tail_length - 1
    return f"{text[:head_length]}…{text[-tail_length:]}"


def ocr_preview_text_key(source: str) -> str:
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"ocr_preview_text_{digest}"


def render_chunk_preview(chunks: Sequence[Document]) -> None:
    """以分頁方式顯示切塊結果，避免一次渲染全部內容。"""

    st.subheader(f"共 {len(chunks):,} 個 Chunks")

    filenames = sorted(
        {
            str(chunk.metadata.get(SOURCE_FILENAME_KEY, "未知檔案"))
            for chunk in chunks
        }
    )
    filename_options = ("全部", *filenames)
    if st.session_state.get("chunk_preview_filename") not in filename_options:
        st.session_state["chunk_preview_filename"] = "全部"

    file_column, page_column, search_column = st.columns([1.2, 1, 2])
    selected_filename = file_column.selectbox(
        "檔案",
        options=filename_options,
        key="chunk_preview_filename",
        on_change=reset_chunk_preview_page,
    )

    pages_for_file = {
        chunk.metadata.get(PAGE_NUMBER_KEY, "未知")
        for chunk in chunks
        if selected_filename == "全部"
        or chunk.metadata.get(SOURCE_FILENAME_KEY) == selected_filename
    }
    page_options = (
        "全部",
        *sorted(
            pages_for_file,
            key=lambda value: (
                (0, int(value)) if str(value).isdigit() else (1, str(value))
            ),
        ),
    )
    if st.session_state.get("chunk_preview_page_filter") not in page_options:
        st.session_state["chunk_preview_page_filter"] = "全部"
    selected_page = page_column.selectbox(
        "頁碼",
        options=page_options,
        key="chunk_preview_page_filter",
        on_change=reset_chunk_preview_page,
    )
    keyword = search_column.text_input(
        "搜尋",
        placeholder="輸入關鍵字",
        key="chunk_preview_search",
        on_change=reset_chunk_preview_page,
    )

    filtered_chunks = filter_chunks(
        chunks,
        filename=None if selected_filename == "全部" else selected_filename,
        page_number=None if selected_page == "全部" else selected_page,
        keyword=keyword,
    )
    if len(filtered_chunks) != len(chunks):
        st.caption(f"符合條件：{len(filtered_chunks):,} 個 Chunks")
    if not filtered_chunks:
        st.info("找不到符合目前篩選條件的 Chunk。")
        return

    size_column, number_column = st.columns(2)
    page_size = size_column.selectbox(
        "每頁",
        options=(5, 10, 20, 50),
        index=2,
        key="chunk_preview_page_size",
        on_change=reset_chunk_preview_page,
    )
    current_page, total_pages, _, _ = get_page_window(
        len(filtered_chunks),
        int(st.session_state.get("chunk_preview_page", 1)),
        page_size,
    )
    st.session_state["chunk_preview_page"] = current_page

    number_column.number_input(
        f"前往頁碼（共 {total_pages} 頁）",
        min_value=1,
        max_value=total_pages,
        step=1,
        key="chunk_preview_page",
    )

    previous_column, status_column, next_column = st.columns([1, 2, 1])
    previous_column.button(
        "← 上一頁",
        disabled=current_page == 1,
        use_container_width=True,
        on_click=change_chunk_preview_page,
        args=(-1, total_pages),
    )
    status_column.markdown(
        f"<p style='text-align:center; margin:0.45rem 0 0'>"
        f"第 {current_page} / {total_pages} 頁</p>",
        unsafe_allow_html=True,
    )
    next_column.button(
        "下一頁 →",
        disabled=current_page == total_pages,
        use_container_width=True,
        on_click=change_chunk_preview_page,
        args=(1, total_pages),
    )

    current_page = int(st.session_state["chunk_preview_page"])
    _, _, start, end = get_page_window(
        len(filtered_chunks), current_page, page_size
    )
    st.caption(f"顯示第 {start + 1}–{end} 筆")
    for original_index, chunk in filtered_chunks[start:end]:
        filename = chunk.metadata.get(SOURCE_FILENAME_KEY, "未知檔案")
        page_number = chunk.metadata.get(PAGE_NUMBER_KEY, "未知")
        method = chunk.metadata.get(TEXT_EXTRACTION_METHOD_KEY, "unknown")
        content_type = chunk.metadata.get(CONTENT_TYPE_KEY)
        if content_type == VISUAL_CONTENT_TYPE or method == VLM_EXTRACTION_METHOD:
            method_label = "VLM 視覺分析"
        else:
            method_label = "OCR" if method == OCR_EXTRACTION_METHOD else "文字擷取"
        content = chunk.page_content.strip()
        short_filename = shorten_filename(filename)
        title = (
            f"Chunk #{original_index}　·　第 {page_number} 頁　·　"
            f"{method_label}　·　{len(content):,} 字　·　{short_filename}"
        )
        with st.expander(title):
            st.caption(f"來源檔案：{filename}")
            confidence = chunk.metadata.get(OCR_CONFIDENCE_KEY)
            if confidence is not None:
                st.caption(f"OCR 信心分數：{float(confidence):.2%}")
            st.write(content)


def render_ocr_preview(ocr_text_by_source: dict[str, str]) -> None:
    st.subheader("OCR 辨識結果")
    if not ocr_text_by_source:
        st.info("目前沒有 OCR 辨識文字。")
        return

    sources = sorted(ocr_text_by_source)
    if st.session_state.get("ocr_preview_source") not in sources:
        st.session_state["ocr_preview_source"] = sources[0]
    selected_source = st.selectbox(
        "OCR 來源",
        options=sources,
        key="ocr_preview_source",
    )
    text = ocr_text_by_source.get(selected_source, "").strip()
    if not text:
        st.warning("這個來源沒有辨識出文字。")
        return
    st.caption(f"{selected_source}｜{len(text):,} 字")
    st.text_area(
        "辨識文字",
        value=text,
        height=420,
        disabled=True,
        key=ocr_preview_text_key(selected_source),
    )


def render_visual_preview(
    visual_text_by_source: dict[str, str],
    failures_by_source: dict[str, str],
    model_name: str,
) -> None:
    st.subheader("VLM 視覺分析")
    all_sources = sorted(set(visual_text_by_source) | set(failures_by_source))
    if not all_sources:
        st.info("目前沒有 VLM 視覺分析結果。")
        return
    if st.session_state.get("vlm_preview_source") not in all_sources:
        st.session_state["vlm_preview_source"] = all_sources[0]
    selected_source = st.selectbox(
        "VLM 來源",
        options=all_sources,
        key="vlm_preview_source",
    )
    st.caption(f"模型：{model_name}｜模型生成分析，重要內容請人工確認。")
    failure = failures_by_source.get(selected_source)
    if failure:
        st.warning(failure)
        return
    visual_text = visual_text_by_source.get(selected_source, "").strip()
    if visual_text:
        st.text_area(
            "格式化視覺描述",
            value=visual_text,
            height=420,
            disabled=True,
            key=f"vlm_preview_text_{hashlib.sha1(selected_source.encode('utf-8')).hexdigest()[:12]}",
        )


def render_ocr_controls(settings: Settings) -> Settings:
    st.subheader("OCR")
    mode_label = OCR_MODE_LABELS.get(settings.ocr_mode, "自動")
    mode = st.selectbox(
        "OCR 模式",
        options=("自動", "強制", "停用"),
        index=("自動", "強制", "停用").index(mode_label),
        help="自動模式只會 OCR 空白或文字過少的 PDF 頁面；圖片檔一定需要 OCR。",
    )
    enable_images = st.checkbox(
        "支援 PNG/JPG 圖片",
        value=settings.ocr_enable_images,
    )
    return replace(
        settings,
        ocr_mode=OCR_MODE_VALUES[mode],
        ocr_enable_images=enable_images,
    )


def render_vlm_controls(settings: Settings) -> Settings:
    st.subheader("VLM 視覺分析")
    enabled = st.checkbox(
        "啟用 VLM 視覺分析",
        value=settings.vlm_enabled,
        help="VLM 會在建庫時分析圖片或 PDF 頁面，因此會增加建庫時間。",
    )
    mode_label = VLM_MODE_LABELS.get(settings.vlm_mode, "自動")
    mode = st.selectbox(
        "VLM 模式",
        options=("自動", "全部頁面", "停用"),
        index=("自動", "全部頁面", "停用").index(mode_label),
        help="自動模式分析圖片、需 OCR 的 PDF 頁及含內嵌圖片的 PDF 頁。",
    )
    st.caption(f"目前模型：{settings.vlm_model}")
    return replace(settings, vlm_enabled=enabled, vlm_mode=VLM_MODE_VALUES[mode])


def render_result(result: RagResult) -> None:
    st.subheader("回答")
    st.write(result.answer)

    if result.sources:
        st.subheader("參考來源")
        for source in result.sources:
            st.markdown(
                f"- [來源 {source.number}｜{source.content_type_label}] "
                f"{source.filename}，第 {source.page_number} 頁"
            )

    with st.expander("查看實際檢索到的原始文字區塊"):
        if not result.chunks:
            st.caption("本次沒有檢索到文字區塊。")
        for source, chunk in zip(result.sources, result.chunks, strict=True):
            st.markdown(
                f"**[來源 {source.number}｜{source.content_type_label}] "
                f"{source.filename}｜第 {source.page_number} 頁**"
            )
            st.text(chunk.page_content.strip())


def render_progress_event(event: RagProgressEvent) -> None:
    st.markdown(f"- **{event.message}**")
    if event.detail:
        st.code(event.detail, language=None)


def render_progress_events(events: Sequence[RagProgressEvent]) -> None:
    for event in events:
        render_progress_event(event)


def render_question_interface(settings: Settings) -> None:
    """顯示可在背景執行及取消的文件問答介面。"""

    job: RagAnswerJob | None = st.session_state.get("rag_job")
    if job is not None and drain_rag_job_updates(job):
        # 重新完整執行一次，讓外層 fragment 停止自動輪詢。
        st.rerun()

    job_active = is_job_active(job)
    question_column, progress_column = st.columns([2.15, 1], gap="large")

    with question_column:
        st.subheader("文件問答")
        question = st.text_area(
            "輸入問題",
            placeholder="例如：使用生成式 AI 時應注意哪些事項？",
            key="rag_question_input",
            disabled=job_active,
        )
        submit_column, stop_column = st.columns(2)
        submit_question = submit_column.button(
            "送出問題",
            type="primary",
            disabled=job_active,
            use_container_width=True,
        )
        stop_question = stop_column.button(
            "停止回答",
            disabled=not job_active,
            use_container_width=True,
        )

        if submit_question:
            if not question.strip():
                st.warning("沒有輸入問題，請先輸入想查詢的內容。")
            elif "vector_store" not in st.session_state:
                st.warning(
                    "尚未建立知識庫，請先上傳 PDF 或圖片文件並按下「建立知識庫」。"
                )
            else:
                st.session_state["rag_job"] = start_rag_job(
                    question,
                    st.session_state["vector_store"],
                    settings,
                )
                st.rerun()

        if stop_question and job is not None and is_job_active(job):
            job.status = "cancelling"
            job.cancel_event.set()
            # 立即更新停止狀態，但不讓完整 App rerun 使 Fragment ID 失效。
            st.rerun(scope="fragment")

        if job is not None:
            if job.status == "cancelling":
                st.info("正在停止回答……")
            elif job.status == "cancelled":
                st.warning("回答已由使用者停止。")
            elif job.status == "failed" and job.error is not None:
                st.error(friendly_error(job.error))

            if job.status in ACTIVE_JOB_STATUSES and job.partial_answer:
                st.subheader("回答（產生中）")
                st.write(f"{job.partial_answer}▌")
            elif job.result is not None and (
                job.status == "completed" or job.result.answer
            ):
                render_result(job.result)

    with progress_column:
        st.subheader("即時流程")
        if job is None:
            st.caption("送出問題後，這裡會顯示檢索與回答產生流程。")
        else:
            label = {
                "running": "正在處理問題……",
                "cancelling": "正在停止回答……",
                "cancelled": "回答已停止",
                "completed": "回答完成",
                "failed": "回答失敗",
            }.get(job.status, "問答狀態")
            state = (
                "running"
                if job.status in ACTIVE_JOB_STATUSES
                else "error" if job.status == "failed" else "complete"
            )
            with st.status(label, state=state, expanded=True):
                render_progress_events(job.progress_events)


def main() -> None:
    st.set_page_config(page_title="本機 RAG 問答助手", page_icon="📚", layout="wide")
    st.title("本機 RAG 問答助手")
    st.write(
        "上傳公開的 PDF 或圖片文件，建立暫存於本次工作階段的知識庫，"
        "並依文件內容進行問答。"
    )
    st.caption("本工具僅供資訊整理與學習，不構成法律或合規建議。")

    try:
        settings = load_settings()
    except ValueError as exc:
        st.error(f"設定錯誤：{exc}")
        st.stop()
    settings = render_ocr_controls(settings)
    settings = render_vlm_controls(settings)

    uploaded_files = st.file_uploader(
        "上傳一份或多份 PDF 或圖片文件（PNG / JPG）",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    current_job: RagAnswerJob | None = st.session_state.get("rag_job")
    if st.button(
        "建立知識庫",
        type="primary",
        disabled=is_job_active(current_job),
    ):
        if not uploaded_files:
            st.warning("尚未上傳 PDF 或圖片文件，請先選擇至少一份檔案。")
        else:
            try:
                with st.spinner(
                    "正在解析 PDF 或圖片文件、切分文字並建立向量索引……"
                ):
                    batch = process_uploaded_pdfs(uploaded_files, settings)
                    vector_store = create_vector_store(batch.chunks, settings)
                st.session_state["vector_store"] = vector_store
                st.session_state["knowledge_stats"] = {
                    "documents": batch.document_count,
                    "pages": batch.page_count,
                    "chunks": len(batch.chunks),
                    "native_pages": batch.extraction_stats.native_page_count,
                    "ocr_pages": batch.extraction_stats.ocr_page_count,
                    "image_pages": batch.extraction_stats.image_page_count,
                    "average_ocr_confidence": (
                        batch.extraction_stats.average_ocr_confidence
                    ),
                    "vlm_pages": batch.extraction_stats.vlm_page_count,
                    "vlm_failures": batch.extraction_stats.vlm_failure_count,
                    "visual_chunks": batch.extraction_stats.visual_chunk_count,
                    "vlm_model": settings.vlm_model,
                }
                st.session_state["knowledge_chunks"] = batch.chunks
                st.session_state["ocr_text_by_source"] = batch.ocr_text_by_source
                st.session_state["visual_text_by_source"] = batch.visual_text_by_source
                st.session_state["vlm_failures_by_source"] = batch.vlm_failures_by_source
                st.session_state["chunk_preview_page"] = 1
                for preview_key in (
                    "chunk_preview_filename",
                    "chunk_preview_page_filter",
                    "chunk_preview_search",
                    "ocr_preview_source",
                    "vlm_preview_source",
                ):
                    st.session_state.pop(preview_key, None)
                st.session_state.pop("rag_result", None)
                st.session_state.pop("rag_progress_events", None)
                st.session_state.pop("rag_job", None)
                st.success("知識庫建立完成。")
            except PDFProcessingError as exc:
                LOGGER.warning("Knowledge base creation failed.", exc_info=True)
                st.error(friendly_error(exc))
            except Exception as exc:
                LOGGER.exception("Knowledge base creation failed unexpectedly.")
                st.error(friendly_error(exc))

    stats = st.session_state.get("knowledge_stats")
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("文件數量", stats["documents"])
        col2.metric("頁數", stats["pages"])
        col3.metric("Chunk 數量", stats["chunks"])
        col4.metric("文字擷取頁", stats["native_pages"])
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("OCR 頁", stats["ocr_pages"])
        col6.metric("VLM 分析頁", stats.get("vlm_pages", 0))
        col7.metric("VLM 失敗頁", stats.get("vlm_failures", 0))
        col8.metric("Visual Chunks", stats.get("visual_chunks", 0))
        if stats.get("average_ocr_confidence") is not None:
            st.caption(f"OCR 平均信心分數：{stats['average_ocr_confidence']:.2%}")

    st.divider()
    question_tab, chunks_tab, ocr_tab, vlm_tab = st.tabs(
        ["文件問答", "Chunk 預覽", "OCR 預覽", "VLM 視覺分析"]
    )

    with question_tab:
        question_job: RagAnswerJob | None = st.session_state.get("rag_job")
        poll_interval = "250ms" if is_job_active(question_job) else None

        @st.fragment(run_every=poll_interval)
        def question_fragment() -> None:
            render_question_interface(settings)

        question_fragment()

    with chunks_tab:
        @st.fragment
        def chunk_preview_fragment() -> None:
            chunks = st.session_state.get("knowledge_chunks")
            if chunks:
                render_chunk_preview(chunks)
            else:
                st.info("建立知識庫後，可在這裡篩選並預覽切塊結果。")

        chunk_preview_fragment()

    with ocr_tab:
        @st.fragment
        def ocr_preview_fragment() -> None:
            render_ocr_preview(st.session_state.get("ocr_text_by_source", {}))

        ocr_preview_fragment()

    with vlm_tab:
        @st.fragment
        def vlm_preview_fragment() -> None:
            render_visual_preview(
                st.session_state.get("visual_text_by_source", {}),
                st.session_state.get("vlm_failures_by_source", {}),
                str(st.session_state.get("knowledge_stats", {}).get("vlm_model", settings.vlm_model)),
            )

        vlm_preview_fragment()


if __name__ == "__main__":
    main()
