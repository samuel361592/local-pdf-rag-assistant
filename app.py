"""本機 PDF RAG 問答助手的 Streamlit 入口。"""

from __future__ import annotations

from math import ceil
from typing import Sequence

import streamlit as st
from langchain_core.documents import Document

from src.rag.config import load_settings
from src.rag.document_loader import (
    PAGE_NUMBER_KEY,
    PDFProcessingError,
    SOURCE_FILENAME_KEY,
    process_uploaded_pdfs,
)
from src.rag.rag_service import RagResult, RagService
from src.rag.vector_store import create_vector_store


def friendly_error(error: Exception) -> str:
    message = str(error)
    lowered = message.lower()
    if "connection" in lowered or "connect" in lowered or "refused" in lowered:
        return "無法連線到 Ollama。請確認 Ollama 已啟動，且 OLLAMA_BASE_URL 設定正確。"
    if "not found" in lowered or "pull model" in lowered or "404" in lowered:
        return "找不到指定的 Ollama 模型。請先下載 bge-m3 與 qwen3:4b。"
    return message or "處理時發生未預期錯誤，請檢查 PDF 與 Ollama 設定。"


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
        content = chunk.page_content.strip()
        short_filename = shorten_filename(filename)
        title = (
            f"Chunk #{original_index}　·　第 {page_number} 頁　·　"
            f"{len(content):,} 字　·　{short_filename}"
        )
        with st.expander(title):
            st.caption(f"來源檔案：{filename}")
            st.write(content)


def render_result(result: RagResult) -> None:
    st.subheader("回答")
    st.write(result.answer)

    if result.sources:
        st.subheader("參考來源")
        for source in result.sources:
            st.markdown(f"- [來源 {source.number}] {source.filename}，第 {source.page_number} 頁")

    with st.expander("查看實際檢索到的原始文字區塊"):
        if not result.chunks:
            st.caption("本次沒有檢索到文字區塊。")
        for source, chunk in zip(result.sources, result.chunks, strict=True):
            st.markdown(f"**[來源 {source.number}] {source.filename}｜第 {source.page_number} 頁**")
            st.text(chunk.page_content.strip())


def main() -> None:
    st.set_page_config(page_title="本機 PDF RAG 問答助手", page_icon="📚", layout="wide")
    st.title("本機 PDF RAG 問答助手")
    st.write("上傳公開 PDF，建立暫存於本次工作階段的知識庫，並依文件內容進行問答。")
    st.caption("本工具僅供資訊整理與學習，不構成法律或合規建議。")

    try:
        settings = load_settings()
    except ValueError as exc:
        st.error(f"設定錯誤：{exc}")
        st.stop()

    uploaded_files = st.file_uploader(
        "上傳一份或多份 PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("建立知識庫", type="primary"):
        if not uploaded_files:
            st.warning("尚未上傳 PDF，請先選擇至少一份檔案。")
        else:
            try:
                with st.spinner("正在解析 PDF、切分文字並建立向量索引……"):
                    batch = process_uploaded_pdfs(uploaded_files, settings)
                    vector_store = create_vector_store(batch.chunks, settings)
                st.session_state["vector_store"] = vector_store
                st.session_state["knowledge_stats"] = {
                    "documents": batch.document_count,
                    "pages": batch.page_count,
                    "chunks": len(batch.chunks),
                }
                st.session_state["knowledge_chunks"] = batch.chunks
                st.session_state["chunk_preview_page"] = 1
                for preview_key in (
                    "chunk_preview_filename",
                    "chunk_preview_page_filter",
                    "chunk_preview_search",
                ):
                    st.session_state.pop(preview_key, None)
                st.session_state.pop("rag_result", None)
                st.success("知識庫建立完成。")
            except PDFProcessingError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(friendly_error(exc))

    stats = st.session_state.get("knowledge_stats")
    if stats:
        col1, col2, col3 = st.columns(3)
        col1.metric("文件數量", stats["documents"])
        col2.metric("頁數", stats["pages"])
        col3.metric("Chunk 數量", stats["chunks"])

    st.divider()
    question_tab, chunks_tab = st.tabs(["文件問答", "Chunk 預覽"])

    with question_tab:
        st.subheader("文件問答")
        question = st.text_area(
            "輸入問題",
            placeholder="例如：使用生成式 AI 時應注意哪些事項？",
        )
        if st.button("送出問題", type="primary"):
            if not question.strip():
                st.warning("沒有輸入問題，請先輸入想查詢的內容。")
            elif "vector_store" not in st.session_state:
                st.warning("尚未建立知識庫，請先上傳 PDF 並按下「建立知識庫」。")
            else:
                try:
                    with st.spinner("正在檢索文件並產生回答……"):
                        service = RagService(
                            st.session_state["vector_store"], settings
                        )
                        st.session_state["rag_result"] = service.answer(question)
                except Exception as exc:
                    st.error(friendly_error(exc))

        result = st.session_state.get("rag_result")
        if result:
            render_result(result)

    with chunks_tab:
        chunks = st.session_state.get("knowledge_chunks")
        if chunks:
            render_chunk_preview(chunks)
        else:
            st.info("建立知識庫後，可在這裡篩選並預覽切塊結果。")


if __name__ == "__main__":
    main()
