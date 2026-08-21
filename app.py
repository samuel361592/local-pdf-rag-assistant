"""本機 PDF RAG 問答助手的 Streamlit 入口。"""

from __future__ import annotations

import streamlit as st

from src.rag.config import load_settings
from src.rag.document_loader import PDFProcessingError, process_uploaded_pdfs
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
    question = st.text_area("輸入問題", placeholder="例如：使用生成式 AI 時應注意哪些事項？")
    if st.button("送出問題"):
        if not question.strip():
            st.warning("沒有輸入問題，請先輸入想查詢的內容。")
        elif "vector_store" not in st.session_state:
            st.warning("尚未建立知識庫，請先上傳 PDF 並按下「建立知識庫」。")
        else:
            try:
                with st.spinner("正在檢索文件並產生回答……"):
                    service = RagService(st.session_state["vector_store"], settings)
                    st.session_state["rag_result"] = service.answer(question)
            except Exception as exc:
                st.error(friendly_error(exc))

    result = st.session_state.get("rag_result")
    if result:
        render_result(result)


if __name__ == "__main__":
    main()
