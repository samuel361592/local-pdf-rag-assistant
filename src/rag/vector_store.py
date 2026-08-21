"""Ollama Embedding 與記憶體內 FAISS 索引。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from .config import Settings


def create_embeddings(settings: Settings) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def create_vector_store(
    chunks: Sequence[Document], settings: Settings, embeddings: Any | None = None
) -> FAISS:
    if not chunks:
        raise ValueError("沒有可建立向量索引的文字區塊。")
    embedding_client = embeddings or create_embeddings(settings)
    return FAISS.from_documents(list(chunks), embedding_client)


def similarity_search(vector_store: Any, question: str, top_k: int = 4) -> list[Document]:
    if not question.strip():
        raise ValueError("問題不可為空白。")
    return list(vector_store.similarity_search(question, k=top_k))
