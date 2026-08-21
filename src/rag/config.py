"""應用程式設定與環境變數驗證。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3"
    chat_model: str = "qwen3:4b"
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 4

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE 必須大於 0。")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP 不可小於 0。")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必須小於 CHUNK_SIZE。")
        if self.top_k <= 0:
            raise ValueError("TOP_K 必須大於 0。")


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} 必須是整數，目前值為 {raw_value!r}。") from exc


def load_settings() -> Settings:
    """從 .env 與環境變數建立並驗證設定。"""

    load_dotenv()
    defaults = Settings()
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", defaults.ollama_base_url),
        embedding_model=os.getenv("EMBEDDING_MODEL", defaults.embedding_model),
        chat_model=os.getenv("CHAT_MODEL", defaults.chat_model),
        chunk_size=_read_int("CHUNK_SIZE", defaults.chunk_size),
        chunk_overlap=_read_int("CHUNK_OVERLAP", defaults.chunk_overlap),
        top_k=_read_int("TOP_K", defaults.top_k),
    )
