"""應用程式設定與環境變數驗證。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


@dataclass(frozen=True, slots=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3"
    chat_model: str = "qwen3:4b"
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 4
    ocr_mode: str = "auto"
    ocr_lang: str = "ch"
    ocr_min_text_chars: int = 30
    ocr_dpi: int = 300
    ocr_enable_images: bool = True
    ocr_enable_mkldnn: bool = False

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE 必須大於 0。")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP 不可小於 0。")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必須小於 CHUNK_SIZE。")
        if self.top_k <= 0:
            raise ValueError("TOP_K 必須大於 0。")
        if self.ocr_mode not in {"auto", "force", "disabled"}:
            raise ValueError("OCR_MODE 必須是 auto、force 或 disabled。")
        if self.ocr_min_text_chars < 0:
            raise ValueError("OCR_MIN_TEXT_CHARS 不可小於 0。")
        if self.ocr_dpi <= 0:
            raise ValueError("OCR_DPI 必須大於 0。")


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} 必須是整數，目前值為 {raw_value!r}。") from exc


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().casefold()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} 必須是布林值，目前值為 {raw_value!r}。")


def load_settings(
    *,
    env_path: Path | None = None,
    override_env: bool = False,
) -> Settings:
    """從 .env 與環境變數建立並驗證設定。"""

    load_dotenv(
        dotenv_path=env_path or PROJECT_ENV_PATH,
        override=override_env,
    )
    defaults = Settings()
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", defaults.ollama_base_url),
        embedding_model=os.getenv("EMBEDDING_MODEL", defaults.embedding_model),
        chat_model=os.getenv("CHAT_MODEL", defaults.chat_model),
        chunk_size=_read_int("CHUNK_SIZE", defaults.chunk_size),
        chunk_overlap=_read_int("CHUNK_OVERLAP", defaults.chunk_overlap),
        top_k=_read_int("TOP_K", defaults.top_k),
        ocr_mode=os.getenv("OCR_MODE", defaults.ocr_mode).strip().casefold(),
        ocr_lang=os.getenv("OCR_LANG", defaults.ocr_lang),
        ocr_min_text_chars=_read_int(
            "OCR_MIN_TEXT_CHARS", defaults.ocr_min_text_chars
        ),
        ocr_dpi=_read_int("OCR_DPI", defaults.ocr_dpi),
        ocr_enable_images=_read_bool(
            "OCR_ENABLE_IMAGES", defaults.ocr_enable_images
        ),
        ocr_enable_mkldnn=_read_bool(
            "OCR_ENABLE_MKLDNN", defaults.ocr_enable_mkldnn
        ),
    )
