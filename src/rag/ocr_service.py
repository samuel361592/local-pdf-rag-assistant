"""PaddleOCR-based OCR utilities for PDF pages and uploaded images."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
from typing import Any

from .config import Settings

OCR_ENGINE_NAME = "PaddleOCR"
SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PADDLEX_CACHE_HOME = PROJECT_ROOT / "storage" / "paddlex"


class OCRError(RuntimeError):
    """代表使用者可理解的 OCR 處理錯誤。"""


@dataclass(frozen=True, slots=True)
class OCRResult:
    text: str
    confidence: float | None = None


def package_version(distribution_name: str) -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "unknown"


def _create_paddle_ocr(settings: Settings) -> Any:
    _configure_paddlex_cache()
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OCRError(
            "需要 OCR 時找不到 PaddleOCR。請先安裝 paddleocr 與 paddlepaddle。"
        ) from exc

    init_attempts = (
        {
            "lang": settings.ocr_lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "enable_mkldnn": settings.ocr_enable_mkldnn,
        },
        {
            "lang": settings.ocr_lang,
            "use_angle_cls": False,
            "enable_mkldnn": settings.ocr_enable_mkldnn,
        },
        {
            "lang": settings.ocr_lang,
            "enable_mkldnn": settings.ocr_enable_mkldnn,
        },
    )
    errors: list[str] = []
    for kwargs in init_attempts:
        try:
            return PaddleOCR(**kwargs)
        except TypeError as exc:
            errors.append(str(exc))
    detail = "; ".join(error for error in errors if error)
    raise OCRError(f"無法初始化 PaddleOCR：{detail or '參數不相容'}。")


def _configure_paddlex_cache() -> None:
    cache_home = Path(
        os.environ.get("PADDLE_PDX_CACHE_HOME", str(DEFAULT_PADDLEX_CACHE_HOME))
    )
    cache_home.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_home)

    # PaddleX reads cache constants at import time. If another import already loaded
    # them, update the module-level values before PaddleOCR builds its pipeline.
    try:
        import paddlex.inference.utils.official_models as official_models_module
        import paddlex.utils.cache as cache_module
    except ImportError:
        return

    cache_module.CACHE_DIR = str(cache_home)
    cache_module.FUNC_CACHE_DIR = str(cache_home / "func_ret")
    cache_module.FILE_LOCK_DIR = str(cache_home / "locks")
    cache_module.TEMP_DIR = str(cache_home / "temp")
    official_models_module.CACHE_DIR = str(cache_home)
    official_models_module.FILE_LOCK_DIR = str(cache_home / "locks")
    official_models_module.official_models._save_dir = cache_home / "official_models"


class OCRService:
    def __init__(self, settings: Settings, engine: Any | None = None) -> None:
        self.settings = settings
        self._engine = engine

    @property
    def engine(self) -> Any:
        if self._engine is None:
            self._engine = _create_paddle_ocr(self.settings)
        return self._engine

    def image_file_to_text(self, image_path: Path) -> OCRResult:
        try:
            raw_result = self._run_ocr(image_path)
        except OCRError:
            raise
        except Exception as exc:
            detail = str(exc).strip() or type(exc).__name__
            raise OCRError(
                f"OCR 辨識圖片失敗：{image_path.name}；{detail}"
            ) from exc
        return _normalize_ocr_result(raw_result)

    def pdf_page_to_text(self, pdf_path: Path, page_index: int) -> OCRResult:
        png_path: Path | None = None
        try:
            try:
                import pymupdf
            except ImportError as exc:
                raise OCRError("需要 OCR PDF 時找不到 PyMuPDF。請先安裝 pymupdf。") from exc
            with pymupdf.open(pdf_path) as pdf:
                page = pdf.load_page(page_index)
                pixmap = page.get_pixmap(dpi=self.settings.ocr_dpi, alpha=False)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as file:
                    png_path = Path(file.name)
                pixmap.save(png_path)
            return self.image_file_to_text(png_path)
        except OCRError:
            raise
        except Exception as exc:
            detail = str(exc).strip() or type(exc).__name__
            raise OCRError(
                f"OCR 辨識 PDF 第 {page_index + 1} 頁失敗：{detail}"
            ) from exc
        finally:
            if png_path is not None:
                png_path.unlink(missing_ok=True)

    def _run_ocr(self, image_path: Path) -> Any:
        engine = self.engine
        if hasattr(engine, "predict"):
            return engine.predict(str(image_path))
        if hasattr(engine, "ocr"):
            try:
                return engine.ocr(str(image_path), cls=False)
            except TypeError:
                return engine.ocr(str(image_path))
        raise OCRError("PaddleOCR engine 沒有可用的 predict 或 ocr 方法。")


def _normalize_ocr_result(raw_result: Any) -> OCRResult:
    texts: list[str] = []
    confidences: list[float] = []
    _collect_ocr_items(raw_result, texts, confidences)
    text = "\n".join(item.strip() for item in texts if item.strip())
    confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else None
    )
    return OCRResult(text=text, confidence=confidence)


def _collect_ocr_items(
    value: Any,
    texts: list[str],
    confidences: list[float],
) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        for key in ("rec_texts", "texts"):
            raw_texts = value.get(key)
            if isinstance(raw_texts, list):
                texts.extend(str(item) for item in raw_texts if str(item).strip())
        for key in ("rec_scores", "scores"):
            raw_scores = value.get(key)
            if isinstance(raw_scores, list):
                confidences.extend(_safe_float(item) for item in raw_scores)
        if texts:
            return
        for item in value.values():
            _collect_ocr_items(item, texts, confidences)
        return
    if isinstance(value, tuple) and len(value) >= 2 and isinstance(value[0], str):
        texts.append(value[0])
        confidences.append(_safe_float(value[1]))
        return
    if isinstance(value, list):
        if len(value) == 2 and isinstance(value[1], tuple) and value[1]:
            text = value[1][0]
            if isinstance(text, str):
                texts.append(text)
                if len(value[1]) > 1:
                    confidences.append(_safe_float(value[1][1]))
                return
        for item in value:
            _collect_ocr_items(item, texts, confidences)
        return
    rec_texts = getattr(value, "rec_texts", None)
    rec_scores = getattr(value, "rec_scores", None)
    if isinstance(rec_texts, list):
        texts.extend(str(item) for item in rec_texts if str(item).strip())
    if isinstance(rec_scores, list):
        confidences.extend(_safe_float(item) for item in rec_scores)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
