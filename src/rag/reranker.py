"""以 FlagEmbedding 對第一階段檢索結果重新評分與排序。"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from math import isfinite
from numbers import Real
from typing import Any

from langchain_core.documents import Document


class RerankerError(RuntimeError):
    """Reranker 載入或計算失敗。"""


class RerankerDependencyError(RerankerError):
    """缺少 FlagEmbedding 相依套件。"""


class RerankerDownloadError(RerankerError):
    """無法從 Hugging Face 下載模型。"""


class RerankerModelNotFoundError(RerankerError):
    """指定的 Hugging Face 模型不存在或無法存取。"""


class RerankerLoadError(RerankerError):
    """模型存在但無法載入。"""


class RerankerComputeError(RerankerError):
    """模型評分失敗。"""


class RerankerMemoryError(RerankerError):
    """載入模型或評分時記憶體不足。"""


def _error_detail(error: BaseException) -> str:
    return str(error).strip() or type(error).__name__


def _is_memory_error(error: BaseException) -> bool:
    detail = _error_detail(error).casefold()
    return isinstance(error, MemoryError) or any(
        marker in detail
        for marker in ("out of memory", "cannot allocate memory", "bad allocation")
    )


@lru_cache(maxsize=None)
def create_reranker(model_name: str, use_fp16: bool) -> Any:
    """延遲載入並快取 FlagReranker；不在模組 import 時載入模型。"""

    clean_model_name = model_name.strip()
    if not clean_model_name:
        raise ValueError("RERANKER_MODEL 不可為空白。")

    try:
        from FlagEmbedding import FlagReranker
    except (ImportError, ModuleNotFoundError) as exc:
        raise RerankerDependencyError(
            "未安裝 FlagEmbedding，無法建立 Reranker。"
        ) from exc

    try:
        return FlagReranker(clean_model_name, use_fp16=use_fp16)
    except Exception as exc:
        detail = _error_detail(exc)
        lowered = detail.casefold()
        if _is_memory_error(exc):
            error_type = RerankerMemoryError
        elif any(
            marker in lowered
            for marker in (
                "repository not found",
                "model not found",
                "not a valid model identifier",
                "404 client error",
            )
        ):
            error_type = RerankerModelNotFoundError
        elif any(
            marker in lowered
            for marker in (
                "connection",
                "connect",
                "network",
                "offline",
                "localentrynotfounderror",
                "timeout",
                "timed out",
                "download",
            )
        ):
            error_type = RerankerDownloadError
        else:
            error_type = RerankerLoadError
        raise error_type(
            f"無法載入 Reranker 模型 {clean_model_name}：{detail}"
        ) from exc


def _normalize_scores(raw_scores: Any) -> list[float]:
    if isinstance(raw_scores, Real):
        values = [raw_scores]
    else:
        converted = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
        values = [converted] if isinstance(converted, Real) else list(converted)

    scores: list[float] = []
    for value in values:
        if not isinstance(value, Real):
            raise ValueError("Reranker 分數必須是數值。")
        score = float(value)
        if not isfinite(score):
            raise ValueError("Reranker 分數必須是有限數值。")
        scores.append(score)
    return scores


def rerank_documents(
    question: str,
    documents: Sequence[Document],
    reranker: Any,
    top_k: int,
) -> list[Document]:
    """依相關性分數穩定排序，並保留原始 Document 與 metadata。"""

    clean_question = question.strip()
    if not clean_question:
        raise ValueError("問題不可為空白。")
    if top_k <= 0:
        raise ValueError("top_k 必須大於 0。")
    if not documents:
        return []

    pairs = [[clean_question, document.page_content] for document in documents]
    try:
        raw_scores = reranker.compute_score(pairs, normalize=True)
    except Exception as exc:
        detail = _error_detail(exc)
        if _is_memory_error(exc):
            raise RerankerMemoryError(
                f"Reranker 計算時記憶體不足：{detail}"
            ) from exc
        raise RerankerComputeError(f"Reranker 計算失敗：{detail}") from exc

    try:
        scores = _normalize_scores(raw_scores)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Reranker 回傳無效分數：{exc}") from exc
    if len(scores) != len(documents):
        raise ValueError(
            "Reranker 分數數量與候選文件數量不同："
            f"{len(scores)} != {len(documents)}。"
        )

    ranked = sorted(
        enumerate(documents),
        key=lambda item: scores[item[0]],
        reverse=True,
    )
    return [document for _, document in ranked[:top_k]]
