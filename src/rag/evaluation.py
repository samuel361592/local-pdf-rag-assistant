"""Retrieval benchmark 的 Golden Dataset 與指標計算。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

from .document_loader import SOURCE_FILENAME_KEY


class DatasetError(ValueError):
    """代表 Golden Dataset 無法載入或內容不合法。"""


@dataclass(frozen=True, slots=True)
class GoldenDatasetItem:
    question: str
    expected_source: str
    expected_text: str


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    item: GoldenDatasetItem
    rank: int | None

    @property
    def is_hit(self) -> bool:
        return self.rank is not None

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.rank is None else 1.0 / self.rank


def load_golden_dataset(dataset_path: Path) -> list[GoldenDatasetItem]:
    """讀取並驗證 JSON 格式的 Golden Dataset。"""

    try:
        raw_dataset = dataset_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DatasetError(f"找不到 Golden Dataset：{dataset_path}") from exc
    except (OSError, UnicodeError) as exc:
        raise DatasetError(f"無法讀取 Golden Dataset：{dataset_path}") from exc

    try:
        data = json.loads(raw_dataset)
    except json.JSONDecodeError as exc:
        raise DatasetError(
            "Golden Dataset JSON 格式錯誤："
            f"第 {exc.lineno} 行、第 {exc.colno} 欄，{exc.msg}。"
        ) from exc

    if not isinstance(data, list):
        raise DatasetError("Golden Dataset 最外層必須是 JSON 陣列。")
    if not data:
        raise DatasetError("Golden Dataset 不可為空，請至少加入一筆測試題。")

    required_fields = ("question", "expected_source", "expected_text")
    items: list[GoldenDatasetItem] = []
    for index, entry in enumerate(data, start=1):
        if not isinstance(entry, dict):
            raise DatasetError(f"Golden Dataset 第 {index} 筆必須是 JSON 物件。")

        missing_fields = [field for field in required_fields if field not in entry]
        if missing_fields:
            raise DatasetError(
                f"Golden Dataset 第 {index} 筆缺少必要欄位："
                f"{', '.join(missing_fields)}。"
            )

        values: dict[str, str] = {}
        for field in required_fields:
            value = entry[field]
            if not isinstance(value, str) or not value.strip():
                raise DatasetError(
                    f"Golden Dataset 第 {index} 筆的 {field} 必須是非空白字串。"
                )
            values[field] = value.strip()

        items.append(GoldenDatasetItem(**values))

    return items


def evaluate_retrieved_chunks(
    item: GoldenDatasetItem, chunks: Sequence[Document]
) -> RetrievalEvaluationResult:
    """回傳第一個同時符合來源檔名與關鍵文字的 Chunk 排名。"""

    for rank, chunk in enumerate(chunks, start=1):
        source_filename = chunk.metadata.get(SOURCE_FILENAME_KEY)
        if (
            source_filename == item.expected_source
            and item.expected_text in chunk.page_content
        ):
            return RetrievalEvaluationResult(item=item, rank=rank)

    return RetrievalEvaluationResult(item=item, rank=None)


def calculate_hit_rate(results: Sequence[RetrievalEvaluationResult]) -> float:
    """計算至少一個正確 Chunk 出現在 Top-K 的題目比例。"""

    if not results:
        raise ValueError("至少需要一筆 Retrieval 結果才能計算 Hit Rate。")
    return sum(result.is_hit for result in results) / len(results)


def calculate_hit_rate_at_k(
    results: Sequence[RetrievalEvaluationResult], k: int
) -> float:
    """計算正確 Chunk 最早出現在前 K 名內的題目比例。"""

    if not results:
        raise ValueError("至少需要一筆 Retrieval 結果才能計算 Hit Rate。")
    if k <= 0:
        raise ValueError("K 必須大於 0。")
    return sum(
        result.rank is not None and result.rank <= k for result in results
    ) / len(results)


def calculate_mrr(results: Sequence[RetrievalEvaluationResult]) -> float:
    """計算所有題目第一個正確 Chunk 的 Mean Reciprocal Rank。"""

    if not results:
        raise ValueError("至少需要一筆 Retrieval 結果才能計算 MRR。")
    return sum(result.reciprocal_rank for result in results) / len(results)
