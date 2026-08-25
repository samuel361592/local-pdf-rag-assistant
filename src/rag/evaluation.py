"""Retrieval benchmark 的 Golden Dataset 與指標計算。"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

from .document_loader import SOURCE_FILENAME_KEY


class DatasetError(ValueError):
    """代表 Golden Dataset 無法載入或內容不合法。"""


class MissReason(str, Enum):
    SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
    EXPECTED_TEXT_NOT_IN_PARSED_PDF = "EXPECTED_TEXT_NOT_IN_PARSED_PDF"
    EXPECTED_TEXT_NOT_IN_ANY_CHUNK = "EXPECTED_TEXT_NOT_IN_ANY_CHUNK"
    GOLD_CHUNK_NOT_IN_TOP_K = "GOLD_CHUNK_NOT_IN_TOP_K"
    SOURCE_MISMATCH = "SOURCE_MISMATCH"
    INTERNAL_EVALUATION_ERROR = "INTERNAL_EVALUATION_ERROR"


@dataclass(frozen=True, slots=True)
class GoldenDatasetItem:
    question: str
    expected_source: str
    expected_text: str


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    item: GoldenDatasetItem
    rank: int | None
    miss_reason: MissReason | None = None

    @property
    def is_hit(self) -> bool:
        return self.rank is not None

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.rank is None else 1.0 / self.rank


@dataclass(frozen=True, slots=True)
class GoldenDatasetCheckResult:
    item: GoldenDatasetItem
    source_exists: bool
    expected_text_in_parsed_pdf: bool
    expected_text_in_any_chunk: bool = False

    @property
    def passed_full_text_check(self) -> bool:
        return self.source_exists and self.expected_text_in_parsed_pdf

    @property
    def passed_chunk_check(self) -> bool:
        return self.passed_full_text_check and self.expected_text_in_any_chunk

    @property
    def miss_reason(self) -> MissReason | None:
        if not self.source_exists:
            return MissReason.SOURCE_NOT_FOUND
        if not self.expected_text_in_parsed_pdf:
            return MissReason.EXPECTED_TEXT_NOT_IN_PARSED_PDF
        if not self.expected_text_in_any_chunk:
            return MissReason.EXPECTED_TEXT_NOT_IN_ANY_CHUNK
        return None


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


def check_golden_dataset_full_text(
    dataset: Sequence[GoldenDatasetItem],
    available_sources: Sequence[str],
    parsed_text_by_source: dict[str, str],
) -> list[GoldenDatasetCheckResult]:
    """嚴格檢查 expected_source 與解析後 PDF 全文，不改寫任何文字。"""

    source_names = set(available_sources)
    return [
        GoldenDatasetCheckResult(
            item=item,
            source_exists=item.expected_source in source_names,
            expected_text_in_parsed_pdf=(
                item.expected_source in source_names
                and item.expected_text
                in parsed_text_by_source.get(item.expected_source, "")
            ),
        )
        for item in dataset
    ]


def check_chunk_evaluability(
    checks: Sequence[GoldenDatasetCheckResult],
    chunks: Sequence[Document],
) -> list[GoldenDatasetCheckResult]:
    """嚴格確認 expected_text 完整存在於來源正確的單一 Chunk。"""

    chunks_by_source: dict[str, list[str]] = {}
    for chunk in chunks:
        source = chunk.metadata.get(SOURCE_FILENAME_KEY)
        if isinstance(source, str):
            chunks_by_source.setdefault(source, []).append(chunk.page_content)

    return [
        replace(
            check,
            expected_text_in_any_chunk=(
                check.passed_full_text_check
                and any(
                    check.item.expected_text in content
                    for content in chunks_by_source.get(
                        check.item.expected_source, []
                    )
                )
            ),
        )
        for check in checks
    ]


def classify_miss_reason(
    item: GoldenDatasetItem,
    retrieved_chunks: Sequence[Document],
    check: GoldenDatasetCheckResult | None = None,
) -> MissReason:
    """依前置檢查與嚴格 Top-K 結果分類 MISS 原因。"""

    if check is not None:
        precheck_reason = check.miss_reason
        if precheck_reason is not None:
            return precheck_reason
        # 已確認正確 Chunk 存在，但嚴格比對仍未命中 Top-K。
        return MissReason.GOLD_CHUNK_NOT_IN_TOP_K

    if any(
        item.expected_text in chunk.page_content
        and chunk.metadata.get(SOURCE_FILENAME_KEY) != item.expected_source
        for chunk in retrieved_chunks
    ):
        return MissReason.SOURCE_MISMATCH
    return MissReason.INTERNAL_EVALUATION_ERROR


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
