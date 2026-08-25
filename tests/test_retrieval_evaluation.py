from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from scripts.evaluate_retrieval import (
    EvaluationSetupError,
    evaluate_questions,
    print_summary,
    validate_evaluation_env_file,
    validate_evaluation_top_k,
    verify_embedding_environment,
)
from src.rag.config import Settings, load_settings
from src.rag.evaluation import (
    DatasetError,
    GoldenDatasetItem,
    RetrievalEvaluationResult,
    calculate_hit_rate,
    calculate_hit_rate_at_k,
    calculate_mrr,
    evaluate_retrieved_chunks,
    load_golden_dataset,
)


@pytest.fixture
def golden_item() -> GoldenDatasetItem:
    return GoldenDatasetItem(
        question="核心功能是什麼？",
        expected_source="expected.pdf",
        expected_text="GOVERN, MAP, MEASURE, and MANAGE",
    )


def make_document(content: str, source: str) -> Document:
    return Document(page_content=content, metadata={"source_filename": source})


class FakeVectorStore:
    def __init__(self, chunks: list[Document]) -> None:
        self.chunks = chunks
        self.requested_k: int | None = None

    def similarity_search(self, _question: str, k: int) -> list[Document]:
        self.requested_k = k
        return self.chunks[:k]


class FakeEmbeddings:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1, 0.2, 0.3]

    def embed_query(self, _text: str) -> list[float]:
        return self.vector


def test_correct_chunk_at_rank_one(golden_item: GoldenDatasetItem) -> None:
    chunks = [
        make_document(
            "The functions are GOVERN, MAP, MEASURE, and MANAGE.",
            "expected.pdf",
        )
    ]

    result = evaluate_retrieved_chunks(golden_item, chunks)

    assert result.is_hit
    assert result.rank == 1
    assert result.reciprocal_rank == 1.0


def test_correct_chunk_at_rank_two(golden_item: GoldenDatasetItem) -> None:
    chunks = [
        make_document("irrelevant", "other.pdf"),
        make_document(
            "The functions are GOVERN, MAP, MEASURE, and MANAGE.",
            "expected.pdf",
        ),
    ]

    result = evaluate_retrieved_chunks(golden_item, chunks)

    assert result.is_hit
    assert result.rank == 2
    assert result.reciprocal_rank == 0.5


def test_top_k_without_correct_chunk_is_a_miss(
    golden_item: GoldenDatasetItem,
) -> None:
    chunks = [
        make_document("irrelevant", "other.pdf"),
        make_document("also irrelevant", "another.pdf"),
    ]

    result = evaluate_retrieved_chunks(golden_item, chunks)

    assert not result.is_hit
    assert result.rank is None
    assert result.reciprocal_rank == 0.0


def test_matching_source_without_expected_text_is_a_miss(
    golden_item: GoldenDatasetItem,
) -> None:
    chunks = [make_document("different content", "expected.pdf")]

    result = evaluate_retrieved_chunks(golden_item, chunks)

    assert not result.is_hit


def test_matching_text_without_expected_source_is_a_miss(
    golden_item: GoldenDatasetItem,
) -> None:
    chunks = [
        make_document(
            "GOVERN, MAP, MEASURE, and MANAGE",
            "other.pdf",
        )
    ]

    result = evaluate_retrieved_chunks(golden_item, chunks)

    assert not result.is_hit


def test_hit_rate_is_calculated_from_hit_count(
    golden_item: GoldenDatasetItem,
) -> None:
    results = [
        RetrievalEvaluationResult(golden_item, rank=1),
        RetrievalEvaluationResult(golden_item, rank=None),
        RetrievalEvaluationResult(golden_item, rank=3),
        RetrievalEvaluationResult(golden_item, rank=None),
    ]

    assert calculate_hit_rate(results) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("cutoff", "expected_hit_rate"),
    [(1, 0.2), (3, 0.4), (5, 0.6), (8, 0.8)],
)
def test_hit_rate_at_k_uses_first_correct_chunk_rank(
    cutoff: int,
    expected_hit_rate: float,
    golden_item: GoldenDatasetItem,
) -> None:
    results = [
        RetrievalEvaluationResult(golden_item, rank=1),
        RetrievalEvaluationResult(golden_item, rank=2),
        RetrievalEvaluationResult(golden_item, rank=4),
        RetrievalEvaluationResult(golden_item, rank=7),
        RetrievalEvaluationResult(golden_item, rank=None),
    ]

    assert calculate_hit_rate_at_k(results, cutoff) == pytest.approx(
        expected_hit_rate
    )


def test_mrr_uses_first_matching_rank_for_each_question(
    golden_item: GoldenDatasetItem,
) -> None:
    results = [
        RetrievalEvaluationResult(golden_item, rank=1),
        RetrievalEvaluationResult(golden_item, rank=2),
        RetrievalEvaluationResult(golden_item, rank=None),
    ]

    assert calculate_mrr(results) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_dataset_rejects_invalid_json(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[{", encoding="utf-8")

    with pytest.raises(DatasetError, match="JSON 格式錯誤"):
        load_golden_dataset(dataset_path)


def test_dataset_rejects_missing_required_field(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        '[{"question": "問題", "expected_source": "source.pdf"}]',
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="缺少必要欄位.*expected_text"):
        load_golden_dataset(dataset_path)


def test_dataset_rejects_empty_list(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")

    with pytest.raises(DatasetError, match="不可為空"):
        load_golden_dataset(dataset_path)


def test_evaluation_env_rejects_missing_required_setting(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("EMBEDDING_MODEL=test-model\n", encoding="utf-8")

    with pytest.raises(EvaluationSetupError, match="缺少必要設定"):
        validate_evaluation_env_file(env_path)


def test_load_settings_can_reload_and_override_old_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "EMBEDDING_MODEL=new-embedding-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EMBEDDING_MODEL", "old-embedding-model")

    settings = load_settings(env_path=env_path, override_env=True)

    assert settings.embedding_model == "new-embedding-model"


def test_embedding_environment_check_returns_vector_dimensions() -> None:
    client = FakeEmbeddings([0.1, 0.2, 0.3, 0.4])

    checked_client, dimensions = verify_embedding_environment(
        Settings(),
        embedding_client=client,
    )

    assert checked_client is client
    assert dimensions == 4


def test_embedding_environment_check_reports_model_error() -> None:
    class FailingEmbeddings:
        def embed_query(self, _text: str) -> list[float]:
            raise RuntimeError("model not found")

    with pytest.raises(
        EvaluationSetupError,
        match="Ollama Embedding 環境檢查失敗.*model not found",
    ):
        verify_embedding_environment(
            Settings(embedding_model="missing-model"),
            embedding_client=FailingEmbeddings(),
        )


def test_evaluation_top_k_must_cover_largest_hit_rate_cutoff() -> None:
    with pytest.raises(EvaluationSetupError, match="TOP_K 必須至少為 8"):
        validate_evaluation_top_k(Settings(top_k=7))


@pytest.mark.parametrize(
    ("chunks", "expected_result"),
    [
        (
            [
                make_document(
                    "GOVERN, MAP, MEASURE, and MANAGE",
                    "expected.pdf",
                )
            ],
            "Result: HIT",
        ),
        ([make_document("irrelevant", "other.pdf")], "Result: MISS"),
    ],
)
def test_console_uses_text_result_without_status_symbols(
    chunks: list[Document],
    expected_result: str,
    golden_item: GoldenDatasetItem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    vector_store = FakeVectorStore(chunks)
    evaluate_questions(vector_store, [golden_item], top_k=40)

    output = capsys.readouterr().out
    assert vector_store.requested_k == 40
    assert "Progress: Evaluating question 1 of 1." in output
    assert f"Question: {golden_item.question}" in output
    assert expected_result in output
    assert "✅" not in output
    assert "❌" not in output


def test_summary_prints_hit_rate_at_each_cutoff(
    golden_item: GoldenDatasetItem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = [
        RetrievalEvaluationResult(golden_item, rank=1),
        RetrievalEvaluationResult(golden_item, rank=2),
        RetrievalEvaluationResult(golden_item, rank=4),
        RetrievalEvaluationResult(golden_item, rank=7),
        RetrievalEvaluationResult(golden_item, rank=None),
    ]

    print_summary(
        document_count=2,
        settings=Settings(top_k=40),
        results=results,
    )

    output = capsys.readouterr().out
    assert output.count("Top-K:") == 1
    assert "Top-K: 40" in output
    assert "Hit Rate@1: 20.00%" in output
    assert "Hit Rate@3: 40.00%" in output
    assert "Hit Rate@5: 60.00%" in output
    assert "Hit Rate@8: 80.00%" in output
    assert "Hit Rate@40: 80.00%" in output
    assert "Hits@" not in output
    assert "MRR:" in output
