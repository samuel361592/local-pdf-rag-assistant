from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pytest
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from scripts.evaluate_retrieval import (
    EvaluationSetupError,
    evaluate_questions,
    print_precheck_results,
    print_summary,
    validate_evaluation_env_file,
    validate_evaluation_top_k,
    verify_embedding_environment,
)
from src.rag.config import Settings, load_settings
from src.rag.evaluation import (
    DatasetError,
    GoldenDatasetCheckResult,
    GoldenDatasetItem,
    MissReason,
    RetrievalEvaluationResult,
    calculate_hit_rate,
    calculate_hit_rate_at_k,
    calculate_mrr,
    check_chunk_evaluability,
    check_golden_dataset_full_text,
    classify_miss_reason,
    evaluate_retrieved_chunks,
    load_golden_dataset,
)
from src.rag.evaluation_cache import (
    EvaluationCacheError,
    build_evaluation_cache_signature,
    load_evaluation_cache,
    save_evaluation_cache,
)
from src.rag.document_loader import DocumentBatch


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


def make_passed_check(item: GoldenDatasetItem) -> GoldenDatasetCheckResult:
    return GoldenDatasetCheckResult(
        item=item,
        source_exists=True,
        expected_text_in_parsed_pdf=True,
        expected_text_in_any_chunk=True,
    )


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


def test_full_text_precheck_uses_strict_source_and_text_matching(
    golden_item: GoldenDatasetItem,
) -> None:
    checks = check_golden_dataset_full_text(
        [
            golden_item,
            GoldenDatasetItem("問題二", "missing.pdf", "Exact Text"),
            GoldenDatasetItem("問題三", "expected.pdf", "govern"),
        ],
        ["expected.pdf"],
        {"expected.pdf": "GOVERN, MAP, MEASURE, and MANAGE"},
    )

    assert checks[0].passed_full_text_check
    assert checks[1].miss_reason is MissReason.SOURCE_NOT_FOUND
    assert (
        checks[2].miss_reason
        is MissReason.EXPECTED_TEXT_NOT_IN_PARSED_PDF
    )


def test_chunk_check_detects_expected_text_split_across_chunks() -> None:
    item = GoldenDatasetItem("問題", "expected.pdf", "boundary text")
    full_text_checks = check_golden_dataset_full_text(
        [item],
        ["expected.pdf"],
        {"expected.pdf": "prefix boundary text suffix"},
    )

    checks = check_chunk_evaluability(
        full_text_checks,
        [
            make_document("prefix boundary", "expected.pdf"),
            make_document(" text suffix", "expected.pdf"),
        ],
    )

    assert checks[0].passed_full_text_check
    assert not checks[0].passed_chunk_check
    assert checks[0].miss_reason is MissReason.EXPECTED_TEXT_NOT_IN_ANY_CHUNK


def test_evaluable_miss_is_gold_chunk_not_in_top_k(
    golden_item: GoldenDatasetItem,
) -> None:
    reason = classify_miss_reason(
        golden_item,
        [make_document("irrelevant", "other.pdf")],
        make_passed_check(golden_item),
    )

    assert reason is MissReason.GOLD_CHUNK_NOT_IN_TOP_K


def test_source_mismatch_is_reported_without_precheck_context(
    golden_item: GoldenDatasetItem,
) -> None:
    reason = classify_miss_reason(
        golden_item,
        [make_document(golden_item.expected_text, "other.pdf")],
    )

    assert reason is MissReason.SOURCE_MISMATCH


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
    assert "Rank:" in output
    assert f"Source: {golden_item.expected_source}" in output
    assert "MISS Reason:" in output


def test_question_error_is_classified_and_evaluation_continues(
    golden_item: GoldenDatasetItem,
) -> None:
    class FailingVectorStore:
        def similarity_search(self, _question: str, k: int) -> list[Document]:
            raise RuntimeError(f"failed at {k}")

    results = evaluate_questions(
        FailingVectorStore(),
        [golden_item],
        top_k=40,
        checks=[make_passed_check(golden_item)],
    )

    assert results[0].rank is None
    assert results[0].miss_reason is MissReason.INTERNAL_EVALUATION_ERROR


def test_precheck_output_includes_question_source_text_and_reason(
    golden_item: GoldenDatasetItem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checks = [
        GoldenDatasetCheckResult(
            item=golden_item,
            source_exists=True,
            expected_text_in_parsed_pdf=True,
            expected_text_in_any_chunk=False,
        )
    ]

    print_precheck_results(checks)

    output = capsys.readouterr().out
    assert "Question Number: 1" in output
    assert f"Source: {golden_item.expected_source}" in output
    assert f"Expected Text: {golden_item.expected_text}" in output
    assert "EXPECTED_TEXT_NOT_IN_ANY_CHUNK" in output


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
    assert "Golden Dataset Questions: 5" in output
    assert "Passed Parsed PDF Check: 5" in output
    assert "Passed Chunk Evaluability Check: 5" in output
    assert "FAISS Cache Used: NO" in output
    assert "INTERNAL_EVALUATION_ERROR: 1" in output


def test_cache_signature_changes_for_pdf_and_index_settings(tmp_path: Path) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"version-one")
    base_settings = Settings(
        embedding_model="embedding-a",
        chunk_size=500,
        chunk_overlap=80,
    )
    base = build_evaluation_cache_signature([pdf_path], base_settings, 1024)

    changed_model = build_evaluation_cache_signature(
        [pdf_path],
        Settings(
            embedding_model="embedding-b",
            chunk_size=500,
            chunk_overlap=80,
        ),
        1024,
    )
    changed_dimensions = build_evaluation_cache_signature(
        [pdf_path], base_settings, 768
    )
    changed_chunk_size = build_evaluation_cache_signature(
        [pdf_path],
        Settings(
            embedding_model="embedding-a",
            chunk_size=600,
            chunk_overlap=80,
        ),
        1024,
    )
    changed_overlap = build_evaluation_cache_signature(
        [pdf_path],
        Settings(
            embedding_model="embedding-a",
            chunk_size=500,
            chunk_overlap=100,
        ),
        1024,
    )
    pdf_path.write_bytes(b"version-two")
    changed_pdf = build_evaluation_cache_signature(
        [pdf_path], base_settings, 1024
    )

    assert len(
        {
            base.cache_key,
            changed_model.cache_key,
            changed_dimensions.cache_key,
            changed_chunk_size.cache_key,
            changed_overlap.cache_key,
            changed_pdf.cache_key,
        }
    ) == 6


def test_faiss_cache_round_trip_and_corruption_detection(tmp_path: Path) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"pdf-content")
    settings = Settings(embedding_model="test-embedding")
    signature = build_evaluation_cache_signature([pdf_path], settings, 3)
    chunk = make_document("cached chunk", "document.pdf")
    index = faiss.IndexFlatL2(3)
    index.add(np.array([[0.1, 0.2, 0.3]], dtype="float32"))
    vector_store = FAISS(
        FakeEmbeddings(),
        index,
        InMemoryDocstore({"chunk-0": chunk}),
        {0: "chunk-0"},
    )
    batch = DocumentBatch(
        document_count=1,
        page_count=1,
        chunks=[chunk],
        parsed_text_by_source={"document.pdf": "cached chunk"},
    )

    cache_dir = save_evaluation_cache(
        tmp_path / "cache", signature, vector_store, batch
    )
    loaded = load_evaluation_cache(
        tmp_path / "cache", signature, FakeEmbeddings()
    )

    assert loaded is not None
    assert loaded.batch.chunks[0].page_content == "cached chunk"
    assert not list(cache_dir.glob("*.pkl"))

    (cache_dir / "documents.json").write_text("{}", encoding="utf-8")
    with pytest.raises(EvaluationCacheError, match="雜湊驗證失敗"):
        load_evaluation_cache(
            tmp_path / "cache", signature, FakeEmbeddings()
        )
