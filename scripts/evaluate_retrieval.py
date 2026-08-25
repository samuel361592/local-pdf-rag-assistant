"""以固定 PDF 與 Golden Dataset 執行 FAISS Retrieval benchmark。"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.config import Settings, load_settings
from src.rag.document_loader import PDFProcessingError, process_uploaded_pdfs
from src.rag.evaluation import (
    DatasetError,
    GoldenDatasetItem,
    RetrievalEvaluationResult,
    calculate_hit_rate_at_k,
    calculate_mrr,
    evaluate_retrieved_chunks,
    load_golden_dataset,
)
from src.rag.vector_store import (
    create_embeddings,
    create_vector_store,
    similarity_search,
)

EVALUATION_DOCUMENTS_DIR = PROJECT_ROOT / "documents" / "evaluation"
DATASET_PATH = PROJECT_ROOT / "evaluation" / "dataset.json"
ENV_PATH = PROJECT_ROOT / ".env"
REQUIRED_ENV_KEYS = (
    "OLLAMA_BASE_URL",
    "EMBEDDING_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
)
HIT_RATE_CUTOFFS = (1, 3, 5, 8)
MINIMUM_EVALUATION_TOP_K = max(HIT_RATE_CUTOFFS)


class EvaluationSetupError(RuntimeError):
    """代表 benchmark 輸入檔案或目錄設定錯誤。"""


class NamedBytesIO(BytesIO):
    """讓本機 PDF 能沿用接受上傳檔案的既有處理函式。"""

    def __init__(self, content: bytes, name: str) -> None:
        super().__init__(content)
        self.name = name


def report_progress(message: str) -> None:
    print(f"Progress: {message}", flush=True)


def validate_evaluation_env_file(env_path: Path) -> None:
    if not env_path.exists():
        raise EvaluationSetupError(f"找不到環境設定檔：{env_path}")
    if not env_path.is_file():
        raise EvaluationSetupError(f"環境設定路徑不是檔案：{env_path}")

    values = dotenv_values(env_path)
    missing_keys = [key for key in REQUIRED_ENV_KEYS if not values.get(key)]
    if missing_keys:
        raise EvaluationSetupError(
            f".env 缺少必要設定：{', '.join(missing_keys)}。"
        )


def print_environment(settings: Settings) -> None:
    print("Environment:")
    print(f"Environment File: {ENV_PATH}")
    print(f"Ollama Base URL: {settings.ollama_base_url}")
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"Chunk Size: {settings.chunk_size}")
    print(f"Chunk Overlap: {settings.chunk_overlap}")
    print(flush=True)


def validate_evaluation_top_k(settings: Settings) -> None:
    if settings.top_k < MINIMUM_EVALUATION_TOP_K:
        raise EvaluationSetupError(
            f"TOP_K 必須至少為 {MINIMUM_EVALUATION_TOP_K}，"
            "才能計算 Hit Rate@1、@3、@5、@8。"
        )


def verify_embedding_environment(
    settings: Settings,
    embedding_client: Any | None = None,
) -> tuple[Any, int]:
    client = embedding_client or create_embeddings(settings)
    try:
        vector = client.embed_query("retrieval evaluation environment check")
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        raise EvaluationSetupError(
            "Ollama Embedding 環境檢查失敗："
            f"{settings.embedding_model} @ {settings.ollama_base_url}；{detail}"
        ) from exc

    if not vector:
        raise EvaluationSetupError(
            f"Embedding 模型 {settings.embedding_model} 沒有回傳向量。"
        )
    return client, len(vector)


def find_evaluation_pdfs(directory: Path) -> list[Path]:
    if not directory.exists():
        raise EvaluationSetupError(f"找不到 Evaluation PDF 目錄：{directory}")
    if not directory.is_dir():
        raise EvaluationSetupError(f"Evaluation PDF 路徑不是目錄：{directory}")

    pdf_paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdf_paths:
        raise EvaluationSetupError(f"Evaluation PDF 目錄中沒有 PDF：{directory}")
    return pdf_paths


def open_evaluation_pdfs(pdf_paths: list[Path]) -> list[NamedBytesIO]:
    pdf_files: list[NamedBytesIO] = []
    try:
        for pdf_path in pdf_paths:
            try:
                content = pdf_path.read_bytes()
            except OSError as exc:
                raise EvaluationSetupError(
                    f"無法讀取 Evaluation PDF：{pdf_path}"
                ) from exc
            pdf_files.append(NamedBytesIO(content, pdf_path.name))
    except Exception:
        for pdf_file in pdf_files:
            pdf_file.close()
        raise
    return pdf_files


def evaluate_questions(
    vector_store: object,
    dataset: list[GoldenDatasetItem],
    top_k: int,
) -> list[RetrievalEvaluationResult]:
    results: list[RetrievalEvaluationResult] = []
    total_questions = len(dataset)
    for index, item in enumerate(dataset, start=1):
        report_progress(f"Evaluating question {index} of {total_questions}.")
        print(f"Question: {item.question}")
        chunks = similarity_search(vector_store, item.question, top_k)
        result = evaluate_retrieved_chunks(item, chunks)
        results.append(result)

        if result.is_hit:
            print("Result: HIT")
            print(f"Rank: {result.rank}")
            print(f"Source: {item.expected_source}")
        else:
            print("Result: MISS")
            print(f"Expected Source: {item.expected_source}")
            print(f"Expected Text: {item.expected_text}")
        print(flush=True)

    return results


def print_summary(
    *,
    document_count: int,
    settings: Settings,
    results: list[RetrievalEvaluationResult],
) -> None:
    mrr = calculate_mrr(results)
    report_cutoffs = sorted({*HIT_RATE_CUTOFFS, settings.top_k})

    print("================================")
    print("Retrieval Evaluation")
    print("================================")
    print()
    print(f"Documents: {document_count}")
    print(f"Questions: {len(results)}")
    print(f"Top-K: {settings.top_k}")
    print()
    for cutoff in report_cutoffs:
        hit_rate = calculate_hit_rate_at_k(results, cutoff)
        print(f"Hit Rate@{cutoff}: {hit_rate:.2%}")
    print(f"MRR: {mrr:.2f}")
    print()
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"Chunk Size: {settings.chunk_size}")
    print(f"Chunk Overlap: {settings.chunk_overlap}")


def run_evaluation() -> None:
    started_at = perf_counter()

    report_progress("Starting retrieval evaluation.")
    report_progress("Checking the environment file.")
    validate_evaluation_env_file(ENV_PATH)
    settings = load_settings(env_path=ENV_PATH, override_env=True)
    validate_evaluation_top_k(settings)
    print_environment(settings)

    report_progress("Checking Ollama and the embedding model.")
    embedding_client, vector_dimensions = verify_embedding_environment(settings)
    report_progress(
        f"Embedding environment is ready with {vector_dimensions} dimensions."
    )

    report_progress("Scanning the evaluation PDF directory.")
    pdf_paths = find_evaluation_pdfs(EVALUATION_DOCUMENTS_DIR)
    report_progress(f"Found {len(pdf_paths)} PDF documents.")

    report_progress("Loading the Golden Dataset.")
    dataset = load_golden_dataset(DATASET_PATH)
    report_progress(f"Loaded {len(dataset)} questions.")

    report_progress("Reading and chunking PDF documents.")
    pdf_files = open_evaluation_pdfs(pdf_paths)
    try:
        batch = process_uploaded_pdfs(pdf_files, settings)
    finally:
        for pdf_file in pdf_files:
            pdf_file.close()

    report_progress(
        f"Created {len(batch.chunks)} chunks from {batch.page_count} pages."
    )
    report_progress(
        f"Creating the FAISS vector store with {settings.embedding_model}. "
        "Embedding may take some time."
    )
    vector_store = create_vector_store(
        batch.chunks,
        settings,
        embeddings=embedding_client,
    )
    report_progress("FAISS vector store is ready.")

    report_progress("Starting question evaluation.")
    results = evaluate_questions(vector_store, dataset, settings.top_k)
    report_progress("Calculating final metrics.")
    print_summary(
        document_count=batch.document_count,
        settings=settings,
        results=results,
    )
    elapsed_seconds = perf_counter() - started_at
    report_progress(f"Evaluation completed in {elapsed_seconds:.1f} seconds.")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    try:
        run_evaluation()
    except (DatasetError, EvaluationSetupError, PDFProcessingError, ValueError) as exc:
        print(f"Evaluation 無法執行：{exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Evaluation 無法存取必要檔案：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Evaluation 執行失敗：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
