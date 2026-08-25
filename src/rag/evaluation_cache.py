"""Retrieval Evaluation 專用的安全 FAISS 磁碟快取。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import Settings
from .document_loader import DocumentBatch, pdf_processing_cache_settings

CACHE_FORMAT_VERSION = 1
INDEX_FILENAME = "index.faiss"
DOCUMENTS_FILENAME = "documents.json"
MANIFEST_FILENAME = "manifest.json"


class EvaluationCacheError(RuntimeError):
    """代表 Evaluation 快取存在但無法安全載入或寫入。"""


@dataclass(frozen=True, slots=True)
class EvaluationCacheSignature:
    cache_key: str
    pdf_hash: str
    settings_hash: str
    pdfs: list[dict[str, object]]
    settings: dict[str, object]


@dataclass(frozen=True, slots=True)
class CachedEvaluationArtifacts:
    vector_store: FAISS
    batch: DocumentBatch


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_evaluation_cache_signature(
    pdf_paths: Sequence[Path],
    settings: Settings,
    embedding_dimensions: int,
) -> EvaluationCacheSignature:
    """以 PDF 內容和所有索引相關設定建立穩定的快取識別。"""

    pdfs = [
        {
            "filename": path.name,
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(pdf_paths, key=lambda candidate: candidate.name)
    ]
    cache_settings = {
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": embedding_dimensions,
        "pdf_processing": pdf_processing_cache_settings(settings),
    }
    pdf_hash = _sha256_bytes(_canonical_json(pdfs).encode("utf-8"))
    settings_hash = _sha256_bytes(
        _canonical_json(cache_settings).encode("utf-8")
    )
    cache_key = _sha256_bytes(f"{pdf_hash}:{settings_hash}".encode("ascii"))
    return EvaluationCacheSignature(
        cache_key=cache_key,
        pdf_hash=pdf_hash,
        settings_hash=settings_hash,
        pdfs=pdfs,
        settings=cache_settings,
    )


def evaluation_cache_directory(
    cache_root: Path, signature: EvaluationCacheSignature
) -> Path:
    return cache_root / signature.cache_key


def _expected_manifest(signature: EvaluationCacheSignature) -> dict[str, object]:
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "cache_key": signature.cache_key,
        "pdf_hash": signature.pdf_hash,
        "settings_hash": signature.settings_hash,
        "pdfs": signature.pdfs,
        "settings": signature.settings,
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationCacheError(f"無法讀取快取檔案 {path.name}：{exc}") from exc


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def load_evaluation_cache(
    cache_root: Path,
    signature: EvaluationCacheSignature,
    embeddings: Any,
) -> CachedEvaluationArtifacts | None:
    """載入並完整驗證快取；不存在時回傳 None，損毀時拋出可恢復錯誤。"""

    cache_dir = evaluation_cache_directory(cache_root, signature)
    if not cache_dir.exists():
        return None

    manifest_path = cache_dir / MANIFEST_FILENAME
    documents_path = cache_dir / DOCUMENTS_FILENAME
    index_path = cache_dir / INDEX_FILENAME
    manifest = _read_json(manifest_path)
    expected_manifest = _expected_manifest(signature)
    if not isinstance(manifest, dict) or any(
        manifest.get(key) != value for key, value in expected_manifest.items()
    ):
        raise EvaluationCacheError("快取 Manifest 與目前 PDF 或設定不一致。")

    for path, checksum_key in (
        (documents_path, "documents_sha256"),
        (index_path, "index_sha256"),
    ):
        if not path.is_file():
            raise EvaluationCacheError(f"快取缺少 {path.name}。")
        try:
            actual_checksum = _sha256_file(path)
        except OSError as exc:
            raise EvaluationCacheError(
                f"無法驗證快取檔案 {path.name}：{exc}"
            ) from exc
        if manifest.get(checksum_key) != actual_checksum:
            raise EvaluationCacheError(f"快取檔案 {path.name} 的雜湊驗證失敗。")

    payload = _read_json(documents_path)
    try:
        raw_chunks = payload["chunks"]
        chunks = [
            Document(
                page_content=entry["page_content"],
                metadata=entry["metadata"],
            )
            for entry in raw_chunks
        ]
        batch = DocumentBatch(
            document_count=int(payload["document_count"]),
            page_count=int(payload["page_count"]),
            chunks=chunks,
            parsed_text_by_source={
                str(key): str(value)
                for key, value in payload["parsed_text_by_source"].items()
            },
        )
        index = faiss.read_index(str(index_path))
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise EvaluationCacheError(f"快取內容格式錯誤：{exc}") from exc

    try:
        expected_dimensions = signature.settings["embedding_dimensions"]
        if index.d != expected_dimensions:
            raise EvaluationCacheError(
                "FAISS 索引維度與目前 Embedding 維度不一致。"
            )
        if index.ntotal != len(chunks):
            raise EvaluationCacheError(
                "FAISS 向量數量與快取 Chunk 數量不一致。"
            )

        document_ids = [f"chunk-{index}" for index in range(len(chunks))]
        docstore = InMemoryDocstore(dict(zip(document_ids, chunks, strict=True)))
        vector_store = FAISS(
            embeddings,
            index,
            docstore,
            dict(enumerate(document_ids)),
        )
    except EvaluationCacheError:
        raise
    except Exception as exc:
        raise EvaluationCacheError(f"無法還原 FAISS 快取：{exc}") from exc
    return CachedEvaluationArtifacts(vector_store=vector_store, batch=batch)


def _replace_text_file(path: Path, content: str) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def save_evaluation_cache(
    cache_root: Path,
    signature: EvaluationCacheSignature,
    vector_store: FAISS,
    batch: DocumentBatch,
) -> Path:
    """以 Manifest 最後提交的方式保存可驗證、無 pickle 的 FAISS 快取。"""

    cache_dir = evaluation_cache_directory(cache_root, signature)
    documents_path = cache_dir / DOCUMENTS_FILENAME
    index_path = cache_dir / INDEX_FILENAME
    manifest_path = cache_dir / MANIFEST_FILENAME
    index_temp_path = index_path.with_name(f"{index_path.name}.tmp")

    payload = {
        "document_count": batch.document_count,
        "page_count": batch.page_count,
        "parsed_text_by_source": batch.parsed_text_by_source,
        "chunks": [
            {
                "page_content": chunk.page_content,
                "metadata": _json_safe(chunk.metadata),
            }
            for chunk in batch.chunks
        ],
    }
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _replace_text_file(documents_path, _canonical_json(payload))
        faiss.write_index(vector_store.index, str(index_temp_path))
        os.replace(index_temp_path, index_path)
        manifest = {
            **_expected_manifest(signature),
            "documents_sha256": _sha256_file(documents_path),
            "index_sha256": _sha256_file(index_path),
        }
        _replace_text_file(manifest_path, _canonical_json(manifest))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise EvaluationCacheError(f"無法保存 Evaluation 快取：{exc}") from exc
    finally:
        index_temp_path.unlink(missing_ok=True)
    return cache_dir
