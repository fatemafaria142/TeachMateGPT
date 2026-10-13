"""
Build the vector index — embed COPE chunks into Qdrant.

Walks ``config.TEXTBOOK_DIR`` for PDFs and text files, runs COPE chunking, embeds with
the local BAAI/bge-m3 model, writes vectors to a persistent Qdrant collection, and
updates an ingest manifest under the data experiment folder. Intended to be run
offline before teacher queries (``python ingestion/embed_and_store.py``).

Ingestion is designed to run one chapter (one file in ``TEXTBOOK_DIR``) at a time:
each run upserts only the chunks for the file(s) currently present, using deterministic
point IDs so re-running the same chapter is idempotent, and merges this run's chunk
counts into the existing manifest's per-file breakdown rather than overwriting it —
so the manifest accumulates into a full-book inventory across chapter-by-chapter runs.
"""
import json
import os
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14",
    category=UserWarning,
)

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import config
from ingestion.educational_chunking import load_and_chunk_textbook, summarize_chunk_stats
from utils.logger import log_info, log_error

_QDRANT_ID_NAMESPACE = uuid.UUID("6f6b1e0a-3c1e-4a8e-9c0a-9b1c3d4e5f60")


def _qdrant_point_id(chunk_uid: str) -> str:
    return str(uuid.uuid5(_QDRANT_ID_NAMESPACE, chunk_uid))


def _prepare_unique_ids(chunks):
    """
    Build stable Qdrant point IDs (UUIDv5 over metadata.chunk_uid).
    Deterministic across reruns of the same chapter -> idempotent upsert.
    """
    ids = []
    seen = {}
    for i, d in enumerate(chunks):
        meta = d.metadata or {}
        base = str(meta.get("chunk_uid") or f"doc_{i}")
        c = seen.get(base, 0)
        seen[base] = c + 1
        key = base if c == 0 else f"{base}__dup{c}"
        ids.append(_qdrant_point_id(key))
    return ids


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": getattr(config, "EMBEDDING_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_qdrant_client() -> QdrantClient:
    Path(config.QDRANT_DB_PATH).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=config.QDRANT_DB_PATH)


def _maybe_reset_collection(client: QdrantClient) -> None:
    if not bool(getattr(config, "RECREATE_COLLECTION_ON_INGEST", False)):
        return
    try:
        client.delete_collection(config.QDRANT_COLLECTION_NAME)
        log_info(
            f"Deleted existing collection '{config.QDRANT_COLLECTION_NAME}' for clean ingest."
        )
    except Exception:
        pass


def _ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if config.QDRANT_COLLECTION_NAME in existing:
        return
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=int(getattr(config, "EMBEDDING_DIM", 1024)),
            distance=Distance.COSINE,
        ),
    )
    log_info(f"Created Qdrant collection '{config.QDRANT_COLLECTION_NAME}'.")


def _load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _write_manifest(manifest_path: Path, by_file: dict) -> dict:
    total = sum(v["total"] for v in by_file.values())
    tier_counts: dict[str, int] = {}
    for v in by_file.values():
        for tier, n in v.get("tier_counts", {}).items():
            tier_counts[tier] = tier_counts.get(tier, 0) + n

    manifest = {
        "collection_name": config.QDRANT_COLLECTION_NAME,
        "qdrant_persist_directory": config.QDRANT_DB_PATH,
        "embedding_model": config.EMBEDDING_MODEL,
        "textbook_dir": config.TEXTBOOK_DIR,
        "pdf_ingestion_mode": getattr(config, "PDF_INGESTION_MODE", ""),
        "total_chunks": total,
        "tier_counts": tier_counts,
        "chapters_ingested": len(by_file),
        "by_file": by_file,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log_info(f"Wrote ingest manifest: {manifest_path} (chapters_ingested={len(by_file)}, total_chunks={total})")
    return manifest


def initialize_vector_store():
    """Embed all textbooks currently in TEXTBOOK_DIR and upsert into Qdrant."""
    try:
        embeddings = _get_embeddings()

        if not os.path.exists(config.TEXTBOOK_DIR):
            os.makedirs(config.TEXTBOOK_DIR)
            log_info(f"Created directory {config.TEXTBOOK_DIR}. Please add textbooks.")
            return

        per_file_chunks: dict[str, list] = {}
        for filename in sorted(os.listdir(config.TEXTBOOK_DIR)):
            if filename.endswith(".pdf") or filename.endswith(".txt"):
                file_path = os.path.join(config.TEXTBOOK_DIR, filename)
                log_info(f"Processing {filename}...")
                chunks = load_and_chunk_textbook(file_path)
                per_file_chunks[filename] = chunks

        all_chunks = [d for chunks in per_file_chunks.values() for d in chunks]
        if not all_chunks:
            log_info("No chunks found to embed.")
            return

        tier_stats = summarize_chunk_stats(all_chunks)
        log_info(
            f"COPE embedding batch: {tier_stats.total} vectors; "
            f"tier_counts={tier_stats.by_tier}"
        )

        client = _get_qdrant_client()
        _maybe_reset_collection(client)
        _ensure_collection(client)

        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=config.QDRANT_COLLECTION_NAME,
            embedding=embeddings,
        )
        ids = _prepare_unique_ids(all_chunks)
        vectorstore.add_documents(documents=all_chunks, ids=ids)
        log_info(
            f"Successfully embedded {len(all_chunks)} chunks into Qdrant "
            f"(unique_ids={len(set(ids))})."
        )

        exp_dir = Path(config.TEXTBOOK_DIR).parent / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = exp_dir / "class8_ingest_manifest.json"
        existing_manifest = _load_manifest(manifest_path)
        by_file = dict(existing_manifest.get("by_file", {}))
        if bool(getattr(config, "RECREATE_COLLECTION_ON_INGEST", False)):
            by_file = {}
        for filename, chunks in per_file_chunks.items():
            stats = summarize_chunk_stats(chunks)
            by_file[filename] = {"total": stats.total, "tier_counts": stats.by_tier}
        _write_manifest(manifest_path, by_file)

    except Exception as e:
        log_error("Vector store initialization failed", e)
        raise

if __name__ == "__main__":
    initialize_vector_store()
