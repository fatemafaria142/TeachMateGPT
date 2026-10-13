"""LLM / embedding providers for the evaluation harness.

Two modes:

* **live**   — real OpenAI calls (requires ``OPENAI_API_KEY``). The only mode whose
  numbers may ever be written into the paper.
* **offline** — deterministic stubs. Lets the whole harness and every baseline be
  exercised without an API key so the plumbing is testable. Results carry
  ``mode="offline"`` and :mod:`eval.report` refuses to publish them.

Both modes count LLM calls, because per-query call count is a reported metric.
"""
from __future__ import annotations

import hashlib
import math
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import config


def has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or getattr(config, "OPENAI_API_KEY", None))


@dataclass
class CallCounter:
    llm_calls: int = 0
    embed_calls: int = 0
    prompt_chars: int = 0

    def reset(self) -> None:
        self.llm_calls = 0
        self.embed_calls = 0
        self.prompt_chars = 0


COUNTER = CallCounter()




class _StubEmbeddings:
    """Deterministic hash-based embeddings.

    Not semantically meaningful — its only job is to make vector plumbing
    (clustering, similarity, store round-trips) exercisable offline. Sharing a
    token between two texts does move them closer together, which is enough for
    RAPTOR's clustering to produce non-degenerate groups in a smoke test.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in (text or "").split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        COUNTER.embed_calls += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        COUNTER.embed_calls += 1
        return self._vec(text)


_hf_embeddings_singleton = None


def get_embeddings(offline: bool):
    """Live embeddings are local BAAI/bge-m3 — the same model COPE indexed with.

    Baselines must embed with the identical model or the comparison measures the
    embedder rather than the retrieval policy. Loaded once per process: bge-m3 is
    ~2 GB and reloading it per config would dominate runtime.
    """
    if offline:
        return _StubEmbeddings()

    global _hf_embeddings_singleton
    if _hf_embeddings_singleton is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        _hf_embeddings_singleton = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": getattr(config, "EMBEDDING_DEVICE", "cpu")},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _hf_embeddings_singleton




class _StubLLM:
    """Returns deterministic, structurally-valid responses.

    Responses are shaped to satisfy the callers' parsers (JSON where JSON is
    expected, a bare label where a label is expected) so that control flow is
    exercised. The *content* is meaningless by construction.
    """

    def __init__(self) -> None:
        self.model = "stub"

    def invoke(self, prompt: str, **_: Any) -> str:
        COUNTER.llm_calls += 1
        COUNTER.prompt_chars += len(str(prompt))
        low = str(prompt).lower()
        digest = hashlib.md5(str(prompt).encode("utf-8")).hexdigest()
        if "json" in low and "score" in low:
            return '{"score": 0.5, "reason": "stub"}'
        if "correct" in low and "incorrect" in low and "ambiguous" in low:
            return ["correct", "ambiguous", "incorrect"][int(digest, 16) % 3]
        if "complexity" in low or "router" in low:
            return ["no_retrieval", "single_step", "multi_step"][int(digest, 16) % 3]
        if "isrel" in low or "issup" in low or "isuse" in low:
            return "relevant supported useful"
        if "summar" in low:
            return f"stub-summary-{digest[:8]}"
        if "entit" in low:
            return '{"entities": [{"name": "stub-entity", "type": "concept"}], "relations": []}'
        return f"stub-response-{digest[:8]}"


class _LiveLLM:
    def __init__(self) -> None:
        from langchain_openai import ChatOpenAI

        self.model = config.OPENAI_MODEL
        self._llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)

    def invoke(self, prompt: str, **kw: Any) -> str:
        COUNTER.llm_calls += 1
        COUNTER.prompt_chars += len(str(prompt))
        resp = self._llm.invoke(prompt, **kw)
        return getattr(resp, "content", str(resp))


def get_llm(offline: bool):
    return _StubLLM() if offline else _LiveLLM()




@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        return str(self.metadata.get("chunk_uid") or "")

    @property
    def tier(self) -> str:
        return str(self.metadata.get("chunk_tier") or "")


_corpus_cache: Optional[list[Chunk]] = None


def load_corpus(force: bool = False) -> list[Chunk]:
    """Load the ingested chunk set from the COPE Qdrant store.

    Payload-only scroll — no vectors, no embedding model — so the corpus can be
    read without loading bge-m3. Every baseline indexes *this exact list*, which
    is what makes the comparison fair.

    Cached per process. Local-mode Qdrant takes an **exclusive** lock on the
    storage directory, and the production retriever keeps its own client open in
    a module singleton for the life of the process. Without this cache, the
    second config in a sweep would try to open the store again and die with
    "already accessed by another instance of Qdrant client". If that singleton is
    already open we borrow it rather than competing for the lock.
    """
    global _corpus_cache
    if _corpus_cache is not None and not force:
        return _corpus_cache

    path = getattr(config, "QDRANT_DB_PATH", "./knowledge_base/qdrant_store")
    collection = getattr(config, "QDRANT_COLLECTION_NAME", "class8_science")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No Qdrant store at {path}. Run ingestion before evaluating."
        )

    from qdrant_client import QdrantClient

    import sys as _sys

    ra = _sys.modules.get("agents.retrieval.agent")
    borrowed = getattr(ra, "_qdrant_client_singleton", None) if ra else None
    client = borrowed or QdrantClient(path=path)
    try:
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                limit=256,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for p in points:
                payload = p.payload or {}
                text = payload.get("page_content", "")
                if not text:
                    continue
                chunks.append(Chunk(text=text, metadata=dict(payload.get("metadata") or {})))
            if offset is None:
                break
    finally:
        if borrowed is None:
            client.close()

    _corpus_cache = chunks
    return chunks


def corpus_fingerprint(chunks: Iterable[Chunk]) -> str:
    """Stable hash of the indexed corpus.

    Recorded in every result file. Two runs are only comparable if their
    fingerprints match; the harness asserts this when aggregating.
    """
    h = hashlib.sha256()
    for c in sorted(chunks, key=lambda c: (c.uid, c.text[:64])):
        h.update(c.uid.encode("utf-8"))
        h.update(c.text.encode("utf-8"))
    return h.hexdigest()[:16]
