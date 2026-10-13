"""Common retriever interface shared by TeachMateGPT and every baseline.

Fairness contract: each baseline receives the *same* chunk list from
``eval.providers.load_corpus`` and returns contexts consumed by the *same*
generator. Only the retrieval policy differs. Anything that breaks that contract
turns the comparison into a generator benchmark and invalidates it.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from eval.providers import Chunk, COUNTER


@dataclass
class RetrievalResult:
    contexts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    hits: list[dict[str, Any]] = field(default_factory=list)
    failed_closed: bool = False
    coverage: Optional[float] = None
    llm_calls: int = 0
    latency_s: float = 0.0
    source_attribution: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)


class Retriever:
    name: str = "base"

    def retrieve(self, query: str) -> RetrievalResult:
        raise NotImplementedError

    def _timed(self, fn, *a, **kw) -> RetrievalResult:
        before = COUNTER.llm_calls
        t0 = time.perf_counter()
        result = fn(*a, **kw)
        result.latency_s = time.perf_counter() - t0
        result.llm_calls = COUNTER.llm_calls - before
        return result



_WORD = re.compile(r"[\wঀ-৿]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def bm25_scores(query: str, corpus_tokens: Sequence[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Plain BM25. Used by baselines that need a lexical leg without pulling in
    the production retriever's topic/anchor machinery."""
    q_terms = tokenize(query)
    n_docs = len(corpus_tokens) or 1
    avgdl = sum(len(d) for d in corpus_tokens) / n_docs or 1.0
    df: dict[str, int] = {}
    for doc in corpus_tokens:
        for t in set(doc):
            df[t] = df.get(t, 0) + 1
    scores = [0.0] * n_docs
    for i, doc in enumerate(corpus_tokens):
        if not doc:
            continue
        dl = len(doc)
        tf: dict[str, int] = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in q_terms:
            if t not in tf:
                continue
            idf = math.log(1 + (n_docs - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            s += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * dl / avgdl))
        scores[i] = s
    return scores


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


def to_hits(chunks: Sequence[Chunk], scores: Optional[Sequence[float]] = None) -> list[dict[str, Any]]:
    hits = []
    for i, c in enumerate(chunks):
        hits.append({
            "rank": i + 1,
            "chunk_text": c.text,
            "chunk_uid": c.uid,
            "chunk_tier": c.tier,
            "source": c.metadata.get("source", "Unknown"),
            "score": float(scores[i]) if scores is not None and i < len(scores) else None,
            "derived": bool(c.metadata.get("derived")),
        })
    return hits


def attribution_ratio(hits: Sequence[dict[str, Any]]) -> float:
    """Share of hits that are verbatim textbook text rather than LLM-derived."""
    if not hits:
        return 0.0
    grounded = sum(0 if h.get("derived") else 1 for h in hits)
    return grounded / len(hits)
