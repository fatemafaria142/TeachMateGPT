"""Vanilla RAG control — dense top-k retrieval, no gates.

Serves two configs:

* ``vanilla``  — the baseline row: flat retrieval, no routing, no CCI/CCR, no
  coverage gate, no verification.
* ``no_cope``  — the COPE ablation: identical retrieval, but drawing on *all*
  tiers flattened, so hierarchy is removed while everything else is held fixed.
"""
from __future__ import annotations

from typing import Sequence

from baselines.base import Retriever, RetrievalResult, cosine, to_hits, attribution_ratio
from eval.providers import Chunk


class VanillaRetriever(Retriever):
    name = "vanilla"

    def __init__(self, corpus: Sequence[Chunk], embeddings, llm, *,
                 top_k: int = 8, flatten_tiers: bool = True) -> None:
        self.corpus = list(corpus) if flatten_tiers else [c for c in corpus if c.tier == "micro"]
        self.embeddings = embeddings
        self.top_k = top_k
        self._vectors: list[list[float]] | None = None

    def _ensure_vectors(self) -> None:
        if self._vectors is None:
            self._vectors = self.embeddings.embed_documents([c.text for c in self.corpus])

    def retrieve(self, query: str) -> RetrievalResult:
        return self._timed(self._retrieve, query)

    def _retrieve(self, query: str) -> RetrievalResult:
        self._ensure_vectors()
        qv = self.embeddings.embed_query(query)
        ranked = sorted(
            ((cosine(qv, v), i) for i, v in enumerate(self._vectors or [])),
            key=lambda t: t[0], reverse=True,
        )[: self.top_k]
        picked = [self.corpus[i] for _, i in ranked]
        hits = to_hits(picked, [s for s, _ in ranked])
        return RetrievalResult(
            contexts=[c.text for c in picked],
            sources=sorted({str(c.metadata.get("source", "Unknown")) for c in picked}),
            hits=hits,
            failed_closed=False,
            source_attribution=attribution_ratio(hits),
        )
