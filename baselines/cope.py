"""Adapter exposing the shipped TeachMateGPT retriever through the baseline interface.

Delegates to :func:`agents.retrieval.retrieve_context` so the evaluated
system is the real pipeline, not a reimplementation. Component ablations are
applied by the harness as `config` overrides before this runs.

Requires a live API key: the production path builds a Chroma store with OpenAI
embeddings and runs LLM-backed multi-query expansion. There is no offline stub
for it, by design — a stubbed COPE would not be the system under test.
"""
from __future__ import annotations

from typing import Any

from baselines.base import Retriever, RetrievalResult, attribution_ratio


class CopeRetriever(Retriever):
    name = "cope"

    def __init__(self, *, offline: bool = False) -> None:
        if offline:
            raise RuntimeError(
                "The COPE retriever is the shipped pipeline and has no offline stub "
                "by design — a stubbed COPE would not be the system under test. "
                "Set OPENAI_API_KEY and use --live, or restrict the sweep to "
                "baseline configs (--baselines)."
            )

    def retrieve(self, query: str) -> RetrievalResult:
        return self._timed(self._retrieve, query)

    def _retrieve(self, query: str) -> RetrievalResult:
        from agents import retrieval

        contexts, sources, hits = retrieval.retrieve_context(query)
        cov = dict(retrieval.LAST_COVERAGE)
        for h in hits:
            h.setdefault("derived", False)
        return RetrievalResult(
            contexts=list(contexts),
            sources=list(sources),
            hits=list(hits),
            failed_closed=not contexts,
            coverage=cov.get("value"),
            source_attribution=attribution_ratio(hits) if hits else 0.0,
            extra={"coverage_gate": cov},
        )
