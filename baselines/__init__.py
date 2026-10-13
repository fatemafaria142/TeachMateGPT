"""Baseline retrieval / verification systems for TeachMateGPT comparison."""
from __future__ import annotations

from typing import Sequence

from baselines.base import Retriever, RetrievalResult
from eval.providers import Chunk


def build_retriever(kind: str, corpus: Sequence[Chunk], embeddings, llm, *,
                    offline: bool = False, **kw) -> Retriever:
    """Factory. `kind` matches RunConfig.retriever."""
    if kind == "cope":
        from baselines.cope import CopeRetriever
        return CopeRetriever(offline=offline, **kw)
    if kind == "vanilla":
        from baselines.vanilla import VanillaRetriever
        return VanillaRetriever(corpus, embeddings, llm, **kw)
    if kind == "raptor":
        from baselines.raptor import RaptorRetriever
        return RaptorRetriever(corpus, embeddings, llm, **kw)
    if kind == "graphrag":
        from baselines.graphrag import GraphRagRetriever
        return GraphRagRetriever(corpus, embeddings, llm, **kw)
    if kind == "crag":
        from baselines.crag import CragRetriever
        return CragRetriever(corpus, embeddings, llm, **kw)
    if kind == "adaptive":
        from baselines.adaptive import AdaptiveRetriever
        return AdaptiveRetriever(corpus, embeddings, llm, **kw)
    raise ValueError(f"Unknown retriever kind: {kind}")


__all__ = ["Retriever", "RetrievalResult", "build_retriever"]
