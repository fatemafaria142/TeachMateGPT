"""Adaptive RAG baseline — query-complexity routing.

Jeong et al., *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
Models through Question Complexity* (NAACL 2024).

A classifier routes each query to one of three strategies:

* ``no_retrieval`` — answer from parametric knowledge alone
* ``single_step``  — one retrieval pass
* ``multi_step``   — iterative retrieve → reason → retrieve

Rival to TeachMateGPT's intent/ambiguity routing. The key difference the harness
measures: Adaptive-RAG routes on *complexity* (how much retrieval is needed),
while TeachMateGPT routes on *admissibility* (whether the request should be
served at all). Adaptive-RAG has no ``harmful``/``off-topic`` terminal state, so
it should show unsafe-content leakage comparable to the ``w/o routing`` ablation.

The paper trains a T5 complexity classifier on silver labels. Here the classifier
is prompted; label it as such in the caption.
"""
from __future__ import annotations

from typing import Any, Sequence

from baselines.base import Retriever, RetrievalResult, bm25_scores, tokenize, to_hits, attribution_ratio
from eval.providers import Chunk

_ROUTER_PROMPT = (
    "নিচের প্রশ্নটির complexity নির্ধারণ করো। শুধু একটি লেবেল দাও:\n"
    "no_retrieval — সাধারণ জ্ঞান, পাঠ্যবই লাগবে না\n"
    "single_step — একবার পাঠ্যবই খুঁজলেই যথেষ্ট\n"
    "multi_step — একাধিক ধাপে তথ্য জোড়া লাগাতে হবে\n\n"
    "প্রশ্ন: {query}\n\nলেবেল:"
)

_FOLLOWUP_PROMPT = (
    "প্রশ্ন: {query}\n\nএ পর্যন্ত পাওয়া তথ্য:\n{context}\n\n"
    "উত্তর সম্পূর্ণ করতে আর কোন তথ্যটি দরকার? শুধু একটি অনুসন্ধান-প্রশ্ন লেখো।\n\nঅনুসন্ধান:"
)

_LABELS = ("no_retrieval", "single_step", "multi_step")


class AdaptiveRetriever(Retriever):
    name = "adaptive"

    def __init__(self, corpus: Sequence[Chunk], embeddings, llm, *,
                 top_k: int = 8, max_steps: int = 3) -> None:
        self.corpus = [c for c in corpus if c.tier == "micro"] or list(corpus)
        self.embeddings = embeddings
        self.llm = llm
        self.top_k = top_k
        self.max_steps = max_steps
        self._tokens = [tokenize(c.text) for c in self.corpus]

    def retrieve(self, query: str) -> RetrievalResult:
        return self._timed(self._retrieve, query)

    def _classify(self, query: str) -> str:
        raw = str(self.llm.invoke(_ROUTER_PROMPT.format(query=query))).strip().lower()
        for label in _LABELS:
            if label in raw:
                return label
        return "single_step"

    def _search(self, query: str, k: int) -> list[Chunk]:
        scores = bm25_scores(query, self._tokens)
        ranked = sorted(zip(scores, range(len(self.corpus))), key=lambda t: t[0], reverse=True)
        return [self.corpus[i] for _, i in ranked[:k]]

    def _retrieve(self, query: str) -> RetrievalResult:
        route = self._classify(query)

        if route == "no_retrieval":
            return RetrievalResult(
                contexts=[], sources=[], hits=[],
                failed_closed=False,
                source_attribution=0.0,
                extra={"adaptive_route": route, "steps": 0},
            )

        if route == "single_step":
            picked = self._search(query, self.top_k)
            hits = to_hits(picked)
            return RetrievalResult(
                contexts=[c.text for c in picked],
                sources=sorted({str(c.metadata.get("source", "Unknown")) for c in picked}),
                hits=hits, failed_closed=not picked,
                source_attribution=attribution_ratio(hits),
                extra={"adaptive_route": route, "steps": 1},
            )

        seen: set[str] = set()
        picked: list[Chunk] = []
        current = query
        steps = 0
        per_step = max(2, self.top_k // self.max_steps)
        for _ in range(self.max_steps):
            steps += 1
            for c in self._search(current, per_step):
                if c.uid not in seen:
                    seen.add(c.uid)
                    picked.append(c)
            if len(picked) >= self.top_k:
                break
            ctx = "\n".join(c.text[:400] for c in picked[-per_step:])
            current = str(self.llm.invoke(_FOLLOWUP_PROMPT.format(query=query, context=ctx))).strip() or query

        picked = picked[: self.top_k]
        hits = to_hits(picked)
        return RetrievalResult(
            contexts=[c.text for c in picked],
            sources=sorted({str(c.metadata.get("source", "Unknown")) for c in picked}),
            hits=hits, failed_closed=not picked,
            source_attribution=attribution_ratio(hits),
            extra={"adaptive_route": route, "steps": steps},
        )
