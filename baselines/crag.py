"""CRAG baseline — Corrective Retrieval-Augmented Generation.

Yan et al., *Corrective Retrieval Augmented Generation* (2024).

A lightweight retrieval evaluator labels the retrieved set
``correct`` / ``ambiguous`` / ``incorrect`` and applies a corrective action:

* **correct**   → knowledge refinement (decompose into strips, drop weak strips)
* **incorrect** → discard and re-retrieve with a rewritten query
* **ambiguous** → combine both

**Deviation from the paper (report this in the caption):** the web-search
fallback is removed. The corpus is closed and curriculum-authorised; permitting
web retrieval would let the baseline ground items in unauthorised sources and
make the comparison meaningless. On ``incorrect`` CRAG therefore rewrites and
re-retrieves internally.

This is the direct rival to the fail-closed coverage gate: CRAG *repairs* weak
retrieval where TeachMateGPT *refuses* it. The contrast is itself a result — CRAG
should show higher coverage and lower faithfulness.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from baselines.base import Retriever, RetrievalResult, bm25_scores, tokenize, to_hits, attribution_ratio
from eval.providers import Chunk

_EVAL_PROMPT = (
    "নিচের প্রশ্ন এবং উদ্ধৃত পাঠ্যাংশ দেখো। পাঠ্যাংশটি প্রশ্নের উত্তর দিতে কতটা প্রাসঙ্গিক?\n"
    "শুধু একটি শব্দে উত্তর দাও: correct, ambiguous, অথবা incorrect।\n\n"
    "প্রশ্ন: {query}\n\nপাঠ্যাংশ:\n{doc}\n\nউত্তর:"
)

_REWRITE_PROMPT = (
    "নিচের প্রশ্নটিকে পাঠ্যবই অনুসন্ধানের জন্য আরও নির্দিষ্ট কীওয়ার্ড-ভিত্তিক আকারে "
    "পুনর্লিখন করো। শুধু পুনর্লিখিত প্রশ্নটি দাও।\n\nপ্রশ্ন: {query}\n\nপুনর্লিখিত:"
)


class CragRetriever(Retriever):
    name = "crag"

    def __init__(self, corpus: Sequence[Chunk], embeddings, llm, *,
                 top_k: int = 8, eval_top_n: int = 5,
                 strip_min_score: float = 0.0) -> None:
        self.corpus = [c for c in corpus if c.tier == "micro"] or list(corpus)
        self.embeddings = embeddings
        self.llm = llm
        self.top_k = top_k
        self.eval_top_n = eval_top_n
        self.strip_min_score = strip_min_score
        self._tokens = [tokenize(c.text) for c in self.corpus]

    def retrieve(self, query: str) -> RetrievalResult:
        return self._timed(self._retrieve, query)

    def _search(self, query: str, k: int) -> list[Chunk]:
        scores = bm25_scores(query, self._tokens)
        ranked = sorted(zip(scores, range(len(self.corpus))), key=lambda t: t[0], reverse=True)
        return [self.corpus[i] for _, i in ranked[:k]]

    def _evaluate(self, query: str, docs: Sequence[Chunk]) -> str:
        """Aggregate per-document evaluator labels into one action."""
        labels = []
        for d in docs[: self.eval_top_n]:
            raw = str(self.llm.invoke(_EVAL_PROMPT.format(query=query, doc=d.text[:1500]))).lower()
            if "incorrect" in raw:
                labels.append("incorrect")
            elif "ambiguous" in raw:
                labels.append("ambiguous")
            elif "correct" in raw:
                labels.append("correct")
            else:
                labels.append("ambiguous")
        if not labels:
            return "incorrect"
        if any(l == "correct" for l in labels):
            return "correct" if labels.count("correct") >= len(labels) / 2 else "ambiguous"
        if all(l == "incorrect" for l in labels):
            return "incorrect"
        return "ambiguous"

    def _refine(self, query: str, docs: Sequence[Chunk]) -> list[Chunk]:
        """Decompose-then-recompose: split into strips, keep query-relevant ones."""
        q_toks = set(tokenize(query))
        refined: list[Chunk] = []
        for d in docs:
            strips = [s.strip() for s in re.split(r"(?<=[।\.\!\?])\s+", d.text) if s.strip()]
            if not strips:
                continue
            kept = [s for s in strips if set(tokenize(s)) & q_toks]
            text = " ".join(kept) if kept else d.text
            meta = dict(d.metadata)
            meta["crag_refined"] = bool(kept)
            refined.append(Chunk(text=text, metadata=meta))
        return refined

    def _retrieve(self, query: str) -> RetrievalResult:
        docs = self._search(query, self.top_k)
        action = self._evaluate(query, docs)

        if action == "correct":
            final = self._refine(query, docs)
        elif action == "incorrect":
            rewritten = str(self.llm.invoke(_REWRITE_PROMPT.format(query=query))).strip()
            final = self._refine(rewritten, self._search(rewritten or query, self.top_k))
        else:
            rewritten = str(self.llm.invoke(_REWRITE_PROMPT.format(query=query))).strip()
            extra = self._search(rewritten or query, max(1, self.top_k // 2))
            seen: set[str] = set()
            merged: list[Chunk] = []
            for c in list(docs) + list(extra):
                if c.uid in seen:
                    continue
                seen.add(c.uid)
                merged.append(c)
            final = self._refine(query, merged[: self.top_k])

        hits = to_hits(final)
        return RetrievalResult(
            contexts=[c.text for c in final],
            sources=sorted({str(c.metadata.get("source", "Unknown")) for c in final}),
            hits=hits,
            failed_closed=not final,
            source_attribution=attribution_ratio(hits),
            extra={"crag_action": action},
        )
