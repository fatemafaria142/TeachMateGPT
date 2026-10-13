"""RAPTOR baseline — recursive abstractive tree over the same chunk set.

Sarthi et al., *RAPTOR: Recursive Abstractive Processing for Tree-Organized
Retrieval* (ICLR 2024).

Direct rival to COPE: RAPTOR *learns* a hierarchy by recursive clustering +
summarisation, where COPE *is given* one by curriculum structure. Built over the
identical micro-tier chunks and identical embedding model so only the tree
differs.

Retrieval uses the **collapsed tree** (all levels pooled into one searchable
set), which the RAPTOR paper reports as the stronger variant.
"""
from __future__ import annotations

import random
from typing import Any, Optional, Sequence

from baselines.base import Retriever, RetrievalResult, cosine, to_hits, attribution_ratio
from eval.providers import Chunk


def _kmeans(vectors: list[list[float]], k: int, seed: int = 13, iters: int = 25) -> list[int]:
    """Small deterministic k-means. Avoids a scikit-learn dependency and keeps
    the clustering reproducible across runs (seeded), which matters because the
    tree shape is a reported result."""
    if k <= 1 or len(vectors) <= k:
        return list(range(len(vectors))) if len(vectors) <= k else [0] * len(vectors)
    rng = random.Random(seed)
    centroids = [list(v) for v in rng.sample(vectors, k)]
    assign = [0] * len(vectors)
    for _ in range(iters):
        moved = False
        for i, v in enumerate(vectors):
            best, best_s = 0, -2.0
            for c_i, c in enumerate(centroids):
                s = cosine(v, c)
                if s > best_s:
                    best, best_s = c_i, s
            if assign[i] != best:
                assign[i] = best
                moved = True
        for c_i in range(k):
            members = [vectors[i] for i, a in enumerate(assign) if a == c_i]
            if members:
                dim = len(members[0])
                centroids[c_i] = [sum(m[d] for m in members) / len(members) for d in range(dim)]
        if not moved:
            break
    return assign


class RaptorRetriever(Retriever):
    name = "raptor"

    def __init__(self, corpus: Sequence[Chunk], embeddings, llm, *,
                 top_k: int = 8, branching: int = 5, max_levels: int = 3,
                 summary_max_chars: int = 1200) -> None:
        self.embeddings = embeddings
        self.llm = llm
        self.top_k = top_k
        self.branching = branching
        self.max_levels = max_levels
        self.summary_max_chars = summary_max_chars
        leaves = [c for c in corpus if c.tier == "micro"] or list(corpus)
        self.leaves = leaves
        self.nodes: list[Chunk] = []
        self.node_vectors: list[list[float]] = []
        self.tree_levels: list[int] = []
        self._built = False


    def build(self) -> dict[str, Any]:
        """Build the collapsed tree. One-time cost; report it in the cost table."""
        texts = [c.text for c in self.leaves]
        vectors = self.embeddings.embed_documents(texts)

        all_nodes: list[Chunk] = list(self.leaves)
        all_vectors: list[list[float]] = list(vectors)
        levels: list[int] = [0] * len(self.leaves)

        current, current_vecs = list(self.leaves), list(vectors)
        cluster_trace: list[dict[str, Any]] = []

        for level in range(1, self.max_levels + 1):
            if len(current) <= self.branching:
                break
            k = max(2, len(current) // self.branching)
            assign = _kmeans(current_vecs, k, seed=13 + level)

            parents: list[Chunk] = []
            for c_i in range(max(assign) + 1 if assign else 0):
                members = [current[i] for i, a in enumerate(assign) if a == c_i]
                if not members:
                    continue
                joined = "\n\n".join(m.text for m in members)[: self.summary_max_chars * 4]
                summary = self.llm.invoke(
                    "নিচের পাঠ্যাংশগুলোর একটি সংক্ষিপ্ত সারাংশ লেখো (বাংলায়)।\n\n"
                    f"{joined}\n\nসারাংশ:"
                )
                parent = Chunk(
                    text=str(summary)[: self.summary_max_chars],
                    metadata={
                        "chunk_uid": f"raptor_l{level}_c{c_i}",
                        "chunk_tier": f"raptor_level_{level}",
                        "source": "RAPTOR summary",
                        "derived": True,
                        "member_uids": [m.uid for m in members],
                        "level": level,
                    },
                )
                parents.append(parent)
                cluster_trace.append({
                    "level": level, "cluster": c_i, "size": len(members),
                    "member_uids": [m.uid for m in members],
                })

            if not parents:
                break
            parent_vecs = self.embeddings.embed_documents([p.text for p in parents])
            all_nodes.extend(parents)
            all_vectors.extend(parent_vecs)
            levels.extend([level] * len(parents))
            current, current_vecs = parents, parent_vecs

        self.nodes, self.node_vectors, self.tree_levels = all_nodes, all_vectors, levels
        self._built = True
        return {
            "n_leaves": len(self.leaves),
            "n_nodes": len(all_nodes),
            "max_level": max(levels) if levels else 0,
            "clusters": cluster_trace,
        }


    def retrieve(self, query: str) -> RetrievalResult:
        if not self._built:
            self.build()
        return self._timed(self._retrieve, query)

    def _retrieve(self, query: str) -> RetrievalResult:
        qv = self.embeddings.embed_query(query)
        scored = sorted(
            ((cosine(qv, v), i) for i, v in enumerate(self.node_vectors)),
            key=lambda t: t[0], reverse=True,
        )[: self.top_k]
        picked = [self.nodes[i] for _, i in scored]
        scores = [s for s, _ in scored]
        hits = to_hits(picked, scores)
        return RetrievalResult(
            contexts=[c.text for c in picked],
            sources=sorted({str(c.metadata.get("source", "Unknown")) for c in picked}),
            hits=hits,
            failed_closed=not picked,
            source_attribution=attribution_ratio(hits),
            extra={"levels_hit": sorted({self.tree_levels[i] for _, i in scored})},
        )


def cluster_chapter_agreement(trace: list[dict[str, Any]], uid_to_chapter: dict[str, str]) -> Optional[float]:
    """Adjusted Rand Index between level-1 RAPTOR clusters and chapter labels.

    Tests the plan's expected finding: on a single textbook, RAPTOR's learned
    clusters should converge toward chapter boundaries — i.e. toward what COPE's
    macro tier already encodes for free.
    """
    lvl1 = [t for t in trace if t["level"] == 1]
    if not lvl1 or not uid_to_chapter:
        return None
    pairs: list[tuple[int, str]] = []
    for c_i, t in enumerate(lvl1):
        for uid in t["member_uids"]:
            ch = uid_to_chapter.get(uid)
            if ch is not None:
                pairs.append((c_i, ch))
    if len(pairs) < 2:
        return None

    from collections import Counter
    def comb2(n: int) -> float:
        return n * (n - 1) / 2

    joint = Counter(pairs)
    a = Counter(c for c, _ in pairs)
    b = Counter(ch for _, ch in pairs)
    n = len(pairs)
    sum_ij = sum(comb2(v) for v in joint.values())
    sum_a = sum(comb2(v) for v in a.values())
    sum_b = sum(comb2(v) for v in b.values())
    expected = sum_a * sum_b / comb2(n) if comb2(n) else 0.0
    max_idx = 0.5 * (sum_a + sum_b)
    denom = max_idx - expected
    return (sum_ij - expected) / denom if denom else None
