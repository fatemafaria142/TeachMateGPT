"""GraphRAG baseline — entity graph + communities + community summaries.

Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused
Summarization* (2024).

Pipeline: per-chunk entity/relation extraction → co-occurrence graph → community
detection → hierarchical community summarisation → local + global search.

Two things this implementation deliberately measures rather than assumes:

1. **Source attribution.** Global search retrieves *community summaries* — LLM-
   derived text, not textbook text. ``source_attribution`` reports the fraction
   of returned context that is traceable to a real chunk. This is the concrete
   form of the SAVER-compatibility concern in IMPLEMENTATION_PLAN.md 5.2.
2. **Community/chapter agreement.** On a single Class-8 textbook the detected
   communities are expected to approximate chapters. Reported, not asserted.

Community detection uses greedy modularity rather than Leiden to avoid a
``igraph``/``leidenalg`` dependency. Both optimise modularity; the plan's claim
concerns what the communities *are*, not the optimiser. Swap in Leiden if the
reviewer asks — the interface is unchanged.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Optional, Sequence

from baselines.base import Retriever, RetrievalResult, bm25_scores, tokenize, to_hits, attribution_ratio
from eval.providers import Chunk

_ENTITY_PROMPT = (
    "নিচের পাঠ্যাংশ থেকে গুরুত্বপূর্ণ বৈজ্ঞানিক entity ও তাদের সম্পর্ক বের করো।\n"
    'JSON ফরম্যাটে উত্তর দাও: {{"entities": [{{"name": "...", "type": "..."}}], '
    '"relations": [{{"source": "...", "target": "...", "description": "..."}}]}}\n\n'
    "পাঠ্যাংশ:\n{text}\n\nJSON:"
)


def _parse_json(raw: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", str(raw), re.DOTALL)
    if not m:
        return {"entities": [], "relations": []}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}
    return {
        "entities": data.get("entities") or [],
        "relations": data.get("relations") or [],
    }


def _greedy_modularity(nodes: list[str], edges: dict[tuple[str, str], float]) -> dict[str, int]:
    """Greedy modularity communities (Clauset-Newman-Moore style, simplified)."""
    if not nodes:
        return {}
    comm = {n: i for i, n in enumerate(nodes)}
    degree: dict[str, float] = defaultdict(float)
    for (u, v), w in edges.items():
        degree[u] += w
        degree[v] += w
    total_w = sum(edges.values()) or 1.0

    improved = True
    while improved:
        improved = False
        for (u, v), w in sorted(edges.items(), key=lambda kv: kv[1], reverse=True):
            cu, cv = comm[u], comm[v]
            if cu == cv:
                continue
            gain = w / total_w - (degree[u] * degree[v]) / (2 * total_w ** 2)
            if gain > 0:
                for n in nodes:
                    if comm[n] == cu:
                        comm[n] = cv
                improved = True
    remap: dict[int, int] = {}
    for n in nodes:
        remap.setdefault(comm[n], len(remap))
        comm[n] = remap[comm[n]]
    return comm


class GraphRagRetriever(Retriever):
    name = "graphrag"

    def __init__(self, corpus: Sequence[Chunk], embeddings, llm, *,
                 top_k: int = 8, mode: str = "hybrid",
                 summary_max_chars: int = 1200) -> None:
        self.corpus = [c for c in corpus if c.tier == "micro"] or list(corpus)
        self.embeddings = embeddings
        self.llm = llm
        self.top_k = top_k
        self.mode = mode
        self.summary_max_chars = summary_max_chars
        self.entity_to_chunks: dict[str, set[str]] = defaultdict(set)
        self.communities: dict[str, int] = {}
        self.community_summaries: list[Chunk] = []
        self.uid_to_chunk: dict[str, Chunk] = {c.uid: c for c in self.corpus}
        self._built = False
        self.index_stats: dict[str, Any] = {}


    def build(self) -> dict[str, Any]:
        edges: dict[tuple[str, str], float] = defaultdict(float)
        n_entities = 0
        for chunk in self.corpus:
            parsed = _parse_json(self.llm.invoke(_ENTITY_PROMPT.format(text=chunk.text[:3000])))
            names = [str(e.get("name", "")).strip() for e in parsed["entities"] if e.get("name")]
            names = [n for n in names if n]
            n_entities += len(names)
            for n in names:
                self.entity_to_chunks[n].add(chunk.uid)
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    edges[tuple(sorted((a, b)))] += 1.0
            for rel in parsed["relations"]:
                s, t = str(rel.get("source", "")).strip(), str(rel.get("target", "")).strip()
                if s and t and s != t:
                    edges[tuple(sorted((s, t)))] += 2.0

        nodes = sorted(self.entity_to_chunks)
        self.communities = _greedy_modularity(nodes, edges)

        by_comm: dict[int, set[str]] = defaultdict(set)
        for ent, c_id in self.communities.items():
            by_comm[c_id] |= self.entity_to_chunks[ent]

        self.community_summaries = []
        for c_id, uids in sorted(by_comm.items()):
            member_texts = [self.uid_to_chunk[u].text for u in sorted(uids) if u in self.uid_to_chunk]
            if not member_texts:
                continue
            joined = "\n\n".join(member_texts)[:6000]
            summary = self.llm.invoke(
                "নিচের সম্পর্কিত পাঠ্যাংশগুলোর একটি সমন্বিত সারাংশ লেখো (বাংলায়)।\n\n"
                f"{joined}\n\nসারাংশ:"
            )
            self.community_summaries.append(Chunk(
                text=str(summary)[: self.summary_max_chars],
                metadata={
                    "chunk_uid": f"graphrag_comm_{c_id}",
                    "chunk_tier": "graphrag_community",
                    "source": "GraphRAG community summary",
                    "derived": True,
                    "member_uids": sorted(uids),
                    "community": c_id,
                },
            ))

        self._built = True
        self.index_stats = {
            "n_chunks_extracted": len(self.corpus),
            "n_entity_mentions": n_entities,
            "n_unique_entities": len(nodes),
            "n_edges": len(edges),
            "n_communities": len(self.community_summaries),
        }
        return self.index_stats


    def retrieve(self, query: str) -> RetrievalResult:
        if not self._built:
            self.build()
        return self._timed(self._retrieve, query)

    def _retrieve(self, query: str) -> RetrievalResult:
        picked: list[Chunk] = []

        if self.mode in ("global", "hybrid") and self.community_summaries:
            toks = [tokenize(c.text) for c in self.community_summaries]
            scores = bm25_scores(query, toks)
            ranked = sorted(zip(scores, self.community_summaries), key=lambda t: t[0], reverse=True)
            n_global = self.top_k if self.mode == "global" else max(1, self.top_k // 2)
            picked.extend(c for _, c in ranked[:n_global])

        if self.mode in ("local", "hybrid"):
            q_toks = set(tokenize(query))
            cand_uids: set[str] = set()
            for ent, uids in self.entity_to_chunks.items():
                if set(tokenize(ent)) & q_toks:
                    cand_uids |= uids
            cands = [self.uid_to_chunk[u] for u in sorted(cand_uids) if u in self.uid_to_chunk]
            if not cands:
                cands = self.corpus
            toks = [tokenize(c.text) for c in cands]
            scores = bm25_scores(query, toks)
            ranked = sorted(zip(scores, cands), key=lambda t: t[0], reverse=True)
            n_local = self.top_k - len(picked)
            picked.extend(c for _, c in ranked[:max(0, n_local)])

        hits = to_hits(picked)
        return RetrievalResult(
            contexts=[c.text for c in picked],
            sources=sorted({str(c.metadata.get("source", "Unknown")) for c in picked}),
            hits=hits,
            failed_closed=not picked,
            source_attribution=attribution_ratio(hits),
            extra={"mode": self.mode, **self.index_stats},
        )


def community_chapter_agreement(retriever: GraphRagRetriever,
                                uid_to_chapter: dict[str, str]) -> Optional[float]:
    """Purity of detected communities w.r.t. chapter labels, in [0, 1].

    High purity supports the plan's argument that on a single-textbook corpus the
    learned graph structure largely recovers the chapter structure COPE is given.
    """
    if not retriever.community_summaries or not uid_to_chapter:
        return None
    total = 0
    correct = 0
    for comm in retriever.community_summaries:
        chapters = [uid_to_chapter.get(u) for u in comm.metadata.get("member_uids", [])]
        chapters = [c for c in chapters if c]
        if not chapters:
            continue
        total += len(chapters)
        correct += max(chapters.count(c) for c in set(chapters))
    return correct / total if total else None
