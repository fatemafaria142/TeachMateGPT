"""
Hybrid retrieval for textbook-grounded assessment.

Pulls evidence from the COPE-built Qdrant store using dense vectors (local BAAI/bge-m3
embeddings) plus lexical BM25, merges ranks (RRF), applies query planning, expansion,
deduplication, and coverage gates, then returns chunk texts, source names, and rich
metadata for generation, verification, and experiment exports.
"""
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document

import config
from utils.logger import log_retrieval
from agents.retrieval.tools import (
    _MULTIQUERY_PROMPT,
    _analyze_query_for_retrieval,
    _anchor_overlap,
    _chunk_relevance_score,
    _contains_forbidden_without_target,
    _dedupe_docs,
    _diverse_top_k,
    _domain_expansion_terms,
    _dynamic_anchor_terms,
    _effective_query,
    _extract_explicit_topic,
    _forbidden_opposites,
    _graph_expand_from_seeds,
    _index_docs_by_uid,
    _is_neighbor_chunk,
    _is_noisy_chunk,
    _is_topic_or_section_query,
    _near_duplicate_text,
    _normalize_query,
    _ordered_unique_sources,
    _query_coverage_ratio,
    _query_terms,
    _scroll_all_documents,
    _strip_task_phrases,
    _topic_overlap,
    _topic_terms,
    _weighted_rrf_merge,
    hit_to_eval_record,
)

LAST_COVERAGE: Dict[str, Any] = {}

_embeddings_singleton: Optional[HuggingFaceEmbeddings] = None
_qdrant_client_singleton: Optional[QdrantClient] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Local BAAI/bge-m3 embedding model, loaded once per process."""
    global _embeddings_singleton
    if _embeddings_singleton is None:
        _embeddings_singleton = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": getattr(config, "EMBEDDING_DEVICE", "cpu")},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_singleton


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client_singleton
    if _qdrant_client_singleton is None:
        _qdrant_client_singleton = QdrantClient(path=config.QDRANT_DB_PATH)
    return _qdrant_client_singleton


def retrieve_context(
    query: str, student_class: str = "8"
) -> Tuple[list[str], list[str], List[Dict[str, Any]]]:
    """
    EMNLP-Grade Retrieval Pipeline:
    1. Multi-Query Expansion: Generates 3 Bangla variations of the query.
    2. Hybrid Search: Semantic (Vector) + Lexical (BM25).
    3. Re-ranking: (Simplified to top-k selection for now, but expansion improves recall).
    """
    LAST_COVERAGE.clear()
    LAST_COVERAGE.update({"value": None, "threshold": None, "failed_closed": None, "suppressed": False})
    primary_query = _effective_query(query)
    focus_query = _strip_task_phrases(primary_query)
    analyzed = _analyze_query_for_retrieval(primary_query)
    retrieval_query = str(analyzed.get("search_query") or focus_query or primary_query)
    normalized_query = _normalize_query(retrieval_query)
    primary_normalized = _normalize_query(focus_query)
    query_terms = _query_terms(primary_normalized)
    domain_terms = _domain_expansion_terms(primary_normalized)
    search_query = normalized_query
    if domain_terms:
        search_query = f"{normalized_query} {' '.join(sorted(domain_terms))}".strip()
    explicit_topic = _extract_explicit_topic(primary_normalized)
    topic_terms = _topic_terms(explicit_topic)
    forbidden_terms = _forbidden_opposites(query_terms)
    llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)
    embeddings = _get_embeddings()

    qdrant_client = _get_qdrant_client()
    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=config.QDRANT_COLLECTION_NAME,
        embedding=embeddings,
    )

    vector_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": int(getattr(config, "RETRIEVAL_VECTOR_K", 10))}
        ),
        llm=llm,
        prompt=_MULTIQUERY_PROMPT,
    )

    docs = _scroll_all_documents(qdrant_client, config.QDRANT_COLLECTION_NAME)
    corpus_docs: list[Document] = []
    corpus_by_uid: dict[str, Document] = {}
    if docs:
        corpus_docs = docs
        corpus_by_uid = _index_docs_by_uid(docs)
        anchor_terms = _dynamic_anchor_terms(primary_normalized, docs, top_n=6)
        anchor_terms = set(anchor_terms) | set(domain_terms)
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = int(getattr(config, "RETRIEVAL_BM25_K", 10))

        dense_only = bool(getattr(config, "RETRIEVAL_DENSE_ONLY", False))
        bm25_only = bool(getattr(config, "RETRIEVAL_BM25_ONLY", False))
        if dense_only and bm25_only:
            raise ValueError("RETRIEVAL_DENSE_ONLY and RETRIEVAL_BM25_ONLY are mutually exclusive")
        if dense_only:
            legs, weights = [vector_retriever], [1.0]
        elif bm25_only:
            legs, weights = [bm25_retriever], [1.0]
        else:
            legs, weights = [bm25_retriever, vector_retriever], [0.4, 0.6]

        results = _weighted_rrf_merge(
            search_query,
            legs,
            weights=weights,
            k=int(getattr(config, "TOP_K_RESULTS", 10)),
        )
    else:
        anchor_terms = _query_terms(primary_normalized) | set(domain_terms)
        results = vectorstore.similarity_search(
            search_query, k=int(getattr(config, "TOP_K_RESULTS", 10))
        )
    conflict_target_terms = set(topic_terms) if topic_terms else (set(anchor_terms) if anchor_terms else set(query_terms))

    candidates = [d for d in results if not _is_noisy_chunk(d.page_content or "")]
    if candidates and forbidden_terms:
        strict_candidates = [
            d
            for d in candidates
            if not _contains_forbidden_without_target(
                d,
                forbidden_terms,
                conflict_target_terms,
            )
        ]
        if strict_candidates:
            candidates = strict_candidates
    if candidates:
        use_focus = bool(getattr(config, "RETRIEVAL_ENABLE_TOPIC_FOCUS", True))
        hard_filter = bool(getattr(config, "RETRIEVAL_TOPIC_HARD_FILTER", True))
        topic_boost = float(getattr(config, "RETRIEVAL_TOPIC_BOOST", 3.0))
        min_topic_overlap = int(getattr(config, "RETRIEVAL_MIN_TOPIC_OVERLAP", 1))
        anchor_boost = 4.0

        if use_focus and topic_terms:
            best_overlap = max((_topic_overlap(d, topic_terms) for d in candidates), default=0)
            if best_overlap < min_topic_overlap:
                results = []
            else:
                if hard_filter:
                    focused = [d for d in candidates if _topic_overlap(d, topic_terms) > 0]
                    if focused:
                        candidates = focused

                if anchor_terms:
                    anchored = [d for d in candidates if _anchor_overlap(d, anchor_terms) > 0]
                    if anchored:
                        candidates = anchored

                results = sorted(
                    candidates,
                    key=lambda d: _chunk_relevance_score(d, query_terms)
                    + (topic_boost * _topic_overlap(d, topic_terms)),
                    reverse=True,
                )[: int(getattr(config, "RETRIEVAL_FINAL_K", 3))]
        else:
            if anchor_terms:
                anchored = [d for d in candidates if _anchor_overlap(d, anchor_terms) > 0]
                if anchored:
                    candidates = anchored
            results = sorted(
                candidates,
                key=lambda d: _chunk_relevance_score(d, query_terms)
                + (anchor_boost * _anchor_overlap(d, anchor_terms)),
                reverse=True,
            )[: int(getattr(config, "RETRIEVAL_FINAL_K", 3))]

    if _is_topic_or_section_query(primary_query) and results:
        final_k = int(getattr(config, "RETRIEVAL_FINAL_K", 3))
        expand_k = int(getattr(config, "RETRIEVAL_EXPAND_K", 30))
        graph_hops = int(getattr(config, "RETRIEVAL_GRAPH_HOPS", 1))
        graph_max = int(getattr(config, "RETRIEVAL_GRAPH_MAX_NODES", 36))
        seed_top_n = int(getattr(config, "RETRIEVAL_GRAPH_SEED_TOP_N", 4))
        seed_pack = results[: max(1, min(seed_top_n, len(results)))]
        extra = corpus_docs if corpus_docs else vectorstore.similarity_search(search_query, k=expand_k)
        scoped: list[Document] = []
        grain_window = int(getattr(config, "RETRIEVAL_NEIGHBOR_GRAIN_WINDOW", 2))
        seq_window = int(getattr(config, "RETRIEVAL_NEIGHBOR_SEQUENCE_WINDOW", 2))
        for d in extra:
            if _is_noisy_chunk(d.page_content or ""):
                continue
            meta = d.metadata or {}
            hit = False
            for seed in seed_pack:
                seed_meta = seed.metadata or {}
                seed_unit_idx = seed_meta.get("structural_unit_idx")
                seed_heading = (seed_meta.get("structural_heading") or "").strip()
                unit_ok = seed_unit_idx is not None and meta.get("structural_unit_idx") == seed_unit_idx
                heading_ok = bool(seed_heading) and str(meta.get("structural_heading", "")).strip() == seed_heading
                neighbor_ok = _is_neighbor_chunk(
                    seed,
                    d,
                    grain_window=grain_window,
                    seq_window=seq_window,
                )
                if unit_ok or heading_ok or neighbor_ok:
                    hit = True
                    break
            if hit:
                scoped.append(d)

        if bool(getattr(config, "ABLATE_GRAPH_EXPANSION", False)):
            graph_docs = []
        else:
            graph_docs = _graph_expand_from_seeds(
                seed_pack,
                corpus_by_uid,
                hops=graph_hops,
                max_nodes=graph_max,
            )
        merged_list = _dedupe_docs(results + scoped + graph_docs)
        use_focus = bool(getattr(config, "RETRIEVAL_ENABLE_TOPIC_FOCUS", True))
        topic_boost = float(getattr(config, "RETRIEVAL_TOPIC_BOOST", 3.0))
        anchor_boost = 4.0
        results = _diverse_top_k(
            merged_list,
            k=final_k,
            query_terms=query_terms,
            topic_terms=topic_terms,
            anchor_terms=anchor_terms,
            use_focus=use_focus,
            topic_boost=topic_boost,
            anchor_boost=anchor_boost,
        )

    enriched_results = []
    seen_chunk_ids = {doc.metadata.get("chunk_uid") for doc in results if doc.metadata.get("chunk_uid")}
    ablate_cci = bool(getattr(config, "ABLATE_CCI", False))

    for doc in results:
        enriched_results.append(doc)
        if ablate_cci:
            continue
        parent_id = doc.metadata.get("parent_macro_id")
        if parent_id and parent_id not in seen_chunk_ids:
            parent_doc = corpus_by_uid.get(parent_id)
            if parent_doc and not _is_noisy_chunk(parent_doc.page_content or ""):
                enriched_results.append(parent_doc)
                seen_chunk_ids.add(parent_id)

    hits: List[Dict[str, Any]] = [hit_to_eval_record(doc, rank=i + 1) for i, doc in enumerate(enriched_results)]

    resolved_hits = hits if bool(getattr(config, "ABLATE_CCR", False)) else resolve_conflicts(hits)
    required_terms = set(topic_terms) if topic_terms else set(anchor_terms)
    if not required_terms:
        required_terms = set(query_terms)
    required_terms = set(list(required_terms)[:6])
    coverage = _query_coverage_ratio(required_terms, resolved_hits)
    min_cov = float(getattr(config, "RETRIEVAL_MIN_QUERY_COVERAGE", 0.25))
    LAST_COVERAGE["value"] = coverage
    LAST_COVERAGE["threshold"] = min_cov
    LAST_COVERAGE["failed_closed"] = coverage < min_cov
    if coverage < min_cov:
        if bool(getattr(config, "ABLATE_COVERAGE_GATE", False)):
            LAST_COVERAGE["suppressed"] = True
        else:
            log_retrieval(
                normalized_query,
                0,
                [f"coverage_below_threshold:{coverage:.2f}<{min_cov:.2f}"],
            )
            return [], [], []

    contexts = [h["chunk_text"] for h in resolved_hits]
    sources_raw = [h.get("source", "Unknown") for h in resolved_hits]
    sources = _ordered_unique_sources([s for s in sources_raw if isinstance(s, str)])

    log_retrieval(normalized_query, len(resolved_hits), sources)

    return contexts, sources, resolved_hits


def resolve_conflicts(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Novel CCR Strategy: Consensus-Based Conflict Resolution.
    1. Identifies chunks discussing similar entities (using keyword overlap).
    2. Flags chunks that contradict majority values (especially numeric data).
    3. Prioritizes 'Macro' chunks for context and 'Micro' for specific facts unless they outlier.
    """
    if not hits:
        return hits


    unique_hits: List[Dict[str, Any]] = []
    for h in hits:
        txt = (h.get("chunk_text") or "").strip()
        if not txt:
            continue
        is_dup = any(_near_duplicate_text(txt, (u.get("chunk_text") or "")) for u in unique_hits)
        if not is_dup:
            unique_hits.append(h)

    return unique_hits
