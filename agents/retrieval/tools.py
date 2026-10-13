"""
Pure helper functions behind hybrid retrieval: query analysis/expansion,
scoring, dedup, neighbor/graph expansion, and coverage measurement.

No module-level singletons live here (those are in ``agent.py``); everything
below is stateless given its arguments, which keeps it easy to unit test.
"""
import hashlib
import json
import math
import re
from typing import Any, Dict, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

import config
from utils.bangla_utils import clean_bangla_text, to_english_digits
from utils.llm_client import chat_completion
from utils.prompt_loader import load_prompt


_MULTIQUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=load_prompt("retrieval_multiquery_prompt.md"),
)


def _doc_fusion_key(doc: Document) -> str:
    meta = doc.metadata or {}
    return f"{meta.get('source', '')}\0{doc.page_content}"


def _scroll_all_documents(client, collection_name: str) -> list[Document]:
    """Fetch every point's page_content + metadata from Qdrant (BM25 corpus build)."""
    docs: list[Document] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in points:
            payload = p.payload or {}
            text = payload.get("page_content", "")
            meta = payload.get("metadata", {}) or {}
            docs.append(Document(page_content=text, metadata=meta))
        if offset is None:
            break
    return docs


def _analyze_query_for_retrieval(query: str) -> Dict[str, Any]:
    """
    Analyze user query first, then use the analysis to drive query expansion.
    """
    q = _normalize_query(query)
    if not q:
        return {"constraints": "", "language": "bn", "search_query": ""}

    system_prompt = load_prompt("retrieval_query_planner_prompt.md")
    user_msg = f"User query:\n{q}"
    try:
        raw = chat_completion(system_prompt, user_msg, model=config.OPENAI_MODEL, temperature=0, json_mode=True)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("non-dict analysis")
        constraints = str(data.get("constraints") or "").strip()
        language = str(data.get("language") or "bn").strip().lower()
        search_query = str(data.get("search_query") or "").strip()
        if not search_query:
            search_query = q
        return {
            "constraints": constraints,
            "language": language if language in {"bn", "en", "mixed"} else "bn",
            "search_query": search_query,
        }
    except Exception:
        return {"constraints": "", "language": "bn", "search_query": q}


def _weighted_rrf_merge(
    query: str,
    retrievers: list,
    weights: list[float],
    *,
    k: int = 5,
    rrf_const: int = 60,
) -> list[Document]:
    """Merge ranked lists from several retrievers (replaces removed EnsembleRetriever)."""
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for retriever, weight in zip(retrievers, weights, strict=True):
        ranked = retriever.invoke(query)
        for rank, doc in enumerate(ranked, start=1):
            key = _doc_fusion_key(doc)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_const + rank)
            by_key[key] = doc
    ordered_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [by_key[key] for key in ordered_keys[:k]]


def hit_to_eval_record(doc: Document, rank: int) -> Dict[str, Any]:
    """One retrieved chunk with stable ids + text for Ragas / LLM-judge / human provenance."""
    meta = dict(doc.metadata or {})
    text = (doc.page_content or "").strip()
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    rec: Dict[str, Any] = {
        "rank": rank,
        "chunk_uid": meta.get("chunk_uid"),
        "source": meta.get("source") or "Unknown",
        "chunk_text": text,
        "text_sha256": h,
        "chunk_tier": meta.get("chunk_tier"),
        "pedagogical_layer": meta.get("pedagogical_layer"),
        "sequence": meta.get("sequence"),
        "structural_unit_idx": meta.get("structural_unit_idx"),
        "grain_idx": meta.get("grain_idx"),
        "structural_heading": meta.get("structural_heading"),
        "lineage_parent_id": meta.get("lineage_parent_id"),
        "parent_macro_id": meta.get("parent_macro_id"),
        "parent_meso_id": meta.get("parent_meso_id"),
        "page": meta.get("page"),
    }
    return {k: v for k, v in rec.items() if v is not None and v != ""}


def _ordered_unique_sources(sources: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for s in sources:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _normalize_query(query: str) -> str:
    q = clean_bangla_text(query or "")
    q = to_english_digits(q)
    return re.sub(r"\s+", " ", q).strip()


def _effective_query(query: str) -> str:
    """
    Prefer explicit teacher follow-up text when present, because it usually carries
    the final narrowed intent for retrieval.
    """
    q = query or ""
    marker = "[Teacher clarification / follow-up]:"
    if marker in q:
        tail = q.split(marker)[-1].strip()
        if tail:
            return tail
    return q


def _strip_task_phrases(query: str) -> str:
    """
    Keep domain concept terms and remove question-generation instruction words
    that can bias retrieval toward exercise-format chunks.
    """
    q = _normalize_query(query)
    if not q:
        return q
    remove_patterns = [
        r"\b(বহুনির্বাচনী|এমসিকিউ|mcq)\b",
        r"\b(সৃজনশীল)\b",
        r"\b(প্রশ্নগুলো|প্রশ্ন|question[s]?)\b",
        r"\b(তৈরি|প্রস্তুত|দিন|দাও|করো)\b",
        r"\b(শিক্ষার্থীদের ধারণা যাচাই করার জন্য|ধারণা যাচাই)\b",
    ]
    cleaned = q
    for p in remove_patterns:
        cleaned = re.sub(p, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
    return cleaned or q


def _query_terms(query: str) -> set[str]:
    q = _normalize_query(query).lower()
    tokens = re.findall(r"[ঀ-৿a-z0-9]+", q)
    stop = {"ও", "এবং", "নিয়ে", "নিয়ে", "সম্পর্কে", "জন্য", "একটি", "করে", "দিন", "করো", "the"}
    return {t for t in tokens if len(t) >= 2 and t not in stop}


def _domain_expansion_terms(query: str) -> set[str]:
    """
    Lightweight curriculum lexicon to stabilize Bangla science retrieval.
    Adds sibling concept anchors when a canonical topic is detected.
    """
    q = _normalize_query(query).lower()
    expanded: set[str] = set()
    if "অমেরুদণ্ডী" in q:
        expanded.update(
            {
                "পর্ব",
                "প্যারিফেরা",
                "নিডারিয়া",
                "নিডারিয়া",
                "প্লাটিহেলমিন্থেস",
                "নেমাটোডা",
                "অ্যানেলিডা",
                "আর্থ্রোপোডা",
                "মলাস্কা",
                "একাইনোডার্মাটা",
            }
        )
    if "মেরুদণ্ডী" in q:
        expanded.update({"মাছ", "উভচর", "সরীসৃপ", "পাখি", "স্তন্যপায়ী", "স্তন্যপায়ী"})
    return expanded


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[ঀ-৿a-z0-9]+", (text or "").lower()))


def _dynamic_anchor_terms(query: str, all_docs: list[Document], *, top_n: int = 6) -> set[str]:
    """
    Dynamically pick query anchors using rarity (IDF-like) over current corpus.
    This avoids static keyword lists and adapts to any chapter/topic.
    """
    q = _normalize_query(query).lower()
    q_terms = _query_terms(q)
    if not q_terms:
        return set()

    quoted = re.findall(r"[“\"']([^”\"']{2,70})[”\"']", q)
    for span in quoted:
        for t in re.findall(r"[ঀ-৿a-z0-9]+", span.lower()):
            if len(t) >= 2:
                q_terms.add(t)

    df = {t: 0 for t in q_terms}
    total_docs = max(1, len(all_docs))
    for d in all_docs:
        txt_tokens = _token_set(d.page_content or "")
        present = {t for t in q_terms if t in txt_tokens}
        for t in present:
            df[t] += 1

    scored: list[tuple[float, str]] = []
    for t in q_terms:
        idf = math.log((1 + total_docs) / (1 + df.get(t, 0))) + 1.0
        score = idf + (0.15 * min(len(t), 10))
        scored.append((score, t))
    scored.sort(reverse=True)
    return {t for _, t in scored[:top_n]}


def _extract_explicit_topic(query: str) -> Optional[str]:
    """
    Extract an explicit chapter/topic focus phrase from Bangla teacher query.
    Examples: "বল ও গতি অধ্যায় থেকে ...", "খাদ্যশৃঙ্খল নিয়ে ..."
    """
    q = _normalize_query(query)
    if not q:
        return None

    quoted = re.findall(r"[“\"']([^”\"']{2,70})[”\"']", q)
    if quoted:
        best = max((x.strip() for x in quoted), key=len, default="")
        if best:
            return best

    m_subject = re.search(r"([ঀ-৿A-Za-z0-9\s]{2,70})\s*বিষ[য়য]টি", q, flags=re.IGNORECASE)
    if m_subject:
        phrase = re.sub(r"\s+", " ", m_subject.group(1)).strip(" -:;,.")
        if phrase:
            return phrase

    patterns = [
        r"[“\"']?([ঀ-৿A-Za-z0-9\s]{2,60}?)[”\"']?\s*অধ্যা(?:য়|য)\s*থেকে",
        r"[“\"']?([ঀ-৿A-Za-z0-9\s]{2,60}?)[”\"']?\s*(নিয়ে|নিয়ে|সম্পর্কে|ভিত্তিক)",
        r"বিষয়\s*[:：]\s*([ঀ-৿A-Za-z0-9\s]{2,60})",
    ]
    for p in patterns:
        m = re.search(p, q, flags=re.IGNORECASE)
        if not m:
            continue
        phrase = re.sub(r"\s+", " ", m.group(1)).strip(" -:;,.")
        if len(phrase) >= 2:
            return phrase
    return None


def _topic_terms(topic: Optional[str]) -> set[str]:
    if not topic:
        return set()
    t = _normalize_query(topic).lower()
    tokens = re.findall(r"[ঀ-৿a-z0-9]+", t)
    stop = {
        "ও",
        "এবং",
        "এর",
        "থেকে",
        "উপর",
        "নিয়ে",
        "নিয়ে",
        "সম্পর্কে",
        "ভিত্তিক",
        "বিষয়টি",
        "বিষয়টি",
        "শিক্ষার্থীরা",
        "শিক্ষার্থীদের",
        "ধারণা",
        "প্রয়োগ",
        "পর্যবেক্ষণ",
        "গুরুত্ব",
    }
    terms = [x for x in tokens if len(x) >= 2 and x not in stop]
    return set(terms[:6])


def _is_noisy_chunk(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t) < 80:
        return True
    if "<<<PAGE_" in t:
        return True
    exam_noise_markers = (
        "নিচের কোনটি সঠিক",
        "সৃজনশীল প্রশ্ন",
        "সাংকেতিক-উত্তর প্রশ্ন",
    )
    if any(m in t for m in exam_noise_markers):
        return True
    bangla_chars = sum(1 for ch in t if "ঀ" <= ch <= "৿")
    alpha_num = sum(1 for ch in t if ch.isalnum())
    symbol_ratio = 1.0 - (alpha_num / max(1, len(t)))
    return bangla_chars < 20 and symbol_ratio > 0.35


def _chunk_relevance_score(doc: Document, q_terms: set[str]) -> float:
    text = (doc.page_content or "").lower()
    meta = doc.metadata or {}
    heading = str(meta.get("structural_heading", "")).lower()
    if not q_terms:
        return 0.0
    txt_tokens = _token_set(text)
    heading_tokens = _token_set(heading)
    overlap_text = sum(1 for t in q_terms if t in txt_tokens)
    overlap_heading = sum(1 for t in q_terms if t in heading_tokens)
    tier = str(meta.get("chunk_tier", "")).lower()
    tier_boost = 0.2 if tier in {"meso", "micro"} else 0.0
    noise_penalty = 1.5 if _is_noisy_chunk(text) else 0.0
    return (overlap_text * 1.0) + (overlap_heading * 1.5) + tier_boost - noise_penalty


def _topic_overlap(doc: Document, topic_terms: set[str]) -> int:
    if not topic_terms:
        return 0
    text = (doc.page_content or "").lower()
    heading = str((doc.metadata or {}).get("structural_heading", "")).lower()
    txt_tokens = _token_set(text)
    heading_tokens = _token_set(heading)
    t_hit = sum(1 for t in topic_terms if t in txt_tokens)
    h_hit = sum(1 for t in topic_terms if t in heading_tokens)
    return t_hit + (2 * h_hit)


def _anchor_overlap(doc: Document, anchors: set[str]) -> int:
    if not anchors:
        return 0
    text = (doc.page_content or "").lower()
    heading = str((doc.metadata or {}).get("structural_heading", "")).lower()
    txt_tokens = _token_set(text)
    heading_tokens = _token_set(heading)
    t_hit = sum(1 for a in anchors if a in txt_tokens)
    h_hit = sum(1 for a in anchors if a in heading_tokens)
    return t_hit + (2 * h_hit)


def _forbidden_opposites(query_terms: set[str]) -> set[str]:
    """
    Build opposite-class tokens to avoid obvious polarity drift in Bangla terms.
    Example: query has 'মেরুদণ্ডী' -> forbid 'অমেরুদণ্ডী' hits.
    """
    forbidden: set[str] = set()
    for t in query_terms:
        if len(t) < 4:
            continue
        if t.startswith("অন") and len(t) > 4:
            forbidden.add(t[2:])
        elif t.startswith("অ") and len(t) > 3:
            forbidden.add(t[1:])
        else:
            forbidden.add(f"অ{t}")
            forbidden.add(f"অন{t}")
    return forbidden


def _contains_forbidden_without_target(
    doc: Document,
    forbidden: set[str],
    target_terms: set[str],
) -> bool:
    if not forbidden:
        return False
    tokens = _token_set((doc.page_content or "") + " " + str((doc.metadata or {}).get("structural_heading", "")))
    has_forbidden = any(f in tokens for f in forbidden)
    has_target = any(t in tokens for t in target_terms)
    return has_forbidden and not has_target


def _query_coverage_ratio(required_terms: set[str], hits: list[dict[str, Any]]) -> float:
    """
    Measure how much retrieved text actually covers core query terms.
    """
    if not required_terms:
        return 1.0
    merged = " ".join((h.get("chunk_text") or "") for h in hits).lower()
    merged_tokens = _token_set(merged)
    covered = sum(1 for t in required_terms if t in merged_tokens)
    return covered / max(1, len(required_terms))


def _normalized_text_for_dedupe(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _near_duplicate_text(a: str, b: str) -> bool:
    """
    Treat chunks as duplicates when one mostly contains the other
    (common for macro/meso/micro from the same section) or they are very similar.
    """
    ta = _normalized_text_for_dedupe(a)
    tb = _normalized_text_for_dedupe(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if short and short in long_:
        if len(short) / max(1, len(long_)) >= 0.78:
            return True
    sa = _token_set(ta)
    sb = _token_set(tb)
    if not sa or not sb:
        return False
    j = len(sa & sb) / max(1, len(sa | sb))
    return j >= 0.92


def _is_topic_or_section_query(query: str) -> bool:
    """
    Heuristic: teacher prompts asking for a whole section/topic (not a single fact)
    need wider evidence than strict keyword match.
    """
    q = _normalize_query(query or "").lower()
    if len(q) < 25:
        return False
    markers = [
        "শ্রেণিবিন্যাস",
        "পর্বের",
        "বিভিন্ন",
        "পার্থক্য",
        "তুলনা",
        "বৈশিষ্ট্য",
        "উদাহরণ",
        "অধ্যায়",
        "অধ্যায়",
        "বিষয়",
        "বিষয়টি",
        "টপিক",
        "সেকশন",
        "উপরের",
        "নিচের",
    ]
    return any(m in q for m in markers)


def _is_neighbor_chunk(seed: Document, candidate: Document, *, grain_window: int, seq_window: int) -> bool:
    sm = seed.metadata or {}
    cm = candidate.metadata or {}
    s_uid = sm.get("chunk_uid")
    c_uid = cm.get("chunk_uid")
    if s_uid and c_uid and s_uid == c_uid:
        return True

    s_heading = str(sm.get("structural_heading") or "").strip()
    c_heading = str(cm.get("structural_heading") or "").strip()
    s_unit = sm.get("structural_unit_idx")
    c_unit = cm.get("structural_unit_idx")

    same_heading = bool(s_heading) and s_heading == c_heading
    same_unit = s_unit is not None and c_unit is not None and s_unit == c_unit
    if not (same_heading or same_unit):
        return False

    s_parents = {
        str(sm.get("lineage_parent_id") or "").strip(),
        str(sm.get("parent_macro_id") or "").strip(),
        str(sm.get("parent_meso_id") or "").strip(),
    }
    c_parents = {
        str(cm.get("lineage_parent_id") or "").strip(),
        str(cm.get("parent_macro_id") or "").strip(),
        str(cm.get("parent_meso_id") or "").strip(),
    }
    s_parents.discard("")
    c_parents.discard("")
    if s_parents and c_parents and (s_parents & c_parents):
        return True

    try:
        sg = int(sm.get("grain_idx"))
        cg = int(cm.get("grain_idx"))
    except Exception:
        sg = cg = None
    try:
        ss = int(sm.get("sequence"))
        cs = int(cm.get("sequence"))
    except Exception:
        ss = cs = None

    if sg is not None and cg is not None and abs(sg - cg) <= max(0, grain_window):
        if ss is None or cs is None:
            return True
        return abs(ss - cs) <= max(0, seq_window)

    if ss is not None and cs is not None:
        return abs(ss - cs) <= max(0, seq_window)
    return False


def _index_docs_by_uid(docs: list[Document]) -> dict[str, Document]:
    by_uid: dict[str, Document] = {}
    for d in docs:
        uid = str((d.metadata or {}).get("chunk_uid") or "").strip()
        if uid:
            by_uid[uid] = d
    return by_uid


def _graph_expand_from_seeds(
    seeds: list[Document],
    by_uid: dict[str, Document],
    *,
    hops: int = 1,
    max_nodes: int = 32,
) -> list[Document]:
    """
    Expand from seed chunks through ingest-time graph links:
    prev/next + sibling neighbors (same parent block).
    """
    if not seeds or not by_uid:
        return []

    out: list[Document] = []
    seen: set[str] = set()
    frontier: list[tuple[str, int]] = []

    for d in seeds:
        uid = str((d.metadata or {}).get("chunk_uid") or "").strip()
        if uid and uid in by_uid:
            frontier.append((uid, 0))

    while frontier and len(out) < max_nodes:
        uid, depth = frontier.pop(0)
        if uid in seen:
            continue
        seen.add(uid)
        doc = by_uid.get(uid)
        if not doc:
            continue
        out.append(doc)
        if depth >= max(0, hops):
            continue
        m = doc.metadata or {}
        neighbors: list[str] = []
        prev_uid = str(m.get("prev_chunk_uid") or "").strip()
        next_uid = str(m.get("next_chunk_uid") or "").strip()
        if prev_uid:
            neighbors.append(prev_uid)
        if next_uid:
            neighbors.append(next_uid)
        raw_neighbors = m.get("graph_neighbors") or []
        if isinstance(raw_neighbors, list):
            neighbors.extend(str(x).strip() for x in raw_neighbors if str(x).strip())
        for n in neighbors:
            if n in by_uid and n not in seen:
                frontier.append((n, depth + 1))
    return out


def _dedupe_docs(docs: list[Document]) -> list[Document]:
    out: list[Document] = []
    seen: set[str] = set()
    for d in docs:
        m = d.metadata or {}
        key = str(
            m.get("chunk_uid")
            or f"{m.get('grain_idx','')}|{m.get('sequence','')}|{m.get('structural_heading','')}|{(d.page_content or '')[:30]}"
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _diverse_top_k(
    docs: list[Document],
    *,
    k: int,
    query_terms: set[str],
    topic_terms: set[str],
    anchor_terms: set[str],
    use_focus: bool,
    topic_boost: float,
    anchor_boost: float,
) -> list[Document]:
    ranked = sorted(
        docs,
        key=lambda d: _chunk_relevance_score(d, query_terms)
        + ((topic_boost * _topic_overlap(d, topic_terms)) if use_focus and topic_terms else (anchor_boost * _anchor_overlap(d, anchor_terms))),
        reverse=True,
    )
    if not ranked:
        return []
    chosen: list[Document] = []
    seen_grains: set[tuple[Any, Any]] = set()
    for d in ranked:
        m = d.metadata or {}
        gk = (m.get("structural_unit_idx"), m.get("grain_idx"))
        same_grain_count = sum(1 for x in chosen if ((x.metadata or {}).get("structural_unit_idx"), (x.metadata or {}).get("grain_idx")) == gk)
        if same_grain_count >= 2 and len(chosen) < k - 1:
            continue
        chosen.append(d)
        seen_grains.add(gk)
        if len(chosen) >= k:
            break
    return chosen[:k]
