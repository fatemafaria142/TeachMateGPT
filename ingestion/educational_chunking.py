"""
COPE textbook chunking — Curriculum-Oriented Pedagogical Embedding.

Splits Class 8 NCTB-style science PDFs into retrieval-sized units that follow how the
book is taught, not arbitrary token windows:

1. **Structure** — Detect Bangla headings and lesson boundaries so segments inherit
   chapter/lesson scope.
2. **Bangla-aware recursion** — ``RecursiveCharacterTextSplitter`` with separators
   tuned for Bangla STEM prose.
3. **Multi-resolution tiers** — Parallel macro / meso / micro (and optional
   token-micro) passes with different sizes and overlaps; each chunk records tier,
   lineage, and neighbor links for hybrid RAG expansion downstream.

Loads pages via ``scanned_pdf_loader`` when needed and emits ``Document`` objects with
rich metadata for ``embed_and_store``.
"""
from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

import config
from ingestion.scanned_pdf_loader import load_pdf_documents
from utils.logger import log_info



STRUCTURAL_ANCHOR = re.compile(
    r"(?m)^(?=[ \t]*(?:"
    r"অধ্যায়|অধ্যায়|পাঠ\s*\d+|পাঠ\s*[-–—]|প্রকল্প|বিজ্ঞান\s+কাজ|অনুশীলনি|"
    r"সার[\s\-]*সংক্ষেপ|বিষয়সূচি|বিষয়সূচি|মূল\s+পয়েন্ট|মূল\s+পয়েন্ট|"
    r"[\(（]?\s*\d{1,2}\s*[\)）]\s*[\.\)]"
    r"))",
)

SECTION_LINE = re.compile(
    r"(?m)^[ \t]*(?:\d{1,2}\.\d{1,2}|\d{1,2}[\.\)]|[(（]\s*[ivx\d]+\s*[)）])",
    re.IGNORECASE,
)


def _stable_id(parts: Sequence[str]) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"blk_{h}"


_BOX_DRAWING = frozenset("│─┌┐└┘├┤┬┴┼╭╮╰╯═║╔╗╝╚╱╲╳")

_NOISE_LINE_PAGE = re.compile(
    r"^\s*(?:page|p\.?)\s*\d{1,4}\b\s*$",
    re.IGNORECASE,
)
_NOISE_LINE_PAGE_FRAC = re.compile(r"^\s*\d{1,4}\s*/\s*\d{1,4}\s*$")
_NOISE_LINE_PAGE_DASH = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")
_NOISE_LINE_RULE = re.compile(r"^[\-–—_=·\.\s]{4,}$")


def _strip_ingest_noise(text: str) -> str:
    """Remove OCR/layout junk before chunking (line- and character-level)."""
    if not text:
        return text
    text = (
        text.replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
    )
    text = text.replace("\ufffd", "")
    text = re.sub(r"(.)\1{12,}", r"\1\1\1\1", text)

    kept: List[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            kept.append("")
            continue
        if _NOISE_LINE_PAGE.match(s) or _NOISE_LINE_PAGE_FRAC.match(s):
            continue
        if _NOISE_LINE_PAGE_DASH.match(s) and len(s) <= 12:
            continue
        if _NOISE_LINE_RULE.match(s):
            continue
        if len(s) >= 4:
            box_n = sum(1 for c in s if c in _BOX_DRAWING)
            if box_n / len(s) > 0.55:
                continue
        kept.append(line.rstrip())

    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = _strip_ingest_noise(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_low_quality_piece(text: str) -> bool:
    """Drop chunks that are too short/noisy for reliable retrieval."""
    t = (text or "").strip()
    if not t:
        return True
    if len(t) < 80:
        return True
    if "<<<PAGE_" in t:
        return True
    if re.search(r"(?:[?|]{10,}|[#%]{15,})", t):
        return True
    bangla_chars = sum(1 for ch in t if "\u0980" <= ch <= "\u09ff")
    alpha_num = sum(1 for ch in t if ch.isalnum())
    symbol_ratio = 1.0 - (alpha_num / max(1, len(t)))
    if bangla_chars < 20 and symbol_ratio > 0.40:
        return True
    printable = sum(1 for ch in t if ch.isprintable() and not ch.isspace())
    if len(t) > 0 and printable / len(t) < 0.45:
        return True
    return False


def split_curriculum_structure(text: str) -> List[Tuple[Optional[str], str]]:
    """
    Layer 1: split full textbook text into coarse structural units.
    Returns list of (heading_snippet_or_None, unit_text).
    """
    text = _normalize_text(text)
    if not text:
        return []

    idxs = [m.start() for m in STRUCTURAL_ANCHOR.finditer(text)]
    if not idxs:
        return [(None, text)]

    units: List[Tuple[Optional[str], str]] = []
    for i, start in enumerate(idxs):
        end = idxs[i + 1] if i + 1 < len(idxs) else len(text)
        segment = text[start:end].strip()
        if not segment:
            continue
        head_line = segment.split("\n", 1)[0].strip()[:120]
        units.append((head_line, segment))

    if not units:
        return [(None, text)]
    return units


def split_section_grains(unit_text: str) -> List[str]:
    """Optional sub-split on numbered lines inside a curriculum unit."""
    text = unit_text.strip()
    if len(text) < 800:
        return [text]

    spans = [m.start() for m in SECTION_LINE.finditer(text)]
    if len(spans) < 2:
        return [text]

    grains: List[str] = []
    for i, start in enumerate(spans):
        end = spans[i + 1] if i + 1 < len(spans) else len(text)
        g = text[start:end].strip()
        if g:
            grains.append(g)
    return grains if grains else [text]


def _bengali_recursive_separators() -> List[str]:
    return list(
        getattr(
            config,
            "BENGALI_RECURSIVE_SEPARATORS",
            [
                "\n\n\n",
                "\n\n",
                "\n",
                "। ",
                "।",
                ". ",
                "! ",
                "? ",
                "; ",
                " — ",
                " – ",
                ", ",
                " ",
                "",
            ],
        )
    )


def _tier_recursive_splitter(tier_name: str) -> RecursiveCharacterTextSplitter:
    layers = getattr(config, "CHUNK_LAYER_RECURSIVE", {})
    tier_cfg = layers.get(tier_name) or {}
    size = int(tier_cfg.get("chunk_size", 600))
    overlap = int(tier_cfg.get("chunk_overlap", 80))
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=_bengali_recursive_separators(),
        keep_separator=True,
    )


def _token_micro_splitter() -> TokenTextSplitter:
    tc = getattr(config, "TOKEN_MICRO_CHUNK", {})
    return TokenTextSplitter(
        encoding_name=tc.get("encoding", "cl100k_base"),
        chunk_size=int(tc.get("chunk_size", 180)),
        chunk_overlap=int(tc.get("chunk_overlap", 24)),
    )


def _document_from_piece(
    piece: str,
    *,
    tier: str,
    pedagogical_layer: str,
    file_basename: str,
    student_class: str,
    structural_heading: Optional[str],
    unit_idx: int,
    grain_idx: int,
    seq: int,
    lineage_parent: Optional[str],
    chunk_uid_override: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Document:
    run_id = str(uuid.uuid4())[:8]
    piece_fingerprint = hashlib.sha256(piece.encode("utf-8")).hexdigest()[:20]
    chunk_uid = chunk_uid_override or _stable_id(
        [
            file_basename,
            str(unit_idx),
            str(grain_idx),
            tier,
            str(seq),
            lineage_parent or "_root_",
            piece_fingerprint,
        ]
    )
    meta: Dict[str, Any] = {
        "source": file_basename,
        "class": student_class,
        "chunk_tier": tier,
        "pedagogical_layer": pedagogical_layer,
        "structural_unit_idx": unit_idx,
        "grain_idx": grain_idx,
        "sequence": seq,
        "chunk_uid": chunk_uid,
        "pipeline_run": run_id,
        "lineage_parent_id": lineage_parent or "",
        "structural_heading": (structural_heading or "")[:200],
        "cope_version": "1.0",
    }
    if extra:
        meta.update(extra)
    return Document(page_content=piece.strip(), metadata=meta)


def _attach_chunk_graph_edges(chunks: List[Document]) -> None:
    """
    Build lightweight graph links inside metadata for retrieval-time expansion.
    Links are directional (prev/next) plus a compact neighbor list.
    """
    if not chunks:
        return

    by_uid: Dict[str, Document] = {}
    for d in chunks:
        uid = str((d.metadata or {}).get("chunk_uid") or "").strip()
        if uid:
            by_uid[uid] = d

    streams: Dict[Tuple[Any, Any, Any], List[Document]] = {}
    for d in chunks:
        m = d.metadata or {}
        key = (m.get("structural_unit_idx"), m.get("grain_idx"), m.get("chunk_tier"))
        streams.setdefault(key, []).append(d)

    for _, docs in streams.items():
        docs.sort(key=lambda x: int((x.metadata or {}).get("sequence", 0)))
        for i, d in enumerate(docs):
            meta = d.metadata or {}
            uid = str(meta.get("chunk_uid") or "").strip()
            if not uid:
                continue
            prev_uid = str((docs[i - 1].metadata or {}).get("chunk_uid") or "").strip() if i > 0 else ""
            next_uid = str((docs[i + 1].metadata or {}).get("chunk_uid") or "").strip() if i + 1 < len(docs) else ""
            if prev_uid:
                meta["prev_chunk_uid"] = prev_uid
            if next_uid:
                meta["next_chunk_uid"] = next_uid
            d.metadata = meta

    parent_groups: Dict[str, List[Document]] = {}
    for d in chunks:
        m = d.metadata or {}
        parent = str(m.get("lineage_parent_id") or m.get("parent_meso_id") or m.get("parent_macro_id") or "").strip()
        if parent:
            parent_groups.setdefault(parent, []).append(d)

    for _, docs in parent_groups.items():
        uids = [str((d.metadata or {}).get("chunk_uid") or "").strip() for d in docs]
        uids = [u for u in uids if u]
        if len(uids) <= 1:
            continue
        for d in docs:
            m = d.metadata or {}
            uid = str(m.get("chunk_uid") or "").strip()
            if not uid:
                continue
            neighbors = [u for u in uids if u != uid][:8]
            if neighbors:
                m["graph_neighbors"] = neighbors
                d.metadata = m

    heading_groups: Dict[Tuple[str, Any], List[Document]] = {}
    for d in chunks:
        m = d.metadata or {}
        heading = str(m.get("structural_heading") or "").strip()
        tier = m.get("chunk_tier")
        if not heading:
            continue
        heading_groups.setdefault((heading, tier), []).append(d)

    for _, docs in heading_groups.items():
        docs.sort(
            key=lambda x: (
                int((x.metadata or {}).get("grain_idx", 0)),
                int((x.metadata or {}).get("sequence", 0)),
            )
        )
        uids = [str((d.metadata or {}).get("chunk_uid") or "").strip() for d in docs]
        for i, d in enumerate(docs):
            uid = str((d.metadata or {}).get("chunk_uid") or "").strip()
            if not uid:
                continue
            window = uids[max(0, i - 2): i] + uids[i + 1: i + 3]
            window = [u for u in window if u and u != uid]
            if not window:
                continue
            m = d.metadata or {}
            existing = m.get("graph_neighbors") or []
            if not isinstance(existing, list):
                existing = []
            merged = []
            seen = set()
            for x in list(existing) + window:
                sx = str(x).strip()
                if not sx or sx in seen:
                    continue
                seen.add(sx)
                merged.append(sx)
            if merged:
                m["graph_neighbors"] = merged[:12]
                d.metadata = m


def run_educational_chunking_pipeline(
    raw_documents: Sequence[Document],
    source_filename: str,
    student_class: str = "8",
) -> List[Document]:
    """
    Full COPE pipeline: structure → grain → multi-tier recursive (+ optional token).
    """
    parts: List[str] = []
    for d in raw_documents:
        pc = (d.page_content or "").strip()
        if not pc:
            continue
        parts.append(pc)
    full_text = _normalize_text("\n\n".join(parts))
    embed_tiers = set(
        getattr(config, "EMBED_CHUNK_TIERS", ["macro", "meso", "micro"])
    )
    use_token = bool(getattr(config, "ENABLE_TOKEN_MICRO_LAYER", False))

    structural_units = split_curriculum_structure(full_text)
    out: List[Document] = []
    base_name = source_filename

    macro_splitter = _tier_recursive_splitter("macro")
    meso_splitter = _tier_recursive_splitter("meso")
    micro_splitter = _tier_recursive_splitter("micro")
    token_splitter = _token_micro_splitter() if use_token else None

    for unit_idx, (heading, unit_text) in enumerate(structural_units):
        grains = split_section_grains(unit_text)
        for grain_idx, grain in enumerate(grains):
            if not grain.strip():
                continue

            macro_parts = macro_splitter.split_text(grain)
            for mi, m_text in enumerate(macro_parts):
                if _is_low_quality_piece(m_text):
                    continue
                parent_macro_id = _stable_id(
                    [base_name, str(unit_idx), str(grain_idx), "macro", str(mi)]
                )
                if "macro" in embed_tiers:
                    out.append(
                        _document_from_piece(
                            m_text,
                            tier="macro",
                            pedagogical_layer="recursive_macro",
                            file_basename=base_name,
                            student_class=student_class,
                            structural_heading=heading,
                            unit_idx=unit_idx,
                            grain_idx=grain_idx,
                            seq=mi,
                            lineage_parent=None,
                            chunk_uid_override=parent_macro_id,
                            extra={"parent_macro_id": parent_macro_id},
                        )
                    )

                meso_parts = meso_splitter.split_text(m_text)
                for sj, ms_text in enumerate(meso_parts):
                    if _is_low_quality_piece(ms_text):
                        continue
                    meso_id = _stable_id([parent_macro_id, "meso", str(sj)])
                    if "meso" in embed_tiers:
                        out.append(
                            _document_from_piece(
                                ms_text,
                                tier="meso",
                                pedagogical_layer="recursive_meso",
                                file_basename=base_name,
                                student_class=student_class,
                                structural_heading=heading,
                                unit_idx=unit_idx,
                                grain_idx=grain_idx,
                                seq=sj,
                                lineage_parent=parent_macro_id,
                                chunk_uid_override=meso_id,
                                extra={
                                    "parent_macro_id": parent_macro_id,
                                    "parent_meso_id": meso_id,
                                },
                            )
                        )

                    micro_parts = micro_splitter.split_text(ms_text)
                    for ti, tx in enumerate(micro_parts):
                        if _is_low_quality_piece(tx):
                            continue
                        if "micro" in embed_tiers:
                            out.append(
                                _document_from_piece(
                                    tx,
                                    tier="micro",
                                    pedagogical_layer="recursive_micro",
                                    file_basename=base_name,
                                    student_class=student_class,
                                    structural_heading=heading,
                                    unit_idx=unit_idx,
                                    grain_idx=grain_idx,
                                    seq=ti,
                                    lineage_parent=meso_id,
                                    extra={
                                        "parent_macro_id": parent_macro_id,
                                        "parent_meso_id": meso_id,
                                    },
                                )
                            )

                        if token_splitter and "token_micro" in embed_tiers:
                            for tki, tk_doc in enumerate(
                                token_splitter.create_documents([tx])
                            ):
                                if _is_low_quality_piece(tk_doc.page_content):
                                    continue
                                out.append(
                                    _document_from_piece(
                                        tk_doc.page_content,
                                        tier="token_micro",
                                        pedagogical_layer="token_aligned",
                                        file_basename=base_name,
                                        student_class=student_class,
                                        structural_heading=heading,
                                        unit_idx=unit_idx,
                                        grain_idx=grain_idx,
                                        seq=tki,
                                        lineage_parent=meso_id,
                                        extra={"parent_meso_id": meso_id},
                                    )
                                )

    if not out:
        fb = _tier_recursive_splitter("macro")
        merged = _normalize_text(full_text) or full_text
        for i, d in enumerate(fb.create_documents([merged])):
            piece = (d.page_content or "").strip()
            if _is_low_quality_piece(piece):
                continue
            out.append(
                _document_from_piece(
                    piece,
                    tier="macro",
                    pedagogical_layer="fallback_recursive_only",
                    file_basename=base_name,
                    student_class=student_class,
                    structural_heading=None,
                    unit_idx=0,
                    grain_idx=0,
                    seq=i,
                    lineage_parent=None,
                    extra={"fallback": True},
                )
            )

    _attach_chunk_graph_edges(out)
    return out


@dataclass
class ChunkingStats:
    by_tier: Dict[str, int]
    total: int


def summarize_chunk_stats(documents: Sequence[Document]) -> ChunkingStats:
    by_tier: Dict[str, int] = {}
    for d in documents:
        t = (d.metadata or {}).get("chunk_tier", "unknown")
        by_tier[t] = by_tier.get(t, 0) + 1
    return ChunkingStats(by_tier=by_tier, total=len(documents))




def load_raw_documents(file_path: str) -> List[Document]:
    """
    Load PDF or text as LangChain documents.

    PDFs use raster + GPT-4o-mini vision when ``PDF_INGESTION_MODE`` is ``vision``
    (alias ``ocr``) or ``auto``; ``native`` uses text extraction only.
    """
    if file_path.endswith(".pdf"):
        mode = getattr(config, "PDF_INGESTION_MODE", "vision")
        if mode == "native":
            return PyPDFLoader(file_path).load()
        return load_pdf_documents(file_path)
    return TextLoader(file_path, encoding="utf-8").load()


def load_and_chunk_textbook(file_path: str) -> List[Document]:
    """
    End-to-end: load one textbook file, then run the COPE multi-resolution pipeline.
    """
    documents = load_raw_documents(file_path)
    basename = os.path.basename(file_path)
    chunks = run_educational_chunking_pipeline(
        documents,
        source_filename=basename,
        student_class="8",
    )
    stats = summarize_chunk_stats(chunks)
    log_info(
        f"COPE chunks for {basename}: total={stats.total} by_tier={stats.by_tier}"
    )
    return chunks
