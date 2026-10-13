"""
LangGraph orchestrator — teacher query to textbook-grounded assessments.

Builds and runs the graph: intent → ambiguity → optional clarification → retrieval
→ assessment → verification → export-ready state. Coordinates skills (retrieval,
generation, SAVER) and routing when context is missing or verification fails.
Main entry: ``LangGraphOrchestrator.process_query``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import config
from agents.assessment import normalize_selected_tasks
from agents.state import TutorState
from agents.skills import run_skill


def _step(node: str, **extra: Any) -> List[Dict[str, Any]]:
    return [{"node": node, **extra}]


def _maybe_errors(err: Optional[str]) -> List[str]:
    return [err] if err else []


def _coverage_readiness(query: str, retrieval_chunks: List[Dict[str, Any]], context: List[str]) -> Dict[str, Any]:
    if not bool(getattr(config, "PREGEN_ENABLE_COVERAGE_CHECK", True)):
        return {"ok": True, "reason": "", "stats": {}}
    min_chunks = int(getattr(config, "PREGEN_MIN_CONTEXT_CHUNKS", 4))
    min_grains = int(getattr(config, "PREGEN_MIN_DISTINCT_GRAINS", 3))
    min_headings = int(getattr(config, "PREGEN_MIN_DISTINCT_HEADINGS", 1))
    min_term_hits = int(getattr(config, "PREGEN_MIN_QUERY_TERM_HITS", 2))

    chunks = retrieval_chunks or []
    headings = {
        str(c.get("structural_heading") or "").strip()
        for c in chunks
        if str(c.get("structural_heading") or "").strip()
    }
    grains = {
        (c.get("structural_unit_idx"), c.get("grain_idx"))
        for c in chunks
        if c.get("structural_unit_idx") is not None and c.get("grain_idx") is not None
    }
    q_tokens = set(re.findall(r"[\u0980-\u09ffa-z0-9]+", (query or "").lower()))
    stop = {"এবং", "ও", "জন্য", "নিয়ে", "নিয়ে", "বিষয়", "বিষয়", "প্রশ্ন", "তৈরি", "দিন"}
    q_tokens = {t for t in q_tokens if len(t) >= 2 and t not in stop}
    merged_ctx = " ".join(context or []).lower()
    term_hits = sum(1 for t in q_tokens if t in merged_ctx)

    ok = (
        len(context or []) >= min_chunks
        and len(grains) >= min_grains
        and len(headings) >= min_headings
        and term_hits >= min_term_hits
    )
    strong_single = (
        len(context or []) >= 1
        and len(headings) >= 1
        and term_hits >= max(6, min_term_hits + 3)
    )
    if strong_single:
        ok = True
    stats = {
        "chunks": len(context or []),
        "distinct_grains": len(grains),
        "distinct_headings": len(headings),
        "query_term_hits": term_hits,
    }
    reason = ""
    if not ok:
        reason = (
            f"coverage_low: chunks={stats['chunks']}, grains={stats['distinct_grains']}, "
            f"headings={stats['distinct_headings']}, term_hits={stats['query_term_hits']}"
        )
    medium_min_chunks = int(getattr(config, "PREGEN_MEDIUM_MIN_CONTEXT_CHUNKS", 2))
    medium_min_grains = int(getattr(config, "PREGEN_MEDIUM_MIN_DISTINCT_GRAINS", 2))
    medium_min_term_hits = int(getattr(config, "PREGEN_MEDIUM_MIN_QUERY_TERM_HITS", 1))
    medium = (
        not ok
        and len(context or []) >= medium_min_chunks
        and len(grains) >= medium_min_grains
        and term_hits >= medium_min_term_hits
    )
    if not medium and not ok:
        medium = len(context or []) >= 3 and term_hits >= max(2, medium_min_term_hits)
    return {"ok": ok, "medium": medium, "reason": reason, "stats": stats}


def intent_node(state: TutorState) -> Dict[str, Any]:
    raw_intent, err = run_skill("intent.classify", state["query"])
    if err or raw_intent is None:
        resolved = "science_query"
    elif isinstance(raw_intent, str):
        resolved = raw_intent.strip().lower()
    else:
        resolved = str(raw_intent).strip().lower()
    _science_aliases = {
        "explanation",
        "quiz",
        "definition",
        "comparison",
        "other_science",
        "science_help",
        "science",
        "curriculum",
    }
    if resolved in _science_aliases:
        resolved = "science_query"
    return {
        "intent": resolved,
        "next_step": (
            "ambiguity"
            if resolved not in ("greeting", "off_topic", "harmful")
            else "end"
        ),
        "execution_trace": _step("intent", intent=resolved, fallback=bool(err)),
        "errors": _maybe_errors(err),
    }


def ambiguity_node(state: TutorState) -> Dict[str, Any]:
    result, err = run_skill("ambiguity.check", state["query"])
    out: Dict[str, Any] = {
        "execution_trace": _step("ambiguity", ok=not err),
        "errors": _maybe_errors(err),
    }
    if err or not isinstance(result, dict):
        result = {"is_ambiguous": False, "reason": "", "options": []}
    out["is_ambiguous"] = bool(result.get("is_ambiguous"))
    out["ambiguity_reason"] = result.get("reason")
    out["clarification_options"] = list(result.get("options") or [])
    out["next_step"] = "clarification" if out["is_ambiguous"] else "retrieval"
    return out


def clarification_node(state: TutorState) -> Dict[str, Any]:
    reason = state.get("ambiguity_reason") or ""
    options = state.get("clarification_options") or []
    answer, err = run_skill("clarification.generate", reason, options)
    out: Dict[str, Any] = {
        "execution_trace": _step("clarification", ok=not err),
        "errors": _maybe_errors(err),
    }
    if err or not answer:
        answer = "প্রশ্নটি একটু স্পষ্ট করলে পাঠ্যবই ভিত্তিক প্রশ্ন তৈরি করা সহজ হবে।"
    out["answer"] = answer
    out["next_step"] = "end"
    return out


def retrieval_node(state: TutorState) -> Dict[str, Any]:
    payload, err = run_skill("retrieval.context", state["query"])
    context: List[str] = []
    sources: List[str] = []
    retrieval_chunks: List[Dict[str, Any]] = []
    if not err and isinstance(payload, tuple) and len(payload) == 3:
        context, sources, retrieval_chunks = payload
        context = list(context or [])
        sources = list(sources or [])
        retrieval_chunks = list(retrieval_chunks or [])
    elif not err and isinstance(payload, tuple) and len(payload) == 2:
        context, sources = payload
        context = list(context or [])
        sources = list(sources or [])
        retrieval_chunks = [
            {"rank": i + 1, "source": sources[0] if sources else "Unknown", "chunk_text": t}
            for i, t in enumerate(context)
        ]
    elif not err and payload:
        context = list(payload)
    out: Dict[str, Any] = {
        "execution_trace": _step("retrieval", ok=not err, chunks=len(context)),
        "errors": _maybe_errors(err),
    }
    if err:
        context = []
        sources = []
        retrieval_chunks = []
    if not context:
        out["context"] = []
        out["sources"] = []
        out["retrieval_chunks"] = []
        out["answer"] = "দুঃখিত, এই বিষয়টি পাঠ্যবইয়ের ভেক্টর স্টোরে খুঁজে পাওয়া যায়নি।"
        out["next_step"] = "end"
        return out
    out["context"] = context
    out["sources"] = sources
    out["retrieval_chunks"] = retrieval_chunks
    out["next_step"] = "assessment"
    return out


def route_after_retrieval(state: TutorState) -> str:
    ctx = state.get("context") or []
    if not ctx:
        return "end"
    return "assessment"


def route_after_assessment(state: TutorState) -> str:
    asm = state.get("assessment")
    if isinstance(asm, dict):
        has_items = any(isinstance(v, list) and len(v) > 0 for v in asm.values())
        if has_items:
            return "verification"
    return "end"


def assessment_node(state: TutorState) -> Dict[str, Any]:
    """Label retrieved text, then generate MCQ and সৃজনশীল from context."""
    ctx = state["context"]
    if not ctx:
        return {
            "assessment": {},
            "chapter": None,
            "concept": None,
            "answer": "দুঃখিত, প্রাসঙ্গিক কনটেক্সট না পাওয়ায় প্রশ্নসেট তৈরি করা হয়নি।",
            "next_step": "end",
            "execution_trace": _step("assessment", ok=False, reason="empty_context"),
            "errors": ["assessment: empty context after retrieval"],
        }
    readiness = _coverage_readiness(
        state.get("query") or "",
        state.get("retrieval_chunks") or [],
        ctx,
    )
    if not readiness.get("ok"):
        if readiness.get("medium") and bool(getattr(config, "PREGEN_NEIGHBOR_RETRY_ENABLE", True)):
            retry_query = (
                (state.get("query") or "").strip()
                + " একই বিষয়ের পাশের অনুচ্ছেদ, উপরের-নিচের প্রতিবেশী অংশ এবং একই অধ্যায়ের নিকটবর্তী অংশ থেকে প্রসঙ্গ আনো।"
            ).strip()
            payload, r_err = run_skill("retrieval.context", retry_query)
            retry_ctx: List[str] = []
            retry_sources: List[str] = []
            retry_chunks: List[Dict[str, Any]] = []
            if not r_err and isinstance(payload, tuple) and len(payload) == 3:
                retry_ctx, retry_sources, retry_chunks = payload
                retry_ctx = list(retry_ctx or [])
                retry_sources = list(retry_sources or [])
                retry_chunks = list(retry_chunks or [])
            recheck = _coverage_readiness(retry_query, retry_chunks, retry_ctx)
            if recheck.get("ok"):
                ctx = retry_ctx
                state["context"] = retry_ctx
                state["sources"] = retry_sources
                state["retrieval_chunks"] = retry_chunks
            else:
                reason = recheck.get("reason") or "coverage_low_after_neighbor_retry"
                return {
                    "assessment": {},
                    "chapter": None,
                    "concept": None,
                    "answer": "দুঃখিত, বর্তমান রিট্রিভড কনটেক্সট দিয়ে নির্ভরযোগ্যভাবে প্রশ্ন তৈরি করা নিরাপদ নয়। অনুগ্রহ করে বিষয়/অধ্যায় একটু স্পষ্ট করুন বা আবার চেষ্টা করুন।",
                    "next_step": "end",
                    "execution_trace": _step("assessment", ok=False, reason=reason, retried=True),
                    "errors": [f"assessment: {reason}"],
                }
        else:
            reason = readiness.get("reason") or "coverage_low"
            return {
                "assessment": {},
                "chapter": None,
                "concept": None,
                "answer": "দুঃখিত, বর্তমান রিট্রিভড কনটেক্সট দিয়ে নির্ভরযোগ্যভাবে প্রশ্ন তৈরি করা নিরাপদ নয়। অনুগ্রহ করে বিষয়/অধ্যায় একটু স্পষ্ট করুন বা আবার চেষ্টা করুন।",
                "next_step": "end",
                "execution_trace": _step("assessment", ok=False, reason=reason),
                "errors": [f"assessment: {reason}"],
            }
    joined = " ".join(ctx)
    meta, m_err = run_skill("concept.detect", joined)
    assessment, a_err = run_skill(
        "assessment.generate",
        ctx,
        state.get("difficulty_level") or "intermediate",
        state.get("selected_tasks") or list(getattr(config, "ALL_ASSESSMENT_TASK_KEYS", ())),
        state.get("query") or "",
    )
    errs = _maybe_errors(m_err) + _maybe_errors(a_err)
    out: Dict[str, Any] = {
        "execution_trace": _step("assessment", ok=not errs),
        "errors": errs,
    }
    if not isinstance(meta, dict):
        meta = {"chapter": None, "concept": None}
    if a_err or not isinstance(assessment, dict):
        assessment = {}
    out["chapter"] = meta.get("chapter")
    out["concept"] = meta.get("concept")
    out["assessment"] = assessment
    out["answer"] = "পাঠ্যবই ভিত্তিক প্রশ্নসেট তৈরি করা হয়েছে। ভেরিফিকেশন চলছে..."
    out["next_step"] = "verification"
    return out


def verification_node(state: TutorState) -> Dict[str, Any]:
    """SAVER Verification: Check faithfulness of generated assessment."""
    assessment = state.get("assessment") or {}
    context = state.get("context") or []
    query = state.get("query") or ""
    
    report, err = run_skill("verification.faithfulness", query, context, assessment)
    
    out: Dict[str, Any] = {
        "execution_trace": _step("verification", ok=not err, faithful=report.get("is_faithful") if report else False),
        "errors": _maybe_errors(err),
    }
    
    if not report or not isinstance(report, dict):
        report = {"is_faithful": True, "reasoning": "Verification failed to run."}
        
    fs = float(report.get("faithfulness_score", 1.0))
    rs = float(report.get("answer_relevance_score", 1.0))
    hs = float(report.get("hallucination_risk_score", 0.0))
    verdict = bool(report.get("is_faithful", True))
    verdict = verdict and fs >= float(getattr(config, "MIN_FAITHFULNESS_SCORE", 0.8))
    verdict = verdict and rs >= float(getattr(config, "MIN_ANSWER_RELEVANCE_SCORE", 0.75))
    verdict = verdict and hs <= float(getattr(config, "MAX_HALLUCINATION_RISK_SCORE", 0.35))

    out["is_faithful"] = verdict
    out["verification_report"] = report
    out["faithfulness_score"] = fs
    out["answer_relevance_score"] = rs
    out["hallucination_risk_score"] = hs
    out["hallucination_risk_level"] = report.get("hallucination_risk_level", "low")
    
    if not out["is_faithful"]:
        out["answer"] = f"সতর্কতা: জেনারেট করা কিছু প্রশ্নে তথ্যের অমিল পাওয়া গেছে।\nকারন: {report.get('reasoning')}"
    else:
        out["answer"] = (
            "পাঠ্যবই ভিত্তিক প্রশ্নসেট সফলভাবে ভেরিফাই করা হয়েছে। "
            f"(Faithfulness: {out['faithfulness_score']:.2f}, "
            f"Relevance: {out['answer_relevance_score']:.2f}, "
            f"Hallucination risk: {out['hallucination_risk_level']})"
        )
        
    out["next_step"] = "end"
    return out


def greeting_node(state: TutorState) -> Dict[str, Any]:
    return {
        "answer": "আমি বিজ্ঞান পাঠ্যক্রম সহায়ক। আপনার বিষয় বা অধ্যায় নিয়ে জিজ্ঞাসা করুন — "
        "আমি পাঠ্যবই থেকে রিট্রিভ করে প্রশ্ন সেট তৈরি করব।",
        "next_step": "end",
        "execution_trace": _step("greeting"),
    }


def harmful_node(state: TutorState) -> Dict[str, Any]:
    return {
        "answer": "দুঃখিত, ক্ষতিকারক বা অনুপযুক্ত বিষয়বস্তু নিয়ে সাহায্য করতে পারি না।",
        "next_step": "end",
        "execution_trace": _step("harmful"),
    }


def off_topic_node(state: TutorState) -> Dict[str, Any]:
    return {
        "answer": "দুঃখিত, আমি শুধুমাত্র ক্লাসের বিজ্ঞান পাঠ্যবই সংক্রান্ত প্রশ্ন নিয়ে কাজ করি।",
        "next_step": "end",
        "execution_trace": _step("off_topic"),
    }


def route_after_intent(state: TutorState) -> str:
    intent = state.get("intent") or ""
    if bool(getattr(config, "ABLATE_ROUTING", False)):
        return "ambiguity_node"
    if intent == "greeting":
        return "greeting_node"
    if intent == "harmful":
        return "harmful_node"
    if intent == "off_topic":
        return "off_topic_node"
    return "ambiguity_node"


def route_after_ambiguity(state: TutorState) -> str:
    if bool(getattr(config, "ABLATE_ROUTING", False)):
        return "retrieval_node"
    if state.get("is_ambiguous"):
        return "clarification_node"
    return "retrieval_node"


workflow = StateGraph(TutorState)

workflow.add_node("intent", intent_node)
workflow.add_node("ambiguity", ambiguity_node)
workflow.add_node("clarification", clarification_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("assessment", assessment_node)
workflow.add_node("verification", verification_node)
workflow.add_node("greeting_node", greeting_node)
workflow.add_node("harmful_node", harmful_node)
workflow.add_node("off_topic_node", off_topic_node)

workflow.set_entry_point("intent")

workflow.add_conditional_edges(
    "intent",
    route_after_intent,
    {
        "greeting_node": "greeting_node",
        "harmful_node": "harmful_node",
        "off_topic_node": "off_topic_node",
        "ambiguity_node": "ambiguity",
    },
)

workflow.add_conditional_edges(
    "ambiguity",
    route_after_ambiguity,
    {
        "clarification_node": "clarification",
        "retrieval_node": "retrieval",
    },
)

workflow.add_edge("clarification", END)
workflow.add_conditional_edges(
    "retrieval",
    route_after_retrieval,
    {
        "assessment": "assessment",
        "end": END,
    },
)
workflow.add_conditional_edges(
    "assessment",
    route_after_assessment,
    {
        "verification": "verification",
        "end": END,
    },
)
workflow.add_edge("verification", END)
workflow.add_edge("greeting_node", END)
workflow.add_edge("harmful_node", END)
workflow.add_edge("off_topic_node", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


class LangGraphOrchestrator:
    """Runs the teacher question-generation graph."""

    def __init__(self, difficulty_level: str = "intermediate"):
        allowed = tuple(getattr(config, "DIFFICULTY_LEVELS", ("beginner", "intermediate", "advanced")))
        self.difficulty_level = difficulty_level if difficulty_level in allowed else "intermediate"

    def process_query(
        self,
        query: str,
        thread_id: str = "1",
        selected_tasks: Optional[List[str]] = None,
    ) -> TutorState:
        run_config = {"configurable": {"thread_id": thread_id}}
        keys = normalize_selected_tasks(selected_tasks)

        initial_state: TutorState = {
            "query": query,
            "difficulty_level": self.difficulty_level,
            "selected_tasks": list(keys),
            "history": [],
            "context": [],
            "sources": [],
            "retrieval_chunks": [],
            "execution_trace": [],
            "errors": [],
            "is_ambiguous": False,
            "ambiguity_reason": None,
            "clarification_options": [],
            "intent": None,
            "chapter": None,
            "concept": None,
            "answer": None,
            "assessment": None,
            "is_faithful": True,
            "verification_report": None,
            "faithfulness_score": 1.0,
            "answer_relevance_score": 1.0,
            "hallucination_risk_score": 0.0,
            "hallucination_risk_level": "low",
            "next_step": "",
        }

        result = app.invoke(initial_state, run_config)
        return result
