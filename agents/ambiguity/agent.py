"""
Ambiguity detection before retrieval and assessment.

Evaluates whether the teacher’s request is specific enough (chapter, concept, or
task type). When the query is too vague, the graph can branch to clarification
instead of pulling weak or misaligned evidence from the textbook store.
"""
from __future__ import annotations

import json
import re

from utils.llm_client import chat_completion
from utils.prompt_loader import load_prompt


_GENERATION_WORDS = (
    "mcq",
    "cq",
    "সৃজনশীল",
    "প্রশ্ন",
    "তৈরি",
    "দিন",
    "করো",
    "করে",
)

_TOPIC_HINTS = (
    "অধ্যায়",
    "অধ্যায়",
    "ধারণা",
    "কনসেপ্ট",
    "বিষয়",
    "নিয়ে",
    "নিয়ে",
    "সম্পর্কে",
    "ভিত্তিক",
    "উপর",
)


def _looks_specific_bangla_request(text: str) -> bool:
    q = (text or "").strip().lower()
    if len(q) < 4:
        return False
    has_generation = any(w in q for w in _GENERATION_WORDS)
    has_topic_hint = any(w in q for w in _TOPIC_HINTS)
    has_topic_phrase = bool(
        re.search(r"[“\"']?[\u0980-\u09ffA-Za-z0-9\s]{2,40}[”\"']?\s*(নিয়ে|নিয়ে|সম্পর্কে|ভিত্তিক)", q)
    )
    return has_generation and (has_topic_hint or has_topic_phrase)


def check_ambiguity(student_query: str) -> dict:
    """Check if the student query is too vague."""
    if _looks_specific_bangla_request(student_query):
        return {"is_ambiguous": False, "reason": "", "options": []}

    system_prompt = load_prompt("ambiguity_prompt.md")
    result_str = chat_completion(system_prompt, student_query, json_mode=True)
    try:
        data = json.loads(result_str)
        if not isinstance(data, dict):
            return {"is_ambiguous": False, "reason": "", "options": []}
        if data.get("is_ambiguous") and _looks_specific_bangla_request(student_query):
            return {"is_ambiguous": False, "reason": "", "options": []}
        return data
    except json.JSONDecodeError:
        return {"is_ambiguous": False, "reason": "", "options": []}
