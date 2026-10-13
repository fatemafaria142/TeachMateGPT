"""
Exam task composer — runs specialists in order for the requested keys.

Given retrieved context and a subset of assessment tasks (for example ``mcqs``,
``creative_questions``), dispatches to the corresponding generators in
``specialists`` with per-task counts from config and merges their outputs into one
assessment dict.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import config
from agents.assessment.exam_specialists import SPECIALIST_ORDER


def _counts_map() -> Dict[str, int]:
    defaults = {
        "mcqs": 5,
        "creative_questions": 1,
    }
    c = getattr(config, "ASSESSMENT_COUNTS", None) or {}
    out: Dict[str, int] = {}
    for key in list(getattr(config, "ALL_ASSESSMENT_TASK_KEYS", ())):
        try:
            out[key] = max(0, int(c.get(key, defaults.get(key, 0))))
        except (TypeError, ValueError):
            out[key] = int(defaults.get(key, 0))
    return out


def generate_assessment_specialized(
    context: List[str],
    *,
    difficulty_level: str = "intermediate",
    selected_task_keys: Optional[Sequence[str]] = None,
    query_text: str = "",
) -> Dict[str, Any]:
    order = list(getattr(config, "ALL_ASSESSMENT_TASK_KEYS", ()))
    if not selected_task_keys:
        keys = order
    else:
        allowed = set(order)
        keys = [k for k in order if k in {str(x).strip() for x in selected_task_keys if str(x).strip() in allowed}]
        if not keys:
            keys = order
    key_set = set(keys)
    counts = _counts_map()
    out: Dict[str, Any] = {}
    ctx = context or []
    lvl = difficulty_level

    for key, fn in SPECIALIST_ORDER:
        if key not in key_set:
            continue
        n = counts.get(key, 0)
        if n <= 0:
            out[key] = []
            continue
        try:
            chunk = fn(ctx, lvl, n, query_text=query_text)
            out[key] = chunk if isinstance(chunk, list) else []
        except Exception:
            out[key] = []

    return out
