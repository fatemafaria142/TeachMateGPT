"""
Assessment generation from retrieved textbook context.

Orchestrates construction of the requested task types (for example MCQs and সৃজনশীল)
at a chosen difficulty—either via one monolithic prompt or by delegating to the
specialist pipeline—respecting ``ASSESSMENT_COUNTS`` and emitting structured dicts
ready for verification and JSON export.
"""
import json
from typing import Any, Dict, Optional, Sequence

import config
from utils.llm_client import chat_completion
from utils.prompt_loader import load_prompt
from agents.assessment.tools import (
    _assessment_counts,
    clip_assessment_to_counts,
    normalize_selected_tasks,
)

__all__ = ["generate_assessment", "normalize_selected_tasks"]


def generate_assessment(
    context: list,
    difficulty_level: str = "intermediate",
    selected_tasks: Optional[Sequence[str]] = None,
    query_text: str = "",
) -> dict:
    """Generate only ``selected_tasks`` formats; omit unrequested keys from the result dict."""
    keys = normalize_selected_tasks(selected_tasks)
    counts_full = _assessment_counts()
    counts = {k: counts_full[k] for k in keys}
    lvl = (difficulty_level or "intermediate").strip().lower()
    if lvl not in getattr(config, "DIFFICULTY_LEVELS", ()):
        lvl = "intermediate"
    label = getattr(config, "LEVEL_LABELS_BN", {}).get(lvl, lvl)

    mode = getattr(config, "ASSESSMENT_GENERATION_MODE", "monolithic").strip().lower()
    if mode == "specialized_agents":
        from agents.assessment.exam_composer import generate_assessment_specialized

        raw = generate_assessment_specialized(
            context,
            difficulty_level=lvl,
            selected_task_keys=keys,
            query_text=query_text,
        )
        return clip_assessment_to_counts(raw, counts)

    system_prompt = load_prompt("assessment_prompt.md")
    user_msg = (
        f"প্রশ্নের স্তর: {label} (key: {lvl}).\n"
        f"শুধুমাত্র এই JSON কীগুলো ব্যবহার করো (অন্য কী দিও না): {json.dumps(keys, ensure_ascii=False)}\n"
        f"প্রতিটি কী-তে লক্ষ্য সংখ্যা: {json.dumps(counts, ensure_ascii=False)}\n\n"
        f"Context:\n{' '.join(context)}"
    )
    result_str = chat_completion(system_prompt, user_msg, json_mode=True)
    try:
        data = json.loads(result_str)
        if not isinstance(data, dict):
            return {}
        out: Dict[str, Any] = {}
        for k in keys:
            v = data.get(k)
            out[k] = v if isinstance(v, list) else []
        return clip_assessment_to_counts(out, counts)
    except Exception:
        return {k: [] for k in keys}
