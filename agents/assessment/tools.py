"""Helpers for selecting assessment task keys and enforcing per-task counts."""
from typing import Any, Dict, List, Optional, Sequence

import config


def _all_keys() -> List[str]:
    k = getattr(config, "ALL_ASSESSMENT_TASK_KEYS", None)
    if k:
        return list(k)
    return ["mcqs", "creative_questions"]


def normalize_selected_tasks(raw: Optional[Sequence[str]]) -> List[str]:
    allowed = set(_all_keys())
    if not raw:
        return _all_keys()
    out = [str(x).strip() for x in raw if str(x).strip() in allowed]
    order = _all_keys()
    ordered = [k for k in order if k in out]
    return ordered or _all_keys()


def _assessment_counts() -> Dict[str, int]:
    defaults = {
        "mcqs": 5,
        "creative_questions": 1,
    }
    c = getattr(config, "ASSESSMENT_COUNTS", None)
    out: Dict[str, int] = {}
    for k in _all_keys():
        try:
            raw = (c or {}).get(k, defaults[k]) if isinstance(c, dict) else defaults[k]
            out[k] = max(0, int(raw))
        except (TypeError, ValueError, KeyError):
            out[k] = int(defaults.get(k, 0))
    return out


def clip_assessment_to_counts(data: Dict[str, Any], counts: Dict[str, int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, n in counts.items():
        v = data.get(k)
        if not isinstance(v, list):
            v = []
        out[k] = v[:n] if n else []
    return out
