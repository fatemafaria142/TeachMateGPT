"""Teacher turns → per-run JSON files + append-only session JSONL for evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple
from uuid import uuid4

import config

from agents.assessment import normalize_selected_tasks


def new_thread_id() -> str:
    return str(uuid4())


def new_run_id() -> str:
    """Stable primary key for one generation run (evaluation / joins)."""
    return str(uuid4())


def sanitize_session_id(raw: str | None) -> str:
    """Filename-safe session label (allows Bangla; strips path/hostile chars)."""
    s = (raw or "default").strip() or "default"
    for ch in '<>:"/\\|?*\n\r\t':
        s = s.replace(ch, "_")
    s = s.strip() or "default"
    return s[:120]


def session_jsonl_path(session_id: str) -> Path:
    root = Path(getattr(config, "RUN_OUTPUT_DIR", "./output"))
    sub = root / getattr(config, "TEACHER_SESSION_SUBDIR", "teacher_sessions")
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{sanitize_session_id(session_id)}.jsonl"


def append_run_to_session_log(record: Dict[str, Any], *, session_id: str) -> Path:
    """Append one JSON object as a single line (UTF-8). Same session = same file, many lines."""
    path = session_jsonl_path(session_id)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
    return path


def iter_session_runs(session_id: str) -> Iterator[Dict[str, Any]]:
    """Read all runs in a session (for evaluation scripts)."""
    path = session_jsonl_path(session_id)
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_session_runs(session_id: str) -> List[Dict[str, Any]]:
    return list(iter_session_runs(session_id))


def build_teacher_turn_record(
    state: Dict[str, Any],
    *,
    run_id: str,
    session_id: str,
    thread_id: str,
) -> Dict[str, Any]:
    """Schema for research / eval. ``run_id`` joins to ``run_<run_id>.json`` filename."""
    now = datetime.now(timezone.utc)
    sid = sanitize_session_id(session_id)
    selected = normalize_selected_tasks(state.get("selected_tasks"))
    full_counts = getattr(config, "ASSESSMENT_COUNTS", {}) or {}
    counts_slice = {k: full_counts[k] for k in selected if k in full_counts}
    order = tuple(getattr(config, "ALL_ASSESSMENT_TASK_KEYS", ()))
    raw_asm = state.get("assessment")
    if isinstance(raw_asm, dict):
        assessment_out = {k: raw_asm[k] for k in order if k in raw_asm}
    else:
        assessment_out = raw_asm
    return {
        "run_id": run_id,
        "session_id": sid,
        "exported_at": now.isoformat().replace("+00:00", "Z"),
        "thread_id": thread_id,
        "query": state.get("query"),
        "difficulty_level": state.get("difficulty_level"),
        "selected_tasks": selected,
        "intent": state.get("intent"),
        "is_ambiguous": state.get("is_ambiguous"),
        "ambiguity_reason": state.get("ambiguity_reason"),
        "chapter": state.get("chapter"),
        "concept": state.get("concept"),
        "answer": state.get("answer"),
        "sources": state.get("sources") or [],
        "retrieval_chunks": state.get("retrieval_chunks") or [],
        "assessment": assessment_out,
        "verification_report": state.get("verification_report"),
        "faithfulness_score": state.get("faithfulness_score"),
        "answer_relevance_score": state.get("answer_relevance_score"),
        "hallucination_risk_score": state.get("hallucination_risk_score"),
        "hallucination_risk_level": state.get("hallucination_risk_level"),
        "assessment_counts": counts_slice,
        "execution_trace": state.get("execution_trace") or [],
        "errors": state.get("errors") or [],
    }


def write_task_folder_artifacts(record: Dict[str, Any]) -> Dict[str, str]:
    """
    For each ``selected_tasks`` key, write ``<RUN_DIR>/<TEACHER_TASK_SUBROOT>/<task>/run_<run_id>.json``
    and append the same object to ``<task>/<session_id>.jsonl``.
    """
    paths: Dict[str, str] = {}
    run_id = record["run_id"]
    sid = record["session_id"]
    asm = record.get("assessment") or {}
    selected = record.get("selected_tasks") or []
    root = Path(getattr(config, "RUN_OUTPUT_DIR", "./output")) / getattr(
        config, "TEACHER_TASK_SUBROOT", "tasks"
    )
    for key in selected:
        items = asm.get(key)
        if not isinstance(items, list):
            items = []
        if not items:
            continue
        sub = root / key
        sub.mkdir(parents=True, exist_ok=True)
        task_rec: Dict[str, Any] = {
            "run_id": run_id,
            "session_id": sid,
            "task_key": key,
            "difficulty_level": record.get("difficulty_level"),
            "exported_at": record.get("exported_at"),
            "query": record.get("query"),
            "selected_tasks": selected,
            "target_count": (record.get("assessment_counts") or {}).get(key),
            "sources": record.get("sources") or [],
            "retrieval_chunks": record.get("retrieval_chunks") or [],
            "chapter": record.get("chapter"),
            "concept": record.get("concept"),
            "items": items,
        }
        jp = sub / f"run_{run_id}.json"
        jp.write_text(json.dumps(task_rec, ensure_ascii=False, indent=2), encoding="utf-8")
        jlog = sub / f"{sid}.jsonl"
        with jlog.open("a", encoding="utf-8") as f:
            f.write(json.dumps(task_rec, ensure_ascii=False) + "\n")
        paths[key] = str(jp.resolve())
    return paths


def save_teacher_turn_json(
    state: Dict[str, Any],
    *,
    thread_id: str,
    run_id: str | None = None,
    session_id: str | None = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Write ``output/run_<run_id>.json`` and append the same object to
    ``output/<TEACHER_SESSION_SUBDIR>/<session_id>.jsonl``.
    """
    rid = run_id or new_run_id()
    sid = sanitize_session_id(session_id)
    out_dir = Path(getattr(config, "RUN_OUTPUT_DIR", "./output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    record = build_teacher_turn_record(state, run_id=rid, session_id=sid, thread_id=thread_id)
    record["task_output_paths"] = write_task_folder_artifacts(record)
    path = out_dir / f"run_{rid}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = append_run_to_session_log(record, session_id=sid)
    return path, record, log_path
