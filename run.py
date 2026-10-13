#!/usr/bin/env python3
"""
CLI: teacher / researcher query → hybrid retrieval → MCQ + Srijonshil question JSON on disk.

Usage:
  python run.py "…"
  python run.py --session my_eval_jan "…"

Output:
  - ``output/run_<run_id>.json`` — one pretty-printed object (includes ``run_id``, ``session_id``).
  - ``output/<TEACHER_SESSION_SUBDIR>/<session_id>.jsonl`` — same object appended as one line per run.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14",
    category=UserWarning,
)

import config

from agents.orchestrator import LangGraphOrchestrator
from utils.teacher_turn_export import (
    build_teacher_turn_record,
    new_run_id,
    save_teacher_turn_json,
)


def main(argv: list[str] | None = None) -> int:
    choices = list(getattr(config, "DIFFICULTY_LEVELS", ("beginner", "intermediate", "advanced")))
    parser = argparse.ArgumentParser(description="Teacher query → RAG → question set JSON")
    parser.add_argument(
        "query_parts",
        nargs="*",
        help="Query text. If empty, reads from stdin or prompt.",
    )
    parser.add_argument(
        "--level",
        default="intermediate",
        choices=choices,
        help="Question profile: beginner | intermediate | advanced",
    )
    parser.add_argument(
        "--session",
        default="cli-default",
        help="Session label: runs append to output/.../<session>.jsonl",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task keys (mcqs,creative_questions). Empty = all.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print JSON to stdout only; do not write files",
    )
    args = parser.parse_args(argv)

    if args.query_parts:
        query = " ".join(args.query_parts).strip()
    else:
        query = sys.stdin.read().strip() if not sys.stdin.isatty() else input("Query: ").strip()

    if not query:
        print("Empty query.", file=sys.stderr)
        return 2

    run_id = new_run_id()
    task_list: list[str] = []
    if (args.tasks or "").strip():
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    orch = LangGraphOrchestrator(difficulty_level=args.level)
    state = orch.process_query(query, thread_id=run_id, selected_tasks=task_list or None)

    if args.no_save:
        rec = build_teacher_turn_record(state, run_id=run_id, session_id=args.session, thread_id=run_id)
        print(json.dumps(rec, ensure_ascii=False, indent=2))
    else:
        path, rec, log_path = save_teacher_turn_json(
            state,
            thread_id=run_id,
            run_id=run_id,
            session_id=args.session,
        )
        out = {
            "run_id": rec["run_id"],
            "session_id": rec["session_id"],
            "run_json": str(path.resolve()),
            "session_jsonl": str(log_path.resolve()),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
