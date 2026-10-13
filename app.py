"""
Streamlit UI — conversational teacher assistant with chat history, task pickers, and per-task exports.
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List, Optional

import streamlit as st

warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14",
    category=UserWarning,
)

import config
from agents.orchestrator import LangGraphOrchestrator
from utils.teacher_turn_export import (
    new_run_id,
    save_teacher_turn_json,
    sanitize_session_id,
    session_jsonl_path,
)

st.set_page_config(
    page_title="Teacher curriculum assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

LEVEL_OPTIONS = [
    ("beginner", "Beginner"),
    ("intermediate", "Intermediate"),
    ("advanced", "Advanced"),
]

LEVEL_HELP_EN = {
    "beginner": "Simpler wording and more direct recall from the textbook context.",
    "intermediate": "Standard classroom depth with light reasoning.",
    "advanced": "Deeper reasoning, comparison, and application-style stems.",
}

TASK_OPTIONS: List[tuple[str, str]] = [
    ("mcqs", "MCQ (multiple choice)"),
    ("creative_questions", "Creative / Srijonshil (সৃজনশীল)"),
]

TASK_EXPANDER_TITLES = {
    "mcqs": "MCQs",
    "creative_questions": "Creative (Srijonshil)",
}


def _inject_chat_css() -> None:
    """Theme: light blue + soft yellow only (backgrounds, borders, accents)."""
    st.markdown(
        """
        <style>
            /* —— Layout: clear space below app chrome (fixes title/toolbar overlap) —— */
            [data-testid="stAppViewContainer"] > .main .block-container {
                padding-top: 1.5rem !important;
                padding-bottom: 2rem !important;
                max-width: 52rem;
                margin-left: auto;
                margin-right: auto;
            }
            [data-testid="stAppViewContainer"] > .main,
            [data-testid="stAppViewContainer"] .stMain {
                background: linear-gradient(180deg, #e0f2fe 0%, #f0f9ff 38%, #fffbeb 72%, #fefce8 100%);
            }
            .stMain .block-container {
                padding-top: 1.5rem !important;
                padding-bottom: 2rem !important;
                max-width: 52rem;
                margin-left: auto;
                margin-right: auto;
            }
            [data-testid="stHeader"] {
                background: linear-gradient(90deg, #e0f2fe 0%, #fef9c3 100%) !important;
                border-bottom: 1px solid #bae6fd !important;
            }
            [data-testid="stDecoration"] { display: none; }

            /* Sidebar */
            [data-testid="stSidebar"] {
                background: linear-gradient(195deg, #f0f9ff 0%, #fffbeb 55%, #fefce8 100%) !important;
                border-right: 2px solid #bae6fd !important;
            }
            [data-testid="stSidebar"] .stMarkdown h1,
            [data-testid="stSidebar"] .stMarkdown h2,
            [data-testid="stSidebar"] .stMarkdown h3 { color: #0369a1 !important; }
            /* Avoid painting every sidebar span (breaks BaseWeb multiselect tag layout). */
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] .stMarkdown span,
            [data-testid="stSidebar"] [data-testid="stCaption"],
            [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
                color: #0c4a6e !important;
            }
            [data-testid="stSidebar"] hr { border-color: #bae6fd !important; }

            /* Multiselect (task types): keep tags in flow — unfocused input is position:absolute in Base Web and can overlap tags in a narrow sidebar. */
            [data-testid="stSidebar"] [data-baseweb="select"]:has([data-baseweb="tag"]) > div:first-child {
                max-height: 16rem !important;
                min-height: 3.5rem !important;
                overflow-y: auto !important;
                align-items: flex-start !important;
            }
            [data-testid="stSidebar"] [data-baseweb="select"]:has([data-baseweb="tag"]) [data-baseweb="value"] {
                display: flex !important;
                flex-wrap: wrap !important;
                align-content: flex-start !important;
                align-items: flex-start !important;
                gap: 0.35rem 0.4rem !important;
                row-gap: 0.45rem !important;
                overflow: visible !important;
                min-height: 2.75rem !important;
            }
            [data-testid="stSidebar"] [data-baseweb="select"]:has([data-baseweb="tag"]) [data-baseweb="tag"] {
                position: relative !important;
                top: auto !important;
                transform: none !important;
                margin-right: 0.35rem !important;
                margin-bottom: 0.2rem !important;
            }

            /* Sidebar app title (above Settings) */
            [data-testid="stSidebar"] .page-title-block {
                background: linear-gradient(135deg, #dbeafe 0%, #fef9c3 55%, #e0f2fe 100%);
                border: 2px solid #7dd3fc;
                border-radius: 16px;
                padding: 0.75rem 0.95rem;
                margin: 0 0 1rem 0;
                box-shadow: 0 2px 0 #fde047;
            }
            [data-testid="stSidebar"] .page-title-block h1 {
                margin: 0 !important;
                padding: 0 !important;
                font-size: 1.2rem !important;
                line-height: 1.25 !important;
                font-weight: 700 !important;
                color: #075985 !important;
                letter-spacing: -0.02em !important;
            }

            /* Chat bubbles */
            [data-testid="stChatMessage"] {
                background-color: #f0f9ff !important;
                border: 2px solid #bae6fd !important;
                border-radius: 16px !important;
                padding: 0.65rem 1rem !important;
                margin-bottom: 0.55rem !important;
                box-shadow: 0 2px 8px rgba(3, 105, 161, 0.08);
            }
            [data-testid="stChatMessage"][data-testid-avatar="user"] {
                background-color: #fffbeb !important;
                border-color: #fbbf24 !important;
                border-left: 5px solid #0ea5e9 !important;
            }
            [data-testid="stChatMessage"][data-testid-avatar="assistant"] {
                background-color: #e0f2fe !important;
                border-color: #7dd3fc !important;
                border-left: 5px solid #facc15 !important;
            }
            [data-testid="stChatInput"] {
                border-radius: 16px !important;
                border: 2px solid #7dd3fc !important;
                background-color: #fffbeb !important;
                box-shadow: 0 2px 10px rgba(14, 165, 233, 0.12);
            }
            [data-testid="stChatInput"] textarea {
                color: #0c4a6e !important;
            }

            /* Expanders + buttons (yellow / blue only) */
            .streamlit-expanderHeader {
                background: #fef9c3 !important;
                border: 1px solid #bae6fd !important;
                border-radius: 10px !important;
                color: #075985 !important;
            }
            section[data-testid="stSidebar"] button[kind="primary"],
            section[data-testid="stSidebar"] button[kind="secondary"] {
                background-color: #fef08a !important;
                color: #0c4a6e !important;
                border: 2px solid #38bdf8 !important;
            }
            .stDownloadButton button {
                background-color: #e0f2fe !important;
                color: #075985 !important;
                border: 2px solid #facc15 !important;
            }

            /* Spinner + alerts in palette */
            .stSpinner > div { border-top-color: #0ea5e9 !important; }
            div[data-testid="stAlert"] {
                background-color: #fef9c3 !important;
                border: 1px solid #38bdf8 !important;
                color: #0c4a6e !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    if "teacher_session_id" not in st.session_state:
        st.session_state.teacher_session_id = new_run_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "awaiting_clarification" not in st.session_state:
        st.session_state.awaiting_clarification = False
    if "vague_query" not in st.session_state:
        st.session_state.vague_query = None


def _trace_nodes(state: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for step in state.get("execution_trace") or []:
        if isinstance(step, dict) and step.get("node"):
            out.append(str(step["node"]))
    return out


def _ran_clarification(state: Dict[str, Any]) -> bool:
    nodes = _trace_nodes(state)
    return "clarification" in nodes and "assessment" not in nodes


def _has_nonempty_assessment(asm: Any) -> bool:
    if not isinstance(asm, dict):
        return False
    return any(isinstance(asm.get(k), list) and len(asm[k]) > 0 for k in asm)


def _assistant_markdown(state: Dict[str, Any], *, rec: Optional[Dict[str, Any]] = None) -> str:
    asm = state.get("assessment") or {}
    selected = state.get("selected_tasks") or list(getattr(config, "ALL_ASSESSMENT_TASK_KEYS", ()))
    parts: List[str] = []
    if _has_nonempty_assessment(asm):
        bits = []
        for k in selected:
            if k in asm and isinstance(asm[k], list):
                bits.append(f"{TASK_EXPANDER_TITLES.get(k, k)}: **{len(asm[k])}**")
        parts.append(
            "**Question set ready** (textbook-grounded). " + " · ".join(bits) if bits else "**Question set ready.**"
        )
        if state.get("chapter") or state.get("concept"):
            parts.append(
                f"_Focus:_ chapter `{state.get('chapter')}` · concept `{state.get('concept')}`"
            )
        parts.append(f"_Tasks this run:_ `{', '.join(selected)}` · _Difficulty:_ `{state.get('difficulty_level')}`")
    elif state.get("answer"):
        parts.append(state["answer"])
    else:
        parts.append("No assistant text was returned for this turn.")
    if rec and rec.get("run_id"):
        parts.append(f"_Run ID:_ `{rec['run_id']}`")
    return "\n\n".join(parts)


def _render_assessment_expanders(asm: Dict[str, Any]) -> None:
    if not asm or not _has_nonempty_assessment(asm):
        return
    for key in getattr(config, "ALL_ASSESSMENT_TASK_KEYS", ()):
        if key not in asm:
            continue
        items = asm.get(key) or []
        if not items:
            continue
        title = TASK_EXPANDER_TITLES.get(key, key)
        with st.expander(f"{title} ({len(items)})", expanded=False):
            st.json(items)


def _clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.awaiting_clarification = False
    st.session_state.vague_query = None


def _sidebar_selected_tasks() -> List[str]:
    keys = [k for k, _ in TASK_OPTIONS]
    default = keys
    picked = st.multiselect(
        "Question types to generate",
        options=keys,
        default=default,
        format_func=lambda k: dict(TASK_OPTIONS).get(k, k),
        help="Only selected formats are sent to the model and saved under output/tasks/<type>/",
    )
    return picked if picked else default


def main() -> None:
    _inject_chat_css()
    _init_session_state()

    with st.sidebar:
        st.markdown(
            """
            <div class="page-title-block">
                <h1>Teacher curriculum assistant</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.header("Settings")
        idx = st.selectbox(
            "Difficulty level",
            range(len(LEVEL_OPTIONS)),
            format_func=lambda i: LEVEL_OPTIONS[i][1],
        )
        level_key = LEVEL_OPTIONS[idx][0]
        st.caption(LEVEL_HELP_EN.get(level_key, ""))
        st.divider()
        selected_tasks = _sidebar_selected_tasks()
        st.caption(
            f"Per-task exports: `{config.RUN_OUTPUT_DIR}/{config.TEACHER_TASK_SUBROOT}/<task>/` "
            f"(run JSON + session JSONL)."
        )
        st.divider()
        st.subheader("Session & logs")
        st.caption(
            f"Full-turn log: `{config.RUN_OUTPUT_DIR}/{config.TEACHER_SESSION_SUBDIR}/<session>.jsonl`"
        )
        sid_input = st.text_input(
            "Session ID",
            value=st.session_state.teacher_session_id,
            help="Reuse one ID for one evaluation batch; change for a new log file.",
        )
        if sid_input.strip() != st.session_state.teacher_session_id:
            st.session_state.teacher_session_id = sanitize_session_id(sid_input)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("New session", use_container_width=True):
                st.session_state.teacher_session_id = new_run_id()
                _clear_chat()
                st.rerun()
        with c2:
            if st.button("Clear chat", use_container_width=True):
                _clear_chat()
                st.rerun()
        st.caption(f"Session: `{st.session_state.teacher_session_id}`")
        st.caption(f"Output: `{config.RUN_OUTPUT_DIR}`")
        full_log = session_jsonl_path(st.session_state.teacher_session_id)
        if full_log.is_file():
            st.download_button(
                "Download session log (JSONL)",
                data=full_log.read_text(encoding="utf-8"),
                file_name=full_log.name,
                mime="application/x-ndjson",
                key="dl_sidebar_jsonl",
            )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content") or "")
            if msg.get("assessment"):
                _render_assessment_expanders(msg["assessment"])
            if msg.get("run_id"):
                st.caption(f"Run ID: `{msg['run_id']}`")

    if prompt := st.chat_input("Type your request (Bangla or English)…"):
        user_visible = str(prompt).strip()
        if st.session_state.awaiting_clarification and st.session_state.vague_query:
            effective_query = (
                f"{st.session_state.vague_query}\n\n"
                f"[Teacher clarification / follow-up]: {user_visible}"
            )
        else:
            effective_query = user_visible

        st.session_state.messages.append({"role": "user", "content": user_visible})
        with st.chat_message("user"):
            st.markdown(user_visible)

        run_id = new_run_id()
        session_id = st.session_state.teacher_session_id

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating…"):
                orch = LangGraphOrchestrator(difficulty_level=level_key)
                state = orch.process_query(
                    effective_query.strip(),
                    thread_id=run_id,
                    selected_tasks=selected_tasks,
                )
            path, rec, log_path = save_teacher_turn_json(
                state,
                thread_id=run_id,
                run_id=run_id,
                session_id=session_id,
            )
            md = _assistant_markdown(state, rec=rec)
            st.markdown(md)
            asm = state.get("assessment") or {}
            if _has_nonempty_assessment(asm):
                _render_assessment_expanders(asm)
            st.caption(f"Full run: `{path.name}` · Session log: `{log_path.name}`")
            tops = rec.get("task_output_paths") or {}
            if tops:
                st.caption("Per-task files: " + " · ".join(f"`{k}`" for k in tops))
            if state.get("errors"):
                st.error(state["errors"])
            st.download_button(
                "Download this run (JSON)",
                data=json.dumps(rec, ensure_ascii=False, indent=2),
                file_name=path.name,
                mime="application/json",
                key=f"dl_run_{run_id}",
            )

        if _ran_clarification(state):
            st.session_state.awaiting_clarification = True
            st.session_state.vague_query = effective_query.strip()
        else:
            st.session_state.awaiting_clarification = False
            st.session_state.vague_query = None

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": md,
                "assessment": asm if _has_nonempty_assessment(asm) else None,
                "run_id": rec.get("run_id"),
            }
        )


if __name__ == "__main__":
    main()
