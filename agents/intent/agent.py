"""
Intent classification for the LangGraph teacher-assistant.

Reads each user message and assigns a routing label (for example ``science_query``,
``greeting``, ``off_topic``, ``harmful``) so the orchestrator only spends retrieval
and generation budget on genuine curriculum requests and can exit early for chit-chat
or unsafe input.
"""
from __future__ import annotations

from utils.llm_client import chat_completion
from utils.prompt_loader import load_prompt


def classify_intent(student_query: str) -> str:
    """Classify the student's intent."""
    system_prompt = load_prompt("intent_prompt.md")
    intent = chat_completion(system_prompt, student_query)
    label = (intent or "").strip().lower()
    return label if label else "science_query"
