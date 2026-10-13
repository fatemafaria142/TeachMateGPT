"""
Clarification messages when a query needs narrowing.

Consumes the ambiguity agent’s reason and optional hint options and produces a short,
polite follow-up—usually Bangla—so the teacher can specify topic, chapter, or desired
question formats before retrieval runs.
"""
from __future__ import annotations

from utils.llm_client import chat_completion
from utils.prompt_loader import load_prompt


def generate_clarification(reason: str, options: list) -> str:
    """Generate a polite Bangla clarification message."""
    system_prompt = load_prompt("clarification_prompt.md")
    user_msg = f"Reason: {reason}\nOptions: {', '.join(options)}"
    return chat_completion(system_prompt, user_msg)
