"""
Chapter and concept labeling from retrieved context.

After retrieval, reads the joined textbook snippets and returns Bangla ``chapter`` and
``concept`` strings for UI display, logging, and exported run artifacts—without
changing retrieval itself.
"""
from __future__ import annotations

import json

from utils.llm_client import chat_completion
from utils.prompt_loader import load_prompt


def detect_concept(context: str) -> dict:
    """Identify the chapter and concept from the retrieved context."""
    system_prompt = load_prompt("concept_prompt.md")
    result_str = chat_completion(system_prompt, context[:2000], json_mode=True)
    try:
        data = json.loads(result_str)
        return data if isinstance(data, dict) else {"chapter": "অজানা", "concept": "অজানা"}
    except json.JSONDecodeError:
        return {"chapter": "অজানা", "concept": "অজানা"}
