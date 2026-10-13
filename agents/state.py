"""
Shared LangGraph state schema for the teacher-assistant graph.

Declares ``TutorState``: the single state dict threaded through intent, ambiguity,
retrieval, assessment, and verification. Reducers append traces and errors instead of
overwriting, so each node reads and updates fields such as ``context``,
``retrieval_chunks``, ``assessment``, and ``verification_report`` consistently.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class TutorState(TypedDict):
    execution_trace: Annotated[List[Dict[str, Any]], operator.add]
    errors: Annotated[List[str], operator.add]

    query: str
    difficulty_level: str
    selected_tasks: List[str]
    next_step: str

    intent: Optional[str]

    is_ambiguous: bool
    ambiguity_reason: Optional[str]
    clarification_options: List[str]

    context: List[str]
    sources: List[str]
    retrieval_chunks: List[Dict[str, Any]]
    chapter: Optional[str]
    concept: Optional[str]
    answer: Optional[str]
    assessment: Optional[Dict[str, Any]]
    is_faithful: bool
    verification_report: Optional[Dict[str, Any]]
    faithfulness_score: float
    answer_relevance_score: float
    hallucination_risk_score: float
    hallucination_risk_level: str

    history: List[Dict[str, Any]]
