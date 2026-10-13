"""
Skill registry — string IDs to agent functions for the LangGraph.

Central map from names like ``intent.classify`` and ``retrieval.context`` to the
underlying Python callables. ``run_skill`` wraps each call and returns
``(result, error)`` so failures land in ``TutorState.errors`` instead of aborting
the whole session.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from agents.ambiguity import check_ambiguity
from agents.assessment import generate_assessment
from agents.clarification import generate_clarification
from agents.concept import detect_concept
from agents.intent import classify_intent
from agents.retrieval import retrieve_context
from agents.verification import verify_faithfulness


@dataclass(frozen=True)
class Skill:
    id: str
    description: str
    fn: Callable[..., Any]


def _default_registry() -> Dict[str, Skill]:
    return {
        "intent.classify": Skill(
            id="intent.classify",
            description="Classify user intent (greeting, harmful, science_query, etc.).",
            fn=classify_intent,
        ),
        "ambiguity.check": Skill(
            id="ambiguity.check",
            description="Detect vague or underspecified questions.",
            fn=check_ambiguity,
        ),
        "clarification.generate": Skill(
            id="clarification.generate",
            description="Produce a clarifying question or options for the student.",
            fn=generate_clarification,
        ),
        "retrieval.context": Skill(
            id="retrieval.context",
            description="Hybrid RAG retrieval from the textbook vector store.",
            fn=retrieve_context,
        ),
        "concept.detect": Skill(
            id="concept.detect",
            description="Infer chapter/concept labels from retrieved context.",
            fn=detect_concept,
        ),
        "assessment.generate": Skill(
            id="assessment.generate",
            description="Build selected question types from retrieved context (args: context, difficulty, selected_tasks).",
            fn=generate_assessment,
        ),
        "verification.faithfulness": Skill(
            id="verification.faithfulness",
            description="Verify if the generated assessment is grounded in context (SAVER strategy).",
            fn=verify_faithfulness,
        ),
    }


class SkillRegistry:
    def __init__(self, skills: Optional[Dict[str, Skill]] = None):
        self._skills: Dict[str, Skill] = dict(skills or _default_registry())

    def list_skills(self) -> List[Skill]:
        return list(self._skills.values())

    def get(self, skill_id: str) -> Skill:
        if skill_id not in self._skills:
            raise KeyError(f"Unknown skill: {skill_id}")
        return self._skills[skill_id]

    def run(self, skill_id: str, *args: Any, **kwargs: Any) -> Tuple[Any, Optional[str]]:
        skill = self.get(skill_id)
        try:
            return skill.fn(*args, **kwargs), None
        except Exception as exc:
            return None, f"{skill_id}: {exc!s}"


_registry: Optional[SkillRegistry] = None


def get_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


def run_skill(skill_id: str, *args: Any, **kwargs: Any) -> Tuple[Any, Optional[str]]:
    return get_registry().run(skill_id, *args, **kwargs)
