"""
Skills package — safe dispatch into registered agents.

Exposes ``run_skill``, ``get_registry``, and types used by the orchestrator; the
concrete skill IDs and callables live in ``agents.skills.registry``.
"""
from agents.skills.registry import Skill, SkillRegistry, get_registry, run_skill

__all__ = ["Skill", "SkillRegistry", "get_registry", "run_skill"]
