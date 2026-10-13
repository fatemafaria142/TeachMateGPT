"""
Teacher-assistant agents — LangGraph nodes, retrieval, assessment, and verification.

This package holds the callable units wired through ``agents.skills.registry`` and
``agents.orchestrator``: intent and ambiguity handling, hybrid textbook retrieval,
concept labeling, assessment generation (including the ``agents.assessment``
exam specialists), SAVER verification, plus ``TutorState`` and the runnable graph
definition. Each agent lives in its own subpackage (``agents.<name>``) exposing
its public API from ``__init__.py``, backed by ``agent.py`` (and ``tools.py``
where helper logic is substantial enough to split out).
"""
