"""RAGAS wrapper: faithfulness, answer relevancy, context precision, context recall.

Context *recall* is reference-based — it needs gold contexts per query. Those live
in ``eval/goldset.json`` and must be authored by hand (see
IMPLEMENTATION_PLAN.md 3). Without them, recall is reported as ``None`` rather
than silently substituted, and the harness records ``ragas_reference=False``.

Note the existing Table 1 is captioned "Reference-Free RAGAS Evaluation" while
also reporting Context Recall. Those are inconsistent: recall is not
reference-free. Building the goldset resolves it in the right direction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass
class RagasScores:
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None

    def to_dict(self) -> dict[str, Optional[float]]:
        return dict(self.__dict__)


def available() -> bool:
    try:
        import ragas
        return True
    except ImportError:
        return False


def score_dataset(samples: Sequence[dict[str, Any]], *, offline: bool) -> list[RagasScores]:
    """Score a list of {question, answer, contexts, ground_truth_contexts?}.

    In offline mode returns all-``None`` — stub text carries no signal and
    fabricating numbers here would be the single easiest way to end up with
    invented results in the paper.
    """
    if offline:
        return [RagasScores() for _ in samples]

    if not available():
        raise RuntimeError(
            "ragas is not installed. `uv pip install -r requirements-eval.txt` "
            "(ragas was split out of requirements.txt)."
        )

    import config
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    if not samples:
        return []

    judge = LangchainLLMWrapper(ChatOpenAI(model=config.OPENAI_MODEL, temperature=0))
    from eval.providers import get_embeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    embedder = LangchainEmbeddingsWrapper(get_embeddings(offline=False))

    has_refs = all(s.get("ground_truth_contexts") for s in samples)
    metrics: list[Any] = [
        Faithfulness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=embedder),
        LLMContextPrecisionWithoutReference(llm=judge),
    ]
    if has_refs:
        metrics.append(LLMContextRecall(llm=judge))

    rows = []
    for s in samples:
        row = SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=list(s["contexts"]) or [""],
        )
        if has_refs:
            row.reference = "\n".join(s["ground_truth_contexts"])
        rows.append(row)

    result = evaluate(EvaluationDataset(samples=rows), metrics=metrics, llm=judge)
    df = result.to_pandas()

    out: list[RagasScores] = []
    for _, row in df.iterrows():
        out.append(RagasScores(
            faithfulness=_get(row, "faithfulness"),
            answer_relevancy=_get(row, "answer_relevancy"),
            context_precision=_get(row, "llm_context_precision_without_reference"),
            context_recall=_get(row, "context_recall") if has_refs else None,
        ))
    return out


def _get(row: Any, key: str) -> Optional[float]:
    try:
        val = row[key]
    except (KeyError, IndexError):
        return None
    return None if val is None or val != val else float(val)
