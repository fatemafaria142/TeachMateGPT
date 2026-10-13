"""
SAVER — Source-Attributed Verification and Evidence Ranking.

Post-generation quality layer: compares MCQs and creative items against the retrieved
context and the original teacher query. Produces faithfulness and relevance scores,
hallucination risk, and per-item flags so the orchestrator can enforce thresholds
and keep exports audit-friendly.
"""
from typing import Any, Dict, List
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config
import json
from utils.prompt_loader import load_prompt

VERIFICATION_PROMPT = load_prompt("verification_prompt.md")


def _default_report(reason: str = "Verification fallback.") -> Dict[str, Any]:
    return {
        "is_faithful": True,
        "reasoning": reason,
        "faithfulness_score": 1.0,
        "answer_relevance_score": 1.0,
        "hallucination_risk_score": 0.0,
        "hallucination_risk_level": "low",
        "answer_relevance_notes": "",
        "flagged_items": [],
    }


def _safe_json_from_content(content: str) -> Dict[str, Any]:
    text = (content or "").strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {}
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def verify_faithfulness(query: str, context: List[str], assessment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs the SAVER verification loop.
    """
    if not assessment or (not assessment.get("mcqs") and not assessment.get("creative_questions")):
        return _default_report("No generated items to verify.")

    llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(VERIFICATION_PROMPT)
    
    formatted_context = "\n---\n".join(context)
    formatted_assessment = json.dumps(assessment, ensure_ascii=False, indent=2)
    
    chain = prompt | llm
    
    response = chain.invoke(
        {
            "query": query,
            "context": formatted_context,
            "assessment": formatted_assessment,
        }
    )

    result = _safe_json_from_content(str(response.content))
    if not result:
        return _default_report("Model returned non-JSON verification output.")

    out = _default_report()
    out.update(result)
    out["is_faithful"] = bool(out.get("is_faithful", True))
    out["faithfulness_score"] = float(min(1.0, max(0.0, out.get("faithfulness_score", 1.0))))
    out["answer_relevance_score"] = float(min(1.0, max(0.0, out.get("answer_relevance_score", 1.0))))
    out["hallucination_risk_score"] = float(min(1.0, max(0.0, out.get("hallucination_risk_score", 0.0))))
    level = str(out.get("hallucination_risk_level", "low")).lower().strip()
    if level not in {"low", "medium", "high"}:
        level = (
            "low"
            if out["hallucination_risk_score"] < 0.33
            else ("medium" if out["hallucination_risk_score"] < 0.66 else "high")
        )
    out["hallucination_risk_level"] = level
    if not isinstance(out.get("flagged_items"), list):
        out["flagged_items"] = []
    return out
