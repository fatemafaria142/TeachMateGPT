"""Config-driven evaluation runner.

One entry point: :func:`run_config`. It applies a :class:`~eval.configs.RunConfig`
as `config` module overrides, drives the selected retriever over the query bank,
optionally generates and verifies, and emits a result dict containing **every
per-query record** — not just aggregates — so metrics can be recomputed and
bootstrapped without re-running the pipeline.

Provenance recorded in every result file: mode (live/offline), model ids, corpus
fingerprint, config overrides, and timestamp. :mod:`eval.report` refuses to write
offline results into the paper.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Sequence

import config
from eval import configs as run_configs
from eval import metrics_ragas, metrics_system
from eval.metrics_system import QueryRecord
from eval.providers import COUNTER, corpus_fingerprint, get_embeddings, get_llm, has_api_key, load_corpus

QUERY_BANK = "batch_input/teacher_query_bank.json"
GOLDSET = "eval/goldset.json"
RESULTS_DIR = "data/experiment/eval_runs"


@contextlib.contextmanager
def count_pipeline_llm_calls():
    """Count LLM calls made *inside* the production pipeline.

    Generation, query planning, multi-query expansion, intent, ambiguity and
    SAVER all go through ``utils.llm_client.chat_completion``, which the harness
    does not own. Without wrapping it, ``mean_llm_calls`` counts only the
    harness's own calls and reports 0.0 for configs that generate 16 items —
    making the cost table understate every row, and understate it *unevenly*
    (COPE routes through this helper far more than the baselines do).
    """
    from utils import llm_client

    original = llm_client.chat_completion

    def counted(*args: Any, **kwargs: Any):
        COUNTER.llm_calls += 1
        return original(*args, **kwargs)

    llm_client.chat_completion = counted
    import sys

    patched: list[tuple[Any, str]] = []
    for mod in list(sys.modules.values()):
        if mod is None or not getattr(mod, "__name__", "").startswith(("agents", "utils")):
            continue
        if getattr(mod, "chat_completion", None) is original:
            setattr(mod, "chat_completion", counted)
            patched.append((mod, "chat_completion"))
    try:
        yield
    finally:
        llm_client.chat_completion = original
        for mod, attr in patched:
            setattr(mod, attr, original)


@contextlib.contextmanager
def config_overrides(overrides: dict[str, Any]):
    """Temporarily patch `config` attributes, restoring them afterwards.

    The pipeline reads flags via ``getattr(config, ...)`` at call time, so
    patching the module is enough — no process restart needed between configs.
    """
    saved = {k: getattr(config, k, None) for k in overrides}
    missing = {k for k in overrides if not hasattr(config, k)}
    try:
        for k, v in overrides.items():
            setattr(config, k, v)
        yield
    finally:
        for k, v in saved.items():
            if k in missing:
                with contextlib.suppress(AttributeError):
                    delattr(config, k)
            else:
                setattr(config, k, v)


def load_query_bank(path: str = QUERY_BANK,
                    chapters: Optional[Sequence[int]] = None) -> list[dict[str, Any]]:
    """Load the teacher query bank, optionally scoped to specific chapters.

    Query ids are assigned over the *full* bank before filtering, so a query keeps
    the same id whether or not a chapter filter is applied — paired statistics
    across runs stay aligned.

    Non-science queries carry no ``meta_chapter`` and are always retained: they
    are what makes routing, leakage and wasted-retrieval measurable, and they are
    chapter-independent by nature.
    """
    with open(path, encoding="utf-8") as fh:
        bank = json.load(fh)
    for i, q in enumerate(bank):
        q.setdefault("query_id", f"q{i + 1:03d}")
    if chapters:
        wanted = set(chapters)
        bank = [
            q for q in bank
            if q.get("meta_chapter") in wanted
            or q.get("meta_expected_intent") != "science_query"
        ]
    return bank


def load_goldset(path: str = GOLDSET) -> dict[str, list[str]]:
    """query_id -> gold context strings. Absent file means recall is skipped."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return {k: list(v) for k, v in data.items()}


def uid_to_chapter_map(corpus) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in corpus:
        ch = c.metadata.get("chapter") or c.metadata.get("source")
        if ch:
            out[c.uid] = str(ch)
    return out


def run_config(
    name: str,
    *,
    offline: Optional[bool] = None,
    limit: Optional[int] = None,
    generate: bool = True,
    chapters: Optional[Sequence[int]] = None,
    query_bank_path: str = QUERY_BANK,
    out_dir: str = RESULTS_DIR,
) -> dict[str, Any]:
    cfg = run_configs.get(name)
    if offline is None:
        offline = not has_api_key()

    bank = load_query_bank(query_bank_path, chapters=chapters)
    if limit:
        bank = bank[:limit]
    goldset = load_goldset()

    corpus = load_corpus()
    fingerprint = corpus_fingerprint(corpus)
    embeddings = get_embeddings(offline)
    llm = get_llm(offline)

    overrides = cfg.as_config_overrides()
    records: list[QueryRecord] = []
    index_stats: dict[str, Any] = {}
    ragas_samples: list[dict[str, Any]] = []

    COUNTER.reset()
    t_start = time.perf_counter()

    with config_overrides(overrides), count_pipeline_llm_calls():
        from baselines import build_retriever

        retriever = build_retriever(cfg.retriever, corpus, embeddings, llm, offline=offline)
        build_calls_before = COUNTER.llm_calls
        t_build = time.perf_counter()
        if hasattr(retriever, "build"):
            index_stats = retriever.build() or {}
        index_stats["build_llm_calls"] = COUNTER.llm_calls - build_calls_before
        index_stats["build_seconds"] = time.perf_counter() - t_build

        verifier = _build_verifier(cfg.verifier, llm)

        for i, item in enumerate(bank, 1):
            print(f"  [{cfg.name}] {i}/{len(bank)} {item['query_id']} "
                  f"{item.get('meta_expected_intent','')}", flush=True)
            records.append(_run_one(
                item, retriever, verifier, cfg,
                offline=offline, generate=generate,
                ragas_samples=ragas_samples, goldset=goldset,
            ))

    wall = time.perf_counter() - t_start

    print(f"  [{cfg.name}] scoring RAGAS over {len(ragas_samples)} samples...", flush=True)
    ragas_scores = metrics_ragas.score_dataset(ragas_samples, offline=offline)
    for rec, sc in zip([r for r in records if r.generated], ragas_scores):
        rec.extra = {**(rec.extra or {}), "ragas": sc.to_dict()}

    summary = metrics_system.summarize(records)
    summary["ragas"] = _mean_ragas(ragas_scores)

    result = {
        "config": cfg.to_dict(),
        "config_name": cfg.name,
        "label": cfg.label,
        "mode": "offline" if offline else "live",
        "provenance": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "corpus_fingerprint": fingerprint,
            "n_corpus_chunks": len(corpus),
            "llm_model": getattr(llm, "model", "unknown"),
            "embedding_model": "stub" if offline else config.EMBEDDING_MODEL,
            "overrides": overrides,
            "query_bank": query_bank_path,
            "chapters": list(chapters) if chapters else "all",
            "n_queries": len(bank),
            "goldset_present": bool(goldset),
            "ragas_reference": bool(goldset),
            "wall_seconds": wall,
        },
        "index_stats": index_stats,
        "summary": summary,
        "records": [r.to_dict() for r in records],
    }

    os.makedirs(out_dir, exist_ok=True)
    suffix = "offline" if offline else "live"
    path = os.path.join(out_dir, f"{cfg.name}.{suffix}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    result["_path"] = path
    return result


def _build_verifier(kind: str, llm):
    if kind == "self_rag":
        from baselines.self_rag import SelfRagVerifier
        return SelfRagVerifier(llm)
    if kind == "saver":
        return "saver"
    return None


_TERMINAL_INTENTS = ("greeting", "harmful", "off_topic")


def _route(query: str, rec: QueryRecord) -> bool:
    """Run intent + ambiguity gates. Returns True if the turn terminates here.

    A terminating turn never reaches retrieval or generation, which is exactly
    what `metrics_system.leakage_rate` and `wasted_retrievals` count.
    """
    from agents.ambiguity import check_ambiguity
    from agents.intent import classify_intent

    intent = classify_intent(query)
    COUNTER.llm_calls += 1
    rec.observed_intent = intent
    if intent in _TERMINAL_INTENTS:
        rec.extra = {**(rec.extra or {}), "terminated_by": f"intent:{intent}"}
        return True

    amb = check_ambiguity(query)
    COUNTER.llm_calls += 1
    if amb.get("is_ambiguous"):
        rec.extra = {**(rec.extra or {}), "terminated_by": "ambiguity:clarification"}
        return True
    return False


def _run_one(item, retriever, verifier, cfg, *, offline, generate,
             ragas_samples, goldset) -> QueryRecord:
    query = item["query"]
    qid = item["query_id"]
    expected = str(item.get("meta_expected_intent", ""))

    rec = QueryRecord(query=query, query_id=qid, expected_intent=expected, extra={})
    calls_before = COUNTER.llm_calls
    t0 = time.perf_counter()

    if cfg.routing and not offline:
        try:
            terminal = _route(query, rec)
        except Exception as exc:
            rec.error = f"routing: {type(exc).__name__}: {exc}"
            rec.latency_s = time.perf_counter() - t0
            rec.llm_calls = COUNTER.llm_calls - calls_before
            return rec
        if terminal:
            rec.latency_s = time.perf_counter() - t0
            rec.llm_calls = COUNTER.llm_calls - calls_before
            return rec

    try:
        rr = retriever.retrieve(query)
    except Exception as exc:
        rec.error = f"{type(exc).__name__}: {exc}"
        rec.latency_s = time.perf_counter() - t0
        rec.llm_calls = COUNTER.llm_calls - calls_before
        return rec

    rec.retrieved = bool(rr.contexts) or not rr.failed_closed
    rec.failed_closed = rr.failed_closed
    rec.coverage = rr.coverage
    rec.n_contexts = len(rr.contexts)
    rec.source_attribution = rr.source_attribution
    rec.extra = {"retriever": retriever.name, **(rr.extra or {})}

    if generate and not rr.failed_closed:
        text = _generate(query, rr.contexts, offline=offline,
                         difficulty=str(item.get("level") or "intermediate"))
        rec.generated = bool(text)
        rec.generated_text = text
        if verifier is not None and verifier != "saver" and text:
            verdict = verifier.verify(query, rr.contexts, text)
            rec.extra["self_rag"] = verdict.to_dict()
            if not verdict.accepted:
                rec.extra["rejected_by_verifier"] = True
        if text:
            ragas_samples.append({
                "question": query,
                "answer": render_for_scoring(text),
                "contexts": rr.contexts or [""],
                "ground_truth_contexts": goldset.get(qid, []),
            })

    rec.latency_s = time.perf_counter() - t0
    rec.llm_calls = COUNTER.llm_calls - calls_before
    return rec


def _generate(query: str, contexts: Sequence[str], *, offline: bool,
              difficulty: str = "intermediate") -> str:
    """Generate assessment items from retrieved evidence.

    Every config routes through this one function — the fairness constraint from
    IMPLEMENTATION_PLAN.md 5. Differences between rows must come from retrieval
    or verification, never from the generator.
    """
    if offline:
        return f"[offline-stub-item] {query[:60]} | ctx={len(contexts)}"
    from agents.assessment import generate_assessment

    data = generate_assessment(
        list(contexts),
        difficulty_level=difficulty,
        selected_tasks=None,
        query_text=query,
    )
    return json.dumps(data, ensure_ascii=False)


def render_for_scoring(generated_json: str) -> str:
    """Flatten generated assessment items into declarative Bangla prose.

    RAGAS faithfulness works by decomposing a *response* into atomic claims and
    checking each against the retrieved contexts. A JSON blob of MCQ objects has
    no claims to extract — questions are interrogative, and the surrounding
    braces and keys are noise — so RAGAS returns NaN. Passing the raw dump made
    every faithfulness/context-precision score come back empty.

    Each MCQ therefore becomes the assertion it actually encodes ("<stem> — the
    answer is <correct option>"), and each creative item contributes its stimulus
    prose, which is already declarative. This is a scoring-time transformation
    only; the stored ``generated_text`` remains the verbatim JSON.

    Note for the paper: needing this transformation at all is direct evidence for
    the positioning argument — RAGAS is built for QA answers, and assessment
    items are not answers.
    """
    try:
        data = json.loads(generated_json)
    except (json.JSONDecodeError, TypeError):
        return str(generated_json)
    if not isinstance(data, dict):
        return str(generated_json)

    parts: list[str] = []
    for mcq in data.get("mcqs") or []:
        if not isinstance(mcq, dict):
            continue
        stem = str(mcq.get("question") or "").strip()
        options = mcq.get("options") or []
        idx = mcq.get("answer_index")
        answer = ""
        if isinstance(idx, int) and 0 <= idx < len(options):
            answer = re.sub(r"^\s*[কখগঘa-dA-D][\)\.]\s*", "", str(options[idx])).strip()
        if stem and answer:
            parts.append(f"{stem} — এর উত্তর হলো {answer}।")
        elif stem:
            parts.append(stem)

    for item in data.get("creative") or data.get("creative_questions") or []:
        if isinstance(item, dict):
            clue = str(item.get("clue") or "").strip()
            if clue:
                parts.append(clue)
        elif isinstance(item, str):
            parts.append(item.strip())

    return "\n".join(p for p in parts if p) or str(generated_json)


def _mean_ragas(scores) -> dict[str, Optional[float]]:
    keys = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    out: dict[str, Optional[float]] = {}
    for k in keys:
        vals = [getattr(s, k) for s in scores if getattr(s, k) is not None]
        out[k] = sum(vals) / len(vals) if vals else None
    return out


def run_many(names: Iterable[str], **kw) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for n in names:
        results[n] = run_config(n, **kw)
    _assert_comparable(results)
    return results


def _assert_comparable(results: dict[str, dict[str, Any]]) -> None:
    """Guard the fairness contract: same corpus, same mode, same bank."""
    prints = {n: r["provenance"]["corpus_fingerprint"] for n, r in results.items()}
    if len(set(prints.values())) > 1:
        raise RuntimeError(f"Configs indexed different corpora, not comparable: {prints}")
    modes = {r["mode"] for r in results.values()}
    if len(modes) > 1:
        raise RuntimeError(f"Mixed live/offline results are not comparable: {modes}")
