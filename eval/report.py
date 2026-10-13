"""Render harness results into `paper/TeachMateGPT - All Tables.md`.

Safety property, enforced in :func:`write_tables`: **offline/stub results are
never written into the paper.** Stub responses are meaningless by construction,
and a number in a results table is indistinguishable from a measured one once
it lands there. Offline runs can only be rendered to stdout for inspection.

Human evaluation (Table 2) is collected from teachers and cannot be produced by
the harness. New configs are added as rows with explicit ``—`` placeholders so a
missing rating is visible rather than absent.
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Iterable, Optional, Sequence

from eval import configs as run_configs
from eval.metrics_stats import bootstrap_ci, format_ci, paired_bootstrap

TABLES_FILE = "paper/TeachMateGPT - All Tables.md"
RESULTS_DIR = "data/experiment/eval_runs"

PENDING = "—"


class OfflineResultsError(RuntimeError):
    """Raised when a caller tries to publish stub numbers into the paper."""


def load_results(results_dir: str = RESULTS_DIR, *, mode: str = "live") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, f"*.{mode}.json"))):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        out[data["config_name"]] = data
    return out


def _fmt(val: Optional[float], digits: int = 2) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return PENDING
    return f"{val:.{digits}f}"


def _pct(val: Optional[float], n: Optional[int] = None, total: Optional[int] = None) -> str:
    if val is None:
        return PENDING
    base = f"{val * 100:.1f}%"
    if n is not None and total:
        return f"{base} ({n}/{total})"
    return base


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _bold_if(name: str, text: str) -> str:
    return f"**{text}**" if name == "full" else text



def render_ragas(results: dict[str, dict[str, Any]]) -> str:
    """RAGAS table, with the scored-sample count in its own column.

    ``n`` is the number of generated items actually scored, not the number of
    queries run — turns that terminate at routing or fail closed produce nothing
    to score. Reporting it inline stops a mean over 2-3 items from being read as
    a population estimate.
    """
    lines = [
        "| Configuration | n | Faithfulness ↑ | Answer Relevancy ↑ | Context Precision ↑ | Context Recall ↑ |",
        "|---|---:|---|---|---|---|",
    ]
    for cfg in run_configs.for_table(run_configs.T_RAGAS):
        res = results.get(cfg.name)
        r = (res or {}).get("summary", {}).get("ragas", {}) or {}
        n_scored = _n_scored(res) if res else None
        lines.append(_row([
            _bold_if(cfg.name, cfg.label),
            str(n_scored) if n_scored is not None else PENDING,
            _bold_if(cfg.name, _fmt(r.get("faithfulness"))),
            _bold_if(cfg.name, _fmt(r.get("answer_relevancy"))),
            _bold_if(cfg.name, _fmt(r.get("context_precision"))),
            _bold_if(cfg.name, _fmt(r.get("context_recall"))),
        ]))
    return "\n".join(lines)


def _n_scored(res: dict[str, Any]) -> int:
    return sum(
        1 for rec in res.get("records", [])
        if ((rec.get("extra") or {}).get("ragas") or {}).get("faithfulness") is not None
    )


def render_human(results: dict[str, dict[str, Any]],
                 ratings_path: str = "data/experiment/teacher_ratings.json") -> str:
    """Teacher ratings come from a hand-authored file, not the harness."""
    ratings: dict[str, dict[str, str]] = {}
    if os.path.exists(ratings_path):
        with open(ratings_path, encoding="utf-8") as fh:
            ratings = json.load(fh)

    lines = [
        "| Configuration | Pedagogical Alignment ↑ | Stimulus Realism ↑ | Linguistic Fluency ↑ | Overall Utility ↑ |",
        "|---|---|---|---|---|",
    ]
    cols = ("pedagogical_alignment", "stimulus_realism", "linguistic_fluency", "overall_utility")
    for cfg in run_configs.for_table(run_configs.T_HUMAN):
        vals = ratings.get(cfg.name, {})
        lines.append(_row([_bold_if(cfg.name, cfg.label)] +
                          [_bold_if(cfg.name, str(vals.get(c, PENDING))) for c in cols]))
    return "\n".join(lines)


def render_retrieval(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Configuration | Fail-Closed Rate | Coverage Ratio | Source Attribution | Notes |",
        "|---|---|---|---|---|",
    ]
    for cfg in run_configs.for_table(run_configs.T_RETRIEVAL):
        s = (results.get(cfg.name) or {}).get("summary", {})
        lines.append(_row([
            _bold_if(cfg.name, cfg.label),
            _pct(s.get("fail_closed_rate"), s.get("fail_closed_n"), s.get("fail_closed_total")),
            _fmt(s.get("coverage_ratio"), 3),
            _fmt(s.get("mean_source_attribution"), 2),
            cfg.notes or ("default" if cfg.name == "full" else ""),
        ]))
    return "\n".join(lines)


def render_routing(results: dict[str, dict[str, Any]]) -> str:
    """Routing ablation — the evidence for the routing contribution (Issue 3)."""
    lines = [
        "| Configuration | Unsafe-Content Leakage | Wasted Retrievals | Mean LLM Calls / Query | Mean Latency (s) |",
        "|---|---|---|---|---|",
    ]
    for name in ("full", "no_routing", "adaptive"):
        cfg = run_configs.REGISTRY.get(name)
        if cfg is None:
            continue
        s = (results.get(name) or {}).get("summary", {})
        wasted = s.get("wasted_retrievals")
        n_ns = s.get("n_non_science")
        lines.append(_row([
            _bold_if(name, cfg.label),
            _pct(s.get("leakage_rate"), s.get("leakage_n"), s.get("leakage_total")),
            f"{wasted}/{n_ns}" if wasted is not None and n_ns else PENDING,
            _fmt(s.get("mean_llm_calls"), 1),
            _fmt(s.get("mean_latency_s"), 2),
        ]))
    return "\n".join(lines)


def render_cost(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Configuration | Mean LLM Calls / Query | Mean Latency (s) | One-Time Index LLM Calls | Index Build (s) |",
        "|---|---|---|---|---|",
    ]
    for cfg in run_configs.for_table(run_configs.T_COST):
        res = results.get(cfg.name) or {}
        s = res.get("summary", {})
        idx = res.get("index_stats", {}) or {}
        lines.append(_row([
            _bold_if(cfg.name, cfg.label),
            _fmt(s.get("mean_llm_calls"), 1),
            _fmt(s.get("mean_latency_s"), 2),
            str(idx.get("build_llm_calls", PENDING)),
            _fmt(idx.get("build_seconds"), 1),
        ]))
    return "\n".join(lines)


def render_significance(results: dict[str, dict[str, Any]],
                        reference: str = "full") -> str:
    """Paired bootstrap of each config against TeachMateGPT on faithfulness."""
    ref = results.get(reference)
    lines = [
        "| Configuration | Δ Faithfulness vs. TeachMateGPT | 95% CI | p (paired bootstrap) | n pairs |",
        "|---|---|---|---|---|",
    ]
    if not ref:
        return "\n".join(lines + [_row(["(reference run missing)", PENDING, PENDING, PENDING, PENDING])])

    ref_by_q = _faithfulness_by_query(ref)
    for name, res in results.items():
        if name == reference:
            continue
        cur = _faithfulness_by_query(res)
        common = [q for q in ref_by_q if q in cur]
        if not common:
            continue
        stats = paired_bootstrap([cur[q] for q in common], [ref_by_q[q] for q in common])
        cfg = run_configs.REGISTRY.get(name)
        lines.append(_row([
            cfg.label if cfg else name,
            _fmt(stats["delta"], 3),
            f"[{stats['lo']:.3f}, {stats['hi']:.3f}]" if stats["n_pairs"] >= 2 else PENDING,
            _fmt(stats["p_value"], 3),
            str(stats["n_pairs"]),
        ]))
    return "\n".join(lines)


def _faithfulness_by_query(res: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for rec in res.get("records", []):
        val = ((rec.get("extra") or {}).get("ragas") or {}).get("faithfulness")
        if val is not None:
            out[rec["query_id"]] = float(val)
    return out



SECTION_RE = r"(^## {heading}\n)(.*?)(?=^## |\Z)"


def _replace_section(text: str, heading: str, body: str) -> str:
    """Replace the body under `## <heading>`, appending the section if absent."""
    pattern = SECTION_RE.format(heading=re.escape(heading))
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    replacement = f"\n{body}\n\n"
    if match:
        return text[: match.end(1)] + replacement + text[match.end(2):]
    return text.rstrip() + f"\n\n## {heading}\n{replacement}"


def write_tables(results: dict[str, dict[str, Any]], *,
                 tables_file: str = TABLES_FILE) -> list[str]:
    """Write rendered tables into the paper file. Returns headings updated.

    There is intentionally no override for the offline check. Publishing stub
    numbers into a results table is unrecoverable in practice — once written they
    are indistinguishable from measurements — so the only way to render them is
    :func:`preview`, which prints to stdout.
    """
    modes = {r.get("mode") for r in results.values()}
    if "offline" in modes:
        raise OfflineResultsError(
            "Refusing to write offline/stub results into the paper. These numbers "
            "are generated by deterministic placeholders and carry no measurement. "
            "Set OPENAI_API_KEY and re-run with --live."
        )

    with open(tables_file, encoding="utf-8") as fh:
        text = fh.read()

    sections = {
        "Table 1: Reference-Free RAGAS Evaluation": render_ragas(results),
        "Table 2: Human Evaluation by Teachers (5-point Likert Scale)": render_human(results),
        "Table 9: Retrieval Ablation Results": render_retrieval(results),
        "Table 14: Routing Ablation and Safety Leakage": render_routing(results),
        "Table 15: Cost and Latency by Configuration": render_cost(results),
        "Table 16: Paired Bootstrap Significance vs. TeachMateGPT": render_significance(results),
    }
    for heading, body in sections.items():
        text = _replace_section(text, heading, body)

    with open(tables_file, "w", encoding="utf-8") as fh:
        fh.write(text)
    return list(sections)


def preview(results: dict[str, dict[str, Any]]) -> str:
    return "\n\n".join([
        "## Table 1: Reference-Free RAGAS Evaluation", render_ragas(results),
        "## Table 9: Retrieval Ablation Results", render_retrieval(results),
        "## Table 14: Routing Ablation and Safety Leakage", render_routing(results),
        "## Table 15: Cost and Latency by Configuration", render_cost(results),
    ])
