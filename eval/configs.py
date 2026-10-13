"""Declarative registry of every run configuration.

One :class:`RunConfig` per row of every results table. Adding a table row means
adding an entry here — never editing a results table by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    name: str
    label: str
    retriever: str = "cope"
    verifier: str = "saver"
    routing: bool = True

    use_cci: bool = True
    use_ccr: bool = True
    use_graph_expansion: bool = True
    use_coverage_gate: bool = True
    dense_only: bool = False
    bm25_only: bool = False

    tables: tuple[str, ...] = ()
    notes: str = ""

    def as_config_overrides(self) -> dict[str, Any]:
        """Map to `config` module attribute overrides applied for the run."""
        return {
            "ABLATE_CCI": not self.use_cci,
            "ABLATE_CCR": not self.use_ccr,
            "ABLATE_ROUTING": not self.routing,
            "ABLATE_COVERAGE_GATE": not self.use_coverage_gate,
            "ABLATE_GRAPH_EXPANSION": not self.use_graph_expansion,
            "RETRIEVAL_DENSE_ONLY": self.dense_only,
            "RETRIEVAL_BM25_ONLY": self.bm25_only,
            "RETRIEVAL_BACKEND": self.retriever,
            "VERIFIER": self.verifier,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T_RAGAS = "ragas"
T_HUMAN = "human"
T_RETRIEVAL = "retrieval"
T_COST = "cost"
T_ROUTING = "routing"


REGISTRY: dict[str, RunConfig] = {}


def _add(cfg: RunConfig) -> RunConfig:
    REGISTRY[cfg.name] = cfg
    return cfg


_add(RunConfig("full", "TeachMateGPT (Full)", tables=(T_RAGAS, T_HUMAN, T_RETRIEVAL, T_COST)))
_add(RunConfig("vanilla", "Vanilla RAG", retriever="vanilla", verifier="none", routing=False,
               use_cci=False, use_ccr=False, use_graph_expansion=False, use_coverage_gate=False,
               tables=(T_RAGAS, T_HUMAN, T_COST), notes="dense top-k + generate, no gates"))
_add(RunConfig("no_cope", "w/o COPE", retriever="vanilla", tables=(T_RAGAS, T_HUMAN),
               notes="flat chunks, no macro/meso/micro tiering"))
_add(RunConfig("no_saver", "w/o SAVER", verifier="none", tables=(T_RAGAS, T_HUMAN)))
_add(RunConfig("no_ccr", "w/o CCR", use_ccr=False, tables=(T_RAGAS, T_HUMAN, T_RETRIEVAL)))

_add(RunConfig("no_cci", "w/o CCI", use_cci=False,
               tables=(T_RAGAS, T_HUMAN, T_RETRIEVAL),
               notes="Issue 3: CCI claimed as contribution, previously unablated in Tables 1-2"))
_add(RunConfig("no_routing", "w/o Routing Agent", routing=False,
               tables=(T_RAGAS, T_HUMAN, T_ROUTING, T_COST),
               notes="Issue 3: routing claimed as contribution, previously unablated"))

_add(RunConfig("dense_only", "Dense-only", dense_only=True, tables=(T_RETRIEVAL,)))
_add(RunConfig("bm25_only", "BM25-only", bm25_only=True, tables=(T_RETRIEVAL,)))
_add(RunConfig("no_coverage_gate", "No coverage gate", use_coverage_gate=False, tables=(T_RETRIEVAL,)))
_add(RunConfig("no_graph", "No graph expansion", use_graph_expansion=False, tables=(T_RETRIEVAL,)))

_add(RunConfig("raptor", "RAPTOR", retriever="raptor", verifier="none", routing=False,
               tables=(T_RAGAS, T_HUMAN, T_RETRIEVAL, T_COST),
               notes="recursive clustering + summarisation, collapsed-tree retrieval"))
_add(RunConfig("graphrag", "GraphRAG", retriever="graphrag", verifier="none", routing=False,
               tables=(T_RAGAS, T_HUMAN, T_RETRIEVAL, T_COST),
               notes="entity graph + Leiden communities + community summaries"))
_add(RunConfig("crag", "CRAG", retriever="crag", verifier="none", routing=False,
               tables=(T_RAGAS, T_HUMAN, T_RETRIEVAL, T_COST),
               notes="retrieval evaluator + query rewrite; web fallback removed (closed corpus)"))
_add(RunConfig("adaptive", "Adaptive RAG", retriever="adaptive", verifier="none", routing=False,
               tables=(T_RAGAS, T_HUMAN, T_RETRIEVAL, T_COST),
               notes="query-complexity router: no-retrieval / single-step / multi-step"))
_add(RunConfig("self_rag", "Self-RAG (prompted)", verifier="self_rag",
               tables=(T_RAGAS, T_HUMAN, T_COST),
               notes="prompted reflection tokens, NOT a trained critic; retrieval held at full hybrid"))


BASELINE_NAMES = ("raptor", "graphrag", "crag", "adaptive", "self_rag")
ABLATION_NAMES = ("no_cope", "no_saver", "no_ccr", "no_cci", "no_routing")


def get(name: str) -> RunConfig:
    if name not in REGISTRY:
        raise KeyError(f"Unknown config '{name}'. Known: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[name]


def for_table(table: str) -> list[RunConfig]:
    return [c for c in REGISTRY.values() if table in c.tables]
