#!/usr/bin/env python3
"""CLI for the TeachMateGPT evaluation harness.

Examples::

    # list every registered configuration
    python scripts/run_eval.py --list

    # smoke-test the plumbing without an API key (numbers are meaningless)
    python scripts/run_eval.py --all --offline --limit 4

    # the real thing (needs OPENAI_API_KEY)
    python scripts/run_eval.py --all --live
    python scripts/run_eval.py --config no_cci --config no_routing --live

    # render collected live results into the paper tables
    python scripts/run_eval.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import configs as run_configs
from eval import harness, report
from eval.providers import has_api_key


LOG_FILE = "logs/eval_run.log"


class _Tee:
    """Write to the terminal and the log file at once, flushing per line so
    `--status` and `tail -f` see progress immediately rather than in buffer-sized
    bursts."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def _status() -> int:
    """Report sweep liveness and per-config completion.

    Deliberately reads the same artefacts a human would: the process table, the
    log file, and the result directory. Nothing is cached, so the answer is
    always the current truth even if a run died without cleaning up.
    """
    import subprocess
    import time

    print("=" * 68)
    proc = subprocess.run(["pgrep", "-af", "run_eval.py"], capture_output=True, text=True)
    running = [l for l in proc.stdout.splitlines() if "--status" not in l and "pgrep" not in l]
    if running:
        print(f"RUNNING  ({len(running)} process)")
        for line in running[:2]:
            pid = line.split()[0]
            print(f"  pid {pid}")
    else:
        print("NOT RUNNING")

    if os.path.exists(LOG_FILE):
        age = time.time() - os.path.getmtime(LOG_FILE)
        print(f"\nLog: {LOG_FILE}  (last written {age:.0f}s ago)")
        with open(LOG_FILE, encoding="utf-8", errors="replace") as fh:
            lines = [l.rstrip() for l in fh if l.strip()]
        progress = [l for l in lines if l.lstrip().startswith("[")]
        for line in progress[-6:]:
            print(f"  {line}")
        if age > 300 and running:
            print("  WARNING: process alive but log is stale >5min — may be hung.")
    else:
        print(f"\nNo log at {LOG_FILE}")

    print(f"\nCompleted result files in {harness.RESULTS_DIR}:")
    found = False
    for mode in ("live", "offline"):
        for name in sorted(run_configs.REGISTRY):
            path = os.path.join(harness.RESULTS_DIR, f"{name}.{mode}.json")
            if not os.path.exists(path):
                continue
            found = True
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            s = data.get("summary", {})
            prov = data.get("provenance", {})
            n_scored = sum(
                1 for r in data.get("records", [])
                if ((r.get("extra") or {}).get("ragas") or {}).get("faithfulness") is not None
            )
            print(f"  {name:16s} [{mode}] n={s.get('n_queries', '?'):>3} "
                  f"scored={n_scored:>2} leakage={s.get('leakage_rate', 0):.2f} "
                  f"errors={s.get('n_errors', '?')} chapters={prov.get('chapters')}")
    if not found:
        print("  (none yet)")
    print("=" * 68)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Run TeachMateGPT evaluation configs.")
    p.add_argument("--config", action="append", default=[], help="config name (repeatable)")
    p.add_argument("--all", action="store_true", help="run every registered config")
    p.add_argument("--ablations", action="store_true", help="run the ablation configs only")
    p.add_argument("--baselines", action="store_true", help="run the external baselines only")
    p.add_argument("--offline", action="store_true", help="force stub mode (no API calls)")
    p.add_argument("--live", action="store_true", help="force live mode (requires API key)")
    p.add_argument("--limit", type=int, default=None, help="only the first N queries")
    p.add_argument("--chapters", type=str, default=None,
                   help="comma-separated chapter numbers, e.g. 1,2 (non-science queries always kept)")
    p.add_argument("--no-generate", action="store_true", help="retrieval only, skip generation")
    p.add_argument("--report", action="store_true", help="write collected results into the paper tables")
    p.add_argument("--preview", action="store_true", help="print tables instead of writing them")
    p.add_argument("--list", action="store_true", help="list configs and exit")
    p.add_argument("--status", action="store_true",
                   help="show whether a sweep is running and which configs have completed")
    args = p.parse_args()

    if args.status:
        return _status()

    if args.list:
        for name, cfg in sorted(run_configs.REGISTRY.items()):
            tables = ",".join(cfg.tables) or "-"
            print(f"{name:18s} {cfg.label:28s} retriever={cfg.retriever:9s} "
                  f"verifier={cfg.verifier:8s} tables=[{tables}]")
        return 0

    if args.report or args.preview:
        if args.report and args.offline:
            print("ERROR: --report never writes stub results into the paper. "
                  "Use --preview --offline to inspect them.", file=sys.stderr)
            return 2
        mode = "offline" if args.offline else "live"
        results = report.load_results(mode=mode)
        if not results:
            print(f"No {mode} results in {report.RESULTS_DIR}. Run configs first.", file=sys.stderr)
            return 1
        if args.preview:
            print(report.preview(results))
            return 0
        try:
            updated = report.write_tables(results)
        except report.OfflineResultsError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"Updated {len(updated)} sections in {report.TABLES_FILE}:")
        for h in updated:
            print(f"  - {h}")
        return 0

    if args.live and args.offline:
        print("ERROR: --live and --offline are mutually exclusive", file=sys.stderr)
        return 2
    offline = True if args.offline else (False if args.live else not has_api_key())
    if not offline and not has_api_key():
        print("ERROR: --live requires OPENAI_API_KEY", file=sys.stderr)
        return 2

    names = list(args.config)
    if args.all:
        names = list(run_configs.REGISTRY)
    if args.ablations:
        names += list(run_configs.ABLATION_NAMES)
    if args.baselines:
        names += list(run_configs.BASELINE_NAMES)
    names = list(dict.fromkeys(names))
    if not names:
        print("Nothing to run. Use --config NAME, --all, --ablations or --baselines.", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    sys.stdout = _Tee(sys.stdout, open(LOG_FILE, "w", encoding="utf-8"))

    if offline:
        print("=" * 72)
        print("OFFLINE MODE — deterministic stubs. Exercises the pipeline only.")
        print("Numbers produced here are NOT measurements and cannot be published.")
        print("=" * 72)

    chapters = None
    if args.chapters:
        chapters = [int(c.strip()) for c in args.chapters.split(",") if c.strip()]
        print(f"Chapter filter: {chapters} (non-science queries retained for routing metrics)")

    failures = 0
    for name in names:
        try:
            res = harness.run_config(
                name, offline=offline, limit=args.limit,
                generate=not args.no_generate, chapters=chapters,
            )
        except Exception as exc:
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1
            continue
        s = res["summary"]
        print(f"[ok] {name:18s} n={s['n_queries']:3d} "
              f"fail_closed={s['fail_closed_rate']:.3f} "
              f"leakage={s['leakage_rate']:.3f} "
              f"calls/q={s['mean_llm_calls']:.1f} "
              f"errors={s['n_errors']} -> {res['_path']}")

    if failures:
        print(f"\n{failures} config(s) failed.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
