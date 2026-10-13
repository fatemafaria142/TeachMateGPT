"""Paired bootstrap confidence intervals and significance.

N=16-30 queries is small. Any table comparing a baseline against TeachMateGPT
needs an interval, or a reviewer will read a 0.02 gap as a result. Pairing is on
query id: both systems see the same queries, so the paired test is the correct
one and is markedly tighter than an unpaired comparison.
"""
from __future__ import annotations

import random
from statistics import mean
from typing import Optional, Sequence


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 17) -> tuple[float, float, float]:
    """Return (point_estimate, lo, hi) for the mean."""
    vals = [v for v in values if v is not None]
    if not vals:
        return (float("nan"),) * 3
    if len(vals) == 1:
        return vals[0], vals[0], vals[0]
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        means.append(mean(vals[rng.randrange(n)] for _ in range(n)))
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return mean(vals), lo, hi


def paired_bootstrap(system: Sequence[Optional[float]], baseline: Sequence[Optional[float]], *,
                     n_boot: int = 10000, alpha: float = 0.05,
                     seed: int = 17) -> dict[str, float]:
    """Paired bootstrap on the per-query difference (system - baseline).

    ``p_value`` is two-sided: the share of resamples whose mean difference has the
    opposite sign to the observed one, doubled.
    """
    pairs = [(s, b) for s, b in zip(system, baseline) if s is not None and b is not None]
    if len(pairs) < 2:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_value": float("nan"), "n_pairs": len(pairs)}
    diffs = [s - b for s, b in pairs]
    observed = mean(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    boots = []
    for _ in range(n_boot):
        boots.append(mean(diffs[rng.randrange(n)] for _ in range(n)))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    if observed >= 0:
        p = 2 * sum(1 for b in boots if b <= 0) / n_boot
    else:
        p = 2 * sum(1 for b in boots if b >= 0) / n_boot
    return {"delta": observed, "lo": lo, "hi": hi, "p_value": min(1.0, p), "n_pairs": n}


def format_ci(point: float, lo: float, hi: float, *, digits: int = 2) -> str:
    if any(v != v for v in (point, lo, hi)):
        return "n/a"
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
