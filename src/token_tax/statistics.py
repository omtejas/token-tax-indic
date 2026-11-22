from __future__ import annotations
import random
from collections.abc import Sequence

def paired_bootstrap_mean_ci(values_a: Sequence[float], values_b: Sequence[float], confidence: float = 0.95, n_boot: int = 5000, seed: int = 42) -> dict:
    """Percentile bootstrap CI for the mean paired difference A-B."""
    if len(values_a) != len(values_b) or not values_a:
        raise ValueError("Paired samples must be non-empty and equal length")
    if not 0 < confidence < 1 or n_boot < 100:
        raise ValueError("Use confidence in (0,1) and at least 100 bootstrap draws")
    diffs = [float(a)-float(b) for a,b in zip(values_a,values_b)]
    rng = random.Random(seed); n=len(diffs); means=[]
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n))/n)
    means.sort(); alpha=(1-confidence)/2
    lo=means[int(alpha*n_boot)]; hi=means[min(int((1-alpha)*n_boot),n_boot-1)]
    return {"n_pairs":n,"mean_difference":sum(diffs)/n,"confidence":confidence,"ci_low":lo,"ci_high":hi,"bootstrap_draws":n_boot,"seed":seed}
