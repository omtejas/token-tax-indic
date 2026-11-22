from __future__ import annotations
from collections.abc import Sequence
import numpy as np

def paired_bootstrap_mean_ci(values_a: Sequence[float], values_b: Sequence[float], confidence: float = 0.95, n_boot: int = 5000, seed: int = 42) -> dict:
    """Vectorized percentile bootstrap CI for the mean paired difference A-B."""
    if len(values_a) != len(values_b) or len(values_a) == 0:
        raise ValueError("Paired samples must be non-empty and equal length")
    if not 0 < confidence < 1 or n_boot < 100:
        raise ValueError("Use confidence in (0,1) and at least 100 bootstrap draws")
    differences = np.asarray(values_a, dtype=np.float64) - np.asarray(values_b, dtype=np.float64)
    if not np.isfinite(differences).all():
        raise ValueError("Paired values must be finite numbers")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(n_boot, len(differences)))
    bootstrap_means = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return {
        "n_pairs": int(len(differences)),
        "mean_difference": float(differences.mean()),
        "confidence": float(confidence),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "bootstrap_draws": int(n_boot),
        "seed": int(seed),
    }
