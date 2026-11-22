from __future__ import annotations
import math
import unicodedata
from collections.abc import Sequence


def normalize_text(text: str, form: str = "none") -> str:
    if form.lower() in {"none", "null", "off"}:
        return text
    if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(f"Unsupported Unicode normalization: {form}")
    return unicodedata.normalize(form, text)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def summarize_token_counts(counts: Sequence[int], english_counts: Sequence[int] | None = None) -> dict:
    values = [int(x) for x in counts]
    if any(x < 0 for x in values):
        raise ValueError("Token counts cannot be negative")
    if not values:
        return {"n": 0, "mean_tokens": None, "median_tokens": None, "total_tokens": 0}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    result = {"n": len(values), "mean_tokens": sum(values) / len(values), "median_tokens": median, "total_tokens": sum(values)}
    if english_counts is not None:
        if len(english_counts) != len(values):
            raise ValueError("Paired English counts must have the same length")
        ratios = [safe_ratio(a, b) for a, b in zip(values, english_counts)]
        valid = [x for x in ratios if x is not None]
        result["mean_ratio_vs_english"] = sum(valid) / len(valid) if valid else None
    return result


def context_capacity(token_counts: Sequence[int], context_window: int) -> int:
    if context_window <= 0:
        raise ValueError("context_window must be positive")
    total = 0
    for count in token_counts:
        if count < 0:
            raise ValueError("Token counts cannot be negative")
        if total + count > context_window:
            break
        total += count
    return total


def estimate_kv_cache_bytes(batch_size: int, sequence_length: int, layers: int, kv_heads: int, head_dim: int, bytes_per_element: int = 2) -> int:
    """Theoretical K+V cache bytes for standard decoder attention; excludes allocator overhead."""
    vals = (batch_size, sequence_length, layers, kv_heads, head_dim, bytes_per_element)
    if any(v <= 0 for v in vals):
        raise ValueError("All KV-cache dimensions and element size must be positive")
    return 2 * batch_size * sequence_length * layers * kv_heads * head_dim * bytes_per_element
