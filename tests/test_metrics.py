import pytest
from token_tax.metrics import normalize_text, safe_ratio, summarize_token_counts, context_capacity, context_window_capacity, estimate_kv_cache_bytes
from token_tax.statistics import paired_bootstrap_mean_ci
from token_tax.costs import estimate_api_input_cost

def test_normalization_nfc():
    assert normalize_text("é", "NFC") == "é"

def test_safe_ratio_zero():
    assert safe_ratio(4, 0) is None

def test_summary_and_pairing():
    result = summarize_token_counts([2, 4, 6], [1, 2, 3])
    assert result["mean_tokens"] == 4
    assert result["mean_ratio_vs_english"] == 2

def test_pairing_length_mismatch():
    with pytest.raises(ValueError):
        summarize_token_counts([1], [1, 2])

def test_context_capacity():
    assert context_capacity([3, 4, 5], 8) == 7

def test_complete_sentences_in_context():
    result = context_window_capacity([3, 4, 5], 8)
    assert result["complete_sentences"] == 2
    assert result["unused_tokens"] == 1

def test_kv_cache_formula():
    assert estimate_kv_cache_bytes(1, 10, 2, 4, 8, 2) == 2560

def test_paired_bootstrap_ci():
    result = paired_bootstrap_mean_ci([3, 4, 5], [1, 2, 3], n_boot=500)
    assert result["mean_difference"] == 2
    assert result["ci_low"] <= 2 <= result["ci_high"]

def test_cost_requires_source():
    with pytest.raises(ValueError):
        estimate_api_input_cost(100, 1.0, "model", "", "2026-11-22")
