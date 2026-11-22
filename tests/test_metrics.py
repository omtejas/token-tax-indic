import pytest
from token_tax.metrics import normalize_text, safe_ratio, summarize_token_counts, context_capacity, estimate_kv_cache_bytes

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

def test_kv_cache_formula():
    assert estimate_kv_cache_bytes(1, 10, 2, 4, 8, 2) == 2560
