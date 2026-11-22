import pandas as pd
from token_tax.failure_analysis import find_failure_examples

def test_failure_analysis_detects_unicode_and_high_fertility(tmp_path):
    path = tmp_path / "tokenizer.csv"
    pd.DataFrame([
        {"sentence_id":"1","tokenizer_id":"tok","language_code":"eng_Latn","normalization":"none","tokens":2,"text":"test"},
        {"sentence_id":"1","tokenizer_id":"tok","language_code":"hin_Deva","normalization":"none","tokens":8,"text":"e\u0301"},
        {"sentence_id":"1","tokenizer_id":"tok","language_code":"hin_Deva","normalization":"NFC","tokens":7,"text":"é"},
    ]).to_csv(path,index=False)
    result=find_failure_examples(path,top_n=5)
    assert "high_fertility" in set(result.failure_type)
    assert "unicode_normalization_changed_text" in set(result.failure_type)
    unicode_row=result[result.failure_type.eq("unicode_normalization_changed_text")].iloc[0]
    assert unicode_row.token_delta_nfc_minus_none == -1
