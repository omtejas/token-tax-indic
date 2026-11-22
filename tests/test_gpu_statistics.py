from token_tax.gpu import summarize_repetitions

def test_repetition_summary_has_mean_std_and_t_confidence_interval():
    rows = [
        {"language_code":"eng_Latn","requested_sequence_length":128,"batch_size":1,"latency":10.0},
        {"language_code":"eng_Latn","requested_sequence_length":128,"batch_size":1,"latency":12.0},
        {"language_code":"eng_Latn","requested_sequence_length":128,"batch_size":1,"latency":14.0},
    ]
    result = summarize_repetitions(rows, ["latency"])[0]
    assert result["repetitions"] == 3
    assert result["latency_mean"] == 12.0
    assert result["latency_std"] == 2.0
    assert result["latency_ci95_low"] < 12.0 < result["latency_ci95_high"]

def test_repetition_summary_preserves_unavailable_metrics_as_null():
    rows = [
        {"language_code":"hin_Deva","requested_sequence_length":256,"batch_size":1,"energy":None},
        {"language_code":"hin_Deva","requested_sequence_length":256,"batch_size":1,"energy":None},
        {"language_code":"hin_Deva","requested_sequence_length":256,"batch_size":1,"energy":None},
    ]
    result = summarize_repetitions(rows, ["energy"])[0]
    assert result["repetitions"] == 3
    assert result["energy_mean"] is None
    assert result["energy_std"] is None
    assert result["energy_ci95_low"] is None
    assert result["energy_ci95_high"] is None
