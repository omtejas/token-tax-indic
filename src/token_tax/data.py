from __future__ import annotations

LANGUAGES = {
    "eng_Latn": "English",
    "hin_Deva": "Hindi",
    "kan_Knda": "Kannada",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
}

def resolve_languages(configured=None):
    """Return ordered language codes and display names from YAML list or mapping."""
    configured = LANGUAGES if configured is None else configured
    if isinstance(configured, dict):
        codes = list(configured)
        names = {code: str(configured[code]) for code in codes}
    else:
        codes = list(configured)
        names = {code: LANGUAGES.get(code, code) for code in codes}
    if not codes or len(codes) != len(set(codes)):
        raise ValueError("Configure a non-empty list of unique language codes")
    return codes, names


def load_flores_parallel(languages=None, split="dev", limit=300, dataset_name="yash9439/flores200", streaming=True):
    """Load consolidated public Parquet FLORES-200 and retain aligned rows."""
    from datasets import load_dataset
    languages = languages or list(LANGUAGES)
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    rows = []
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        missing = [lang for lang in languages if lang not in row]
        if missing:
            raise KeyError(f"FLORES columns missing: {missing}. Available fields: {list(row)[:20]}")
        sentence_id = row.get("id", index + 1)
        values = {lang: row[lang] for lang in languages}
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ValueError(f"FLORES row {sentence_id!r} contains an empty or non-text translation")
        rows.append({"sentence_id": sentence_id, **values})
    if not rows:
        raise ValueError(f"No rows loaded from {dataset_name} split={split}")
    return rows
