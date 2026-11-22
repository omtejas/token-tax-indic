from __future__ import annotations

LANGUAGES = {
    "eng_Latn": "English",
    "hin_Deva": "Hindi",
    "kan_Knda": "Kannada",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
}

def load_flores_parallel(languages=None, split="dev", limit=300, dataset_name="yash9439/flores200"):
    """Load consolidated public Parquet FLORES-200 and retain aligned rows."""
    from datasets import load_dataset
    languages = languages or list(LANGUAGES)
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    rows = []
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        missing = [lang for lang in languages if lang not in row]
        if missing:
            raise KeyError(f"FLORES columns missing: {missing}. Available fields: {list(row)[:20]}")
        sentence_id = row.get("id", index + 1)
        rows.append({"sentence_id": sentence_id, **{lang: row[lang] for lang in languages}})
    if not rows:
        raise ValueError(f"No rows loaded from {dataset_name} split={split}")
    return rows
