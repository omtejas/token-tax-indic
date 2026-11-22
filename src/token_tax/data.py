from __future__ import annotations

LANGUAGES = {"eng_Latn": "English", "hin_Deva": "Hindi", "kan_Knda": "Kannada", "tam_Taml": "Tamil", "tel_Telu": "Telugu"}


def load_flores_parallel(languages=None, split="dev", limit=300):
    """Load aligned FLORES-200 rows. Dataset access/licence must be reviewed before use."""
    from datasets import load_dataset
    languages = languages or list(LANGUAGES)
    # FLORES-200 is commonly distributed as language-specific configs.
    # We load each config and join by row index; record dataset revision for publication.
    frames = {}
    for lang in languages:
        dataset = load_dataset("facebook/flores", lang, split=split, streaming=True, trust_remote_code=False)
        rows = []
        for i, row in enumerate(dataset):
            if limit is not None and i >= limit:
                break
            text = row.get("sentence") or row.get("text")
            if text is None:
                raise KeyError(f"No sentence field in FLORES row for {lang}: {list(row)}")
            rows.append({"sentence_id": row.get("id", i), "text": text})
        frames[lang] = rows
    # Do not silently pair different row IDs.
    ids = [[str(row["sentence_id"]) for row in rows] for rows in frames.values()]
    if ids and any(group != ids[0] for group in ids[1:]):
        raise ValueError("FLORES language rows are not aligned by ID; refusing positional pairing")
    return [{"sentence_id": row["sentence_id"], **{lang: frames[lang][i]["text"] for lang in languages}} for i, row in enumerate(next(iter(frames.values()), []))]
