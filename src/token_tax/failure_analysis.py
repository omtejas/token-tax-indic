from __future__ import annotations
import pandas as pd

REQUIRED = {"sentence_id", "tokenizer_id", "language_code", "normalization", "tokens", "text"}

def find_failure_examples(csv_path, top_n=20):
    """Build auditable tokenizer edge-case rows; never infer truncation or GPU OOM."""
    df = pd.read_csv(csv_path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    frames = []

    # High fertility relative to the same tokenizer's English sentence.
    base = df[df.normalization.eq("none")].copy()
    english = base[base.language_code.eq("eng_Latn")][["sentence_id", "tokenizer_id", "tokens"]].rename(columns={"tokens": "english_tokens"})
    indic = base[~base.language_code.eq("eng_Latn")].merge(english, on=["sentence_id", "tokenizer_id"], how="inner", validate="many_to_one")
    indic["token_ratio_vs_english"] = indic.apply(lambda r: r.tokens / r.english_tokens if r.english_tokens else None, axis=1)
    high = indic.sort_values("token_ratio_vs_english", ascending=False).groupby(["tokenizer_id", "language_code"], group_keys=False).head(top_n).copy()
    high["failure_type"] = "high_fertility"
    frames.append(high)

    # Text whose code-point sequence changes under NFC, with token-count impact.
    keys = ["sentence_id", "tokenizer_id", "language_code"]
    none = df[df.normalization.eq("none")][keys + ["text", "tokens"]].rename(columns={"text":"text_none", "tokens":"tokens_none"})
    nfc = df[df.normalization.eq("NFC")][keys + ["text", "tokens"]].rename(columns={"text":"text_nfc", "tokens":"tokens_nfc"})
    unicode = none.merge(nfc, on=keys, how="inner", validate="one_to_one")
    unicode = unicode[unicode.text_none.fillna("").ne(unicode.text_nfc.fillna(""))].copy()
    if not unicode.empty:
        unicode["token_delta_nfc_minus_none"] = unicode.tokens_nfc - unicode.tokens_none
        unicode["failure_type"] = "unicode_normalization_changed_text"
        frames.append(unicode)

    # Empty and zero-token cases are explicit diagnostics.
    empty = df[df.text.isna() | df.text.astype(str).str.strip().eq("")].copy()
    if not empty.empty:
        empty["failure_type"] = "empty_or_whitespace_text"
        frames.append(empty)
    zero = df[df.text.notna() & df.text.astype(str).str.strip().ne("") & df.tokens.fillna(0).eq(0)].copy()
    if not zero.empty:
        zero["failure_type"] = "nonempty_text_zero_tokens"
        frames.append(zero)
    replacement = df[df.text.astype(str).str.contains(chr(0xFFFD), regex=False, na=False)].copy()
    if not replacement.empty:
        replacement["failure_type"] = "unicode_replacement_character"
        frames.append(replacement)

    if not frames:
        return pd.DataFrame(columns=list(df.columns) + ["failure_type"])
    result = pd.concat(frames, ignore_index=True, sort=False)
    # Stable unique diagnostic rows; the same row can legitimately have different failure types.
    return result.drop_duplicates(subset=["failure_type", "sentence_id", "tokenizer_id", "language_code", "normalization"])

def find_high_fertility_examples(csv_path, top_n=20):
    result = find_failure_examples(csv_path, top_n)
    return result[result.failure_type.eq("high_fertility")]

def save_failure_examples(csv_path, output_path, top_n=20):
    result = find_failure_examples(csv_path, top_n)
    result.to_csv(output_path, index=False)
    return output_path
