from __future__ import annotations
import pandas as pd

def find_high_fertility_examples(csv_path, top_n=20):
    """Return sentence-level Indic/English token-ratio outliers for manual review."""
    df=pd.read_csv(csv_path)
    required={"sentence_id","tokenizer_id","language_code","normalization","tokens","text"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    base=df[df.normalization=="none"]
    english=base[base.language_code=="eng_Latn"][["sentence_id","tokenizer_id","tokens"]].rename(columns={"tokens":"english_tokens"})
    indic=base[base.language_code!="eng_Latn"].merge(english,on=["sentence_id","tokenizer_id"],how="inner",validate="many_to_one")
    indic["token_ratio_vs_english"]=indic.apply(lambda r: r.tokens/r.english_tokens if r.english_tokens else None,axis=1)
    return indic.sort_values("token_ratio_vs_english",ascending=False).groupby(["tokenizer_id","language_code"],group_keys=False).head(top_n)

def save_failure_examples(csv_path, output_path, top_n=20):
    result=find_high_fertility_examples(csv_path,top_n)
    result.to_csv(output_path,index=False)
    return output_path
