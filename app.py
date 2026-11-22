from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="The Token Tax", page_icon="🌐", layout="wide")
st.title("The Token Tax")
st.caption("Tokenizer efficiency across English, Hindi, Kannada, Tamil and Telugu")
files = sorted(Path("results").glob("tokenizer_metrics_*.csv"), reverse=True)
if not files:
    st.info("No measured runs found yet. Run the tokenizer benchmark to populate this dashboard.")
    st.code("python -m token_tax.cli tokenize --config configs/experiment.yaml", language="powershell")
    st.caption("No synthetic or illustrative benchmark data is displayed.")
    st.stop()
selected = st.selectbox("Experiment run", files, format_func=lambda p: p.name)
df = pd.read_csv(selected)
left, right = st.columns(2)
with left:
    langs = sorted(df["language"].dropna().unique())
    chosen_langs = st.multiselect("Languages", langs, default=langs)
with right:
    toks = sorted(df["tokenizer_id"].dropna().unique())
    chosen_toks = st.multiselect("Tokenizers", toks, default=toks)
view = df[(df.language.isin(chosen_langs)) & (df.tokenizer_id.isin(chosen_toks)) & (df.normalization == "none")]
if view.empty:
    st.warning("No rows match these filters.")
else:
    summary = view.groupby(["language", "tokenizer_id"], as_index=False).agg(mean_tokens=("tokens", "mean"), median_tokens=("tokens", "median"), tokens_per_character=("tokens_per_character", "mean"))
    st.subheader("Average tokens per aligned sentence")
    st.bar_chart(summary, x="language", y="mean_tokens", color="tokenizer_id")
    st.subheader("Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", view.to_csv(index=False).encode("utf-8"), file_name="token_tax_filtered.csv", mime="text/csv")
