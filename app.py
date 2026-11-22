from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="The Token Tax", page_icon="🌐", layout="wide")
st.title("The Token Tax")
st.caption("Tokenizer efficiency across English, Hindi, Kannada, Tamil and Telugu")
files=sorted(Path("results").glob("tokenizer_metrics_*.csv"),reverse=True)
if not files:
    st.info("No measured tokenizer runs yet.")
    st.code("python -m token_tax.cli tokenize --config configs/experiment.yaml",language="powershell")
    st.caption("This dashboard never displays fabricated sample results.")
else:
    selected=st.selectbox("Tokenizer run",files,format_func=lambda p:p.name)
    df=pd.read_csv(selected)
    a,b=st.columns(2)
    with a: langs=st.multiselect("Languages",sorted(df.language.dropna().unique()),default=sorted(df.language.dropna().unique()))
    with b: toks=st.multiselect("Tokenizers",sorted(df.tokenizer_id.dropna().unique()),default=sorted(df.tokenizer_id.dropna().unique()))
    view=df[(df.language.isin(langs))&(df.tokenizer_id.isin(toks))&(df.normalization=="none")]
    summary=view.groupby(["language","tokenizer_id"],as_index=False).agg(mean_tokens=("tokens","mean"),median_tokens=("tokens","median"),tokens_per_character=("tokens_per_character","mean"))
    st.subheader("Average tokens per aligned sentence")
    st.bar_chart(summary,x="language",y="mean_tokens",color="tokenizer_id")
    st.dataframe(summary,use_container_width=True,hide_index=True)
    st.download_button("Download filtered tokenizer CSV",view.to_csv(index=False).encode("utf-8"),file_name="token_tax_tokenizer_results.csv",mime="text/csv")

st.divider()
st.subheader("GPU prefill benchmark")
gpu_files=sorted(Path("results").glob("gpu_metrics_*.csv"),reverse=True)
if not gpu_files:
    st.info("No measured GPU runs yet. Run the native-model GPU benchmark on a CUDA-enabled device.")
else:
    gpu_path=st.selectbox("GPU run",gpu_files,format_func=lambda p:p.name)
    gpu=pd.read_csv(gpu_path)
    st.dataframe(gpu,use_container_width=True,hide_index=True)
    st.download_button("Download GPU CSV",gpu.to_csv(index=False).encode("utf-8"),file_name="token_tax_gpu_results.csv",mime="text/csv")
