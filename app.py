"""Results-first dashboard for The Token Tax; displays measured files only."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path("results")

def newest_valid(pattern: str, required=()):
    for path in sorted(ROOT.glob(pattern), key=lambda p: p.name, reverse=True):
        try:
            frame = pd.read_csv(path)
            if set(required).issubset(frame.columns) and not frame.empty:
                return path, frame
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue
    return None, None

def newest_json(pattern: str):
    for path in sorted(ROOT.glob(pattern), key=lambda p: p.name, reverse=True):
        try:
            return path, json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None, None

st.set_page_config(page_title="The Token Tax", page_icon="◈", layout="wide")
st.title("The Token Tax")
st.caption("Indic-language tokenizer efficiency · native-model GPU inference · reproducible measurements")
tok_path, tok = newest_valid("tokenizer_metrics_*.csv", {"sentence_id","tokenizer_id","language_code","normalization","tokens"})
gpu_path, gpu = newest_valid("gpu_metrics_*.csv", {"language_code","requested_sequence_length","batch_size"})
rep_path, reps = newest_valid("gpu_repetition_summary_*.csv", {"language_code","requested_sequence_length","batch_size","repetitions"})
fail_path, failures = newest_valid("failure_examples.csv", {"tokenizer_id","language_code","sentence_id"})
meta_path, meta = newest_json("gpu_metadata_*.json")
run_path, run_meta = newest_json("run_metadata_*.json")

with st.sidebar:
    st.subheader("Latest valid run")
    st.write("Tokenizer:", tok_path.name if tok_path else "Not found")
    st.write("GPU:", gpu_path.name if gpu_path else "Not found")
    st.caption("Newest parseable result files are selected automatically.")

tabs = st.tabs(["Overview","Tokenizer comparison","Normalization ablation","GPU prefill/decode","Repeated-run statistics","Failure analysis","Reproducibility metadata"])

with tabs[0]:
    a,b,c,d=st.columns(4)
    a.metric("Tokenizer run", "Ready" if tok is not None else "Missing")
    b.metric("GPU run", "Ready" if gpu is not None else "Missing")
    c.metric("GPU repetitions", int(reps.repetitions.min()) if reps is not None else "Pending")
    d.metric("Languages", int(tok.language_code.nunique()) if tok is not None else 0)
    if tok is not None:
        st.write(f"Tokenizer rows: **{len(tok):,}** · aligned sentences: **{tok.sentence_id.nunique():,}** · tokenizers: **{tok.tokenizer_id.nunique()}**")
    if gpu is not None:
        st.write(f"GPU records: **{len(gpu):,}** · models: **{gpu.model_id.nunique() if 'model_id' in gpu else 'legacy run'}**")
    st.info("Only locally generated experiment artifacts are displayed. Empty or unavailable telemetry is never replaced with fabricated values.")

with tabs[1]:
    st.subheader("Tokenizer comparison")
    if tok is None: st.warning("No valid tokenizer CSV found.")
    else:
        base=tok[tok.normalization.eq("none")]
        summary=base.groupby(["tokenizer_id","language_code"],as_index=False).agg(mean_tokens=("tokens","mean"),median_tokens=("tokens","median"),sentences=("sentence_id","nunique"))
        fig=px.bar(summary,x="language_code",y="mean_tokens",color="tokenizer_id",barmode="group",labels={"mean_tokens":"Mean tokens / sentence","language_code":"Language"})
        st.plotly_chart(fig,width="stretch")
        st.dataframe(summary,width="stretch",hide_index=True)
        st.download_button("Export tokenizer summary",summary.to_csv(index=False),"tokenizer_summary.csv","text/csv")

with tabs[2]:
    st.subheader("Unicode normalization ablation")
    if tok is None or not {"none","NFC"}.issubset(set(tok.normalization.dropna())):
        st.warning("Both none and NFC normalization records are required.")
    else:
        ab=tok.groupby(["tokenizer_id","language_code","normalization"],as_index=False).agg(mean_tokens=("tokens","mean"),median_tokens=("tokens","median"),sentences=("sentence_id","nunique"))
        pivot=ab.pivot_table(index=["tokenizer_id","language_code"],columns="normalization",values="mean_tokens").reset_index()
        if {"none","NFC"}.issubset(pivot.columns):
            pivot["NFC_minus_none"]=pivot["NFC"]-pivot["none"]
            pivot["NFC_percent_change"]=100*pivot["NFC_minus_none"]/pivot["none"].replace(0,pd.NA)
        st.dataframe(pivot,width="stretch",hide_index=True)
        st.plotly_chart(px.bar(ab,x="language_code",y="mean_tokens",color="normalization",barmode="group",facet_col="tokenizer_id"),width="stretch")

with tabs[3]:
    st.subheader("GPU prefill and generation")
    if gpu is None: st.warning("No valid GPU metrics CSV found.")
    else:
        latency_cols=[c for c in ["prefill_latency_ms_median","multi_token_generation_end_to_end_latency_ms_median","one_token_generation_end_to_end_latency_ms_median","decode_latency_ms_median"] if c in gpu]
        if latency_cols:
            long=gpu.melt(id_vars=["language_code","requested_sequence_length","batch_size"],value_vars=latency_cols,var_name="stage",value_name="latency_ms")
            st.plotly_chart(px.line(long,x="requested_sequence_length",y="latency_ms",color="stage",facet_col="language_code",markers=True),width="stretch")
        st.dataframe(gpu,width="stretch",hide_index=True)
        st.download_button("Export GPU measurements",gpu.to_csv(index=False),"gpu_measurements.csv","text/csv")

with tabs[4]:
    st.subheader("Repeated-run statistics")
    if reps is None: st.warning("Repeated-run summary not found. Legacy one-run CSVs do not count as repetitions.")
    else:
        st.dataframe(reps,width="stretch",hide_index=True)
        metrics=[c for c in reps.columns if c.endswith("_mean") and c.removesuffix("_mean")+"_std" in reps.columns]
        if metrics:
            metric=st.selectbox("Metric",metrics)
            st.plotly_chart(px.bar(reps,x="language_code",y=metric,error_y=metric.removesuffix("_mean")+"_std",color="requested_sequence_length",barmode="group"),width="stretch")
        st.download_button("Export repetition statistics",reps.to_csv(index=False),"gpu_repetition_statistics.csv","text/csv")

with tabs[5]:
    st.subheader("Failure analysis")
    if fail_path is not None:
        st.dataframe(failures,width="stretch",hide_index=True)
        st.download_button("Export failure examples",failures.to_csv(index=False),"failure_examples.csv","text/csv")
    else:
        st.warning("No failure_examples.csv found. Generate high-fertility examples and add Unicode, truncation, empty-text and GPU-OOM diagnostics as they are captured.")
    if gpu is not None:
        if "peak_allocated_bytes" in gpu:
            st.write("GPU memory failures are not inferred from successful rows; inspect run logs for OOM records.")

with tabs[6]:
    st.subheader("Reproducibility metadata")
    if meta is not None:
        st.markdown("**GPU run metadata**")
        st.json(meta)
    elif run_meta is not None:
        st.json(run_meta)
    else:
        st.warning("No run metadata JSON found.")
    if meta_path: st.caption(f"Source: {meta_path.name}")
