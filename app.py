"""Local, results-first dashboard for The Token Tax study."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RESULTS_DIR = Path("results")
TOKENIZER_COLUMNS = {"language", "language_code", "tokenizer_id", "normalization", "tokens"}
GPU_COLUMNS = {"language", "requested_sequence_length", "batch_size", "prefill_latency_ms_median"}


def available_runs(pattern: str) -> list[Path]:
    return sorted(RESULTS_DIR.glob(pattern), reverse=True)


def read_result(path: Path, required: set[str]) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read {path.name}: {exc}")
        return None
    missing = required - set(frame.columns)
    if missing:
        st.error(f"{path.name} is not compatible. Missing: {', '.join(sorted(missing))}.")
        return None
    return frame


def empty_state() -> None:
    st.markdown("### Ready for your first measured run")
    st.write("The interface is ready. It intentionally stays empty until it finds a CSV created by the experiment pipeline.")
    st.code(".\\run_experiments.ps1", language="powershell")
    st.caption("Results remain local by default. No sample or estimated benchmark values are shown as measurements.")


def tokenizer_dashboard(paths: list[Path]) -> None:
    st.subheader("Tokenizer analysis")
    if not paths:
        empty_state()
        return
    path = st.selectbox("Tokenizer result", paths, format_func=lambda item: item.name, key="tokenizer_run")
    data = read_result(path, TOKENIZER_COLUMNS)
    if data is None:
        return
    data = data[data["normalization"].eq("none")].copy()
    if data.empty:
        st.warning("This run contains no unnormalized records to compare.")
        return
    left, right = st.columns(2)
    with left:
        languages = st.multiselect("Languages", sorted(data.language.dropna().unique()), default=sorted(data.language.dropna().unique()))
    with right:
        tokenizers = st.multiselect("Tokenizers", sorted(data.tokenizer_id.dropna().unique()), default=sorted(data.tokenizer_id.dropna().unique()))
    view = data[data.language.isin(languages) & data.tokenizer_id.isin(tokenizers)]
    if view.empty:
        st.info("Choose at least one language and tokenizer to display results.")
        return
    summary = view.groupby(["language", "tokenizer_id"], as_index=False).agg(
        mean_tokens=("tokens", "mean"), median_tokens=("tokens", "median"), total_tokens=("tokens", "sum"), sentences=("sentence_id", "nunique")
    )
    a, b, c = st.columns(3)
    a.metric("Aligned sentences", f"{view.sentence_id.nunique():,}")
    b.metric("Tokenizer families", f"{view.tokenizer_id.nunique():,}")
    c.metric("Mean tokens / sentence", f"{summary.mean_tokens.mean():.2f}")
    chart = px.bar(summary, x="language", y="mean_tokens", color="tokenizer_id", barmode="group", title="Tokenization cost across aligned sentences", labels={"mean_tokens": "Mean tokens per sentence", "language": "Language", "tokenizer_id": "Tokenizer"})
    chart.update_layout(legend_title_text="Tokenizer", margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(summary.sort_values(["language", "mean_tokens"]), use_container_width=True, hide_index=True)
    st.download_button("Download filtered tokenizer data", view.to_csv(index=False).encode("utf-8"), "token_tax_tokenizer_results.csv", "text/csv")


def gpu_dashboard(paths: list[Path]) -> None:
    st.subheader("GPU prefill benchmark")
    if not paths:
        st.info("No measured GPU run is available yet. Run the benchmark on a CUDA-enabled device after setup completes.")
        return
    path = st.selectbox("GPU result", paths, format_func=lambda item: item.name, key="gpu_run")
    data = read_result(path, GPU_COLUMNS)
    if data is None:
        return
    a, b, c = st.columns(3)
    a.metric("Measured scenarios", f"{len(data):,}")
    b.metric("Median prefill latency", f"{data.prefill_latency_ms_median.median():.1f} ms")
    throughput = data.get("throughput_nonpad_tokens_per_second")
    c.metric("Median throughput", f"{throughput.median():.1f} tok/s" if throughput is not None else "Unavailable")
    chart = px.line(data, x="requested_sequence_length", y="prefill_latency_ms_median", color="language", symbol="batch_size", markers=True, title="Measured GPU prefill latency", labels={"requested_sequence_length": "Requested sequence length", "prefill_latency_ms_median": "Median prefill latency (ms)"})
    chart.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(data, use_container_width=True, hide_index=True)
    st.download_button("Download GPU data", data.to_csv(index=False).encode("utf-8"), "token_tax_gpu_results.csv", "text/csv")


st.set_page_config(page_title="The Token Tax", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>.stApp { background: #f7f8fc; } [data-testid="stMetricValue"] { color: #5348d8; }</style>""", unsafe_allow_html=True)
st.title("The Token Tax")
st.caption("A reproducible dashboard for multilingual tokenizer efficiency and native-model GPU measurements.")
with st.sidebar:
    st.header("Study status")
    tokenizer_runs, gpu_runs = available_runs("tokenizer_metrics_*.csv"), available_runs("gpu_metrics_*.csv")
    st.success(f"{len(tokenizer_runs)} tokenizer run(s) found") if tokenizer_runs else st.warning("No tokenizer run found")
    st.success(f"{len(gpu_runs)} GPU run(s) found") if gpu_runs else st.warning("No GPU run found")
    st.divider()
    st.caption("Only files generated by this project are visualized. GPU figures are hardware-specific measurements, not general performance claims.")

tokenizer_dashboard(tokenizer_runs)
st.divider()
gpu_dashboard(gpu_runs)
