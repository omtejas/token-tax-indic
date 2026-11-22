from __future__ import annotations
import csv, json, platform, subprocess
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
from .data import load_flores_parallel, LANGUAGES
from .metrics import normalize_text, summarize_token_counts, context_window_capacity
from .statistics import paired_bootstrap_mean_ci
from .tokenizers import load_hf_tokenizer, count_tokens


def environment_metadata() -> dict:
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "datasets", "numpy"):
        try: packages[name] = version(name)
        except PackageNotFoundError: packages[name] = None
    gpu = {"available": False, "name": None, "total_vram_bytes": None}
    try:
        import torch
        gpu["available"] = torch.cuda.is_available()
        if gpu["available"]:
            props = torch.cuda.get_device_properties(0)
            gpu.update(name=props.name, total_vram_bytes=props.total_memory, cuda_version=torch.version.cuda)
    except Exception as exc:
        gpu["error"] = str(exc)
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "platform": platform.platform(), "packages": packages, "gpu": gpu}


def run_tokenizer_benchmark(config: dict, output_dir: str | Path | None = None) -> tuple[Path, Path]:
    output = Path(output_dir or config.get("project", {}).get("output_dir", "results")); output.mkdir(parents=True, exist_ok=True)
    data_cfg = config.get("dataset", {})
    languages = list(data_cfg.get("languages", LANGUAGES))
    rows = load_flores_parallel(languages, data_cfg.get("split", "dev"), data_cfg.get("max_parallel_rows", 300), data_cfg.get("name", "Muennighoff/flores200"))
    specs = config.get("tokenizers", {}).get("candidates", [])
    if len(specs) < 5:
        raise ValueError("Configure at least five tokenizer candidates for the required study")
    records = []
    for spec in specs:
        tokenizer = load_hf_tokenizer(spec)
        for row in rows:
            for lang in languages:
                original = row[lang]
                for form in config.get("normalization", {}).get("forms", ["none"]):
                    text = normalize_text(original, form)
                    count = count_tokens(tokenizer, text)
                    records.append({"sentence_id": row["sentence_id"], "tokenizer_id": spec["id"], "tokenizer_model": spec["model_id"], "language_code": lang, "language": languages[lang], "normalization": form, "characters": len(text), "whitespace_words": len(text.split()), "tokens": count, "tokens_per_character": count / max(len(text), 1), "tokens_per_word": count / max(len(text.split()), 1)})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output / f"tokenizer_metrics_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]) if records else ["sentence_id", "tokenizer_id", "language_code", "tokens"]); writer.writeheader(); writer.writerows(records)
    summary = {}
    for spec in specs:
        for lang in languages:
            vals = [r["tokens"] for r in records if r["tokenizer_id"] == spec["id"] and r["language_code"] == lang and r["normalization"] == "none"]
            english = [r["tokens"] for r in records if r["tokenizer_id"] == spec["id"] and r["language_code"] == "eng_Latn" and r["normalization"] == "none"]
            summary[f"{spec['id']}::{lang}"] = summarize_token_counts(vals, english if lang != "eng_Latn" and len(vals) == len(english) else None)
            if lang != "eng_Latn" and len(vals) == len(english) and vals:
                summary[f"{spec['id']}::{lang}"]["paired_difference_ci_vs_english"] = paired_bootstrap_mean_ci(vals, english)
    context_window = int(config.get("benchmark", {}).get("context_window", 8192))
    context_summary = {}
    for spec in specs:
        english_counts = [r["tokens"] for r in records if r["tokenizer_id"] == spec["id"] and r["language_code"] == "eng_Latn" and r["normalization"] == "none"]
        english_capacity = context_window_capacity(english_counts, context_window)
        for lang in languages:
            counts = [r["tokens"] for r in records if r["tokenizer_id"] == spec["id"] and r["language_code"] == lang and r["normalization"] == "none"]
            report = context_window_capacity(counts, context_window)
            report["effective_context_loss_pct_vs_english"] = (100 * (english_capacity["complete_sentences"] - report["complete_sentences"]) / english_capacity["complete_sentences"]) if lang != "eng_Latn" and english_capacity["complete_sentences"] else 0.0
            context_summary[f"{spec['id']}::{lang}"] = report
    meta = {"run_id": stamp, "status": "measured", "experiment": "tokenizer_comparison", "environment": environment_metadata(), "row_count": len(rows), "record_count": len(records), "summary": summary, "context_efficiency": context_summary, "config": config}
    json_path = output / f"run_metadata_{stamp}.json"; json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path
