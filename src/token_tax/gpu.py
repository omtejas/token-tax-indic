from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import load_flores_parallel, resolve_languages
from .tokenizers import load_hf_tokenizer
from .benchmark import environment_metadata, _hub_model_revision, _dataset_revision


def gpu_info():
    try:
        import torch
    except ImportError:
        return {"torch_installed": False, "cuda_available": False, "message": "Install CUDA-enabled PyTorch to run GPU experiments."}
    info = {"torch_installed": True, "torch_version": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if info["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        info.update(name=props.name, total_memory_bytes=total, free_memory_bytes=free)
    return info


def _dtype(name, torch):
    options = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in options:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return options[name]


def _read_nvidia_smi():
    """Best-effort Windows/Linux telemetry; N/A fields remain unavailable."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=utilization.gpu,power.draw,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True,
        )
        parts = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]
        if len(parts) < 3:
            return None

        def number(value):
            try:
                parsed = float(value.split()[0])
                return parsed if math.isfinite(parsed) else None
            except (ValueError, IndexError):
                return None

        return {"gpu_utilization_percent": number(parts[0]), "power_watts": number(parts[1]), "gpu_memory_used_mib": number(parts[2])}
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


class _TelemetrySampler:
    """Samples nvidia-smi during a timed section; energy is explicitly estimated."""
    def __init__(self, interval_seconds=0.25):
        self.interval = max(0.1, float(interval_seconds))
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None
        self.started = None
        self.finished = None

    def _sample_loop(self):
        while not self.stop_event.is_set():
            sample = _read_nvidia_smi()
            if sample is not None:
                self.samples.append(sample)
            self.stop_event.wait(self.interval)
        sample = _read_nvidia_smi()
        if sample is not None:
            self.samples.append(sample)

    def __enter__(self):
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finished = time.perf_counter()
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)

    def result(self):
        end = self.finished if self.finished is not None else time.perf_counter()
        start = self.started if self.started is not None else end
        elapsed = max(0.0, end - start)
        utilization = [s["gpu_utilization_percent"] for s in self.samples if s["gpu_utilization_percent"] is not None]
        power = [s["power_watts"] for s in self.samples if s["power_watts"] is not None]
        memory = [s["gpu_memory_used_mib"] for s in self.samples if s["gpu_memory_used_mib"] is not None]
        avg_power = float(np.mean(power)) if power else None
        return {
            "telemetry_source": "nvidia-smi",
            "telemetry_sample_count": len(self.samples),
            "gpu_utilization_mean_percent": float(np.mean(utilization)) if utilization else None,
            "gpu_utilization_max_percent": float(np.max(utilization)) if utilization else None,
            "gpu_power_mean_watts": avg_power,
            "gpu_memory_used_mean_mib": float(np.mean(memory)) if memory else None,
            "energy_joules_estimated": avg_power * elapsed if avg_power is not None else None,
            "energy_method": "mean_nvidia_smi_power_watts_times_timed_wall_seconds" if avg_power is not None else "unavailable",
            "telemetry_status": "measured_samples" if self.samples else "unavailable",
        }


def _write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _t_critical_95(df):
    values = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    return values.get(df, 1.96)


def summarize_repetitions(rows, metrics):
    """Summarize repetition-level measurements with two-sided Student-t 95% CIs."""
    groups = {}
    keys = ("language_code", "requested_sequence_length", "batch_size")
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    summaries = []
    for key, group in groups.items():
        summary = dict(zip(keys, key))
        summary["repetitions"] = len(group)
        for metric in metrics:
            values = [float(r[metric]) for r in group if r.get(metric) not in (None, "")]
            if not values:
                summary.update({f"{metric}_mean": None, f"{metric}_std": None, f"{metric}_ci95_low": None, f"{metric}_ci95_high": None})
                continue
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else None
            margin = _t_critical_95(len(values) - 1) * std / math.sqrt(len(values)) if std is not None else None
            summary.update({
                f"{metric}_mean": mean,
                f"{metric}_std": std,
                f"{metric}_ci95_low": mean - margin if margin is not None else None,
                f"{metric}_ci95_high": mean + margin if margin is not None else None,
            })
        summaries.append(summary)
    return summaries


def run_gpu_benchmark(experiment_config, model_config, output_dir=None):
    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch. Check the installed wheel and NVIDIA driver.")
    mc = model_config.get("model", {})
    if not mc.get("enabled", False):
        raise ValueError("Set model.enabled=true only after reviewing model licence and revision.")
    model_id = mc["model_id"]
    revision = mc.get("revision", "main")
    tok = load_hf_tokenizer({"model_id": model_id, "revision": revision, "trust_remote_code": False})
    tok.padding_side = "left"  # decoder-only generation with fixed-length batches
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif tok.unk_token is not None:
            tok.pad_token = tok.unk_token
        else:
            vocab = tok.get_vocab()
            for candidate in ("<|endoftext|>", "</s>", "<pad>", "[PAD]"):
                if candidate in vocab:
                    tok.pad_token = candidate
                    break
            if tok.pad_token is None:
                raise ValueError("The model tokenizer has no usable padding token.")
    dtype = _dtype(mc.get("torch_dtype", "float16"), torch)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=False).to("cuda").eval()
    dc = experiment_config.get("dataset", {})
    language_codes, languages = resolve_languages(dc.get("languages"))
    rows = load_flores_parallel(language_codes, dc.get("split", "dev"), dc.get("max_parallel_rows", 300), dc.get("name", "yash9439/flores200"), dc.get("streaming", True))
    bc = experiment_config.get("benchmark", {})
    warmups = int(bc.get("warmup_runs", 2))
    trials = int(bc.get("timed_runs", 5))
    repetitions = int(bc.get("repetitions", 3))
    if repetitions < 3:
        raise ValueError("Faculty protocol requires at least three independent GPU repetitions.")
    if trials < 1 or warmups < 0:
        raise ValueError("timed_runs must be >=1 and warmup_runs must be >=0.")
    lengths = [int(x) for x in bc.get("max_sequence_lengths", [128, 256, 512])]
    batches = [int(x) for x in bc.get("batch_sizes", [1])]
    generation_tokens = int(bc.get("generation_tokens", 16))
    telemetry_interval = float(bc.get("telemetry_interval_seconds", 0.25))
    if generation_tokens < 2:
        raise ValueError("generation_tokens must be at least 2 for multi-token decode.")
    output = Path(output_dir or experiment_config.get("project", {}).get("output_dir", "results"))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    cfg = model.config
    baseline_allocated = int(torch.cuda.memory_allocated())
    layers = int(cfg.num_hidden_layers)
    attention_heads = int(cfg.num_attention_heads)
    kv_heads = int(getattr(cfg, "num_key_value_heads", attention_heads))
    head_dim = int(getattr(cfg, "head_dim", int(cfg.hidden_size) // attention_heads))
    bytes_per = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    resolved_revision = getattr(model.config, "_commit_hash", None) or _hub_model_revision(model_id, revision)
    model.generation_config.pad_token_id = tok.pad_token_id

    for lang, label in languages.items():
        corpus = chr(10).join(row[lang] for row in rows)
        for seq_len in lengths:
            encoded = tok(corpus, return_tensors="pt", truncation=True, max_length=seq_len, add_special_tokens=True)
            actual = int(encoded["attention_mask"].sum().item())
            encoded = tok.pad(encoded, padding="max_length", max_length=seq_len, return_tensors="pt")
            for batch in batches:
                inputs = {key: value.repeat(batch, 1).to("cuda") for key, value in encoded.items()}
                for repetition in range(1, repetitions + 1):
                    with torch.inference_mode():
                        for _ in range(warmups):
                            model(**inputs, use_cache=True)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        prefill_samples = []
                        with _TelemetrySampler(telemetry_interval) as telemetry:
                            for _ in range(trials):
                                start = time.perf_counter()
                                model(**inputs, use_cache=True)
                                torch.cuda.synchronize()
                                prefill_samples.append((time.perf_counter() - start) * 1000)
                        prefill_telemetry = telemetry.result()

                        one_token_samples = []
                        for _ in range(warmups):
                            model.generate(**inputs, max_new_tokens=1, min_new_tokens=1, do_sample=False, use_cache=True)
                        torch.cuda.synchronize()
                        with _TelemetrySampler(telemetry_interval) as telemetry:
                            for _ in range(trials):
                                start = time.perf_counter()
                                model.generate(**inputs, max_new_tokens=1, min_new_tokens=1, do_sample=False, use_cache=True)
                                torch.cuda.synchronize()
                                one_token_samples.append((time.perf_counter() - start) * 1000)
                        one_token_telemetry = telemetry.result()

                        multi_samples = []
                        for _ in range(warmups):
                            model.generate(**inputs, max_new_tokens=generation_tokens, min_new_tokens=generation_tokens, do_sample=False, use_cache=True)
                        torch.cuda.synchronize()
                        with _TelemetrySampler(telemetry_interval) as telemetry:
                            for _ in range(trials):
                                start = time.perf_counter()
                                model.generate(**inputs, max_new_tokens=generation_tokens, min_new_tokens=generation_tokens, do_sample=False, use_cache=True)
                                torch.cuda.synchronize()
                                multi_samples.append((time.perf_counter() - start) * 1000)
                        multi_telemetry = telemetry.result()
                    peak_allocated = int(torch.cuda.max_memory_allocated())
                    generated_per_trial = generation_tokens * batch
                    median_multi_ms = float(np.median(multi_samples))
                    record = {
                        "model_id": model_id,
                        "model_revision": revision,
                        "resolved_model_revision": resolved_revision,
                        "language_code": lang,
                        "language": label,
                        "requested_sequence_length": seq_len,
                        "actual_nonpad_tokens": actual,
                        "batch_size": batch,
                        "repetition_id": repetition,
                        "precision": str(dtype).replace("torch.", ""),
                        "timed_trials": trials,
                        "generation_tokens_per_sequence": generation_tokens,
                        "prefill_latency_ms_median": float(np.median(prefill_samples)),
                        "prefill_latency_ms_mean": float(np.mean(prefill_samples)),
                        "prefill_latency_ms_std_within_repetition": float(np.std(prefill_samples, ddof=1)) if len(prefill_samples) > 1 else None,
                        "prefill_latency_ms_p95": float(np.percentile(prefill_samples, 95)),
                        "prefill_throughput_nonpad_tokens_per_second": float(batch * actual / (float(np.median(prefill_samples)) / 1000)),
                        "one_token_generation_end_to_end_latency_ms_median": float(np.median(one_token_samples)),
                        "one_token_generation_end_to_end_latency_ms_p95": float(np.percentile(one_token_samples, 95)),
                        "multi_token_generation_end_to_end_latency_ms_median": median_multi_ms,
                        "multi_token_generation_end_to_end_latency_ms_p95": float(np.percentile(multi_samples, 95)),
                        "multi_token_generated_tokens_per_second": float(generated_per_trial / (median_multi_ms / 1000)),
                        "baseline_model_allocated_bytes": baseline_allocated,
                        "peak_allocated_bytes": peak_allocated,
                        "peak_allocated_delta_bytes": max(0, peak_allocated - baseline_allocated),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                        "theoretical_kv_cache_bytes": 2 * batch * seq_len * layers * kv_heads * head_dim * bytes_per,
                        "kv_cache_status": "theoretical_from_model_config",
                    }
                    for stage, telemetry_result in (
                        ("prefill", prefill_telemetry),
                        ("one_token_generation", one_token_telemetry),
                        ("multi_token_generation", multi_telemetry),
                    ):
                        for key, value in telemetry_result.items():
                            record[f"{stage}_{key}"] = value
                    energy = multi_telemetry["energy_joules_estimated"]
                    record["multi_token_generation_energy_joules_per_generated_token_estimated"] = energy / generated_per_trial if energy is not None and generated_per_trial else None
                    record["multi_token_generation_energy_status"] = "estimated_from_nvidia_smi_power" if record["multi_token_decode_energy_joules_per_generated_token_estimated"] is not None else "unavailable"
                    records.append(record)
                del inputs
                torch.cuda.empty_cache()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_csv = output / f"gpu_metrics_{stamp}.csv"
    _write_csv(raw_csv, records)
    summary_metrics = [
        "prefill_latency_ms_median",
        "one_token_generation_end_to_end_latency_ms_median",
        "multi_token_generation_end_to_end_latency_ms_median",
        "multi_token_generated_tokens_per_second",
        "peak_allocated_delta_bytes",
        "prefill_gpu_utilization_mean_percent",
        "multi_token_generation_gpu_utilization_mean_percent",
        "multi_token_decode_energy_joules_per_generated_token_estimated",
    ]
    summaries = summarize_repetitions(records, summary_metrics)
    summary_csv = output / f"gpu_repetition_summary_{stamp}.csv"
    _write_csv(summary_csv, summaries)
    metadata = {
        "run_id": stamp,
        "status": "measured",
        "experiment": "native_model_prefill_decode_repeated",
        "environment": environment_metadata(),
        "model": {
            "id": model_id,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "tokenizer_resolved_revision": getattr(tok, "init_kwargs", {}).get("_commit_hash") or _hub_model_revision(model_id, revision),
            "config": model.config.to_dict(),
        },
        "dataset": {
            "id": dc.get("name", "yash9439/flores200"),
            "split": dc.get("split", "dev"),
            "revision": _dataset_revision(dc.get("name", "yash9439/flores200")),
            "rows": len(rows),
        },
        "config": {
            "max_sequence_lengths": lengths,
            "batch_sizes": batches,
            "repetitions": repetitions,
            "warmup_runs_per_repetition": warmups,
            "timed_trials_per_repetition": trials,
            "generation_tokens": generation_tokens,
            "telemetry_interval_seconds": telemetry_interval,
            "precision": str(dtype).replace("torch.", ""),
        },
        "telemetry": {
            "source": "nvidia-smi",
            "fields": ["gpu_utilization_percent", "power_watts", "memory_used_mib"],
            "energy_note": "Estimated as mean sampled board power multiplied by timed wall duration; not a direct energy-meter measurement. Null means unavailable.",
        },
        "repetitions": repetitions,
        "records": len(records),
        "summary_records": len(summaries),
        "raw_results_file": str(raw_csv),
        "repetition_summary_file": str(summary_csv),
    }
    meta_path = output / f"gpu_metadata_{stamp}.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return raw_csv, meta_path
