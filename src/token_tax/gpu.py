from __future__ import annotations
import time


def gpu_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch_installed": False, "cuda_available": False, "message": "Install a CUDA-enabled PyTorch build to run GPU experiments."}
    info = {"torch_installed": True, "torch_version": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if info["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        info.update(name=p.name, total_memory_bytes=p.total_memory, free_memory_bytes=torch.cuda.mem_get_info(0)[0])
    return info


def benchmark_forward(model, encoded, warmup=3, trials=20):
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is not available to PyTorch")
    model.eval(); encoded = {k: v.to("cuda") for k, v in encoded.items()}
    with torch.inference_mode():
        for _ in range(warmup): model(**encoded)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(trials):
            start = time.perf_counter(); model(**encoded); torch.cuda.synchronize(); samples.append((time.perf_counter()-start)*1000)
    samples.sort()
    return {"trials": trials, "latency_ms_median": samples[len(samples)//2], "latency_ms_p95": samples[min(int(.95*len(samples)), len(samples)-1)], "latency_ms_mean": sum(samples)/len(samples), "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "batch_size": int(next(iter(encoded.values())).shape[0]), "input_tokens": int(next(iter(encoded.values())).shape[-1])}
