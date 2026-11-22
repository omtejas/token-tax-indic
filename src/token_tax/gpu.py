from __future__ import annotations
import csv, json, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .data import load_flores_parallel, LANGUAGES
from .tokenizers import load_hf_tokenizer
from .benchmark import environment_metadata

def gpu_info():
    try: import torch
    except ImportError: return {"torch_installed":False,"cuda_available":False,"message":"Install CUDA-enabled PyTorch to run GPU experiments."}
    info={"torch_installed":True,"torch_version":torch.__version__,"cuda_build":torch.version.cuda,"cuda_available":torch.cuda.is_available()}
    if info["cuda_available"]:
        p=torch.cuda.get_device_properties(0); free,total=torch.cuda.mem_get_info(0)
        info.update(name=p.name,total_memory_bytes=total,free_memory_bytes=free)
    return info

def _dtype(name, torch):
    options={"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}
    if name not in options: raise ValueError(f"Unsupported torch dtype: {name}")
    return options[name]

def run_gpu_benchmark(experiment_config, model_config, output_dir=None):
    import torch
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is not available to PyTorch. Check the installed wheel and NVIDIA driver.")
    mc=model_config.get("model",{})
    if not mc.get("enabled",False): raise ValueError("Set model.enabled=true only after reviewing model licence and revision.")
    model_id=mc["model_id"]; revision=mc.get("revision","main")
    tok=load_hf_tokenizer({"model_id":model_id,"revision":revision,"trust_remote_code":False})
    dtype=_dtype(mc.get("torch_dtype","float16"),torch)
    model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,torch_dtype=dtype,low_cpu_mem_usage=True,trust_remote_code=False).to("cuda").eval()
    dc=experiment_config.get("dataset",{}); languages=dc.get("languages",LANGUAGES)
    rows=load_flores_parallel(list(languages),dc.get("split","dev"),dc.get("max_parallel_rows",300),dc.get("name","yash9439/flores200"))
    bc=experiment_config.get("benchmark",{}); warmups=int(bc.get("warmup_runs",3)); trials=int(bc.get("timed_runs",20))
    lengths=bc.get("max_sequence_lengths",[128,256,512]); batches=bc.get("batch_sizes",[1])
    output=Path(output_dir or experiment_config.get("project",{}).get("output_dir","results")); output.mkdir(parents=True,exist_ok=True)
    records=[]; cfg=model.config
    layers=int(cfg.num_hidden_layers); heads=int(getattr(cfg,"num_key_value_heads",cfg.num_attention_heads)); dim=int(cfg.hidden_size)//int(cfg.num_attention_heads); bytes_per=2 if dtype in (torch.float16,torch.bfloat16) else 4
    for lang, label in languages.items():
        corpus=chr(10).join(r[lang] for r in rows)
        for seq_len in lengths:
            encoded=tok(corpus,return_tensors="pt",truncation=True,max_length=int(seq_len),padding="max_length",add_special_tokens=True)
            actual=int(encoded["attention_mask"].sum().item())
            for batch in batches:
                inputs={k:v.repeat(int(batch),1).to("cuda") for k,v in encoded.items()}
                with torch.inference_mode():
                    for _ in range(warmups): model(**inputs,use_cache=True)
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); samples=[]
                    for _ in range(trials):
                        start=time.perf_counter(); model(**inputs,use_cache=True); torch.cuda.synchronize(); samples.append((time.perf_counter()-start)*1000)
                samples.sort(); p95=samples[min(int(.95*len(samples)),len(samples)-1)]
                records.append({"model_id":model_id,"model_revision":revision,"resolved_model_revision":getattr(model.config,"_commit_hash",None),"language_code":lang,"language":label,"requested_sequence_length":int(seq_len),"actual_nonpad_tokens":actual,"batch_size":int(batch),"precision":str(dtype).replace("torch.",""),"trials":trials,"prefill_latency_ms_median":float(np.median(samples)),"prefill_latency_ms_p95":float(p95),"prefill_latency_ms_mean":float(np.mean(samples)),"peak_allocated_bytes":int(torch.cuda.max_memory_allocated()),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved()),"throughput_nonpad_tokens_per_second":float(int(batch)*actual/(float(np.median(samples))/1000)), "theoretical_kv_cache_bytes":2*int(batch)*int(seq_len)*layers*heads*dim*bytes_per,"kv_cache_status":"theoretical_from_model_config"})
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    csv_path=output/f"gpu_metrics_{stamp}.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    metadata={"run_id":stamp,"status":"measured","experiment":"native_model_prefill","environment":environment_metadata(),"model":{"id":model_id,"requested_revision":revision,"resolved_revision":getattr(model.config,"_commit_hash",None),"tokenizer_resolved_revision":getattr(tok,"init_kwargs",{}).get("_commit_hash"),"config":model.config.to_dict()},"dataset":{"id":dc.get("name","yash9439/flores200"),"split":dc.get("split","dev")},"records":len(records),"warmup_runs":warmups,"timed_runs":trials,"results_file":str(csv_path)}
    meta_path=output/f"gpu_metadata_{stamp}.json"; meta_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    del model; torch.cuda.empty_cache()
    return csv_path,meta_path
