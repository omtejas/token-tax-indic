import argparse, json
from pathlib import Path
from .config import load_config
from .benchmark import run_tokenizer_benchmark, environment_metadata
from .gpu import gpu_info


def main():
    parser = argparse.ArgumentParser(prog="token-tax")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("tokenize"); p.add_argument("--config", default="configs/experiment.yaml")
    p = sub.add_parser("gpu-info")
    p = sub.add_parser("gpu-benchmark"); p.add_argument("--config", default="configs/experiment.yaml"); p.add_argument("--model-config", default="configs/model.yaml")
    args = parser.parse_args()
    if args.command == "gpu-info": print(json.dumps(gpu_info(), indent=2)); return
    if args.command == "tokenize":
        csv_path, json_path = run_tokenizer_benchmark(load_config(args.config)); print(f"CSV: {csv_path}\nMetadata: {json_path}"); return
    if args.command == "gpu-benchmark":
        model_cfg = load_config(args.model_config).get("model", {})
        if not model_cfg.get("enabled", False): raise SystemExit("GPU model benchmark is disabled. Verify model access/licence and set model.enabled=true in configs/model.yaml.")
        raise SystemExit("Model benchmark adapter is intentionally gated until a native tokenizer/model pair is verified. See README and paper protocol.")

if __name__ == "__main__": main()
