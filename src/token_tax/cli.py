import argparse,json
from .config import load_config
from .benchmark import run_tokenizer_benchmark
from .gpu import gpu_info,run_gpu_benchmark

def main():
    parser=argparse.ArgumentParser(prog="token-tax"); sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("tokenize"); p.add_argument("--config",default="configs/experiment.yaml")
    sub.add_parser("gpu-info")
    p=sub.add_parser("gpu-benchmark"); p.add_argument("--config",default="configs/experiment.yaml"); p.add_argument("--model-config",default="configs/model.yaml")
    args=parser.parse_args()
    if args.command=="gpu-info": print(json.dumps(gpu_info(),indent=2)); return
    if args.command=="tokenize":
        csv_path,meta_path=run_tokenizer_benchmark(load_config(args.config)); print(f"CSV: {csv_path}\\nMetadata: {meta_path}"); return
    csv_path,meta_path=run_gpu_benchmark(load_config(args.config),load_config(args.model_config)); print(f"CSV: {csv_path}\\nMetadata: {meta_path}")

if __name__=="__main__": main()
