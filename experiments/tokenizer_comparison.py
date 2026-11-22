from token_tax.config import load_config
from token_tax.benchmark import run_tokenizer_benchmark

if __name__ == "__main__":
    csv_path, metadata_path = run_tokenizer_benchmark(load_config("configs/experiment.yaml"))
    print(csv_path)
    print(metadata_path)
