from token_tax.config import load_config
from token_tax.gpu import run_gpu_benchmark

if __name__ == "__main__":
    csv_path, metadata_path = run_gpu_benchmark(load_config("configs/experiment.yaml"), load_config("configs/model.yaml"))
    print(csv_path)
    print(metadata_path)
