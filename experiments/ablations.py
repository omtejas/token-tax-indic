from token_tax.config import load_config
from token_tax.benchmark import run_tokenizer_benchmark

if __name__ == "__main__":
    config = load_config("configs/experiment.yaml")
    # The configured normalization forms (none and NFC) are both run and retained in the CSV.
    config["normalization"]["forms"] = ["none", "NFC"]
    csv_path, metadata_path = run_tokenizer_benchmark(config)
    print(csv_path)
    print(metadata_path)
