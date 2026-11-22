# The Token Tax
### Measuring tokenizer inefficiency and inference cost across Indian languages

A reproducible research project comparing how tokenizers represent equivalent English, Hindi, Kannada, Tamil, and Telugu text, then measuring the downstream inference implications with compatible tokenizer–model pairs.

> **Research integrity:** this repository contains experiment code and empty result templates. No benchmark result is claimed until it is produced by a recorded run on actual hardware.

## Research questions
1. How do token counts differ across languages and tokenizer families on aligned parallel sentences?
2. How does tokenization affect context-window occupancy and the number of equivalent sentences that fit?
3. For compatible model/tokenizer pairs, how do prompt length, prefill latency, throughput, and peak GPU memory differ?
4. How sensitive are results to Unicode normalization and input length?

## Experimental design
- Dataset: FLORES-200 parallel translations; preserve sentence IDs and document dataset revision, licence, and split.
- Languages: English (`eng_Latn`), Hindi (`hin_Deva`), Kannada (`kan_Knda`), Tamil (`tam_Taml`), Telugu (`tel_Telu`).
- Tokenizer study: at least five tokenizer configurations, pinned by model/tokenizer revision. Tokenizer-only comparisons do not require GPU.
- Model study: only use each model with its native tokenizer and embedding vocabulary. Cross-model system comparisons are not described as a causal tokenizer-only effect.
- GPU study: warm-up, synchronized timed runs, repeated trials, batch/sequence sweeps, peak allocated/reserved memory, and hardware/software metadata.
- Statistics: paired sentence-level comparisons and bootstrap confidence intervals. Random seeds are recorded for stochastic model operations; deterministic token counts are not artificially assigned seeds.
- Ablations: NFC normalization on/off and sentence-length groups. Normalization effects are reported, not assumed harmless.

## Quick start (Windows PowerShell)
Python 3.14 may not yet be supported by every optional ML dependency. Use Python 3.11 or 3.12 in a dedicated environment for the most predictable Windows GPU stack.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Run the unit tests:
```powershell
python -m pytest
```

Run the tokenizer benchmark after configuring `configs/experiment.yaml`:
```powershell
python -m token_tax.cli tokenize --config configs/experiment.yaml
```

Run a GPU capability check:
```powershell
python -m token_tax.cli gpu-info
```

Run the model benchmark only after setting a verified, accessible model ID in `configs/model.yaml`:
```powershell
python -m token_tax.cli gpu-benchmark --config configs/experiment.yaml --model-config configs/model.yaml
```

Launch the dashboard:
```powershell
streamlit run app.py
```

## Hardware profile
The initial target is an NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB VRAM. The benchmark detects the actual device at runtime and does not assume a fixed CUDA toolkit version. Leave VRAM headroom for Windows. Start with small models and batch size 1; increase only after a successful warm-up.

## Repository layout
- `src/token_tax/`: reusable data, tokenizer, metric, GPU, analysis, and CLI modules.
- `configs/`: explicit experiment and model configuration.
- `notebooks/`: Colab-compatible workflow.
- `experiments/`: executable experiment entry points.
- `tests/`: offline unit tests.
- `results/`: generated outputs (not committed by default).
- `paper/`: manuscript scaffold and reproducibility checklist.

## Data, model, and licence notes
FLORES-200, pretrained models, and tokenizer vocabularies retain their own terms and licences. The repository's code licence does not relicense those assets. Download data through the supported Hugging Face datasets interface; do not commit dataset dumps or model weights.

## Reproducibility outputs
Each run writes a machine-readable JSON metadata file and CSV metrics including timestamp, git revision when available, configuration, dataset/model/tokenizer revisions, Python/package versions, GPU name/VRAM, precision, batch size, sequence lengths, trial count, and warnings. Results are tagged as measured, estimated, or unavailable.

## Limitations
- Cross-language token counts are not a measure of translation quality.
- Different model families differ in architecture, training, and tokenizer; comparisons between them are system-level.
- GPU timings are hardware-, driver-, thermal-, and load-dependent.
- Energy metrics are omitted unless reliable power telemetry is available.
- API cost is only estimated when a matching provider/model tokenizer and a dated public price are explicitly configured.

## Project status
Foundation implementation. Actual local GPU and full FLORES runs are pending execution on the target PC.
