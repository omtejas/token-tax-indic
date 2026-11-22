# The Token Tax: Measuring Tokenizer Inefficiency and Inference Cost for Indian Languages

**Authors:** To be completed  
**Status:** Experimental protocol; results pending

## Abstract
This study measures tokenization efficiency for aligned English, Hindi, Kannada, Tamil, and Telugu text and investigates the inference implications using native tokenizer–model pairs. Quantitative findings will be inserted only after reproducible runs.

## 1. Introduction
Define the practical problem of token-based context and inference costs for multilingual users. State research questions and hypotheses without presenting expected ranges as findings.

## 2. Related Work
TODO: add verified primary sources on subword tokenization, multilingual language models, FLORES-200, and inference/KV-cache memory. Record access dates and licences.

## 3. Dataset and Methods
- Dataset: FLORES-200; exact release/revision, licence, split and row count: PENDING.
- Languages: English, Hindi, Kannada, Tamil, Telugu.
- Alignment: sentence IDs; verify one-to-one pairing.
- Tokenizers: at least five; exact IDs, revisions, vocabulary sizes and licences: PENDING.
- GPU models: native tokenizer/model pairs only; model architecture, parameter count, precision and licence: PENDING.
- Hardware/software: record actual GPU, VRAM, driver, CUDA build, framework versions and thermals/load where available.
- Ablations: Unicode normalization and length buckets.

## 4. Metrics
Tokens/sentence, tokens/character, tokens/whitespace-word, paired token ratio vs English, context occupancy, prefill latency, throughput, peak allocated/reserved GPU memory, and theoretical KV-cache bytes from exact architecture. Clearly separate measured from estimated metrics.

## 5. Experimental Protocol
Warm-up and synchronized timed runs; batch/sequence-length sweeps; repeated trials; report median, p95, dispersion and bootstrap confidence intervals. Use paired sentence-level analysis. Do not assign meaningless random seeds to deterministic tokenizer counts.

## 6. Results
PENDING — populate from committed raw CSV/JSON run artifacts. Include sample sizes and confidence intervals.

## 7. Failure Analysis
PENDING — inspect tokenizer edge cases, Unicode normalization differences, truncation, unsupported characters, loading failures, OOM events, and misaligned examples.

## 8. Discussion and Limitations
Distinguish tokenizer effects from model-family effects. Discuss dataset domain, translation variation, architecture, hardware, quantization, timing noise and generalizability.

## 9. Reproducibility
Provide repository commit, config files, exact dataset/model/tokenizer revisions, environment metadata, run commands and data access instructions.

## 10. Conclusion
PENDING — answer research questions based on measured evidence only.
