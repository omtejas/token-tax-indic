# Dataset handling

The configured source is the public consolidated Parquet mirror `yash9439/flores200`. It stores one row per aligned FLORES-200 sentence group with language-code columns such as `eng_Latn`, `hin_Deva`, `kan_Knda`, `tam_Taml`, and `tel_Telu`. The development split contains 997 examples according to its dataset card.

The loader reads a single split and selects the five columns from each row, preserving alignment by construction. It does not execute remote dataset scripts. The run metadata records the dataset repository revision when Hugging Face Hub metadata is available.

FLORES-200 originates from the NLLB/FLORES work and is licensed CC-BY-SA 4.0. The consolidated mirror is a third-party redistribution: verify its provenance and preserve the original attribution/share-alike obligations before distributing results or derived data. Do not commit downloaded corpus files or caches.
