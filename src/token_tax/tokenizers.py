from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any

@dataclass
class TokenizerSpec:
    id: str
    label: str
    model_id: str
    revision: str = "main"
    trust_remote_code: bool = False


def load_hf_tokenizer(spec: dict[str, Any]):
    from transformers import AutoTokenizer
    if spec.get("trust_remote_code", False):
        raise ValueError("trust_remote_code is disabled by default; review the repository code before opting in")
    kwargs = {
        "revision": spec.get("revision", "main"),
        "trust_remote_code": False,
    }
    model_id = str(spec.get("model_id", "")).strip()
    tokenizer_id = str(spec.get("id", "")).strip().lower()
    if tokenizer_id == "gpt2" or model_id.lower().rstrip("/").endswith("gpt2"):
        # Avoid Transformers' tokenizer_config.json parser. A truncated 51-byte
        # metadata file can be cached even when the actual GPT-2 vocab is valid.
        from huggingface_hub import hf_hub_download
        from transformers import GPT2TokenizerFast
        vocab_file = hf_hub_download(model_id or "gpt2", "vocab.json", revision=kwargs["revision"])
        merges_file = hf_hub_download(model_id or "gpt2", "merges.txt", revision=kwargs["revision"])
        return GPT2TokenizerFast(vocab_file=vocab_file, merges_file=merges_file)
    if tokenizer_id == "mbert" or model_id == "bert-base-multilingual-cased":
        from huggingface_hub import hf_hub_download
        from transformers import BertTokenizerFast
        vocab_file = hf_hub_download(model_id, "vocab.txt", revision=kwargs["revision"])
        return BertTokenizerFast(vocab_file=vocab_file, do_lower_case=False)
    if tokenizer_id == "xlm-roberta" or model_id == "xlm-roberta-base":
        from huggingface_hub import hf_hub_download
        from transformers import XLMRobertaTokenizerFast
        vocab_file = hf_hub_download(model_id, "sentencepiece.bpe.model", revision=kwargs["revision"])
        return XLMRobertaTokenizerFast(vocab_file=vocab_file)
    if tokenizer_id.startswith("qwen") or model_id.lower().startswith("qwen/"):
        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast
        tokenizer_file = hf_hub_download(model_id, "tokenizer.json", revision=kwargs["revision"])
        return PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    if tokenizer_id == "indicbertv2" or "indicbert" in model_id.lower():
        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast
        tokenizer_file = hf_hub_download(model_id, "tokenizer.json", revision=kwargs["revision"])
        return PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    try:
        return AutoTokenizer.from_pretrained(spec["model_id"], **kwargs)
    except (json.JSONDecodeError, ValueError) as exc:
        # Interrupted downloads can leave an empty tokenizer_config.json in the
        # HF cache. Force one clean retry instead of failing the whole sweep.
        if "JSON" not in str(exc) and not isinstance(exc, json.JSONDecodeError):
            raise
        try:
            return AutoTokenizer.from_pretrained(spec["model_id"], force_download=True, **kwargs)
        except json.JSONDecodeError:
            # Some Transformers/HF-cache combinations fail while parsing the
            # tiny GPT-2 tokenizer_config.json even though vocab files are valid.
            # The concrete tokenizer class can load the required vocab directly.
            if spec["model_id"].lower() == "gpt2":
                from transformers import GPT2TokenizerFast
                return GPT2TokenizerFast.from_pretrained(
                    spec["model_id"], force_download=True, **kwargs
                )
            raise


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))
