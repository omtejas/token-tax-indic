from __future__ import annotations
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
    return AutoTokenizer.from_pretrained(spec["model_id"], revision=spec.get("revision", "main"), trust_remote_code=False)


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))
