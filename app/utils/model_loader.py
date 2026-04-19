from __future__ import annotations

from typing import Literal, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from app.utils.config import QWEN3_MODEL

_CACHE: dict[str, Tuple] = {}


def _bnb_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_qwen3(quantization: Literal["4bit", "fp16"] = "4bit"):
    cache_key = f"{QWEN3_MODEL}:{quantization}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL, trust_remote_code=True)

    if quantization == "4bit":
        model = AutoModelForCausalLM.from_pretrained(
            QWEN3_MODEL,
            quantization_config=_bnb_4bit_config(),
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            QWEN3_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    _CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model
