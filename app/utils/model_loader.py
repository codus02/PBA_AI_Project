from __future__ import annotations

from typing import Iterable, Literal, Tuple

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from app.utils.config import QWEN3_EMBED_MODEL, QWEN3_MODEL

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


def load_qwen3_embedding():
    cache_key = f"{QWEN3_EMBED_MODEL}:embed"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    tokenizer = AutoTokenizer.from_pretrained(
        QWEN3_EMBED_MODEL, padding_side="left", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        QWEN3_EMBED_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    _CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


@torch.no_grad()
def embed_texts(
    texts: Iterable[str],
    batch_size: int = 16,
    max_length: int = 512,
    instruction: str | None = None,
) -> torch.Tensor:
    """Qwen3-Embedding 인코딩. 마지막 토큰 풀링 + L2 정규화."""
    tokenizer, model = load_qwen3_embedding()
    device = next(model.parameters()).device

    items = list(texts)
    if instruction:
        items = [f"Instruct: {instruction}\nQuery: {t}" for t in items]

    out_chunks: list[torch.Tensor] = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden = model(**enc).last_hidden_state  # (B, L, H)
        # left padding이라 마지막 실제 토큰 = 시퀀스 마지막 위치
        last = hidden[:, -1]
        last = torch.nn.functional.normalize(last, p=2, dim=1)
        out_chunks.append(last.float().cpu())

    return torch.cat(out_chunks, dim=0)
