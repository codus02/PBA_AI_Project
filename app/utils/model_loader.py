from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Literal, Tuple

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from app.utils.config import (
    DIALOGUE_LLM_MODEL,
    LLM_MODEL,
    QWEN3_EMBED_MODEL,
    SLOT_EXTRACTOR_ADAPTER_PATH,
    SLOT_EXTRACTOR_BACKEND,
    SLOT_EXTRACTOR_MODEL,
)

_CACHE: dict[str, Tuple] = {}

_DEFAULT_QUANT = os.getenv("LLM_QUANT", "4bit").lower()

_NETWORK_ERR_KEYWORDS = (
    "connect",
    "resolve",
    "dns",
    "timeout",
    "timed out",
    "offline",
    "no internet",
    "failed to fetch",
    "name or service",
    "temporary failure",
)


def _offline_mode() -> bool:
    for var in ("LLM_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.getenv(var, "").lower() in ("1", "true", "yes"):
            return True
    return False


def _load_with_offline_fallback(loader_fn, model_id: str, **kwargs):
    """네트워크 로드 → 연결 오류 시 local_files_only=True 재시도."""
    if _offline_mode():
        kwargs.setdefault("local_files_only", True)
        return loader_fn(model_id, **kwargs)
    try:
        return loader_fn(model_id, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in _NETWORK_ERR_KEYWORDS):
            retry_kwargs = dict(kwargs)
            retry_kwargs["local_files_only"] = True
            return loader_fn(model_id, **retry_kwargs)
        raise


def _bnb_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def _bnb_8bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(load_in_8bit=True)


def _resolve_adapter_path(adapter_path: str | None) -> str | None:
    if not adapter_path:
        return None
    expanded = str(Path(adapter_path).expanduser())
    return expanded or None


def _load_causal_llm(
    model_id: str,
    quantization: Literal["4bit", "8bit", "fp16"] | None = None,
    adapter_path: str | None = None,
):
    quant = (quantization or _DEFAULT_QUANT).lower()
    resolved_adapter = _resolve_adapter_path(adapter_path)
    cache_key = f"{model_id}:{quant}:{resolved_adapter or ''}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    tokenizer = _load_with_offline_fallback(
        AutoTokenizer.from_pretrained,
        model_id,
        trust_remote_code=True,
    )

    if quant == "4bit":
        model = _load_with_offline_fallback(
            AutoModelForCausalLM.from_pretrained,
            model_id,
            quantization_config=_bnb_4bit_config(),
            device_map="auto",
            trust_remote_code=True,
        )
    elif quant == "8bit":
        model = _load_with_offline_fallback(
            AutoModelForCausalLM.from_pretrained,
            model_id,
            quantization_config=_bnb_8bit_config(),
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = _load_with_offline_fallback(
            AutoModelForCausalLM.from_pretrained,
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    if resolved_adapter:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise RuntimeError(
                "SLOT_EXTRACTOR_ADAPTER_PATH is set but 'peft' is not installed."
            ) from e
        model = PeftModel.from_pretrained(model, resolved_adapter)

    model.eval()
    _CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


def load_dialogue_llm(quantization: Literal["4bit", "8bit", "fp16"] | None = None):
    return _load_causal_llm(DIALOGUE_LLM_MODEL, quantization=quantization)


def load_slot_extractor_llm(quantization: Literal["4bit", "8bit", "fp16"] | None = None):
    adapter_path = None
    if SLOT_EXTRACTOR_BACKEND == "adapter":
        adapter_path = SLOT_EXTRACTOR_ADAPTER_PATH or None
    return _load_causal_llm(
        SLOT_EXTRACTOR_MODEL,
        quantization=quantization,
        adapter_path=adapter_path,
    )


def load_llm(quantization: Literal["4bit", "8bit", "fp16"] | None = None):
    """Legacy alias: current default LLM = dialogue model."""
    return load_dialogue_llm(quantization=quantization)


def _apply_template(tokenizer, messages, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )


def render_chat(tokenizer, system: str, user: str, add_generation_prompt: bool = True) -> str:
    """모델-agnostic chat template 렌더링.

    표준 [system, user] 형식을 먼저 시도하고, 템플릿이 system role 을 거부하면
    (Gemma 계열 등) system 내용을 user 앞에 프리펜드해 단일 user 메시지로 폴백한다.
    이름 기반 분기를 피해 로컬 스냅샷·alias 경로에서도 안전하다.
    """
    try:
        return _apply_template(
            tokenizer,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            add_generation_prompt,
        )
    except Exception as e:
        msg = str(e).lower()
        if "system" in msg and ("role" in msg or "not support" in msg or "unsupported" in msg):
            return _apply_template(
                tokenizer,
                [{"role": "user", "content": f"{system}\n\n{user}"}],
                add_generation_prompt,
            )
        raise


def load_qwen3_embedding():
    cache_key = f"{QWEN3_EMBED_MODEL}:embed"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    tokenizer = _load_with_offline_fallback(
        AutoTokenizer.from_pretrained,
        QWEN3_EMBED_MODEL,
        padding_side="left",
        trust_remote_code=True,
    )
    model = _load_with_offline_fallback(
        AutoModel.from_pretrained,
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
