from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Literal, Tuple

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from app.utils.config import (
    DIALOGUE_LLM_MODEL,
    LLM_MODEL,
    LLM_BACKEND,
    OLLAMA_BASE_URL,
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
    """모델-agnostic chat template 렌더링."""
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


def build_chat_prompt(tokenizer, messages: list[dict], **kwargs) -> str:
    """apply_chat_template 래퍼 — 모델별 미지원 파라미터 자동 제거."""
    if "qwen3" not in DIALOGUE_LLM_MODEL.lower():
        kwargs.pop("enable_thinking", None)
    return tokenizer.apply_chat_template(messages, **kwargs)


def llm_chat(
    messages: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    model_id: str | None = None,
) -> str:
    """백엔드(Ollama / HF)에 무관하게 chat completion → 텍스트 반환."""
    target_model = model_id or DIALOGUE_LLM_MODEL
    if LLM_BACKEND == "ollama":
        import requests

        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": repetition_penalty,
            },
        }
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    if target_model == DIALOGUE_LLM_MODEL:
        tokenizer, model = load_dialogue_llm()
    elif target_model == SLOT_EXTRACTOR_MODEL:
        tokenizer, model = load_slot_extractor_llm()
    else:
        tokenizer, model = _load_causal_llm(target_model)

    rendered = _apply_template(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    do_sample = temperature > 0.0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
    if torch.cuda.is_available():
        del inputs, out
        torch.cuda.empty_cache()
    return raw


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
        hidden = model(**enc).last_hidden_state
        last = hidden[:, -1]
        last = torch.nn.functional.normalize(last, p=2, dim=1)
        out_chunks.append(last.float().cpu())

    return torch.cat(out_chunks, dim=0)
