"""
QLoRA 어댑터 기반 슬롯 추출기.

models/slot_extractor_adapter/ 가 없으면 import 해도 아무 일 없음.
어댑터가 있을 때만 로드 + 추론 수행.

사용:
    from app.utils.slot_extractor import adapter_extract_slots, adapter_available
    if adapter_available():
        result = adapter_extract_slots(user_text)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch

from app.utils.config import SLOT_EXTRACTOR_ADAPTER_PATH

logger = logging.getLogger(__name__)

# 어댑터 경로 (프로젝트 루트 기준)
_DEFAULT_ADAPTER_DIR = Path(__file__).parent.parent.parent / "models" / "slot_extractor_adapter"
_ADAPTER_DIR = Path(SLOT_EXTRACTOR_ADAPTER_PATH).expanduser() if SLOT_EXTRACTOR_ADAPTER_PATH else _DEFAULT_ADAPTER_DIR

# 싱글턴 캐시
_MODEL_CACHE: dict = {}


def adapter_available() -> bool:
    """어댑터 디렉토리가 존재하는지 확인."""
    backend = os.getenv("SLOT_EXTRACTOR_BACKEND", "").strip().lower()
    if backend != "adapter":
        return False
    return (_ADAPTER_DIR / "adapter_config.json").exists()


def _load_adapter_model():
    """어댑터 + base model 로드 (최초 1회만). (tokenizer, model) 반환."""
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["tokenizer"], _MODEL_CACHE["model"]

    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    logger.info("슬롯 추출 어댑터 로드: %s", _ADAPTER_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(_ADAPTER_DIR))
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(_ADAPTER_DIR),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    _MODEL_CACHE["tokenizer"] = tokenizer
    _MODEL_CACHE["model"] = model
    logger.info("어댑터 로드 완료")
    return tokenizer, model


def adapter_extract_slots(user_text: str, system_prompt: str, max_new_tokens: int = 180) -> str:
    """어댑터 모델로 슬롯 추출 → raw JSON 문자열 반환."""
    tokenizer, model = _load_adapter_model()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

    if torch.cuda.is_available():
        del inputs, out
        torch.cuda.empty_cache()

    return raw
