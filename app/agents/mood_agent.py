"""공간 이미지 → mood atom 분포.

파이프라인:
    이미지 → BLIP-large (영어 캡션)
           → SentenceTransformer(all-MiniLM-L6-v2)
           → 14개 combo 임베딩과 코사인 유사도
           → softmax → combo 확률분포 (sum=1)
           → atom 단위로 marginalize → {atom: p}

저장 포맷 (party_space_analysis.mood_tags_json):
    {"bright": 0.72, "casual": 0.61, "playful": 0.30, ...}   # atom → prob

오케스트레이션(score_cocktail)에서는 cocktail.mood_tag 를 "|" 로 쪼개
각 atom 확률의 평균을 mood bonus 로 쓴다.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np
import torch
from PIL import Image as PILImage

from app.db.database import SessionLocal
from app.db.models import Cocktail
from app.utils.model_loader import _load_with_offline_fallback

_BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"
_EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_CAPTION_PROMPT = "a photo of a space with"
_SOFTMAX_TEMPERATURE = 0.1   # 코사인 유사도(≈0.2~0.4) → 분포 sharpening
_EMBED_MAX_LEN = 256

_CACHE: dict[str, Any] = {}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_blip():
    if "blip" in _CACHE:
        return _CACHE["blip"]
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = _load_with_offline_fallback(BlipProcessor.from_pretrained, _BLIP_MODEL_ID)
    model = _load_with_offline_fallback(
        BlipForConditionalGeneration.from_pretrained, _BLIP_MODEL_ID
    ).to(_device())
    model.eval()
    _CACHE["blip"] = (processor, model)
    return processor, model


def _load_embedder():
    """all-MiniLM-L6-v2 = BERT 백본 + mean pooling + L2 정규화 (공식 구성).

    sentence-transformers 의존 없이 transformers 로 직접 로드한다.
    """
    if "embedder" in _CACHE:
        return _CACHE["embedder"]
    from transformers import AutoModel, AutoTokenizer

    tokenizer = _load_with_offline_fallback(AutoTokenizer.from_pretrained, _EMBED_MODEL_ID)
    model = _load_with_offline_fallback(AutoModel.from_pretrained, _EMBED_MODEL_ID).to(_device())
    model.eval()
    _CACHE["embedder"] = (tokenizer, model)
    return tokenizer, model


def _mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)
    return summed / count


@torch.no_grad()
def _encode(texts: list[str]) -> np.ndarray:
    """텍스트 리스트 → (N, D) numpy, L2 정규화된 임베딩."""
    tokenizer, model = _load_embedder()
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=_EMBED_MAX_LEN,
        return_tensors="pt",
    ).to(model.device)
    out = model(**enc).last_hidden_state
    pooled = _mean_pool(out, enc["attention_mask"])
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.float().cpu().numpy()


def _load_combo_vocab() -> tuple[list[str], np.ndarray]:
    """DB cocktails.mood_tag distinct 값 + L2 정규화된 임베딩."""
    if "combo_vocab" in _CACHE:
        return _CACHE["combo_vocab"]

    db = SessionLocal()
    try:
        rows = (
            db.query(Cocktail.mood_tag)
            .filter(Cocktail.mood_tag.isnot(None))
            .filter(Cocktail.is_active.is_(True))
            .distinct()
            .all()
        )
        combos = sorted({r[0].strip() for r in rows if r[0] and r[0].strip()})
    finally:
        db.close()

    if not combos:
        raise RuntimeError("cocktails.mood_tag 에 유효한 값이 없습니다.")

    cleaned = [c.replace("|", " ") for c in combos]
    vecs = _encode(cleaned).astype(np.float32)   # 이미 L2-normalized
    _CACHE["combo_vocab"] = (combos, vecs)
    return combos, vecs


def _caption_image(image_path: str) -> str:
    processor, model = _load_blip()
    image = PILImage.open(image_path).convert("RGB")
    inputs = processor(image, text=_CAPTION_PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60)
    return processor.decode(out[0], skip_special_tokens=True).strip()


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = x / max(temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _combo_to_atom_marginal(combos: list[str], combo_probs: np.ndarray) -> dict[str, float]:
    """p(atom) = Σ p(combo) for combo ∋ atom. 각 atom 값은 [0, 1]."""
    marginal: dict[str, float] = {}
    for combo, p in zip(combos, combo_probs):
        for atom in combo.split("|"):
            atom = atom.strip()
            if not atom:
                continue
            marginal[atom] = marginal.get(atom, 0.0) + float(p)
    return marginal


def analyze_space_image(
    image_path: str,
    temperature: float = _SOFTMAX_TEMPERATURE,
) -> dict[str, Any]:
    """공간 이미지 → mood atom 분포.

    Returns:
        {
            "caption_en": "a photo of a space with ...",
            "best_mood_tag": "social|modern|casual",
            "mood_tags_json": {"social": 0.54, "modern": 0.47, "casual": 0.61, ...},
        }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    caption = _caption_image(image_path)

    combos, combo_vecs = _load_combo_vocab()
    q = _encode([caption])[0].astype(np.float32)   # 이미 L2-normalized

    sims = combo_vecs @ q               # (C,) cosine (둘 다 normalized)
    combo_probs = _softmax(sims, temperature=temperature)
    atom_marginal = _combo_to_atom_marginal(combos, combo_probs)

    best_idx = int(np.argmax(sims))
    return {
        "caption_en": caption,
        "best_mood_tag": combos[best_idx],
        "mood_tags_json": atom_marginal,
    }


# ============================================================================
# img2tag (Gemini 기반): 공간 이미지 → mood_tag 3-tuple
# FE 전용 경량 엔드포인트(/space/img2tag) 용.
# ============================================================================

_GEMINI_MODEL_ID = "gemini-2.0-flash"


def _build_gemini_client(api_key: str):
    """주어진 키로 google-genai 클라이언트를 생성. 키별로 인스턴스 분리해 캐시.

    각 키마다 별개 클라이언트 인스턴스라야 quota 회계가 키 단위로 정확해진다.
    """
    cache_key = f"gemini::{api_key[-6:]}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    from google import genai

    client = genai.Client(api_key=api_key)
    _CACHE[cache_key] = client
    return client


def _to_pil(image_input) -> PILImage.Image:
    """파일경로(str/Path), PIL Image, bytes 모두 수용."""
    if isinstance(image_input, PILImage.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, (bytes, bytearray)):
        return PILImage.open(io.BytesIO(image_input)).convert("RGB")
    return PILImage.open(image_input).convert("RGB")


def _pil_to_part(pil_img: PILImage.Image):
    """PIL Image → google-genai Part (JPEG bytes). PIL 직전달은 SDK 버전에 따라 무시될 수 있어서 명시적으로 변환."""
    from google.genai import types
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def img2tag(image_input) -> tuple[str, ...]:
    """공간 이미지 → mood_tag 튜플 (예: ("lively", "bright", "spacious")).

    Gemini 에게 DB mood atom 목록을 주고 직접 3개를 고르게 함.
    임베딩 유사도 비교 불필요 — 같은 태그를 가진 칵테일이 여럿이어도 영향 없음.
    """
    from app.services import gemini_key_pool

    # DB에서 사용 가능한 mood atom 집합 구성
    combos, _ = _load_combo_vocab()
    atoms = sorted({atom.strip() for combo in combos for atom in combo.split("|") if atom.strip()})
    atom_list = ", ".join(atoms)

    classify_prompt = (
        f"You are classifying the mood of a party venue photo. "
        f"Available mood tags: [{atom_list}]. "
        f"Select exactly 3 tags from the list that best match this space's atmosphere. "
        f"Output ONLY the 3 tags as a comma-separated list, nothing else. "
        f"Example: warm, intimate, cozy"
    )

    pil_img = _to_pil(image_input)
    image_part = _pil_to_part(pil_img)
    last_exc: Exception | None = None

    keys = gemini_key_pool._parse_keys_from_env()
    max_attempts = max(len(keys), 1)

    for _ in range(max_attempts):
        label, api_key, key_id = gemini_key_pool.acquire()
        client = _build_gemini_client(api_key)
        try:
            response = client.models.generate_content(
                model=_GEMINI_MODEL_ID,
                contents=[classify_prompt, image_part],
            )
            raw = response.text.strip()
            tags = [t.strip().lower() for t in raw.split(",") if t.strip()][:3]
            valid = [t for t in tags if t in atoms] or tags
            return tuple(valid)
        except Exception as e:
            if gemini_key_pool.is_gemini_quota_error(e):
                gemini_key_pool.mark_quota_exceeded(key_id)
                last_exc = e
                continue
            raise

    raise gemini_key_pool.QuotaExhaustedError(
        "모든 Gemini 키에서 quota 에러 발생"
    ) from last_exc
