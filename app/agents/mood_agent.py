"""공간 이미지 분석.

1) analyze_space_image:
   BLIP-large + MiniLM 으로 공간 이미지를 mood atom 분포로 변환한다.
   추천 오케스트레이션이 사용하는 `mood_tags_json` 저장용.

2) img2tag:
   Gemini 로 공간 분위기 설명을 생성한 뒤, 사전 계산 임베딩 코퍼스에 매핑해
   프론트용 mood_tag 3개를 반환한다.
"""
from __future__ import annotations

import io
import os
import pathlib
import threading
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage

from app.db.database import SessionLocal
from app.db.models import Cocktail
from app.utils.model_loader import _load_with_offline_fallback

_BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"
_EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_CAPTION_PROMPT = "a photo of a space with"
_SOFTMAX_TEMPERATURE = 0.1
_EMBED_MAX_LEN = 256

_GEMINI_MODEL_ID = "gemini-2.0-flash"
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_IMG2TAG_EMBED_PATH = _PROJECT_ROOT / "modeling/image/final/cocktail_embeddings_st.npy"
_IMG2TAG_IDS_PATH = _PROJECT_ROOT / "modeling/image/final/cocktail_ids_st.npy"
_IMG2TAG_PROMPT = (
    "Describe the mood and atmosphere of this space in 1-2 sentences. "
    "Focus on the emotional vibe (e.g. cozy, lively, romantic), "
    "the lighting and visual tone (e.g. warm, bright, dark), "
    "and the sense of space (e.g. intimate, open, casual). "
    "Do not describe objects."
)

_RPM_LIMIT = 5
_RPD_LIMIT = 20
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
    vecs = _encode(cleaned).astype(np.float32)
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
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    caption = _caption_image(image_path)

    combos, combo_vecs = _load_combo_vocab()
    q = _encode([caption])[0].astype(np.float32)

    sims = combo_vecs @ q
    combo_probs = _softmax(sims, temperature=temperature)
    atom_marginal = _combo_to_atom_marginal(combos, combo_probs)

    best_idx = int(np.argmax(sims))
    return {
        "caption_en": caption,
        "best_mood_tag": combos[best_idx],
        "mood_tags_json": atom_marginal,
    }


class _GeminiKeyManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._usage: dict[str, dict[str, Any]] = {}

    def _ensure_entry(self, key: str) -> dict[str, Any]:
        entry = self._usage.get(key)
        if entry is None:
            entry = {"minute": deque(), "day": None, "day_count": 0}
            self._usage[key] = entry
        return entry

    def _normalize_day(self, entry: dict[str, Any], now: datetime) -> None:
        day_key = now.strftime("%Y-%m-%d")
        if entry["day"] != day_key:
            entry["day"] = day_key
            entry["day_count"] = 0

    def _prune_minute(self, entry: dict[str, Any], now_ts: float) -> None:
        minute_window = entry["minute"]
        while minute_window and now_ts - minute_window[0] >= 60.0:
            minute_window.popleft()

    def reserve_key(self, keys: list[str]) -> str:
        now = datetime.now()
        now_ts = now.timestamp()
        with self._lock:
            for key in keys:
                entry = self._ensure_entry(key)
                self._normalize_day(entry, now)
                self._prune_minute(entry, now_ts)
                if len(entry["minute"]) >= _RPM_LIMIT:
                    continue
                if entry["day_count"] >= _RPD_LIMIT:
                    continue
                entry["minute"].append(now_ts)
                entry["day_count"] += 1
                return key
        raise RuntimeError("사용 가능한 Gemini API 키가 없습니다. 분당/일일 한도를 모두 초과했습니다.")


_KEY_MANAGER = _GeminiKeyManager()


def _load_gemini_api_keys() -> list[str]:
    keys: list[str] = []
    for idx in range(1, 11):
        value = os.environ.get(f"GOOGLE_API_KEY{idx}", "").strip()
        if value:
            keys.append(value)
    legacy = os.environ.get("GOOGLE_API_KEY", "").strip()
    if legacy and legacy not in keys:
        keys.append(legacy)
    if not keys:
        raise RuntimeError("GOOGLE_API_KEY1~10 또는 GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
    return keys


def _get_gemini_client(api_key: str):
    cache_key = f"gemini:{api_key}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    from google import genai

    client = genai.Client(api_key=api_key)
    _CACHE[cache_key] = client
    return client


def _load_img2tag_corpus() -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    if "img2tag_corpus" in _CACHE:
        return _CACHE["img2tag_corpus"]

    for p in (_IMG2TAG_EMBED_PATH, _IMG2TAG_IDS_PATH):
        if not p.exists():
            raise FileNotFoundError(f"img2tag 참조 파일 없음: {p}")

    embeds = np.load(_IMG2TAG_EMBED_PATH)
    ids = np.load(_IMG2TAG_IDS_PATH)

    db = SessionLocal()
    try:
        rows = db.query(Cocktail.cocktail_id, Cocktail.mood_tag).all()
        id_to_tag = {int(cid): (tag or "") for cid, tag in rows}
    finally:
        db.close()

    _CACHE["img2tag_corpus"] = (embeds, ids, id_to_tag)
    return embeds, ids, id_to_tag


def _to_pil(image_input) -> PILImage.Image:
    if isinstance(image_input, PILImage.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, (bytes, bytearray)):
        return PILImage.open(io.BytesIO(image_input)).convert("RGB")
    return PILImage.open(image_input).convert("RGB")


def analyze_image(image_input) -> str:
    keys = _load_gemini_api_keys()
    errors: list[str] = []

    for _ in range(len(keys)):
        api_key = _KEY_MANAGER.reserve_key(keys)
        client = _get_gemini_client(api_key)
        pil_img = _to_pil(image_input)
        try:
            response = client.models.generate_content(
                model=_GEMINI_MODEL_ID,
                contents=[_IMG2TAG_PROMPT, pil_img],
            )
            text = (response.text or "").strip()
            if text:
                return text
            errors.append("Gemini 응답이 비어 있습니다.")
        except Exception as exc:
            errors.append(str(exc))

    raise RuntimeError("; ".join(errors) if errors else "Gemini 호출에 실패했습니다.")


def img2tag(image_input) -> tuple[str, ...]:
    vlm_text = analyze_image(image_input)
    query_emb = _encode([vlm_text])[0]

    embeds, ids, id_to_tag = _load_img2tag_corpus()
    norms = np.linalg.norm(embeds, axis=1) * np.linalg.norm(query_emb)
    sims = embeds @ query_emb / np.where(norms == 0, 1e-9, norms)

    best_cid = int(ids[int(np.argmax(sims))])
    mood_tag = id_to_tag.get(best_cid, "")
    if not mood_tag:
        raise RuntimeError(f"cocktail_id {best_cid} 에 해당하는 mood_tag 가 DB 에 없습니다.")
    tags = tuple(tag.strip() for tag in mood_tag.split("|") if tag.strip())
    if not tags:
        raise RuntimeError("img2tag 결과를 생성하지 못했습니다.")
    return tags
