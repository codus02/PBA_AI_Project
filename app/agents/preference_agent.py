"""
preference_agent.py
역할:
  - LLM(Qwen3) 단일 호출로 슬롯 추출 + 종료 판단 + 다음 질문 생성
  - 피드백 인텐트 분류 및 선호 벡터 조정 (rule-based MVP — 추후 별도 리팩토링)
  - DB → 통합 프로필 빌드
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.db.crud import (
    get_guest_session,
    get_initial_tag_response,
    get_preference_slot,
    get_preference_vector,
    get_latest_space_analysis_by_party,
)

logger = logging.getLogger(__name__)


# ============================================================
# 슬롯 스키마 (영문 enum + intensity dict)
# ============================================================

CURRENT_MOOD_VALUES = {"good", "soso", "bad"}

PARTY_PURPOSE_VALUES = {
    "celebration",
    "date",
    "business",
    "solo",
    "hangout",
}

STRENGTH_VALUES = {"zero", "light", "medium", "strong"}

TASTE_KEYS = {"sweet", "sour", "bitter", "body", "creamy", "freshness"}
AROMA_KEYS = {"woody", "minty", "fruity", "citrus", "floral", "coffee", "herbal"}
INTENSITY_VALUES = {"low", "medium", "high"}

DISLIKED_BASE_VALUES = {"whiskey", "gin", "rum", "vodka", "tequila"}

SLOT_KEYS = [
    "current_mood",
    "party_purpose",
    "taste_profile",
    "aroma_profile",
    "strength_preference",
    "disliked_bases",
    "favorite_drinks",
]


# ============================================================
# 종료 판단
# ============================================================

MIN_FILLED_SLOTS_FOR_PROCEED = 4
MAX_USER_TURNS = 7


def _count_filled_slots(slots: dict) -> int:
    n = 0
    for k in SLOT_KEYS:
        v = slots.get(k)
        if v is None:
            continue
        if isinstance(v, (list, dict)) and len(v) == 0:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        n += 1
    return n


def _calc_effective_completion(slots: dict) -> float:
    return round(_count_filled_slots(slots) / len(SLOT_KEYS) * 100, 2)


def should_move_to_recommendation(
    merged_slots: dict,
    user_turn_count: int,
    user_msg: str,
    llm_should_stop: bool = False,
    llm_stop_reason: str = "",
) -> tuple[bool, str]:
    """LLM 판단 + 턴/슬롯 상한 조합으로 종료 결정."""
    if llm_should_stop:
        return True, llm_stop_reason or "llm_stop"
    if user_turn_count >= MAX_USER_TURNS:
        return True, "turn_limit"
    if _count_filled_slots(merged_slots) >= MIN_FILLED_SLOTS_FOR_PROCEED:
        # LLM이 더 물어볼 게 있다고 판단 안 했으면 계속 진행
        return False, "enough_slots_but_continue"
    return False, "keep_collecting"


# ============================================================
# 피드백 인텐트 분류 / 벡터 업데이트 (rule-based MVP — 별도 리팩토링 예정)
# ============================================================

def classify_intent(feedback_text: str) -> str:
    text = feedback_text.lower()
    accept_keywords = [
        "좋아", "완벽", "이걸로", "맛있어", "마음에 들어",
        "ok", "오케이", "선택", "확정", "맞아", "딱이야",
    ]
    adjust_keywords = [
        "더", "조금", "살짝", "약간", "달게", "쓰게", "시게",
        "강하게", "약하게", "바꿔", "수정", "변경", "달았으면", "썼으면",
    ]
    for kw in accept_keywords:
        if kw in text:
            return "ACCEPT"
    for kw in adjust_keywords:
        if kw in text:
            return "ADJUST"
    return "REJECT"


def update_vector(before_vec: dict[str, float], feedback_text: str) -> dict[str, float]:
    updated = dict(before_vec)
    text = feedback_text.lower()

    def clamp(v: float) -> float:
        return max(0.0, min(5.0, v))

    adjustments: list[tuple[list[str], str, float]] = [
        (["더 달", "달게", "달았으면"], "sweetness_score", +0.5),
        (["덜 달", "덜달", "달지 않게"], "sweetness_score", -0.5),
        (["더 쓰", "쓰게", "썼으면"], "bitterness_score", +0.5),
        (["덜 쓰", "덜쓰"], "bitterness_score", -0.5),
        (["더 시", "시게"], "sourness_score", +0.5),
        (["덜 시", "덜시"], "sourness_score", -0.5),
        (["더 강", "강하게"], "alcohol_score", +0.5),
        (["약하게", "더 약", "가볍게"], "alcohol_score", -0.5),
        (["청량", "상큼"], "freshness_score", +0.5),
    ]
    for keywords, field, delta in adjustments:
        if any(kw in text for kw in keywords):
            updated[field] = clamp(updated.get(field, 2.5) + delta)
    return updated


# ============================================================
# Qwen 프롬프트 구성
# ============================================================

_SYSTEM_PROMPT = """
너는 칵테일 추천 시스템의 대화 컨트롤러다.
반드시 JSON 객체 하나만 출력한다. 설명, 마크다운, 코드블록 금지.

출력 형식:
{
  "extracted_slots": {
    "current_mood": null 또는 "good"|"soso"|"bad",
    "party_purpose": null 또는 "celebration"|"date"|"business"|"solo"|"hangout",
    "taste_profile": { "sweet"|"sour"|"bitter"|"body"|"creamy"|"freshness": "low"|"medium"|"high", ... },
    "aroma_profile": { "woody"|"minty"|"fruity"|"citrus"|"floral"|"coffee"|"herbal": "low"|"medium"|"high", ... },
    "strength_preference": null 또는 "zero"|"light"|"medium"|"strong",
    "disliked_bases": ["whiskey"|"gin"|"rum"|"vodka"|"tequila", ...],
    "favorite_drinks": [문자열, ...]
  },
  "should_stop": true 또는 false,
  "stop_reason": "user_requested"|"enough_info"|"",
  "next_question": "한국어 한 문장"
}

규칙:
1. 최근 USER 발화에서 새롭게 드러난 정보만 extracted_slots에 넣어라.
2. USER가 말하지 않은 것은 추측하지 마라.
3. "없어/없음/상관없어/가리는 거 없어/다 잘 마셔"는 비선호 베이스 질문에 대한 응답이면 disliked_bases = [] 로 둔다.
4. 맛/향에서 싫어함은 같은 key의 low intensity로 표현한다.
5. next_question은 한국어 존댓말 한 문장으로 자연스럽게 작성한다.
6. 사용자가 "이제 추천해줘", "그만", "바로 추천"처럼 말하면 should_stop=true.

예시:
- "친구 생일파티" -> {"party_purpose":"celebration"}
- "회사 사람들이랑" -> {"party_purpose":"business"}
- "친구들이랑 놀러" -> {"party_purpose":"hangout"}
- "기분 좋아" -> {"current_mood":"good"}
- "그냥 그저 그래" -> {"current_mood":"soso"}
- "기분 안 좋아" -> {"current_mood":"bad"}
- "술 안 들어간 걸로" -> {"strength_preference":"zero"}
- "무알콜로 부탁해" -> {"strength_preference":"zero"}
- "술고래야" -> {"strength_preference":"strong"}
- "풀냄새 좋아" -> {"aroma_profile":{"herbal":"high"}}
- "우유맛 싫어" -> {"taste_profile":{"creamy":"low"}}
- "가리는 거 없어" -> {"disliked_bases":[]}

반드시 JSON만 출력해라.
""".strip()


def _format_history(history: list[dict], max_turns: int = 6) -> str:
    if not history:
        return "(대화 없음)"
    rows = history[-max_turns:]
    lines = []
    for r in rows:
        role = r.get("speaker_role", "?")
        text = r.get("utterance_text") or r.get("message_text") or ""
        tag = "LLM" if role == "LLM" else "USER"
        lines.append(f"{tag}: {text}")
    return "\n".join(lines)

def _last_llm_question(history: list[dict]) -> str:
    for row in reversed(history or []):
        if row.get("speaker_role") == "LLM":
            return (row.get("utterance_text") or row.get("message_text") or "").strip()
    return ""


def _build_user_prompt(history: list[dict], slots: dict, user_msg: str) -> str:
    last_q = _last_llm_question(history)
    return (
        f"Current accumulated slots (JSON):\n{json.dumps(slots, ensure_ascii=False)}\n\n"
        f"Most recent assistant question:\n{last_q or '(없음)'}\n\n"
        f"Recent dialogue:\n{_format_history(history)}\n\n"
        f"Latest user message:\n{user_msg}\n\n"
        "The user may answer indirectly or contextually.\n"
        "Interpret the latest user message relative to the most recent assistant question.\n"
        "Extract ONLY what the latest user message newly says.\n"
        "Return JSON now."
    )


# ============================================================
# JSON 파서 + 검증
# ============================================================

def _extract_json_object(text: str) -> Optional[dict]:
    # 가장 바깥쪽 중괄호 추출 (greedy)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _validate_intensity_dict(raw: Any, allowed_keys: set[str]) -> dict:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if k not in allowed_keys:
            continue
        if not isinstance(v, str):
            continue
        vv = v.strip().lower()
        if vv in INTENSITY_VALUES:
            out[k] = vv
    return out


def _validate_enum(raw: Any, allowed: set[str]) -> Optional[str]:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    v = raw.strip().lower()
    return v if v in allowed else None


def _validate_list_enum(raw: Any, allowed: set[str]) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        v = item.strip().lower()
        if v in allowed and v not in out:
            out.append(v)
    return out


def _validate_free_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, (str, int, float)):
            continue
        s = str(item).strip()
        if s and s not in out:
            out.append(s)
    return out


def validate_extracted_slots(raw: dict) -> dict:
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, Any] = {}

    if "current_mood" in raw:
        v = _validate_enum(raw["current_mood"], CURRENT_MOOD_VALUES)
        if v:
            cleaned["current_mood"] = v

    if "party_purpose" in raw:
        v = _validate_enum(raw["party_purpose"], PARTY_PURPOSE_VALUES)
        if v:
            cleaned["party_purpose"] = v

    if "strength_preference" in raw:
        v = _validate_enum(raw["strength_preference"], STRENGTH_VALUES)
        if v:
            cleaned["strength_preference"] = v

    if "taste_profile" in raw:
        d = _validate_intensity_dict(raw["taste_profile"], TASTE_KEYS)
        if d:
            cleaned["taste_profile"] = d

    if "aroma_profile" in raw:
        d = _validate_intensity_dict(raw["aroma_profile"], AROMA_KEYS)
        if d:
            cleaned["aroma_profile"] = d

    if "disliked_bases" in raw:
        if isinstance(raw["disliked_bases"], list) and len(raw["disliked_bases"]) == 0:
            cleaned["disliked_bases"] = []
        else:
            lst = _validate_list_enum(raw["disliked_bases"], DISLIKED_BASE_VALUES)
            if lst:
                cleaned["disliked_bases"] = lst

    if "favorite_drinks" in raw:
        lst = _validate_free_list(raw["favorite_drinks"])
        if lst:
            cleaned["favorite_drinks"] = lst

    return cleaned


# ============================================================
# Qwen 단일 호출 — 추출 + 종료 + 다음 질문
# ============================================================

def _parse_llm_output(raw: str) -> dict:
    parsed = _extract_json_object(raw) or {}
    extracted = validate_extracted_slots(parsed.get("extracted_slots") or {})
    should_stop = bool(parsed.get("should_stop"))
    stop_reason = str(parsed.get("stop_reason") or "")
    next_q = str(parsed.get("next_question") or "").strip() or "조금 더 알려주실 만한 게 있을까요?"
    return {
        "extracted_slots": extracted,
        "should_stop": should_stop,
        "stop_reason": stop_reason,
        "next_question": next_q,
        "raw": raw,
    }


def _analyze_via_ollama(history: list[dict], slots: dict, user_msg: str, model_name: str) -> dict:
    import ollama
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(history, slots, user_msg)},
    ]
    resp = ollama.chat(model=model_name, messages=messages, options={"temperature": 0})
    raw = resp["message"]["content"].strip()
    result = _parse_llm_output(raw)
    result["source"] = f"ollama:{model_name}"
    return result


def _analyze_via_qwen(history: list[dict], slots: dict, user_msg: str, max_new_tokens: int) -> dict:
    from app.utils.model_loader import load_qwen3
    import torch

    tokenizer, model = load_qwen3()
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(history, slots, user_msg)},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
    result = _parse_llm_output(raw)
    result["source"] = "qwen"
    return result


def analyze_user_turn(
    history: list[dict],
    slots: dict,
    user_msg: str,
    max_new_tokens: int = 512,
) -> dict:
    """LLM 단일 호출로 추출/종료/질문을 한 번에 얻는다.

    QWEN3_MODEL 환경변수가 'name:tag' 형식이면 Ollama로, 아니면 HuggingFace로 로드.
    """
    from app.utils.config import QWEN3_MODEL
    try:
        if ":" in QWEN3_MODEL:
            return _analyze_via_ollama(history, slots, user_msg, QWEN3_MODEL)
        else:
            return _analyze_via_qwen(history, slots, user_msg, max_new_tokens)
    except Exception as e:
        logger.warning("analyze_user_turn failed: %r", e, exc_info=True)
        return {
            "extracted_slots": {},
            "should_stop": False,
            "stop_reason": "",
            "next_question": "조금 더 알려주실 만한 게 있을까요?",
            "source": "fallback",
            "raw": "",
        }


def generate_opening_question() -> str:
    """대화 첫 질문 — 정해진 오프너. (Qwen 호출 비용 아끼려고 고정)"""
    return "오늘 어떤 자리이고 어떤 분위기를 원하세요? 원하는 맛이나 향도 같이 알려주시면 좋아요."


# ============================================================
# 슬롯 병합
# ============================================================

def merge_slots(current: dict, extracted: dict) -> dict:
    merged = dict(current) if current else {}

    for key in ("current_mood", "party_purpose", "strength_preference"):
        if key in extracted and extracted[key]:
            merged[key] = extracted[key]

    for key in ("taste_profile", "aroma_profile"):
        if key in extracted and isinstance(extracted[key], dict):
            base = dict(merged.get(key) or {})
            base.update(extracted[key])
            merged[key] = base

    for key in ("disliked_bases", "favorite_drinks"):
        if key in extracted and isinstance(extracted[key], list):
            base = list(merged.get(key) or [])
            for item in extracted[key]:
                if item not in base:
                    base.append(item)
            merged[key] = base

    return merged


# ============================================================
# User profile builder
# ============================================================

def _empty_slot_dict() -> dict:
    return {k: None for k in SLOT_KEYS}


def build_user_profile(db: Session, guest_session_id: str) -> dict:
    guest = get_guest_session(db, guest_session_id)
    tags = get_initial_tag_response(db, guest_session_id)
    slot = get_preference_slot(db, guest_session_id)
    vector = get_preference_vector(db, guest_session_id)
    space = get_latest_space_analysis_by_party(db, guest.party_session_id) if guest else None

    merged_slots = _empty_slot_dict()
    if slot:
        merged_slots.update({
            "current_mood": slot.current_mood,
            "party_purpose": slot.party_purpose,
            "taste_profile": slot.taste_profile_json or {},
            "aroma_profile": slot.aroma_profile_json or {},
            "strength_preference": slot.strength_preference,
            "disliked_bases": slot.disliked_bases_json or [],
            "favorite_drinks": slot.favorite_drinks_json or [],
        })

    return {
        "guest": guest,
        "initial_tags": tags,
        "slot": slot,
        "vector": vector,
        "space": space,
        "merged_slots": merged_slots,
        "effective_completion": _calc_effective_completion(merged_slots),
    }