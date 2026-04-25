"""
preference_agent.py
역할:
  - LLM 단일 호출로 슬롯 추출 + 종료 판단 + 다음 질문 생성
  - 피드백 인텐트 분류 및 선호 벡터 조정 (rule-based MVP — 추후 별도 리팩토링)
  - DB → 통합 프로필 빌드
"""
from __future__ import annotations

import json
import logging
import os
import re
from types import SimpleNamespace
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


def _dialogue_trace_enabled() -> bool:
    return os.getenv("PBA_TRACE_DIALOGUE", "").strip().lower() in {"1", "true", "yes", "on"}


def _dialogue_relaxed_mode_enabled() -> bool:
    # relaxed 모드는 실제 품질보다 추출 안정성을 더 크게 해쳐서 폐기.
    # 남아 있는 호출 경로는 모두 기본 모드로 수렴시킨다.
    return False


def _trace_clone(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return value


def _log_dialogue_trace(user_msg: str, trace: dict) -> None:
    if not _dialogue_trace_enabled():
        return
    payload = {
        "user_msg": user_msg,
        "pipeline": trace,
    }
    logger.info("dialogue_trace=%s", json.dumps(payload, ensure_ascii=False))


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
INTENSITY_VALUES = {"zero", "low", "medium", "high"}
_INTENSITY_ALIASES = {
    "strong": "high",
    "weak": "low",
    "light": "low",
    "normal": "medium",
}

# LLM이 영어 일반명사로 흘리는 걸 스키마 키로 흡수
_TASTE_KEY_ALIASES = {
    "refreshing": "freshness",
    "fresh": "freshness",
    "refresh": "freshness",
    "sweetness": "sweet",
    "sourness": "sour",
    "bitterness": "bitter",
    "bodied": "body",
    "full_body": "body",
    "cream": "creamy",
    "creaminess": "creamy",
}
_AROMA_KEY_ALIASES = {
    "wood": "woody",
    "mint": "minty",
    "fruit": "fruity",
    "citrusy": "citrus",
    "flower": "floral",
    "flowery": "floral",
    "herb": "herbal",
    "herby": "herbal",
}
# zero  = "완전 비선호" → 해당 축이 두드러진 칵테일은 추천에서 HARD-EXCLUDE
# low   = "별로/싫어" → rerank 에서 감점
# medium= "보통/적당히"
# high  = "매우 선호"

DISLIKED_BASE_VALUES = {"whiskey", "gin", "rum", "vodka", "tequila"}

# LLM이 disliked_bases 에 잘못 넣는 "스피릿 아닌" 재료 → taste/aroma 축으로 리매핑.
# enum 위반으로 드롭되기 전에 여기서 의미를 살린다 (우유·크림 비선호가 사라져버리는 버그 방지).
_DISLIKED_BASE_REMAP_TO_CREAMY_ZERO = {
    "milk", "cream", "dairy", "우유", "크림", "milky", "creamy",
}

SLOT_KEYS = [
    "current_mood",
    "party_purpose",
    "taste_profile",
    "aroma_profile",
    "strength_preference",
    "disliked_bases",
    "favorite_drinks",
]

SLOT_ASK_ORDER = [
    "party_purpose",
    "current_mood",
    "strength_preference",
    "taste_profile",
    "aroma_profile",
]
# favorite_drinks 는 더 이상 필수 질문이 아니다. 사용자가 먼저 언급하면 추출만 된다.
# 비선호 맛·향은 별도 슬롯이 아니라 taste_profile/aroma_profile 에 "zero"/"low" 로 들어간다.
# Followup 프롬프트가 대화 중 한 번 비선호를 확인하도록 지시한다.
# disliked_bases는 사용자가 먼저 언급하지 않으면 굳이 묻지 않는다 (SLOT_KEYS엔 남아있으므로 추출은 됨).

USER_STOP_PATTERNS = [
    "그만 물어",
    "그만 물어봐",
    "그만 묻",
    "이제 그만 물어",
    "이제 그만 묻",
    "꼬치꼬치",
    "귀찮게 묻지마",
    "귀찮게 묻지 마",
    "귀찮게 질문하지마",
    "귀찮게 질문하지 마",
    "질문 그만",
    "질문그만",
    # "알아서" 류는 축-단위 위임("바디감은 너가 알아서 해줘")과 충돌한다.
    # 전체 위임 의미가 분명한 표현만 잡는다.
    "알아서 다 해줘",
    "알아서 다해줘",
    "다 알아서 해",
    "다 알아서 골라",
    "전부 알아서",
    "그냥 알아서 골라",
    "그냥 알아서 해줘",
    # 자연스러운 "추천해줘" 류 — 사용자가 충분히 말했다고 느낄 때 바로 종료.
    "추천해줘",
    "추천 해줘",
    "추천해 줘",
    "이제 추천해",
    "이제 추천 해",
    "이제 추천",
    "그냥 추천",
    "바로 추천해",
    "바로 추천 해",
    "지금 바로 추천",
    "추천으로 넘어가",
    "추천 단계로",
    "이제 추천 단계로",
    "이제 골라줘",
    "이제 골라 줘",
    "그냥 골라",
    "이제 뽑아",
    "그냥 뽑아",
    "이제 보여줘",
    "보여줘 이제",
]
# 사용자가 명확히 "그만 묻고 바로 추천"의 의사를 보일 때 종료한다.

# 빈 list/dict가 "명시적 답변"(예: disliked_bases=[] = "없음")으로 인정되는 슬롯
# current_mood 는 사용자가 자연스럽게 말하면 받되, 추천 전 필수로 캐묻지는 않는다.
_EMPTY_OK_SLOTS = {"current_mood", "disliked_bases", "favorite_drinks"}

_NO_PREFERENCE_PATTERNS = [
    "없어", "없음", "상관없어", "상관 없어", "가리는 거 없어", "가리는거 없어",
    "딱히 없어", "딱히없어", "다 잘 마셔", "다잘마셔", "다 잘 먹", "다 잘마심",
    "다 괜찮", "다좋아", "아무거나", "뭐든", "안 가려", "안가려", "전부 괜찮",
]

_NO_FAVORITE_PATTERNS = [
    "없어", "없음", "딱히", "딱히 없어", "딱히없어", "생각 안 나", "생각안나",
    "잘 모르겠", "모르겠", "없는데", "없어요",
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", _sanitize_user_text(text).lower())


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ANSI_ESC_SINGLE_RE = re.compile(r"\x1b[@-_]")
_CARET_ANSI_RE = re.compile(r"\^\[\[[0-9;?]*[ -/]*[@-~]")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _sanitize_user_text(text: str) -> str:
    """터미널 escape/control 문자와 방향키 흔적을 제거한다."""
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = _ANSI_ESCAPE_RE.sub(" ", cleaned)
    cleaned = _ANSI_ESC_SINGLE_RE.sub(" ", cleaned)
    cleaned = _CARET_ANSI_RE.sub(" ", cleaned)
    cleaned = _CONTROL_CHAR_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


# Gemma/Qwen 이 칵테일/대화 맥락에서 자주 튀어나오는 한자 → 한글 치환.
# 단순 제거 시 "家族들이랑" → "들이랑" 처럼 문장이 깨지므로, 의미 유지 위해 치환을 우선한다.
_HANJA_TO_HANGUL = {
    "家族": "가족", "家": "집",
    "友人": "친구", "友": "친구", "仲間": "동료",
    "柑橘": "시트러스", "檸檬": "레몬", "萊姆": "라임", "葡萄柚": "자몽", "橙子": "오렌지",
    "草莓": "딸기", "桃": "복숭아", "芒果": "망고", "蘋果": "사과", "鳳梨": "파인애플",
    "水果": "과일", "果": "과일",
    "香": "향", "味": "맛",
    "酸": "신맛", "甘": "단맛", "甜": "단맛", "苦": "쓴맛", "辛": "매운맛",
    "酒": "술", "飲料": "음료", "飲み物": "음료", "杯": "잔",
    "薄荷": "민트", "花": "꽃", "草": "허브",
    "中": "중간", "强": "강하게", "強": "강하게", "弱": "약하게",
    "溫": "따뜻한", "冷": "차가운", "清爽": "상큼한", "新鮮": "신선한",
    "今日": "오늘", "今晩": "오늘 밤", "晩": "저녁",
}

# 치환 대상 아닌 한자/일본어/이모지는 그냥 제거.
_NON_KOREAN_CHAR_RE = re.compile(
    r"[一-鿿"                  # CJK 통합 한자
    r"㐀-䶿"                   # CJK 확장 A
    r"぀-ゟ゠-ヿ"      # 히라가나 / 카타카나
    r"\U0001F300-\U0001F9FF"           # 이모지 (symbols & pictographs)
    r"\U0001FA00-\U0001FAFF"           # 추가 이모지
    r"\U0001F600-\U0001F64F"           # 감정 이모지
    r"\U0001F680-\U0001F6FF"           # 교통/기호 이모지
    r"☀-⛿✀-➿"      # misc symbols / dingbats (✨ ☕ 등)
    r"]+",
    flags=re.UNICODE,
)


def _strip_non_korean_tokens(text: str) -> str:
    """LLM reply 후처리: (1) 자주 튀어나오는 한자 단어는 한글로 치환,
    (2) 나머지 한자/일본어/이모지/기호는 제거, (3) 공백 정리.

    영어 단어는 유지. 단순 제거만 하면 "家族들이랑" → "들이랑" 처럼 어색해지므로
    치환 사전을 먼저 적용한다.
    """
    if not text:
        return text
    cleaned = text
    # 1. 긴 단어 먼저 (家族 > 家) 로 치환
    for hanja, hangul in sorted(_HANJA_TO_HANGUL.items(), key=lambda x: -len(x[0])):
        if hanja in cleaned:
            cleaned = cleaned.replace(hanja, hangul)
    # 2. 남은 한자/이모지/일본어 제거
    cleaned = _NON_KOREAN_CHAR_RE.sub("", cleaned)
    # 3. 공백 정리
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?~])", r"\1", cleaned)
    return cleaned.strip()


def _contains_any(text: str, patterns: list[str]) -> bool:
    normalized = _normalize_text(text)
    return any(_normalize_text(p) in normalized for p in patterns)


def _has_non_pending_value(v) -> bool:
    if isinstance(v, dict):
        return any(val != INTENSITY_PENDING for val in v.values())
    return bool(v)


def _missing_slots(slots: dict) -> list[str]:
    missing: list[str] = []
    for key in SLOT_ASK_ORDER:
        if key not in slots:
            missing.append(key)
            continue
        v = slots.get(key)
        if v is None:
            missing.append(key)
        elif isinstance(v, (list, dict)) and len(v) == 0:
            if key not in _EMPTY_OK_SLOTS:
                missing.append(key)
        elif isinstance(v, dict) and not _has_non_pending_value(v):
            # 초기 태그 seed만 있고 강도가 아직 대화로 확정 안 됨
            missing.append(key)
        elif key in ("taste_profile", "aroma_profile") and isinstance(v, dict):
            # medium(초기 태그 seed) 만 있고 high/low/zero 로 확정된 값이 하나도 없으면 "미확정"
            if not any(sub in INTENSITY_VALUES for sub in v.values()):
                missing.append(key)
        elif isinstance(v, str) and not v.strip():
            missing.append(key)
    return missing

def _next_slot_to_ask(slots: dict) -> Optional[str]:
    miss = _missing_slots(slots)
    return miss[0] if miss else None


_PENDING_AXIS_KO = {
    "sweet": "단맛", "sour": "신맛", "bitter": "쓴맛",
    "body": "바디감", "creamy": "크리미한 질감", "freshness": "청량감",
    "woody": "우디향", "minty": "민트향", "fruity": "과일향",
    "citrus": "시트러스향", "floral": "꽃향", "coffee": "커피향", "herbal": "허브향",
}
_AXIS_OUTER_BY_KEY = {
    "sweet": "taste_profile",
    "sour": "taste_profile",
    "bitter": "taste_profile",
    "body": "taste_profile",
    "creamy": "taste_profile",
    "freshness": "taste_profile",
    "woody": "aroma_profile",
    "minty": "aroma_profile",
    "fruity": "aroma_profile",
    "citrus": "aroma_profile",
    "floral": "aroma_profile",
    "coffee": "aroma_profile",
    "herbal": "aroma_profile",
}
_PURPOSE_ACK = {
    "celebration": "생일 자리면 기분 좋게 마무리되는 쪽이 잘 맞겠네요.",
    "date": "데이트 자리면 너무 거칠지 않게 분위기 맞추는 게 좋겠네요.",
    "business": "회식 자리면 너무 튀지 않으면서도 인상 남는 쪽이 좋겠네요.",
    "solo": "혼자 마시는 자리면 취향이 더 또렷하게 보여도 좋죠.",
    "hangout": "친구들이랑 마시는 자리면 편하게 즐길 수 있는 쪽이 좋겠네요.",
}
_MOOD_ACK = {
    "good": "기분 좋은 날이면 산뜻하게 가도 잘 받죠.",
    "soso": "무난하게 즐기기 좋은 방향으로 맞춰볼 수 있겠네요.",
    "bad": "기분이 좀 처질 때는 취향 맞는 한 잔이 꽤 중요하죠.",
}
_STRENGTH_ACK = {
    "zero": "술 기운은 빼고 맛 중심으로 보면 되겠네요.",
    "light": "도수는 가볍게 가는 쪽이 좋겠네요.",
    "medium": "도수는 적당히 있는 쪽이 좋겠네요.",
    "strong": "도수는 확실히 있는 쪽을 원하시는군요.",
}
_INTENSITY_ACK = {
    "high": "확실한 쪽이 좋으시군요.",
    "medium": "적당한 쪽이 좋으시군요.",
    "low": "은은한 쪽이 좋으시군요.",
    "zero": "없는 쪽이 좋으시군요.",
}
_AXIS_NATURAL_QUESTION = {
    "sweet": "그럼 단맛은 확실히 느껴지는 쪽이 좋아요, 아니면 살짝만 받쳐주는 쪽이 좋아요?",
    "sour": "그럼 신맛은 새콤함이 또렷한 쪽이 좋아요, 아니면 적당히만 있는 쪽이 좋아요?",
    "bitter": "그럼 쓴맛은 포인트로 느껴지는 쪽이 좋아요, 아니면 부담 없게 적당한 쪽이 좋아요?",
    "body": "그럼 바디감은 묵직한 쪽이 좋아요, 아니면 가볍게 넘어가는 쪽이 좋아요?",
    "creamy": "그럼 크리미한 질감은 확실히 부드러운 쪽이 좋아요, 아니면 살짝만 느껴지는 쪽이 좋아요?",
    "freshness": "그럼 청량감은 상큼함이 확 오는 쪽이 좋아요, 아니면 은은하게만 있는 쪽이 좋아요?",
    "woody": "그럼 우디향은 존재감 있게 느껴지는 쪽이 좋아요, 아니면 은은하게 스치는 쪽이 좋아요?",
    "minty": "그럼 민트향은 시원하게 확 느껴지는 쪽이 좋아요, 아니면 살짝만 있는 쪽이 좋아요?",
    "fruity": "그럼 과일향은 또렷하게 올라오는 쪽이 좋아요, 아니면 은은하게 도는 쪽이 좋아요?",
    "citrus": "그럼 시트러스향은 레몬이나 라임처럼 또렷한 쪽이 좋아요, 아니면 은은한 쪽이 좋아요?",
    "floral": "그럼 꽃향은 향긋하게 분명한 쪽이 좋아요, 아니면 은은하게만 감도는 쪽이 좋아요?",
    "coffee": "그럼 커피향은 존재감 있게 느껴지는 쪽이 좋아요, 아니면 끝에 살짝 남는 쪽이 좋아요?",
    "herbal": "그럼 허브향은 또렷한 쪽이 좋아요, 아니면 은은하게만 느껴지는 쪽이 좋아요?",
}
_OUTER_CATEGORY_EXAMPLE = {
    "taste_profile": "맛으로는 단맛, 신맛, 쓴맛, 청량감, 크리미한 질감 같은 쪽으로 많이 나뉘어요.",
    "aroma_profile": "향으로는 과일향, 시트러스향, 허브향, 우디향, 꽃향 같은 쪽으로 많이 나뉘어요.",
}
_SLOT_LEVEL_QUESTION = {
    "party_purpose": "오늘은 어떤 자리에서 마시는 건지 조금만 더 알려주실래요?",
    "current_mood": "오늘 기분은 들뜬 쪽인지, 편하게 쉬고 싶은 쪽인지 궁금해요.",
    "strength_preference": "도수는 가볍게 갈까요, 아니면 술맛이 어느 정도 느껴져도 괜찮아요?",
    "taste_profile": "맛은 달달한 쪽, 상큼한 쪽, 쌉쌀한 쪽 중에 어디가 더 가까워요?",
    "aroma_profile": "향은 과일향, 시트러스향, 허브향, 우디향 중에 어떤 쪽이 더 끌리세요?",
}


def _first_pending_key(slots: dict) -> Optional[str]:
    for outer in ("taste_profile", "aroma_profile"):
        v = slots.get(outer) or {}
        if not isinstance(v, dict):
            continue
        for k, val in v.items():
            if val == INTENSITY_PENDING:
                return k
    return None


def _reply_targets_axis(reply: str, axis_key: str) -> bool:
    ko = _PENDING_AXIS_KO.get(axis_key, axis_key)
    return ko in (reply or "")


_AXIS_FOLLOWUP_HINTS = (
    "어느정도", "강하게", "강한", "은은", "적당", "확드러", "약하게", "약한",
    "빼고", "없었", "확실", "분명", "또렷", "강도", "좋으세요",
)


def _reply_is_specific_axis_followup(reply: str, axis_key: str) -> bool:
    normalized = _normalize_text(reply)
    axis_label = _normalize_text(_PENDING_AXIS_KO.get(axis_key, axis_key))
    if not normalized or axis_label not in normalized:
        return False
    return any(hint in normalized for hint in _AXIS_FOLLOWUP_HINTS)


def _pending_ask_template(axis_key: str) -> str:
    ko = _PENDING_AXIS_KO.get(axis_key, axis_key)
    return f"그럼 {ko} 쪽은 어느 정도가 좋으세요? 확실한 쪽, 적당한 쪽, 아니면 은은한 쪽이요?"


def _infer_last_asked_axis_key(history: list[dict]) -> Optional[str]:
    question = _normalize_text(_last_llm_question(history))
    if not question:
        return None
    for axis_key, label in _PENDING_AXIS_KO.items():
        if _normalize_text(label) in question:
            return axis_key
    return None


def _build_followup_ack(extracted: dict) -> str:
    if not isinstance(extracted, dict) or not extracted:
        return ""

    purpose = extracted.get("party_purpose")
    if purpose in _PURPOSE_ACK:
        return _PURPOSE_ACK[purpose]

    mood = extracted.get("current_mood")
    if mood in _MOOD_ACK:
        return _MOOD_ACK[mood]

    strength = extracted.get("strength_preference")
    if strength in _STRENGTH_ACK:
        return _STRENGTH_ACK[strength]

    for axis in ("taste_profile", "aroma_profile"):
        values = extracted.get(axis) or {}
        if not isinstance(values, dict):
            continue
        for key, intensity in values.items():
            if intensity in _INTENSITY_ACK:
                label = _PENDING_AXIS_KO.get(key, key)
                return f"{label}은 {_INTENSITY_ACK[intensity]}"

    return ""


def _build_ack_segments(extracted: dict) -> list[str]:
    if not isinstance(extracted, dict):
        return []
    parts: list[str] = []

    purpose = extracted.get("party_purpose")
    if purpose in _PURPOSE_ACK:
        parts.append(_PURPOSE_ACK[purpose])

    mood = extracted.get("current_mood")
    if mood in _MOOD_ACK:
        parts.append(_MOOD_ACK[mood])

    strength = extracted.get("strength_preference")
    if strength in _STRENGTH_ACK:
        parts.append(_STRENGTH_ACK[strength])

    for axis in ("taste_profile", "aroma_profile"):
        values = extracted.get(axis) or {}
        if not isinstance(values, dict):
            continue
        for key, intensity in values.items():
            if intensity in _INTENSITY_ACK:
                label = _PENDING_AXIS_KO.get(key, key)
                parts.append(f"{label}은 {_INTENSITY_ACK[intensity]}")

    bases = extracted.get("disliked_bases") or []
    if isinstance(bases, list) and bases:
        if len(bases) == 1:
            parts.append(f"{bases[0]} 베이스는 빼둘게요.")
        else:
            parts.append("말씀하신 베이스는 빼고 볼게요.")

    return parts


def _build_axis_question(axis_key: str) -> str:
    return _AXIS_NATURAL_QUESTION.get(axis_key, _pending_ask_template(axis_key))


def _build_question_answer_prefix(axis_key: Optional[str]) -> str:
    outer = _AXIS_OUTER_BY_KEY.get(axis_key or "")
    if outer in _OUTER_CATEGORY_EXAMPLE:
        return _OUTER_CATEGORY_EXAMPLE[outer]
    return "취향은 몇 가지 방향으로 좁혀가면 생각보다 금방 잡혀요."


def _build_unknown_guidance(axis_key: Optional[str]) -> str:
    label = _PENDING_AXIS_KO.get(axis_key or "", "그 느낌")
    if axis_key == "creamy":
        return "괜찮아요. 평소에 라떼처럼 부드러운 쪽이 편한지, 아니면 깔끔한 쪽이 편한지만 떠올려보셔도 돼요."
    if axis_key == "freshness":
        return "괜찮아요. 에이드처럼 상큼한 쪽이 끌리는지, 아니면 너무 튀지 않는 쪽이 편한지만 생각해보셔도 돼요."
    if _AXIS_OUTER_BY_KEY.get(axis_key or "") == "aroma_profile":
        return f"괜찮아요. {label}이 아예 없는 것보단 살짝 있는 쪽이 좋은지 정도만 봐도 충분해요."
    return f"괜찮아요. {label}이 확실한 쪽이 편한지, 은은한 쪽이 편한지 느낌만 말씀해주셔도 돼요."


def _build_guided_reply(
    history: list[dict],
    current_slots: dict,
    extracted: dict,
    user_intent: str,
    fallback_reply: str,
) -> str:
    merged = merge_slots(current_slots, extracted or {})
    pending_axes = _pending_intensity_keys(merged)
    target_axis = pending_axes[0] if pending_axes else None
    next_slot = _next_slot_to_ask(merged)

    ack_parts = _build_ack_segments(extracted)
    ack = " ".join(ack_parts[:2]).strip()

    if user_intent == "QUESTION":
        answer = _build_question_answer_prefix(target_axis or _infer_last_asked_axis_key(history))
        if target_axis:
            question = _build_axis_question(target_axis)
            return f"{answer} {question}".strip()
        if next_slot:
            question = _SLOT_LEVEL_QUESTION.get(next_slot)
            if question:
                if ack:
                    return f"{ack} {question}".strip()
                return question
        if ack:
            return f"{ack} {answer}".strip()
        return answer

    if user_intent == "UNKNOWN" and target_axis:
        prefix = ack or _build_unknown_guidance(target_axis)
        question = _build_axis_question(target_axis)
        return f"{prefix} {question}".strip()

    if target_axis:
        question = _build_axis_question(target_axis)
        if ack:
            return f"{ack} {question}".strip()
        if fallback_reply and not _reply_is_specific_axis_followup(fallback_reply, target_axis):
            cleaned = re.sub(r"\s+", " ", fallback_reply).strip()
            cleaned = re.sub(r"[?？].*$", "", cleaned).strip()
            if cleaned:
                return f"{cleaned.rstrip('.! ')} {question}".strip()
        return question

    if next_slot:
        question = _SLOT_LEVEL_QUESTION.get(next_slot)
        if question:
            if ack:
                return f"{ack} {question}".strip()
            return question

    if ack:
        return ack
    return fallback_reply.strip()


def _build_pending_followup_reply(
    axis_key: str,
    extracted: dict,
    fallback_reply: str = "",
) -> str:
    question = _pending_ask_template(axis_key)
    if fallback_reply and _reply_is_specific_axis_followup(fallback_reply, axis_key):
        return fallback_reply

    ack = _build_followup_ack(extracted)
    if ack:
        return f"{ack} {question}"

    cleaned = (fallback_reply or "").strip()
    if cleaned:
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"[?？].*$", "", cleaned).strip()
        if cleaned and not _reply_targets_axis(cleaned, axis_key):
            return f"{cleaned.rstrip('.! ')} {question}"

    return question

def _explicit_user_stop(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(p in t for p in USER_STOP_PATTERNS)

def _drop_unchanged_slots(extracted: dict, current: dict) -> dict:
    cleaned: dict[str, Any] = {}
    for key, value in extracted.items():
        old = current.get(key)

        if isinstance(value, dict) and isinstance(old, dict):
            diff = {k: v for k, v in value.items() if old.get(k) != v}
            if diff:
                cleaned[key] = diff
            continue

        if isinstance(value, list) and isinstance(old, list):
            if value != old:
                cleaned[key] = value
            continue

        if value != old:
            cleaned[key] = value

    return cleaned

# ============================================================
# 종료 판단
# ============================================================

MIN_FILLED_SLOTS_FOR_PROCEED = 4
MAX_USER_TURNS = 10
RELAXED_MAX_USER_TURNS = 5
_PROFILE_COMPLETION_TAG_CAP = 2

# 초기 태그에서 시드된 taste/aroma 강도를 "대화로 확정해야 할" 값으로 표시.
# merge_slots 에서 LLM이 high/medium/low 를 넣으면 자연스럽게 덮어써진다.
INTENSITY_PENDING = "pending"


def _count_filled_slots(slots: dict) -> int:
    n = 0
    for k in SLOT_KEYS:
        if k not in slots:
            continue
        v = slots.get(k)
        if v is None:
            continue
        if isinstance(v, (list, dict)) and len(v) == 0:
            if k in _EMPTY_OK_SLOTS:
                n += 1
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if isinstance(v, dict) and not _has_non_pending_value(v):
            continue  # 강도 미확정(pending only)이면 filled 아님
        n += 1
    return n


# current_mood 는 완전 옵셔널 — 사용자가 자발적으로 말해야만 들어가는 항목.
# "모르겠다"가 잦고 추천 품질에 결정적이지 않아서 completion 분모에서 제외.
_COMPLETION_OPTIONAL_SLOTS = {"disliked_bases", "favorite_drinks"}

# 대화로 "확정"된 것으로 볼 강도 값 = INTENSITY_VALUES.
# 초기 태그 시드는 INTENSITY_PENDING("pending") 으로 넣고, 대화에서 값이 붙으면 확정으로 본다.

MIN_RECOMMEND_COMPLETION = 80.0


def _calc_effective_completion(slots: dict) -> float:
    """Flat tag-level completion.

    분모 구성:
      - party_purpose: 1단위 (scalar)
      - strength_preference: 1단위 (scalar)
      - taste_profile: 최대 2개 태그까지만 분모 반영
      - aroma_profile: 최대 2개 태그까지만 분모 반영

    분자 가중:
      - 확정된 scalar = 1.0
      - 확정된 강도 태그 (zero/low/medium/high) = 1.0
      - pending 태그 = 0 (사용자가 아직 강도를 말 안 한 상태)
      - null/empty = 0

    NOTE: pending 은 retrieval/score 에서는 soft-medium prior 로 반영하되,
    completion 계산에는 포함하지 않는다. 사용자가 실제로 발화한 정보만이
    "추천 준비 완료도" 의 정당한 근거이기 때문. pending 을 카운트하면 초기
    태그만 선택한 상태에서도 완성도가 인위적으로 높아져 대화 유도가 깨진다.

    선택 슬롯(current_mood / disliked_bases / favorite_drinks)은 분모 제외.

    예: scalar 2 확정 + taste[freshness=high, body=pending] + aroma[fruity=high]
      → denom=4, filled=3 → 75.0% (body pending 은 filled 에 안 들어감)
    """
    denom = 0.0
    filled = 0.0

    for k in ("party_purpose", "strength_preference"):
        denom += 1.0
        v = slots.get(k)
        if isinstance(v, str) and v.strip():
            filled += 1.0

    for k in ("taste_profile", "aroma_profile"):
        v = slots.get(k)
        if not isinstance(v, dict) or not v:
            continue  # 태그 0개면 축 자체 분모 제외
        confirmed = sum(1 for val in v.values() if val in INTENSITY_VALUES)
        cap = min(len(v), _PROFILE_COMPLETION_TAG_CAP)
        denom += float(cap)
        filled += float(min(confirmed, cap))

    if denom <= 0:
        return 0.0
    return round(filled / denom * 100, 2)


def _has_pending_intensity(slots: dict) -> bool:
    """taste_profile/aroma_profile 안에 강도 미확정(pending) 축이 남아 있나?"""
    for key in ("taste_profile", "aroma_profile"):
        v = slots.get(key) or {}
        if isinstance(v, dict) and any(val == INTENSITY_PENDING for val in v.values()):
            return True
    return False


def _pending_intensity_keys(slots: dict) -> list[str]:
    """강도 미확정 taste/aroma 축 목록."""
    pending: list[str] = []
    for key in ("taste_profile", "aroma_profile"):
        v = slots.get(key) or {}
        if not isinstance(v, dict):
            continue
        for sub_key, sub_val in v.items():
            if sub_val == INTENSITY_PENDING and sub_key not in pending:
                pending.append(sub_key)
    return pending


def should_move_to_recommendation(
    merged_slots: dict,
    user_turn_count: int,
    user_msg: str,
    llm_should_stop: bool = False,
    llm_stop_reason: str = "",
) -> tuple[bool, str]:
    """추천 단계 진입 게이트 — 최소화된 3 조건.

    LLM 의 판단을 신뢰한다. completion/pending 같은 내부 메트릭으로 대화를
    끊지 않는다 (사용자가 질문해도 completion 만 높으면 종료되던 문제 방지).

      1) 사용자가 명시적으로 추천을 원함 → 즉시 통과
      2) 대화 5턴 초과 → 강제 추천 (무한 루프 방지)
      3) LLM 이 RECOMMEND 판단 → 통과
    """
    if _explicit_user_stop(user_msg):
        return True, "user_requested"
    if user_turn_count >= 5:
        return True, "turn_limit"
    if llm_should_stop:
        return True, llm_stop_reason or "llm_recommend"
    return False, "keep_collecting"


_PROCEED_REASON_TO_OPENER = {
    "completion_threshold": "취향을 충분히 파악했어요. 맞춤 칵테일을 골라드릴게요.",
    "user_requested":       "네, 바로 추천해드릴게요.",
    "turn_limit":           "지금까지 말씀해주신 내용으로 추천해드릴게요.",
    "llm_recommend":        "이 정도면 충분할 것 같아요. 이제 추천으로 넘어갈게요.",
}

# 80% 미만에서도 통과시키는 reason → /recommend-sample 에 force=True 가 필요함.
_PROCEED_REASONS_FORCE = {"user_requested", "turn_limit"}


def transition_opener(reason: str) -> str:
    """proceed_to_recommendation 직전에 사용자에게 보낼 전환 멘트."""
    return _PROCEED_REASON_TO_OPENER.get(reason, _PROCEED_REASON_TO_OPENER["llm_recommend"])


def proceed_requires_force(reason: str) -> bool:
    """이 reason 은 completion<80 에서도 통과 → /recommend-sample 에 force 필요."""
    return reason in _PROCEED_REASONS_FORCE


# ============================================================
# 피드백 인텐트 분류 + 벡터 델타 (LLM 단일 호출)
# ============================================================

FEEDBACK_INTENTS = {"ACCEPT", "ADJUST", "REJECT"}

VECTOR_FIELDS = {
    "sweetness_score":  (1.0, 5.0),
    "bitterness_score": (1.0, 5.0),
    "sourness_score":   (1.0, 5.0),
    "freshness_score":  (1.0, 5.0),
    "body_score":       (1.0, 5.0),
    "herbal_score":     (0.0, 3.0),
    "citrus_score":     (0.0, 3.0),
    "alcohol_score":    (0.0, 5.0),
}

_FEEDBACK_SYSTEM_PROMPT = """
너는 칵테일 추천 시스템의 피드백 분석기다.
사용자가 추천된 샘플 칵테일에 남긴 한국어 피드백을 보고 의도(intent)와 선호 벡터 변화량(deltas)을 추론한다.
반드시 JSON 객체 하나만 출력. 설명·마크다운·코드블록 금지.

[핵심 원칙 — 방향 해석, 절대 혼동 금지]
사용자가 "X하다" 형태로 감상을 말했으면 그 축이 **이미 과하다**는 뜻이다.
→ 해당 축의 vector_delta 는 반드시 **음수(-)**.
  "달아/달다/달콤해서" → sweetness_score **-**  (절대 + 아니다)
  "써/쓰다/씁쓸해"      → bitterness_score **-**
  "시다/시어/셔서"      → sourness_score **-**
  "독해/세다/쎄다"      → alcohol_score **-**
  "진해/무거워"          → body_score **-**
"더 X게 해줘"/"더 X하게" 꼴만 증가(+). "너무/좀/조금 + X하다" 꼴은 전부 감소(-).
"좀 달아" 는 "달아서 부담스럽다" 의 축약이다 — 감소(-) 이지 절대 증가(+) 아니다.

[어휘 구분 — 도수 vs 신맛]
- "세다/쎄다"  = 도수가 강하다  → alcohol_score 감소
- "시다/시어"  = 신맛이 강하다  → sourness_score 감소
"너무 세" 는 alcohol, "너무 시" 는 sour. 자음 하나 차이니 주의.

[intent 정의 — 매우 중요]
- ACCEPT: 추천을 그대로 받겠다는 의사. 만족/수용 표현.
  예) "좋아", "이걸로 갈게", "딱이야", "이대로 받을게"
- ADJUST: **같은 칵테일을 마셨는데 특정 축이 과하거나 부족하다고 느낌.** "너무/좀 X하다" 식의 감상 포함.
  이건 "다른 걸 달라"가 아니라 "같은 음료를 이 방향으로 조정해서 다시 만들어달라"는 뜻으로 해석해라.
  예) "더 달게", "덜 시게", "좀 더 강한 걸로", "부드럽게 갔으면", "쨍한 느낌으로",
      "조금만 더 묵직하게", "허브향 좀 빼줘",
      ★ "너무 달아" → sweetness_score -0.5 (줄여달라는 뜻),
      ★ "너무 써" → bitterness_score -0.5,
      ★ "좀 달아" / "조금 달아" → sweetness_score -0.3,
      ★ "너무 약해" → alcohol_score +0.5,
      ★ "너무 진해" → body_score -0.5,
      ★ "신맛이 너무 세" → sourness_score -0.5
- REJECT: **강하고 명시적인** 거부 + "다른 거 달라"는 명시적 요구.
  반드시 둘 다 충족: (강한 부정) AND (교체 요구).
  예) "이거 완전 노맛 다른거 줘", "이거 맛없어 바꿔줘", "이건 별로야 다른거 줘",
      "내 취향 아냐 새로 추천해줘", "아예 다른 거 달라"
  ★ "너무 X하다" 만 있는 건 REJECT 아님 → ADJUST.
  ★ "맛없어" / "별로야" 만 있고 "다른거 달라" 없으면 ADJUST (부정 방향 추론).

[vector_deltas]
- ADJUST일 때만 의미가 있다. ACCEPT/REJECT면 빈 dict {} 를 출력해라.
- 키는 다음 중 하나여야 한다:
  sweetness_score, bitterness_score, sourness_score, freshness_score,
  body_score, herbal_score, citrus_score, alcohol_score
- 값은 -1.0 ~ +1.0 사이의 float. 단계 가이드:
    살짝/조금/약간 → ±0.3
    그냥 "더/덜"     → ±0.5
    훨씬/확/완전     → ±1.0
- 한 피드백에 여러 축이 동시에 언급되면 모두 넣어라.
- 사용자가 직접 말하지 않은 축은 절대 추측하지 마라.

[키워드 → 축 매핑 가이드]
- "달", "달콤", "달달"        → sweetness_score
- "쓰", "쌉싸름", "씁쓸"       → bitterness_score
- "시", "시큼", "신맛"         → sourness_score
- "상큼", "청량", "쨍", "깔끔"  → freshness_score
- "묵직", "무겁", "바디"       → body_score
- "허브", "풀내", "민트"       → herbal_score
- "시트러스", "레몬", "라임"    → citrus_score
- "강", "쎄", "독", "약", "가볍" → alcohol_score
- "부드럽" 단독은 body_score -0.3 (약하게 달라는 의미)

[출력 형식]
{
  "intent": "ACCEPT" | "ADJUST" | "REJECT",
  "vector_deltas": { "<score_field>": <float>, ... }
}

예시:
USER: "좋아 이걸로 갈게"
→ {"intent":"ACCEPT","vector_deltas":{}}

USER: "이대로 받아도 될 것 같아"
→ {"intent":"ACCEPT","vector_deltas":{}}

USER: "굳이 더 손볼 필요는 없어 보여"
→ {"intent":"ACCEPT","vector_deltas":{}}

USER: "괜찮네"
→ {"intent":"ACCEPT","vector_deltas":{}}

USER: "괜찮아"
→ {"intent":"ACCEPT","vector_deltas":{}}

USER: "굿" / "이거 굿" / "오 굿" / "ㄱㄱ"
→ {"intent":"ACCEPT","vector_deltas":{}}

USER: "ㅇㅋ" / "오키" / "오케이"
→ {"intent":"ACCEPT","vector_deltas":{}}

USER: "완전히 야르다" / "야르다" / "개꿀" / "대박" / "쩐다" / "존맛"
→ {"intent":"ACCEPT","vector_deltas":{}}
# 이런 슬랭·감탄사는 모두 강한 긍정이다. REJECT 절대 아니다.

USER: "조금 더 달게 해줘"
→ {"intent":"ADJUST","vector_deltas":{"sweetness_score":0.3}}

USER: "훨씬 강하게 가줘"
→ {"intent":"ADJUST","vector_deltas":{"alcohol_score":1.0}}

USER: "덜 시고 좀 더 묵직하게"
→ {"intent":"ADJUST","vector_deltas":{"sourness_score":-0.5,"body_score":0.5}}

USER: "허브향 좀 빼고 시트러스 살려줘"
→ {"intent":"ADJUST","vector_deltas":{"herbal_score":-0.5,"citrus_score":0.5}}

USER: "쨍하고 상큼한 느낌으로"
→ {"intent":"ADJUST","vector_deltas":{"freshness_score":0.5}}

USER: "너무 달아"
→ {"intent":"ADJUST","vector_deltas":{"sweetness_score":-0.5}}

USER: "좀 달다" / "조금 달아"
→ {"intent":"ADJUST","vector_deltas":{"sweetness_score":-0.3}}

USER: "너무 시어"
→ {"intent":"ADJUST","vector_deltas":{"sourness_score":-0.5}}

USER: "이거 너무 독해"
→ {"intent":"ADJUST","vector_deltas":{"alcohol_score":-0.5}}

USER: "이거 완전 노맛 다른거 줘"
→ {"intent":"REJECT","vector_deltas":{}}

USER: "이거 맛없어 바꿔줘"
→ {"intent":"REJECT","vector_deltas":{}}

USER: "이건 별로야 다른거 줘"
→ {"intent":"REJECT","vector_deltas":{}}

USER: "내 취향이랑 너무 달라, 아예 다른거 줘"
→ {"intent":"REJECT","vector_deltas":{}}

반드시 위 형식의 JSON 한 줄만 출력해라.
""".strip()


def _clamp_score(field: str, value: float) -> float:
    lo, hi = VECTOR_FIELDS.get(field, (0.0, 5.0))
    return max(lo, min(hi, value))


_FEEDBACK_NEG_FEELING_RULES: list[tuple[str, list[str]]] = [
    ("sweetness_score",  ["달아", "달다", "달아서", "달콤해서", "달콤하다"]),
    ("bitterness_score", ["쓴맛", "쓴 맛", "쓴데", "쓴", "써서", "쓰다", "쓰네", "씁쓸해", "씁쓸하다", "씁쓸해서"]),
    ("sourness_score",   ["시다", "시어", "시어서", "셔서", "시큼해", "시큼하다"]),
    ("alcohol_score",    ["독해", "독하다", "독해서", "세다", "세네", "세서",
                          "쎄다", "쎄네", "쎄서"]),
    ("body_score",       ["진해", "진하다", "진해서", "무거워", "무거워서"]),
]

_FEEDBACK_DEGREE_MAG: list[tuple[list[str], float]] = [
    (["훨씬", "확", "완전", "엄청", "지나치게"], 1.0),
    (["너무", "매우", "아주", "진짜"],           0.5),
    (["좀", "조금", "약간", "살짝"],             0.3),
]

# "신맛이 너무 세" 같은 섞임 표현에서 alcohol 오판 방지
_FEEDBACK_SUBJECT_CONFLICT: dict[str, list[str]] = {
    "alcohol_score": ["신맛", "단맛", "쓴맛", "바디", "산미", "크리미"],
}

# 원문 증거 체크 — LLM 이 룰 적용 가능 축에 델타를 넣었는데 룰이 fire 안 했고
# 이 키워드도 원문에 없으면 LLM 오분류로 보고 드롭
_FEEDBACK_AXIS_TEXT_EVIDENCE: dict[str, list[str]] = {
    "sweetness_score":  ["달", "단맛", "달콤"],
    "bitterness_score": ["쓴", "쓴맛", "쓰", "써", "쌉", "씁"],
    "sourness_score":   ["시", "신맛", "셔", "새콤", "시큼"],
    "alcohol_score":    ["강", "쎄", "세", "독", "약해", "약한", "가볍", "도수"],
    "body_score":       ["묵직", "무거", "바디", "진", "부드"],
}
_FEEDBACK_RULE_AXIS_PATTERNS: dict[str, list[str]] = {
    "sweetness_score": ["달", "단맛", "달콤", "달달"],
    "bitterness_score": ["쓴맛", "쓴", "쌉", "씁"],
    "sourness_score": ["신맛", "시큼", "시", "산미"],
    "freshness_score": ["청량", "상큼", "깔끔", "쨍"],
    "body_score": ["바디", "묵직", "무거", "진", "부드럽"],
    "herbal_score": ["허브", "민트", "풀내"],
    "citrus_score": ["시트러스", "레몬", "라임", "자몽", "오렌지"],
    "alcohol_score": ["도수", "술맛", "독", "강", "쎄", "세", "가볍"],
}
_FEEDBACK_MORE_PATTERNS = [
    "더", "올려", "올려줘", "늘려", "늘려줘", "넣어", "넣어줘",
    "살려", "살려줘", "강하게", "강한게", "강한걸", "강한 거",
    "진하게", "쎄게", "세게",
]
_FEEDBACK_LESS_PATTERNS = [
    "덜", "빼", "빼줘", "빼고", "줄여", "줄여줘", "낮춰", "낮춰줘",
    "약하게", "약한게", "약한걸", "약한 거", "은은하게",
]


def _apply_feedback_sign_rules(text: str, deltas: dict[str, float]) -> dict[str, float]:
    """LLM 부호 오독을 룰로 교정.

    1) "너무/좀 + X하다" 형 감상어 감지 → 해당 축 델타를 규칙 기반 음수로 강제.
    2) 룰 적용 가능 축에 LLM 이 델타를 넣었지만 룰도 fire 안 하고 원문 증거도 없으면
       오분류로 보고 드롭 ("너무 세다" 에 LLM 이 sourness 를 넣은 경우 등).
    """
    if not text:
        return deltas
    out = dict(deltas or {})
    rule_set: set[str] = set()

    for axis, tokens in _FEEDBACK_NEG_FEELING_RULES:
        hit_pos = -1
        for tok in tokens:
            p = text.find(tok)
            if p >= 0:
                hit_pos = p
                break
        if hit_pos < 0:
            continue
        pre_window = text[max(0, hit_pos - 15): hit_pos]
        if any(w in pre_window for w in _FEEDBACK_SUBJECT_CONFLICT.get(axis, [])):
            continue
        # "더" 가 앞에 있으면 "좀 더 달아", "더 달아도 좋을" 같은 positive request →
        # 불평 룰로 덮어쓰지 말고 LLM 의 부호를 그대로 유지한다.
        if "더" in pre_window:
            continue
        mag = 0.3
        for adverbs, m in _FEEDBACK_DEGREE_MAG:
            if any(a in pre_window for a in adverbs):
                mag = m
                break
        out[axis] = -mag
        rule_set.add(axis)

    rule_axes = {a for a, _ in _FEEDBACK_NEG_FEELING_RULES}
    for axis in list(out.keys()):
        if axis in rule_set or axis not in rule_axes:
            continue
        kws = _FEEDBACK_AXIS_TEXT_EVIDENCE.get(axis, [])
        if kws and not any(k in text for k in kws):
            out.pop(axis, None)

    return out


def _feedback_request_magnitude(normalized_text: str) -> float:
    if any(tok in normalized_text for tok in [_normalize_text(t) for t in ("훨씬", "확", "완전", "엄청", "매우")]):
        return 1.0
    if any(tok in normalized_text for tok in [_normalize_text(t) for t in ("좀", "조금", "약간", "살짝")]):
        return 0.3
    return 0.5


def _apply_feedback_request_rules(text: str, deltas: dict[str, float]) -> dict[str, float]:
    """LLM 이 놓친 '더/덜/빼줘/올려줘' 조정 요청을 룰로 복구한다."""
    if not text:
        return deltas

    normalized = _normalize_text(text)
    out = dict(deltas or {})
    more = any(_normalize_text(p) in normalized for p in _FEEDBACK_MORE_PATTERNS)
    less = any(_normalize_text(p) in normalized for p in _FEEDBACK_LESS_PATTERNS)
    if more == less:
        return out

    magnitude = _feedback_request_magnitude(normalized)
    signed = magnitude if more else -magnitude

    for axis, patterns in _FEEDBACK_RULE_AXIS_PATTERNS.items():
        if axis in out:
            continue
        if any(_normalize_text(pattern) in normalized for pattern in patterns):
            out[axis] = signed

    return out


def _validate_feedback_output(parsed: dict) -> tuple[str, dict[str, float]]:
    intent_raw = str(parsed.get("intent", "")).strip().upper()
    intent = intent_raw if intent_raw in FEEDBACK_INTENTS else "REJECT"

    deltas_raw = parsed.get("vector_deltas") or {}
    deltas: dict[str, float] = {}
    if isinstance(deltas_raw, dict) and intent == "ADJUST":
        for k, v in deltas_raw.items():
            if k not in VECTOR_FIELDS:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            fv = max(-1.0, min(1.0, fv))
            if fv != 0.0:
                deltas[k] = fv
    return intent, deltas


# "맛없어/별로야" 처럼 축 정보 없는 부정 발화는 ADJUST 로 잡혀도 deltas 가 비어
# 의미 있는 조정이 불가능하다. 이런 경우 REJECT 로 재분류해 다른 칵테일 추천으로 넘긴다.
_GENERIC_NEGATIVE_PATTERNS = (
    "맛없", "맛 없", "별로야", "별로네", "별로다", "별로인", "별로였",
    "안 좋아", "안좋아", "안 좋네", "안좋네", "마음에 안", "마음에안",
    "싫다", "싫네", "싫어졌", "끌리지 않", "끌리지않", "땡기지 않", "땡기지않",
    "이건 아니", "이건아니", "별로 안", "별로안",
)


def _is_generic_negative_without_axis(text: str, deltas: dict) -> bool:
    """축 언급 없는 일반적 부정 표현인지 판정.

    - deltas 비어 있음 (구체 축/방향 추출 실패)
    - 부정 표현 키워드 포함
    - taste/aroma 축 키워드 **없음** (있으면 ADJUST 유지해서 LLM 재시도 가능)
    """
    if deltas:
        return False
    if not text:
        return False
    low = text.strip().lower()
    if not any(p in low for p in _GENERIC_NEGATIVE_PATTERNS):
        return False
    # 축 키워드 하나라도 있으면 ADJUST 유지 (LLM 이 delta 잘못 뽑은 케이스는 별개 문제)
    for kws in _AXIS_SUBKEY_TO_KEYWORDS.values():
        if _contains_axis_keyword(text, kws):
            return False
    return True


def analyze_feedback(
    before_vec: dict[str, float],
    feedback_text: str,
    max_new_tokens: int = 192,
) -> dict:
    """LLM 단일 호출로 피드백 intent + 벡터 델타 분석.

    반환:
      {"intent": "ACCEPT|ADJUST|REJECT",
       "deltas": {field: delta, ...},
       "updated_vec": {field: clamped_score, ...},
       "raw": "<원문>"}
    """
    feedback_text = _sanitize_user_text(feedback_text)
    try:
        import torch

        tokenizer, model, render_chat = _load_dialogue_llm_resources()
        rendered = render_chat(
            tokenizer,
            _FEEDBACK_SYSTEM_PROMPT,
            f"USER 피드백: {feedback_text}\n\nJSON으로 답해라.",
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
        parsed = _extract_json_object(raw) or {}
        intent, deltas = _validate_feedback_output(parsed)

        if intent == "ADJUST":
            deltas = _apply_feedback_sign_rules(feedback_text, deltas)
            deltas = _apply_feedback_request_rules(feedback_text, deltas)

        # 축 없는 일반 부정("맛없어/별로야") 은 ADJUST deltas={} 로 잡혀도 의미가 없다.
        # 그냥 다른 칵테일 추천으로 넘기게 REJECT 로 재분류.
        if intent == "ADJUST" and _is_generic_negative_without_axis(feedback_text, deltas):
            logger.info(
                "feedback: reclassifying ADJUST → REJECT (generic negative without axis): %r",
                feedback_text,
            )
            intent = "REJECT"
            deltas = {}

        updated = dict(before_vec or {})
        for field, delta in deltas.items():
            updated[field] = _clamp_score(field, float(updated.get(field, 2.5)) + delta)

        return {
            "intent": intent,
            "deltas": deltas,
            "updated_vec": updated,
            "raw": raw,
        }
    except Exception as e:
        logger.warning("LLM analyze_feedback failed: %r", e, exc_info=True)
        return {
            "intent": "REJECT",
            "deltas": {},
            "updated_vec": dict(before_vec or {}),
            "raw": "",
        }


def classify_intent(feedback_text: str) -> str:
    """후방호환 — 단독 인텐트만 필요할 때. 내부적으로 analyze_feedback을 호출."""
    return analyze_feedback({}, feedback_text)["intent"]


def update_vector(before_vec: dict[str, float], feedback_text: str) -> dict[str, float]:
    """후방호환 — 분석+델타 적용 결과 벡터를 반환."""
    return analyze_feedback(before_vec, feedback_text)["updated_vec"]


# ============================================================
# LLM 프롬프트 구성
# ============================================================


def _format_history(history: list[dict], max_turns: int = 4) -> str:
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

def _infer_last_asked_slot(history: list[dict]) -> Optional[str]:
    last_q = _last_llm_question(history)
    q = _normalize_text(last_q)
    if not q:
        return None

    if any(k in q for k in ["어떤자리", "자리", "모임", "데이트", "혼자", "회식"]):
        return "party_purpose"
    if any(k in q for k in ["기분", "다운", "그냥저냥"]):
        return "current_mood"
    if any(k in q for k in ["도수", "가볍게", "확가는", "세게", "무알콜"]):
        return "strength_preference"
    if any(k in q for k in ["맛은", "달달", "상큼", "쌉싸름", "쓴맛", "단맛"]):
        return "taste_profile"
    if any(k in q for k in ["향은", "과일향", "허브향", "나무향", "커피향", "플로럴"]):
        return "aroma_profile"
    if any(k in q for k in ["베이스", "위스키", "보드카", "데킬라", "럼", "진", "피하고싶으신술"]):
        return "disliked_bases"
    if any(k in q for k in ["즐겨드시는", "자주드시는", "좋아하시는술", "칵테일이나술", "평소에"]):
        return "favorite_drinks"
    return None


_MOOD_KEYWORDS = {
    "good": ["기분 좋", "기분좋", "신나", "신난", "설레", "들떠",
             "째진", "째져", "째짐", "째지", "쩐다", "쩔어",
             "텐션", "업됐", "업됨"],
    "bad": ["기분 별로", "꿀꿀", "우울", "다운", "지쳐", "피곤", "쳐져", "안 좋", "안좋",
            "구리다", "구려", "구림", "구리네", "구려서", "꾸리다"],
    "soso": ["그냥 그래", "그냥그래", "쏘쏘", "그저 그래", "그저그래", "보통이", "그럭저럭", "무난"],
}
_PURPOSE_KEYWORDS = {
    "celebration": ["생파", "생일", "축하", "돌잔치", "기념일"],
    "date": ["데이트", "여친", "남친", "썸녀", "썸남", "둘이"],
    "business": ["회식", "거래처", "직장 동료", "회사 동료", "비즈니스"],
    "solo": ["혼술", "혼자 마시", "혼자 왔", "나 혼자"],
    "hangout": ["친구들이랑", "놀러", "모임", "캐주얼", "친구랑", "친구와",
                "가족", "가족들", "가족이랑", "가족과", "가족분"],
}
_STRENGTH_KEYWORDS = {
    "strong": ["독하게", "세게", "쎄게", "강하게", "달리자", "달릴", "술고래", "취하고 싶"],
    "light": ["약하게", "가볍게", "순한", "살살", "부드럽게", "잘 못 마셔"],
    "zero": ["무알콜", "논알콜", "알콜 빼", "술 빼"],
    "medium": ["적당히", "보통으로", "중간으로"],
}
_STRENGTH_EXPLICIT_PATTERNS = [
    "도수", "알콜", "알코올", "무알콜", "논알콜", "술 ",
    "술이", "술은", "술로", "취하고", "취하", "독한 거", "순한 거",
]
_AFFIRMATIVE_PREFERENCE_PATTERNS = [
    "좋아", "좋지", "좋네", "좋은", "원해", "원하지", "원하는",
    "끌려", "끌리는", "맞아", "그렇지", "그쪽", "그 쪽", "느낌이지",
    "느낌", "스타일", "취향", "가고 싶", "가고싶",
]


def _backfill_enum(user_msg: str, patterns: dict[str, list[str]]) -> Optional[str]:
    text = (user_msg or "").lower()
    for enum_val, kws in patterns.items():
        for kw in kws:
            if kw in text:
                return enum_val
    return None


def _has_explicit_strength_signal(user_msg: str) -> bool:
    normalized = _normalize_text(user_msg)
    if any(_normalize_text(p) in normalized for p in _STRENGTH_EXPLICIT_PATTERNS):
        return True
    for kws in _STRENGTH_KEYWORDS.values():
        if any(_normalize_text(kw) in normalized for kw in kws):
            return True
    return False


def _has_explicit_scalar_signal(user_msg: str, key: str) -> bool:
    if key == "current_mood":
        return _backfill_enum(user_msg, _MOOD_KEYWORDS) is not None or _contains_any(user_msg, ["기분"])
    if key == "party_purpose":
        return _backfill_enum(user_msg, _PURPOSE_KEYWORDS) is not None
    if key == "strength_preference":
        return _has_explicit_strength_signal(user_msg)
    return False


def _suppress_unanchored_scalar_overrides(current_slots: dict, user_msg: str, extracted: dict) -> dict:
    """직접적인 근거 없이 기존 scalar 슬롯을 덮어쓰거나 지우는 걸 막는다."""
    fixed = dict(extracted or {})
    for key in ("current_mood", "party_purpose", "strength_preference"):
        if key not in fixed:
            continue
        current_val = (current_slots or {}).get(key)
        next_val = fixed.get(key)
        if current_val in (None, "", "null") or current_val == next_val:
            continue
        if _has_correction_signal(user_msg):
            continue
        if _has_explicit_scalar_signal(user_msg, key):
            continue
        fixed.pop(key, None)
    return fixed


def _apply_rule_based_slot_guards(
    history: list[dict],
    user_msg: str,
    extracted: dict,
) -> dict:
    """LLM 오추출을 줄이기 위한 안전장치."""
    fixed = dict(extracted or {})
    last_slot = _infer_last_asked_slot(history)

    if last_slot == "disliked_bases" and _contains_any(user_msg, _NO_PREFERENCE_PATTERNS):
        fixed["disliked_bases"] = []

    if last_slot == "favorite_drinks" and _contains_any(user_msg, _NO_FAVORITE_PATTERNS):
        fixed["favorite_drinks"] = []

    # 스칼라 enum 백업 — 사용자가 명시적으로 키워드를 내뱉으면 last_slot 과
    # 무관하게 채워준다. (예: 첫 턴에 "회식해. 기분 구리다" 면 mood/purpose 둘 다
    # 잡혀야 하지만 LLM 이 한 축만 뽑는 케이스가 잦음.)
    if "current_mood" not in fixed:
        v = _backfill_enum(user_msg, _MOOD_KEYWORDS)
        if v:
            fixed["current_mood"] = v
    if "party_purpose" not in fixed:
        v = _backfill_enum(user_msg, _PURPOSE_KEYWORDS)
        if v:
            fixed["party_purpose"] = v
    if "strength_preference" not in fixed:
        v = _backfill_enum(user_msg, _STRENGTH_KEYWORDS)
        if v:
            fixed["strength_preference"] = v

    # 도수 질문이 아니거나, 사용자 발화에 도수 관련 명시 표현이 없으면 strength 오추출을 버린다.
    # 단, 사용자가 도수-직결 키워드("독하게", "가볍게", "무알콜" 등)를 직접 말했으면 명시로 본다.
    if "strength_preference" in fixed:
        normalized = _normalize_text(user_msg)
        explicit_strength = _has_explicit_strength_signal(user_msg)
        if last_slot != "strength_preference" and not explicit_strength:
            fixed.pop("strength_preference", None)

    return fixed


# ─── CONFIRM 발화 → 직전 LLM 제안값 복원 ─────────────────────────
# 바텐더가 "청량함 강하게, 크리미 약하게로 갈까요?" 제안 후
# 사용자가 "ㅇㅇ" / "응" / "그래" 하면 EXTRACT 는 rule 8 대로 {} 를 내뱉는다.
# 그러면 제안값이 슬롯에 반영되지 않는 설계 갭이 생긴다.
# → 아래 룰 기반 후처리로 직전 LLM 발화에서 "축 + 강도" 패턴을 복구한다.

_PROPOSAL_WORD_TO_KEY: dict[str, tuple[str, str]] = {
    # taste
    "단맛": ("taste_profile", "sweet"),
    "달달": ("taste_profile", "sweet"),
    "달콤": ("taste_profile", "sweet"),
    "신맛": ("taste_profile", "sour"),
    "새콤": ("taste_profile", "sour"),
    "쓴맛": ("taste_profile", "bitter"),
    "쌉싸름": ("taste_profile", "bitter"),
    "씁쓸": ("taste_profile", "bitter"),
    "바디감": ("taste_profile", "body"),
    "묵직": ("taste_profile", "body"),
    "크리미": ("taste_profile", "creamy"),
    "부드러": ("taste_profile", "creamy"),
    "청량": ("taste_profile", "freshness"),
    "상큼": ("taste_profile", "freshness"),
    # aroma (향 단어는 더 길게 매칭해서 오염 방지)
    "우디향": ("aroma_profile", "woody"),
    "우디한 향": ("aroma_profile", "woody"),
    "나무향": ("aroma_profile", "woody"),
    "민트향": ("aroma_profile", "minty"),
    "민트감": ("aroma_profile", "minty"),
    "과일향": ("aroma_profile", "fruity"),
    "과일 향": ("aroma_profile", "fruity"),
    "프루티": ("aroma_profile", "fruity"),
    "시트러스향": ("aroma_profile", "citrus"),
    "시트러스 향": ("aroma_profile", "citrus"),
    "시트러스": ("aroma_profile", "citrus"),
    "레몬향": ("aroma_profile", "citrus"),
    "자몽향": ("aroma_profile", "citrus"),
    "라임향": ("aroma_profile", "citrus"),
    "플로럴": ("aroma_profile", "floral"),
    "플로럴한 느낌": ("aroma_profile", "floral"),
    "꽃향": ("aroma_profile", "floral"),
    "커피향": ("aroma_profile", "coffee"),
    "커피 느낌": ("aroma_profile", "coffee"),
    "커피감": ("aroma_profile", "coffee"),
    "허브향": ("aroma_profile", "herbal"),
    "허브 느낌": ("aroma_profile", "herbal"),
    "허벌한 느낌": ("aroma_profile", "herbal"),
    "허벌한": ("aroma_profile", "herbal"),
}

_INTENSITY_WORDS: list[tuple[str, list[str]]] = [
    ("high", ["강하게", "세게", "쎄게", "확", "진하게", "짱", "듬뿍"]),
    ("low", ["약하게", "살짝", "은은하게", "옅게", "연하게", "약간"]),
    ("medium", ["적당히", "적당하게", "보통", "중간"]),
    ("zero", ["빼고", "없이", "질색"]),
]

_CONFIRM_TOKEN = (
    r"(?:"
    r"맞아요?|맞네|맞습니다|맞죠|응|어|네|예|ㅇㅇ+|ㅇㅋ+|오케이?|ok|okay|"
    r"좋아요?|좋습니다|그래|그대로|그렇게(?:\s*해(?:줘)?)?|그거(?:로|로요)?|"
    r"너가?\s*말한\s*대로|네?\s*그렇게\s*해주세요"
    r")"
)
_CONFIRM_ONLY_RE = re.compile(
    r"^\s*" + _CONFIRM_TOKEN + r"(?:[,\s!.?~]+" + _CONFIRM_TOKEN + r")*\s*[!.?~]*\s*$",
    re.IGNORECASE,
)
_CONFIRM_PREFIX_RE = re.compile(
    r"^\s*" + _CONFIRM_TOKEN + r"(?:\s*[,.!?~]+\s*|\s+)",
    re.IGNORECASE,
)

# 양자택일 질문 감지 — "A 아니면 B", "A 좋으세요 B 좋으세요", 슬래시 옵션 등.
# ("A, B 로 갈까요" 같은 '복수축 동시제안' 은 제안이므로 여기서 걸리면 안 된다.)
_ALTERNATIVE_Q_RE = re.compile(
    r"아니면|또는"
    r"|좋으세요.{0,15}좋으세요"
    r"|좋아하세요.{0,15}좋아하세요"
    r"|어떠세요.{0,15}어떠세요"
    r"|(강하게|세게|쎄게).{0,10}(아니면|또는|아님)"
    r"|\s/\s"
)


def _is_confirm_only(user_msg: str) -> bool:
    return bool(_CONFIRM_ONLY_RE.match((user_msg or "").strip()))


def _has_leading_confirmation(user_msg: str) -> bool:
    text = (user_msg or "").strip()
    if not text:
        return False
    return bool(_CONFIRM_PREFIX_RE.match(text))


def _parse_llm_proposal(text: str) -> dict:
    """LLM 발화에서 '축 + 강도' 제안 패턴을 뽑아 slot dict 반환.

    - 양자택일 질문(강하게/약하게 제시)이면 {} 반환 (모호).
    - 축 키워드 뒤 20자 이내에서 가장 가까운 강도 단어 1개만 채택.
    """
    if not text or _ALTERNATIVE_Q_RE.search(text):
        return {}

    result: dict[str, dict] = {}
    for kw, (axis, key) in _PROPOSAL_WORD_TO_KEY.items():
        for m in re.finditer(re.escape(kw), text):
            window = text[m.end(): m.end() + 20]
            best_pos = len(window) + 1
            best_intensity: Optional[str] = None
            for intensity, iwords in _INTENSITY_WORDS:
                for iw in iwords:
                    idx = window.find(iw)
                    if idx >= 0 and idx < best_pos:
                        best_pos = idx
                        best_intensity = intensity
            if best_intensity:
                result.setdefault(axis, {}).setdefault(key, best_intensity)
    return result


# ─── CORRECTION null guard ─────────────────────────────────────
# EXTRACT LLM 이 STOP/UNKNOWN 발화에도 기존 슬롯들에 무관한 null 을
# 뱉어내는 패턴을 차단. CORRECTION 시그널이 명시적으로 있을 때만 null 통과.

_CORRECTION_PATTERNS = [
    "안 했", "안했",
    "한 적 없", "한적 없", "한적없",
    "말한 적", "말 한 적",
    "말 안 했", "말안했",
    "내가 언제",
    "그런 말", "그런 건 아니", "그런건 아니",
    "그런 적",
    "얘기 안 했", "얘기안했", "얘기 안 한",
    "그게 아니",
]

_REPEAT_ACK_PATTERNS = [
    "아까 말했", "아까말했",
    "이미 말했", "이미말했",
    "방금 말했", "방금말했",
    "위에서 말했", "위에서말했",
    "앞에서 말했", "앞에서말했",
    "벌써 말했", "벌써말했",
    "다시 묻", "또 묻",
]


def _has_correction_signal(user_msg: str) -> bool:
    text = _normalize_text(user_msg)
    return any(_normalize_text(p) in text for p in _CORRECTION_PATTERNS)


def _has_repeat_ack_signal(user_msg: str) -> bool:
    text = _normalize_text(user_msg)
    return any(_normalize_text(p) in text for p in _REPEAT_ACK_PATTERNS)


def _guard_correction_null(user_msg: str, extracted: dict) -> dict:
    """CORRECTION 시그널 없는 발화에서 LLM 이 내뱉은 null 들을 제거.

    사용자가 "추천해줘/모르겠어" 같은 STOP/UNKNOWN 발화 시 EXTRACT LLM 이
    기존 확정 슬롯들을 null 로 뱉어내는 버그 방지.
    """
    if _has_correction_signal(user_msg):
        return extracted

    fixed = dict(extracted or {})
    for key in ("current_mood", "party_purpose", "strength_preference"):
        if key in fixed and fixed[key] is None:
            del fixed[key]
    for key in ("taste_profile", "aroma_profile"):
        if key in fixed and isinstance(fixed[key], dict):
            cleaned_sub = {k: v for k, v in fixed[key].items() if v is not None}
            if cleaned_sub:
                fixed[key] = cleaned_sub
            else:
                del fixed[key]
    return fixed


def _suppress_repeat_ack_slot_changes(
    history: list[dict],
    user_msg: str,
    extracted: dict,
) -> dict:
    """'아까 말했어' 류 메타 응답에서는 직전 질문 슬롯의 변경을 막는다."""
    if not _has_repeat_ack_signal(user_msg):
        return extracted

    fixed = dict(extracted or {})
    last_slot = _infer_last_asked_slot(history)
    if not last_slot:
        return fixed

    if last_slot in ("taste_profile", "aroma_profile"):
        fixed.pop(last_slot, None)
    else:
        fixed.pop(last_slot, None)
    return fixed


def _apply_confirmation_from_proposal(
    history: list[dict],
    user_msg: str,
    extracted: dict,
) -> dict:
    """사용자가 '응/ㅇㅇ/그래'만 했을 때, 직전 LLM 제안값을 slot 으로 복원."""
    if not (_is_confirm_only(user_msg) or _has_leading_confirmation(user_msg)):
        return extracted
    fixed = dict(extracted or {})

    last_llm = _last_llm_question(history)
    proposals = _parse_llm_proposal(last_llm)
    if not proposals:
        return fixed

    for axis, d in proposals.items():
        base = dict(fixed.get(axis) or {})
        for k, v in d.items():
            base.setdefault(k, v)
        fixed[axis] = base
    return fixed


# ─── 룰 기반 salvage & 환각 차단 ────────────────────────────────
# EXTRACT LLM 이 다축 발화를 한 축만 뽑거나, 엉뚱한 축을 환각으로 넣는 케이스를
# 룰로 보정한다. `_PROPOSAL_WORD_TO_KEY` 를 역매핑해 재활용.

# subkey -> [한국어 키워드]  (사용자 발화/LLM 제안 텍스트 매칭용)
# NOTE: 사용자는 "과일향" 말고 "과일" 처럼 줄여 말하거나, "딸기/레몬" 같은 구체 과일명을
# 던진다. _drop_hallucinated 가 키워드 매칭 실패로 정당한 추출을 떨어뜨리는 걸 막기 위해
# 관대하게 확장한다.
_AXIS_SUBKEY_TO_KEYWORDS: dict[tuple[str, str], list[str]] = {}
for _kw, (_axis, _sub) in _PROPOSAL_WORD_TO_KEY.items():
    _AXIS_SUBKEY_TO_KEYWORDS.setdefault((_axis, _sub), []).append(_kw)
# taste 구어형 (bare "단"은 false positive 위험으로 제외)
_AXIS_SUBKEY_TO_KEYWORDS.setdefault(("taste_profile", "creamy"), []).extend(
    ["우유", "유제품", "밀크", "부드러운", "크리미한"]
)
_AXIS_SUBKEY_TO_KEYWORDS.setdefault(("taste_profile", "sweet"), []).extend(
    ["달달한", "달콤한", "달게", "단 거", "단 칵테일", "단 맛", "단 술", "달달"]
)
_AXIS_SUBKEY_TO_KEYWORDS.setdefault(("taste_profile", "sour"), []).extend(
    ["새콤한", "시큼", "시게", "신 거", "신 칵테일", "신 맛", "신 술", "새콤", "톡 쏘는", "톡쏘는"]
)
_AXIS_SUBKEY_TO_KEYWORDS.setdefault(("taste_profile", "bitter"), []).extend(
    ["쌉쌀", "씁쓰", "쓴 거", "쓴 칵테일", "쓴 맛", "쌉싸름한", "쓰다"]
)
_AXIS_SUBKEY_TO_KEYWORDS.setdefault(("taste_profile", "freshness"), []).extend(
    ["청량한", "상큼한", "시원한", "깔끔한", "개운한", "톡 쏘는 느낌", "탄산"]
)
_AXIS_SUBKEY_TO_KEYWORDS.setdefault(("taste_profile", "body"), []).extend(
    ["묵직한", "진한", "깊은", "농밀", "무게감"]
)
# aroma: 사용자가 던지는 줄임말/구체 과일명/재료명 전부 수용
_AXIS_SUBKEY_TO_KEYWORDS.setdefault(("aroma_profile", "fruity"), []).extend(
    ["과일", "프루티한", "딸기", "복숭아", "망고", "파인애플", "사과", "배", "포도", "체리", "베리", "열대과일", "트로피컬"]
)
_AXIS_SUBKEY_TO_KEYWORDS.setdefault(("aroma_profile", "citrus"), []).extend(
    ["오렌지향", "시트러스한", "레몬", "라임", "자몽", "오렌지", "유자", "귤"]
)
_AXIS_SUBKEY_TO_KEYWORDS.setdefault(("aroma_profile", "minty"), []).extend(
    ["민트", "박하", "시원한 향", "청량한 향"]
)
_AXIS_SUBKEY_TO_KEYWORDS.setdefault(("aroma_profile", "herbal"), []).extend(
    ["허브", "허벌", "허브향이", "로즈마리", "바질", "식물"]
)
_AXIS_SUBKEY_TO_KEYWORDS.setdefault(("aroma_profile", "woody"), []).extend(
    ["나무", "우디한", "오크", "숲", "삼나무"]
)
_AXIS_SUBKEY_TO_KEYWORDS.setdefault(("aroma_profile", "floral"), []).extend(
    ["꽃", "플로럴한", "장미", "라벤더", "자스민", "엘더플라워"]
)
_AXIS_SUBKEY_TO_KEYWORDS.setdefault(("aroma_profile", "coffee"), []).extend(
    ["커피", "에스프레소", "모카"]
)

_CONSTRAINED_INTENSITY_CHOICES = ("zero", "low", "medium", "high", "null")
_TOKEN_TRIE_END = "__end__"
_CHOICE_TRIE_CACHE: dict[tuple[int, tuple[str, ...]], dict] = {}


# salvage 전용 — _INTENSITY_WORDS 보다 구어체 강도 표현 포함.
# NOTE: "좋아/좋네/좋음" 같은 generic 선호 표현은 **여기 포함시키지 마라**.
# 사용자가 "은은한게 좋아" 라고 하면 low 를 원하는 건데, "좋아" 가 high 로
# 매칭되면 nearest_intensity 가 low 대신 high 를 고른다 (실제 Turn 3 버그).
# 선호 축이 pending 인지 확인하는 신호는 `_salvage_affirmed_pending_axes` 에서
# 별도로 처리되므로, 여기서는 "강도 단어만" 엄격히 포함한다.
_USER_INTENSITY_WORDS: list[tuple[str, list[str]]] = [
    ("high",   ["강하게", "세게", "쎄게", "확", "진하게", "짱", "듬뿍",
                "엄청", "완전", "너무", "매우", "많이", "진짜",
                "확실", "분명", "또렷", "뚜렷", "강했으면", "강한 편",
                "세면", "강하면"]),
    ("low",    ["약하게", "살짝", "은은하게", "은은한", "은은해", "옅게", "연하게",
                "약간", "조금만", "은은하면", "약했으면", "약한 편",
                "강하지 않게", "강하지 않게요", "강하지 않았으면",
                "과하지 않게", "튀지 않게", "높지 않았으면"]),
    ("medium", ["적당히", "적당하게", "보통", "중간", "중간정도", "그냥",
                "적당하면", "중간 정도", "균형 잡힌", "어느 정도"]),
    ("zero",   ["빼고", "없이", "질색", "싫어", "싫음", "별로",
                "없었으면", "안 났으면", "안났으면", "안 들어갔으면"]),
]
_AXIS_UNKNOWN_PATTERNS = ["모르겠", "잘 모르", "잘모르", "애매", "글쎄", "딱히 모르"]


def _has_explicit_intensity_hint(text: str) -> bool:
    normalized = _normalize_text(text)
    for _intensity, words in _USER_INTENSITY_WORDS:
        if any(_normalize_text(word) in normalized for word in words):
            return True
    return False


def _has_affirmative_preference_signal(text: str) -> bool:
    return _contains_any(text, _AFFIRMATIVE_PREFERENCE_PATTERNS)


def _has_local_unknown_near_axis(text: str, kw_start: int, kw_end: int) -> bool:
    before = _normalize_text(text[max(0, kw_start - 6): kw_start])
    after = _normalize_text(text[kw_end: kw_end + 18])
    return any(_normalize_text(pattern) in before or _normalize_text(pattern) in after for pattern in _AXIS_UNKNOWN_PATTERNS)


def _axis_is_explicitly_unknown(text: str, keywords: list[str]) -> bool:
    for kw in keywords:
        for match in re.finditer(re.escape(kw), text):
            if _has_local_unknown_near_axis(text, match.start(), match.end()):
                return True
    return False


def _detect_requested_intensity(text: str) -> Optional[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return None
    best_match: tuple[int, str] | None = None
    for intensity, words in _USER_INTENSITY_WORDS:
        for word in words:
            normalized_word = _normalize_text(word)
            if normalized_word == "그냥":
                continue
            if normalized_word and normalized_word in normalized:
                score = len(normalized_word)
                if best_match is None or score > best_match[0]:
                    best_match = (score, intensity)
    return best_match[1] if best_match else None


def _nearest_intensity(text: str, kw_start: int, kw_end: int) -> Optional[str]:
    """축 키워드 기준 앞 10자 + 뒤 25자 양방향에서 가장 가까운 강도 단어.

    (한국어는 "단맛 쎈게"(뒤) 와 "엄청 단"(앞) 양방향 다 나오므로 둘 다 본다.
    직전 축의 강도 오염 위험이 있어 앞 윈도우는 더 짧게.
    동거리일 때는 앞쪽(수식어)이 뒤쪽(선호 시그널)보다 축의 intensity 를 더
    정확히 묘사하므로 앞쪽 우선. 그래서 before 를 먼저 스캔한다.)
    """
    best_dist = 10**9
    best: Optional[str] = None

    # 앞 윈도우는 짧게(3자) — "엄청 단", "너무 달게" 같은 즉각 수식어만 잡고
    # 이전 문장의 강도어("단맛 쎈게 좋지. 신맛…"에서 '좋지')가 끼어드는 건 차단.
    before_start = max(0, kw_start - 3)
    before = text[before_start: kw_start]
    for intensity, iwords in _USER_INTENSITY_WORDS:
        for iw in iwords:
            idx = before.rfind(iw)
            if idx < 0:
                continue
            dist = len(before) - (idx + len(iw))
            if dist < best_dist:
                best_dist = dist
                best = intensity

    after = text[kw_end: kw_end + 25]
    for intensity, iwords in _USER_INTENSITY_WORDS:
        for iw in iwords:
            idx = after.find(iw)
            if idx >= 0 and idx < best_dist:  # < 로 동거리시 before 승
                best_dist = idx
                best = intensity

    return best


def _salvage_affirmed_pending_axes(
    history: list[dict],
    current_slots: dict,
    user_msg: str,
    extracted: dict,
) -> dict:
    """강도 단어가 없더라도, pending 축에 대한 명시적 긍정 답변은 high 로 보정한다."""
    if not user_msg or _has_correction_signal(user_msg):
        return extracted
    if _has_explicit_intensity_hint(user_msg):
        return extracted
    if not _has_affirmative_preference_signal(user_msg):
        return extracted

    last_slot = _infer_last_asked_slot(history)
    fixed = dict(extracted or {})

    for axis in ("taste_profile", "aroma_profile"):
        current = dict(fixed.get(axis) or {}) if isinstance(fixed.get(axis), dict) else {}
        existing = dict((current_slots or {}).get(axis) or {})
        for (candidate_axis, sub), keywords in _AXIS_SUBKEY_TO_KEYWORDS.items():
            if candidate_axis != axis:
                continue
            if current.get(sub) in INTENSITY_VALUES:
                continue
            if existing.get(sub) != INTENSITY_PENDING and last_slot != axis:
                continue
            if not _contains_axis_keyword(user_msg, keywords):
                continue
            if _axis_is_explicitly_unknown(user_msg, keywords):
                continue
            current[sub] = "high"
        if current:
            fixed[axis] = current

    return fixed


def _anchor_last_asked_axis_reply(
    history: list[dict],
    current_slots: dict,
    user_msg: str,
    extracted: dict,
) -> dict:
    """축 키워드 없는 짧은 응답도 직전 질문 축에 앵커링한다.

    예:
    - "과일향은 어느 정도?" -> "강한게 좋아" => fruity=high
    - "신맛은 어느 정도?"   -> "좋아"         => sour=high
    """
    if not user_msg or _has_correction_signal(user_msg) or _ALTERNATIVE_Q_RE.search(user_msg):
        return extracted

    axis_key = _infer_last_asked_axis_key(history)
    outer_key = _AXIS_OUTER_BY_KEY.get(axis_key or "")
    if not axis_key or not outer_key:
        return extracted

    fixed = dict(extracted or {})
    current = dict(fixed.get(outer_key) or {}) if isinstance(fixed.get(outer_key), dict) else {}
    if current.get(axis_key) in INTENSITY_VALUES:
        return fixed

    existing = dict((current_slots or {}).get(outer_key) or {})
    if existing.get(axis_key) not in (None, INTENSITY_PENDING):
        return fixed

    keywords = _AXIS_SUBKEY_TO_KEYWORDS.get((outer_key, axis_key), [])
    if keywords and _axis_is_explicitly_unknown(user_msg, keywords):
        return fixed

    explicit_axis_found = False
    different_axis_found = False
    for (candidate_outer, sub), keywords in _AXIS_SUBKEY_TO_KEYWORDS.items():
        if not _contains_axis_keyword(user_msg, keywords):
            continue
        explicit_axis_found = True
        if candidate_outer != outer_key or sub != axis_key:
            different_axis_found = True
    if explicit_axis_found and different_axis_found:
        return fixed

    intensity = _detect_requested_intensity(user_msg)
    if intensity is None and (
        _is_confirm_only(user_msg)
        or _has_leading_confirmation(user_msg)
        or _has_affirmative_preference_signal(user_msg)
    ):
        intensity = "high"
    if intensity is None:
        return fixed

    current[axis_key] = intensity
    fixed[outer_key] = current
    return fixed


def _build_choice_trie(tokenizer, choices: tuple[str, ...]) -> dict:
    cache_key = (id(tokenizer), choices)
    cached = _CHOICE_TRIE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    trie: dict = {}
    for choice in choices:
        for variant in (choice, f" {choice}", f"\n{choice}"):
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if not token_ids:
                continue
            node = trie
            for tok in token_ids:
                node = node.setdefault(tok, {})
            node[_TOKEN_TRIE_END] = choice

    _CHOICE_TRIE_CACHE[cache_key] = trie
    return trie


def _allowed_trie_tokens(trie: dict, prefix: list[int], eos_token_id: Optional[int]) -> list[int]:
    node = trie
    for tok in prefix:
        node = node.get(tok)
        if node is None:
            return [eos_token_id] if eos_token_id is not None else []

    allowed = [tok for tok in node.keys() if tok != _TOKEN_TRIE_END]
    if _TOKEN_TRIE_END in node and eos_token_id is not None:
        allowed.append(eos_token_id)
    return allowed or ([eos_token_id] if eos_token_id is not None else [])


def _contains_axis_keyword(user_msg: str, keywords: list[str]) -> bool:
    normalized = _normalize_text(user_msg)
    return any(_normalize_text(kw) in normalized for kw in keywords)


def _build_intensity_choice_prompt(user_msg: str, axis: str, sub: str) -> tuple[str, str]:
    axis_label = _PENDING_AXIS_KO.get(sub, sub)
    profile_label = "맛" if axis == "taste_profile" else "향"
    system = (
        "너는 칵테일 취향 강도 분류기다. "
        "반드시 다음 다섯 값 중 하나만 출력해라: zero, low, medium, high, null.\n"
        "zero=빼고 싶음/싫음/없었으면 좋겠음, "
        "low=은은하게/약하게/강하지 않게, "
        "medium=적당히/중간/보통, "
        "high=강하게/확실하게/분명하게/또렷하게, "
        "null=이번 발화에 그 축 언급 없음 또는 모호함."
    )
    user = (
        f"사용자 발화에서 {profile_label} 축 '{axis_label}'의 강도만 고르세요.\n"
        f"발화: {user_msg}\n"
        "답:"
    )
    return system, user


def _load_dialogue_llm_resources():
    from app.utils.model_loader import load_dialogue_llm, render_chat

    tokenizer, model = load_dialogue_llm()
    return tokenizer, model, render_chat


def _load_slot_extractor_resources():
    from app.utils.model_loader import load_slot_extractor_llm, render_chat

    tokenizer, model = load_slot_extractor_llm()
    return tokenizer, model, render_chat


def _decode_constrained_choice(system_prompt: str, user_prompt: str, choices: tuple[str, ...]) -> Optional[str]:
    try:
        import torch

        tokenizer, model, render_chat = _load_slot_extractor_resources()
        rendered = render_chat(tokenizer, system_prompt, user_prompt)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        trie = _build_choice_trie(tokenizer, choices)
        eos_token_id = tokenizer.eos_token_id
        max_choice_tokens = max(
            len(tokenizer.encode(choice, add_special_tokens=False))
            for choice in choices
        ) + 2

        def prefix_allowed_tokens_fn(_batch_id, input_ids):
            prefix = input_ids[input_len:].tolist()
            return _allowed_trie_tokens(trie, prefix, eos_token_id)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_choice_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=eos_token_id,
                eos_token_id=eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )

        raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip().lower()
        del inputs, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return raw if raw in choices else None
    except Exception as e:
        logger.debug("constrained choice decode failed: %r", e, exc_info=True)
        return None


def _apply_constrained_intensity_rescue(user_msg: str, extracted: dict) -> dict:
    """명시적으로 언급된 taste/aroma 축이 비어 있을 때 small-choice decoding 으로 복구.

    완전한 JSON grammar 제약 대신, 실제 손실이 큰 강도 축만 `zero|low|medium|high|null`
    중 하나로 강제 선택하게 해서 schema drift 를 줄인다.
    """
    if not user_msg:
        return extracted
    if _ALTERNATIVE_Q_RE.search(user_msg) or _has_correction_signal(user_msg):
        return extracted

    fixed = dict(extracted or {})

    for axis in ("taste_profile", "aroma_profile"):
        current = dict(fixed.get(axis) or {}) if isinstance(fixed.get(axis), dict) else {}
        for (candidate_axis, sub), keywords in _AXIS_SUBKEY_TO_KEYWORDS.items():
            if candidate_axis != axis:
                continue
            if sub in current and current[sub] in INTENSITY_VALUES:
                continue
            if not _contains_axis_keyword(user_msg, keywords):
                continue
            if _axis_is_explicitly_unknown(user_msg, keywords):
                continue
            system_prompt, user_prompt = _build_intensity_choice_prompt(user_msg, axis, sub)
            choice = _decode_constrained_choice(
                system_prompt,
                user_prompt,
                _CONSTRAINED_INTENSITY_CHOICES,
            )
            if choice in INTENSITY_VALUES:
                current[sub] = choice
        if current:
            fixed[axis] = current

    return fixed


def _salvage_taste_aroma_from_text(user_msg: str, extracted: dict) -> dict:
    """사용자 발화에 명시된 taste/aroma 축 중 LLM 이 놓친 것을 룰로 복구.

    - 강도 단어가 근처에 없으면 복구하지 않음 (환각 방지).
    - 이미 추출된 축은 덮어쓰지 않음 (LLM 결과 우선).
    - correction 발화("좋아한다고 안 했는데")에서는 salvage 전체 skip.
    """
    if not user_msg:
        return extracted
    if _ALTERNATIVE_Q_RE.search(user_msg):
        return extracted
    if _has_correction_signal(user_msg):
        return extracted

    text = user_msg
    fixed = dict(extracted or {})
    taste = dict(fixed.get("taste_profile") or {}) if isinstance(fixed.get("taste_profile"), dict) else {}
    aroma = dict(fixed.get("aroma_profile") or {}) if isinstance(fixed.get("aroma_profile"), dict) else {}

    for (axis, sub), kws in _AXIS_SUBKEY_TO_KEYWORDS.items():
        target = taste if axis == "taste_profile" else aroma
        current_value = target.get(sub)
        for kw in kws:
            for m in re.finditer(re.escape(kw), text):
                if _has_local_unknown_near_axis(text, m.start(), m.end()):
                    continue
                intensity = _nearest_intensity(text, m.start(), m.end())
                if intensity:
                    if current_value in (None, INTENSITY_PENDING) or current_value != intensity:
                        target[sub] = intensity
                        current_value = intensity
                    break
            if sub in target and target[sub] not in (None, INTENSITY_PENDING):
                break

    if taste:
        fixed["taste_profile"] = taste
    if aroma:
        fixed["aroma_profile"] = aroma
    return fixed


def _drop_hallucinated_taste_aroma(
    history: list[dict],
    user_msg: str,
    extracted: dict,
) -> dict:
    """사용자 발화에 근거가 없는 taste/aroma 축 드롭.

    보존 조건 (하나라도 만족):
      1) 사용자 발화에 그 축 키워드가 있음 (ex. "단맛", "과일", "딸기")
      2) 사용자 발화가 **짧은 응답** (≤ 15자) 이고, 직전 LLM 질문에 해당 축
         키워드가 있음 (ex. LLM "단맛 어느 정도?" → 사용자 "좋아해" → sweet=high 보존)
      3) confirm-only 발화이고 직전 LLM 제안에 해당 축 있음
    """
    fixed = dict(extracted or {})
    last_llm = _last_llm_question(history) or ""
    text_user = user_msg or ""
    confirm_mode = _is_confirm_only(text_user) or _has_leading_confirmation(text_user)
    short_reply = len(text_user.strip()) <= 15   # "좋아해", "응 좋아", "적당히" 같은 짧은 답

    for axis in ("taste_profile", "aroma_profile"):
        sub_dict = fixed.get(axis)
        if not isinstance(sub_dict, dict):
            continue
        kept: dict = {}
        for sub, val in sub_dict.items():
            kws = _AXIS_SUBKEY_TO_KEYWORDS.get((axis, sub), [])
            if not kws:
                kept[sub] = val  # 매핑 모르는 축은 일단 보존
                continue
            if _contains_axis_keyword(text_user, kws):
                kept[sub] = val
                continue
            # 짧은 응답일 때: 직전 LLM 질문의 축을 그대로 받은 것으로 간주해 보존
            if short_reply and _contains_axis_keyword(last_llm, kws):
                kept[sub] = val
                continue
            if confirm_mode and _contains_axis_keyword(last_llm, kws):
                kept[sub] = val
                continue
            logger.debug(
                "drop hallucinated %s.%s=%r (no keyword in user msg; confirm=%s short=%s)",
                axis, sub, val, confirm_mode, short_reply,
            )
        if kept:
            fixed[axis] = kept
        else:
            fixed.pop(axis, None)
    return fixed


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


def _summarize_slots_for_prompt(slots: dict) -> str:
    parts: list[str] = []
    mood_map = {"good": "기분 좋음", "soso": "그냥저냥", "bad": "다운"}
    purpose_map = {
        "celebration": "모임/축하자리",
        "date": "데이트",
        "business": "비즈니스 자리",
        "solo": "혼술",
        "hangout": "친구들과 노는 자리",
    }
    strength_map = {"zero": "무알콜", "light": "가볍게", "medium": "적당히", "strong": "센 쪽"}

    if slots.get("party_purpose") in purpose_map:
        parts.append(f"자리: {purpose_map[slots['party_purpose']]}")
    if slots.get("current_mood") in mood_map:
        parts.append(f"기분: {mood_map[slots['current_mood']]}")
    if slots.get("strength_preference") in strength_map:
        parts.append(f"도수: {strength_map[slots['strength_preference']]}")
    taste = slots.get("taste_profile") or {}
    if taste:
        confirmed = [f"{k}={v}" for k, v in taste.items() if v != INTENSITY_PENDING]
        pending = [k for k, v in taste.items() if v == INTENSITY_PENDING]
        if confirmed:
            parts.append("맛(확정): " + ", ".join(confirmed))
        if pending:
            parts.append("맛(강도 미확정 — 대화로 확인 필요): " + ", ".join(pending))
    aroma = slots.get("aroma_profile") or {}
    if aroma:
        confirmed = [f"{k}={v}" for k, v in aroma.items() if v != INTENSITY_PENDING]
        pending = [k for k, v in aroma.items() if v == INTENSITY_PENDING]
        if confirmed:
            parts.append("향(확정): " + ", ".join(confirmed))
        if pending:
            parts.append("향(강도 미확정 — 대화로 확인 필요): " + ", ".join(pending))
    favs = slots.get("favorite_drinks") or []
    if favs:
        parts.append("즐겨드심: " + ", ".join(favs))
    bases = slots.get("disliked_bases") or []
    if bases:
        parts.append("비선호 베이스: " + ", ".join(bases))
    return " / ".join(parts) if parts else "(아직 파악된 선호 없음)"



# ============================================================
# JSON 파서 + 검증
# ============================================================

def _sanitize_json_text(s: str) -> str:
    # // 또는 # 주석 제거 (라인 단위)
    s = re.sub(r"//[^\n]*", "", s)
    s = re.sub(r"(?m)^\s*#[^\n]*", "", s)
    # } 또는 ] 바로 뒤에 붙은 stray : 제거  e.g. "{}:"
    s = re.sub(r"([}\]])\s*:(?=\s*[,\}\]])", r"\1", s)
    # trailing comma 제거
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    # '%' 로 시작하는 주석형 잡음 제거
    s = re.sub(r"%[^\n,}\]]*", "", s)
    return s


def _regex_field_fallback(text: str) -> dict:
    """완전 JSON 파싱이 실패해도 주요 필드만이라도 건진다."""
    out: dict[str, Any] = {}
    m = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if m:
        try:
            out["reply"] = json.loads(f'"{m.group(1)}"')
        except Exception:
            out["reply"] = m.group(1)
    m = re.search(r'"action"\s*:\s*"([A-Z_]+)"', text)
    if m:
        out["action"] = m.group(1)
    m = re.search(r'"user_intent"\s*:\s*"([A-Z_]+)"', text)
    if m:
        out["user_intent"] = m.group(1)
    m = re.search(r'"extracted_slots"\s*:\s*(\{.*?\})', text, re.DOTALL)
    if m:
        try:
            out["extracted_slots"] = json.loads(_sanitize_json_text(m.group(1)))
        except Exception:
            pass
    return out


def _extract_json_object(text: str) -> Optional[dict]:
    # 가장 바깥쪽 중괄호 추출 (greedy)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return _regex_field_fallback(text) or None
    raw = match.group(0)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    try:
        obj = json.loads(_sanitize_json_text(raw))
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    fb = _regex_field_fallback(raw)
    return fb or None


def _validate_intensity_dict(raw: Any, allowed_keys: set[str], aliases: dict[str, str] | None = None) -> dict:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Optional[str]] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        key = k.strip().lower()
        if aliases and key not in allowed_keys:
            key = aliases.get(key, key)
        if key not in allowed_keys:
            continue
        # CORRECTION: null/"null"/"none"/"" 은 삭제 시그널로 통과시킨다.
        # merge_slots 가 None 을 받으면 해당 태그를 pop.
        if v is None:
            out[key] = None
            continue
        if isinstance(v, str):
            vv = v.strip().lower()
            vv = _INTENSITY_ALIASES.get(vv, vv)
            if vv in ("", "null", "none"):
                out[key] = None
                continue
            if vv in INTENSITY_VALUES:
                out[key] = vv
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


def _normalize_single_enum_like(raw: Any) -> Any:
    if isinstance(raw, str):
        return raw.strip().lower()
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], str):
        return raw[0].strip().lower()
    return raw


def _normalize_current_mood_value(raw: Any) -> Any:
    normalized = _normalize_single_enum_like(raw)
    if normalized == "medium":
        return "soso"
    return normalized


def _normalize_profile_key(raw_key: Any, aliases: dict[str, str]) -> Optional[str]:
    if not isinstance(raw_key, str):
        return None
    key = raw_key.strip().lower()
    return aliases.get(key, key)


def _rescue_profile_misplacements(
    raw_taste: Any,
    raw_aroma: Any,
) -> tuple[Any, Any, Optional[str]]:
    taste = dict(raw_taste) if isinstance(raw_taste, dict) else raw_taste
    aroma = dict(raw_aroma) if isinstance(raw_aroma, dict) else raw_aroma
    rescued_strength: Optional[str] = None

    if isinstance(taste, dict):
        strength_candidate = _normalize_single_enum_like(taste.pop("strength_preference", None))
        if isinstance(strength_candidate, str) and strength_candidate in STRENGTH_VALUES:
            rescued_strength = strength_candidate
        for raw_key in list(taste.keys()):
            taste_key = _normalize_profile_key(raw_key, _TASTE_KEY_ALIASES)
            aroma_key = _normalize_profile_key(raw_key, _AROMA_KEY_ALIASES)
            if aroma_key not in AROMA_KEYS or taste_key in TASTE_KEYS:
                continue
            if not isinstance(aroma, dict):
                aroma = {}
            value = taste.pop(raw_key)
            aroma.setdefault(aroma_key, value)

    if isinstance(aroma, dict):
        strength_candidate = _normalize_single_enum_like(aroma.pop("strength_preference", None))
        if rescued_strength is None and isinstance(strength_candidate, str) and strength_candidate in STRENGTH_VALUES:
            rescued_strength = strength_candidate
        for raw_key in list(aroma.keys()):
            aroma_key = _normalize_profile_key(raw_key, _AROMA_KEY_ALIASES)
            taste_key = _normalize_profile_key(raw_key, _TASTE_KEY_ALIASES)
            if taste_key not in TASTE_KEYS or aroma_key in AROMA_KEYS:
                continue
            if not isinstance(taste, dict):
                taste = {}
            value = aroma.pop(raw_key)
            taste.setdefault(taste_key, value)

    return taste, aroma, rescued_strength


def validate_extracted_slots(raw: dict) -> dict:
    """LLM extracted_slots → enum/스키마 검증된 dict. 유효하지 않은 값은 제거하고 warning 로그.

    스키마 위반이 무시되고 사용자 턴이 공회전하는 문제를 드러내기 위해
    드롭된 값들을 logger.warning 으로 남긴다.
    """
    if not isinstance(raw, dict):
        if raw:
            logger.warning("validate_extracted_slots: non-dict input dropped: %r", raw)
        return {}
    cleaned: dict[str, Any] = {}
    dropped: list[str] = []

    rescued_taste, rescued_aroma, rescued_strength = _rescue_profile_misplacements(
        raw.get("taste_profile"),
        raw.get("aroma_profile"),
    )

    _MISSING = object()

    def _scalar(
        key: str,
        allowed: set[str],
        *,
        fallback: Any = _MISSING,
        normalizer=None,
    ) -> None:
        # 키가 raw 에 아예 없고, fallback 도 없으면 스킵.
        if key not in raw and fallback is _MISSING:
            return
        # 키가 raw 에 없고 fallback 이 None 이면 "정보 없음" 으로 간주 → 스킵.
        # (과거엔 cleaned[key] = None 을 넣어서 merge_slots 가 기존 슬롯을 삭제하던 버그)
        if key not in raw and fallback is None:
            return
        val = raw.get(key, _MISSING)
        if val is _MISSING:
            val = fallback
        # 사용자가 raw 에 명시적으로 null 을 넣은 경우만 null 확정 (CORRECTION 의도).
        if val is None:
            cleaned[key] = None
            return
        if normalizer is not None:
            val = normalizer(val)
        v = _validate_enum(val, allowed)
        if v:
            cleaned[key] = v
        else:
            dropped.append(f"{key}={val!r} (허용값: {sorted(allowed)})")

    _scalar("current_mood", CURRENT_MOOD_VALUES, normalizer=_normalize_current_mood_value)
    _scalar("party_purpose", PARTY_PURPOSE_VALUES)
    _scalar(
        "strength_preference",
        STRENGTH_VALUES,
        fallback=rescued_strength,
        normalizer=_normalize_single_enum_like,
    )

    for profile_key, allowed, aliases in (
        ("taste_profile", TASTE_KEYS, _TASTE_KEY_ALIASES),
        ("aroma_profile", AROMA_KEYS, _AROMA_KEY_ALIASES),
    ):
        raw_val = rescued_taste if profile_key == "taste_profile" else rescued_aroma
        if raw_val is None:
            raw_val = raw.get(profile_key)
        if raw_val is None and profile_key not in raw:
            continue
        if not isinstance(raw_val, dict):
            dropped.append(f"{profile_key}: non-dict {raw_val!r}")
            continue
        d = _validate_intensity_dict(raw_val, allowed, aliases)
        invalid = {k: v for k, v in raw_val.items() if k not in d and aliases.get(str(k).strip().lower(), str(k).strip().lower()) not in d}
        if invalid:
            dropped.append(f"{profile_key} invalid entries: {invalid}")
        if d:
            cleaned[profile_key] = d

    if "disliked_bases" in raw:
        rv = raw["disliked_bases"]
        if isinstance(rv, list) and len(rv) == 0:
            cleaned["disliked_bases"] = []
        else:
            lst = _validate_list_enum(rv, DISLIKED_BASE_VALUES)
            invalid = [x for x in (rv if isinstance(rv, list) else []) if not (isinstance(x, str) and x.strip().lower() in DISLIKED_BASE_VALUES)]
            # 우유/크림류 구제: LLM이 disliked_bases에 milk/cream 넣은 경우 → taste_profile.creamy=zero
            remap_creamy = any(
                isinstance(x, str) and x.strip().lower() in _DISLIKED_BASE_REMAP_TO_CREAMY_ZERO
                for x in invalid
            )
            if remap_creamy:
                tp = cleaned.get("taste_profile") or {}
                if "creamy" not in tp:  # 이미 명시값 있으면 덮어쓰지 않음
                    tp["creamy"] = "zero"
                    cleaned["taste_profile"] = tp
                    dropped.append("disliked_bases milk/cream → taste_profile.creamy=zero (remapped)")
            if invalid and not remap_creamy:
                dropped.append(f"disliked_bases invalid: {invalid}")
            elif invalid and remap_creamy:
                leftover = [x for x in invalid if isinstance(x, str) and x.strip().lower() not in _DISLIKED_BASE_REMAP_TO_CREAMY_ZERO]
                if leftover:
                    dropped.append(f"disliked_bases invalid: {leftover}")
            if lst:
                cleaned["disliked_bases"] = lst

    if "favorite_drinks" in raw:
        rv = raw["favorite_drinks"]
        if isinstance(rv, list) and len(rv) == 0:
            cleaned["favorite_drinks"] = []
        else:
            lst = _validate_free_list(rv)
            if lst:
                cleaned["favorite_drinks"] = lst

    if dropped:
        logger.warning("validate_extracted_slots dropped: %s", " | ".join(dropped))

    return cleaned


# ============================================================
# Pass 1 — 슬롯 추출 전용 LLM 호출
# ============================================================
#
# 기존 `_BARTENDER_SYSTEM_PROMPT` 는 한 번의 generate 로 (추출 + reply + action)
# 전부 뽑으려다 EXAONE 이 JSON 자체를 놓치는 failure 가 반복됨 (2턴 silent
# extraction). 역할을 쪼개서 추출은 짧고 결정적인 프롬프트로만 돌린다.

_EXTRACT_SYSTEM_PROMPT = """
너는 한국어 칵테일 취향 슬롯 추출기다. 사용자 발화에서 명시적으로 말한 정보만 뽑아 JSON 한 객체로만 출력한다.

[출력 스키마 — 이 한 객체만. 설명·공감·질문·마크다운·코드블록·이모지 금지]
{"extracted_slots": { ... }}

[슬롯 필드]
- current_mood: "good"|"soso"|"bad"|null
- party_purpose: "celebration"|"date"|"business"|"solo"|"hangout"|null
- taste_profile: {"sweet"|"sour"|"bitter"|"body"|"creamy"|"freshness": "zero"|"low"|"medium"|"high"}
- aroma_profile: {"woody"|"minty"|"fruity"|"citrus"|"floral"|"coffee"|"herbal": "zero"|"low"|"medium"|"high"}
- strength_preference: "zero"|"light"|"medium"|"strong"|null
- disliked_bases: ["whiskey"|"gin"|"rum"|"vodka"|"tequila"]
- favorite_drinks: [자유 문자열]

[원칙]
1. **이번 발화에 명시된 축만 추출.** 추측·유도 금지. 한 발화에 여러 축이면 모두 넣어라.
2. 질문·되물음·"모르겠다/딱히" 로만 답한 발화는 {"extracted_slots":{}}.
3. **축 혼용·nesting**
   - taste_profile 키: {sweet, sour, bitter, body, creamy, freshness} 만.
   - aroma_profile 키: {woody, minty, fruity, citrus, floral, coffee, herbal} 만.
   - 헷갈리기 쉬운 것:
     · "청량함/상큼/시원함" = freshness (taste, 절대 aroma 아님)
     · "우유/크리미/밀키/밀크" = creamy (taste, disliked_bases 아님)
     · "바디감/묵직" = body (taste, aroma 아님)
     · "라임/레몬/자몽" = citrus (aroma, 절대 taste 아님)
     · "파인애플/망고/프루티" = fruity (aroma)
   - 모든 강도는 반드시 taste_profile/aroma_profile 안에 nesting.
     top-level 에 "body":"medium" 같은 키 직접 넣으면 드롭. 반드시 {"taste_profile":{"body":"medium"}}.
4. 선호 극성 유지: "좋아/적당히/살짝" = 선호(high/medium/low 강도 차), "싫어/별로/빼줘/질색" = zero.
   low/medium/high 전부 "선호" 범주. 비선호는 오직 zero.
5. **enum 외 값 금지.** 강도는 {zero,low,medium,high} 만. "mint"/"medium-high" 같은 표현 금지.
   ⚠️ **"강하게" → "high"** (taste/aroma 값으로 "strong" 절대 금지 — "strong" 은 strength_preference 전용).
   ⚠️ **"확실하게/분명하게/또렷하게" → "high"**, **"은은하게/약하게/강하지 않게" → "low"**, **"없었으면/안 났으면" → "zero"**.
6. **CORRECTION (부정/부인/반문):** "X 한 적 없어 / 얘기 안 한 것 같은데 / 내가 언제 X 라고 했어?"
   → 해당 축만 null 로 넣어라.
   ⚠️ 무관한 축에 null 뱉지 마라. 사용자가 언급 안 한 축은 그냥 extracted 에서 빼라.
   ⚠️ STOP 발화("추천해줘/모르겠다/그냥 뽑아줘") 는 CORRECTION 이 아니다 → null 금지, {} 출력.
7. **CONFIRM (응/ㅇㅇ/네/맞아/그대로/너가 말한 대로):**
   - 단독 CONFIRM → {"extracted_slots":{}}.
   - 단, 직전 봇이 "X 강하게, Y 약하게로 할까요?" 처럼 구체 강도값을 제안했다면 그 값을 추출.
   - 봇이 양자택일("강하게 좋으세요 약하게 좋으세요?") 후 "응"이면 모호 → {}.
8. **이전 턴 값 재출력 금지.** 봇이 축 A 를 물어봐도 사용자가 A 키워드 안 썼으면 A 포함 금지.
   이번 발화에 A 키워드가 없으면 A 는 없는 것이다.
9. **"분위기/느낌/무드" ≠ current_mood.** "조용한 분위기 / 편안한 느낌 / 가벼운 무드" 같은 자리·공간 스타일은 current_mood 에 넣지 마라.
   current_mood 는 사용자 본인 기분 표현("기분 좋아/신나/우울/피곤") 만. 애매하면 넣지 말 것.
10. **★ 없는 축 추가 금지.** 사용자가 이번 발화에 **명시적으로 말한 축만** 넣어라.
    [최근 대화] 에 나온 축이라도 이번 발화에 그 한국어 키워드가 없으면 절대 추가 금지.
    초기 태그의 pending 축(예: "fruity":"pending")도 사용자가 이번 발화에서 그 축을 언급하지 않았으면 손대지 마라.
    "친구 생일파티" 라는 발화에 추출할 건 party_purpose 뿐 — aroma.fruity 같은 축은 **절대 추가 금지**.
11. **★ 한 발화에 taste 축이 여러 개면 모두 추출.** "단맛 X, 신맛 Y" 면 둘 다 필수. aroma 축도 마찬가지.
    taste 와 aroma 가 섞여 있어도 각각 올바른 profile 에 넣어라. 크리미·바디·단/신/쓴/청량은 taste, 우디/민트/과일/시트러스/꽃/커피/허브는 aroma.

[강도 매핑] — low/medium/high 는 전부 "선호" 강도. zero 는 배제(비선호).
- "확/완전/엄청/강하게/세게/짱/듬뿍" → high (강하게 선호)
- "적당히/중간/보통" → medium (적당히 선호)
- "살짝/약간/조금/은은하게/옅게/살며시" → low (약하게 선호 — 있으면 좋지만 필수 아님)
- "별로/덜/안 땡겨/싫어/완전 싫어/질색/알레르기/빼줘" → zero (비선호/배제)

[축 키워드 매핑]
- taste: 단맛→sweet, 신맛/새콤→sour, 쓴맛/씁쓸→bitter, 바디감/묵직→body, 크리미/부드러움/우유/밀크→creamy, 청량감/상큼/시원함→freshness
- aroma: 우디/나무/우디한 향→woody, 민트/민트감→minty, 과일/과일향/프루티/파인애플/망고→fruity, 시트러스/시트러스 향/레몬/자몽/라임→citrus, 꽃/플로럴/플로럴한 느낌→floral, 커피/커피향/커피 느낌→coffee, 허브/허브향/허벌한 느낌→herbal
- purpose: 혼자/혼술→solo, 회식/거래처→business, 생일/기념/축하/돌잔치→celebration, 데이트/썸/둘이→date, 친구/모임/놀러/가족/가족들→hangout
- mood: 기분 좋아/신나/설레→good, 기분 별로/우울/힘들/안 좋→bad
- strength: 무알콜/논알콜→zero, 약하게/가볍게→light, 보통/적당히→medium, 세게/강하게→strong

[예시]
USER: "상큼한 거 짱 좋아해. 과일향도 진짜 좋아해. 묵직한 맛은 중간정도면 좋겠어"
→ {"extracted_slots":{"taste_profile":{"freshness":"high","body":"medium"},"aroma_profile":{"fruity":"high"}}}
(상큼=freshness(taste), 과일향=fruity(aroma), 묵직=body(taste) — 축 절대 섞지 마라.)

USER: "바디감 중간정도로"
→ {"extracted_slots":{"taste_profile":{"body":"medium"}}}
(body 는 반드시 taste_profile 안에 nesting. top-level 에 "body":"medium" 절대 금지.)

USER: "단맛은 강하게, 크리미한 맛은 완전 싫어"
→ {"extracted_slots":{"taste_profile":{"sweet":"high","creamy":"zero"}}}
(비선호="싫어/질색/빼줘" → zero. 선호 강도(high/medium/low) 와 구분해라.)

USER: "위스키는 싫어"
→ {"extracted_slots":{"disliked_bases":["whiskey"]}}
(베이스 술(whiskey/gin/rum/vodka/tequila)이 싫으면 disliked_bases. taste_profile 에 넣지 마라.)

USER: "응" (직전 봇: "청량함은 강하게, 크리미는 약하게로 갈까요?")
→ {"extracted_slots":{"taste_profile":{"freshness":"high","creamy":"low"}}}
(봇이 구체적 값을 제안했고 사용자가 확인한 경우만 추출.)

USER: "응" (직전 봇: "청량함은 강하게가 좋으세요 약하게가 좋으세요?")
→ {"extracted_slots":{}}
(양자택일 질문은 모호하므로 추출 금지.)

USER: "단맛 좋아한다고 안 했는데?"
→ {"extracted_slots":{"taste_profile":{"sweet":null}}}
(부정/부인/정정은 null 로 지워라. 긍정값으로 넣지 마라.)

USER: "크리미한 맛 얘기 안 한 것 같은데"
→ {"extracted_slots":{"taste_profile":{"creamy":null}}}
("~ 얘기 안 한 것 같은데 / ~ 말 안 했어 / ~ 한 적 없어" 도 정정이다.)

USER: "과일향이랑 민트향은 강하게"
→ {"extracted_slots":{"aroma_profile":{"fruity":"high","minty":"high"}}}
(이번 발화에 없는 축은 절대 재출력하지 마라. 이전 턴 값 복제 금지.)

USER: "단맛 쎈게 좋지. 신맛은 중간정도가 좋아"
→ {"extracted_slots":{"taste_profile":{"sweet":"high","sour":"medium"}}}
(taste 두 축 모두 필수. 사용자가 향/도수를 언급 안 했으니 aroma_profile·strength_preference 는 절대 추가 금지.)

USER: "크리미한 질감 엄청 좋아해. 우디향 은은하게"
→ {"extracted_slots":{"taste_profile":{"creamy":"high"},"aroma_profile":{"woody":"low"}}}
(크리미=taste.creamy (절대 aroma 아님), 우디향=aroma.woody. "은은하게"=low.)

USER: "우디한 향이 분명했으면 좋겠어요. 민트감은 은은하면 돼요"
→ {"extracted_slots":{"aroma_profile":{"woody":"high","minty":"low"}}}
(우디한 향/민트감 모두 aroma. "분명했으면"=high, "은은하면"=low.)

USER: "커피 느낌은 약했으면 좋겠어요. 시트러스 향은 강하지 않게요"
→ {"extracted_slots":{"aroma_profile":{"coffee":"low","citrus":"low"}}}
(커피 느낌/시트러스 향 모두 aroma. "약했으면/강하지 않게"=low.)

USER: "오늘 친구 생일파티야"
→ {"extracted_slots":{"party_purpose":"celebration"}}
(축하 분위기라고 해서 aroma.fruity·taste.sweet 같은 축 **절대 추가 금지** — 사용자가 명시한 축만.)

JSON 한 객체만. 다른 어떤 텍스트도 출력하지 마라.
""".strip()


def _build_extract_user_prompt(history: list[dict], user_msg: str) -> str:
    """추출 전용 프롬프트 — 최근 맥락 + 방금 발화만."""
    return (
        f"[최근 대화]\n{_format_history(history, max_turns=3)}\n\n"
        f"[사용자의 방금 발화]\n{user_msg}\n\n"
        f"위 발화에서 명시적으로 나타난 슬롯만 추출. JSON 한 객체만 출력."
    )


def _extract_slots_llm(history: list[dict], user_msg: str) -> tuple[dict, str]:
    """Pass 1: 슬롯 추출 전용. 반환 = (extracted_slots_raw_dict, raw_text)."""
    try:
        import torch

        tokenizer, model, render_chat = _load_slot_extractor_resources()
        rendered = render_chat(
            tokenizer,
            _EXTRACT_SYSTEM_PROMPT,
            _build_extract_user_prompt(history, user_msg),
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        del inputs, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        parsed = _extract_json_object(raw) or {}
        es = parsed.get("extracted_slots")
        if not isinstance(es, dict):
            # LLM 이 flat 하게 출력한 경우 (예: {"taste_profile": {...}}) — 전체를 슬롯으로 간주
            es = parsed if isinstance(parsed, dict) else {}
            es = {k: v for k, v in es.items() if k != "extracted_slots"}
        return es, raw
    except Exception as e:
        logger.warning("LLM _extract_slots_llm failed: %r", e, exc_info=True)
        return {}, ""


# ============================================================
# Pass 2 — reply / action 생성 (추출은 Pass 1 결과를 주입받음)
# ============================================================

_BARTENDER_SYSTEM_PROMPT = """
[역할]
당신은 10년차 한국인 바텐더다. 바 카운터에서 손님과 자연스럽게 대화하며 오늘 마실 칵테일을 같이 고른다.
설문조사원처럼 꼬치꼬치 묻지 마라. 친구가 카운터에서 같이 골라주는 느낌으로 말해라.

[대화 규칙]
1. 손님 발화의 구체 단어("생일파티", "회식", "오늘 너무 힘들어" 등)에 **먼저 공감하고** 그 흐름에서 자연스럽게 맛/향/도수 중 하나를 넌지시 떠봐라.
2. 슬롯과 무관한 잡담(날씨/근황/음악)도 한두 문장 받아주고 부드럽게 이어라. 끊지 마라.
3. 손님이 "모르겠다"고 하면 무리해서 묻지 말고 일상 취향(음식/카페/향수)으로 우회하거나 바로 제안해라.
4. 감이 충분히 잡히면 action 을 RECOMMEND 로 바꿔라. 완벽하지 않아도 된다.
5. "좋아/응/그래" 같은 짧은 동의는 직전 질문 축에 대한 긍정 답변으로 이해하고 이어가라.
6. 한 턴에 질문 1개. "A / B / C 중에?" 선다형은 가끔만.

[출력 언어 — 엄격]
- 순수 한글 한국어만. 한자·중국어·일본어·이모지·영어 단어 섞지 마라.
- 축 이름은 내부에서 쓰는 용어일 뿐이다. 손님에게 메뉴처럼 나열하지 마라.

[절대 금지]
- ASK 단계에서 칵테일 이름 언급 (마티니/모히토 등 실재·가상 둘 다). 추천 고르기는 다음 단계가 한다.
- extracted_slots 값 채우기 — 항상 {} 로 출력. 추출은 외부 파이프라인이 한다.

[출력 — JSON 한 덩어리만]
{
  "extracted_slots": {},
  "user_intent": "SLOT" | "QUESTION" | "UNKNOWN" | "CORRECTION" | "STOP" | "OTHER",
  "action": "ASK" | "RECOMMEND",
  "reply": "손님한테 보일 2~3문장 친근한 존댓말 한국어"
}
""".strip()


_RELAXED_DIALOGUE_SYSTEM_PROMPT = """
너는 10년 차 한국인 바텐더다. 손님의 기분과 맥락을 읽고 자연스럽게 대화하며 취향을 파악한다.
설문조사원처럼 체크리스트를 채우지 말고, 손님이 방금 한 말을 먼저 짚은 뒤 맛/향/도수 중 하나만 부드럽게 이어서 물어라.

[핵심]
- 추출은 다른 파이프라인이 이미 처리 중이다. extracted_slots 는 {} 로 둬도 된다.
- 이미 말한 정보를 다시 확인 질문으로 반복하지 마라.
- "좋아/좋지/그래/맞아" 같은 짧은 동의는 직전 질문 축에 대한 긍정 답변으로 해석해도 된다.
- 3~5턴 안에 감이 잡히면 action 을 RECOMMEND 로 바꿔라.
- 모르면 억지로 캐묻지 말고 예시를 주거나 가볍게 제안해라.
- 칵테일 이름은 추천 단계 전까지 직접 말하지 마라.

[출력]
JSON 객체 하나만 출력:
{
  "extracted_slots": {},
  "reply": "손님에게 건넬 친근한 한두 문장",
  "user_intent": "SLOT" | "QUESTION" | "UNKNOWN" | "CORRECTION" | "STOP" | "OTHER",
  "action": "ASK" | "RECOMMEND"
}
""".strip()


def _locked_slots_description(slots: dict) -> str:
    """이미 확정된 슬롯 목록 — LLM 에게 '절대 재질문 금지' 용으로 제공."""
    items: list[str] = []
    if slots.get("party_purpose"):
        items.append(f"자리(party_purpose)={slots['party_purpose']}")
    if slots.get("strength_preference"):
        items.append(f"도수(strength_preference)={slots['strength_preference']}")
    if slots.get("current_mood"):
        items.append(f"기분(current_mood)={slots['current_mood']}")
    taste = slots.get("taste_profile") or {}
    confirmed_taste = [k for k, v in taste.items() if v in INTENSITY_VALUES]
    if confirmed_taste:
        items.append("맛 강도 확정: " + ", ".join(f"{k}={taste[k]}" for k in confirmed_taste))
    aroma = slots.get("aroma_profile") or {}
    confirmed_aroma = [k for k, v in aroma.items() if v in INTENSITY_VALUES]
    if confirmed_aroma:
        items.append("향 강도 확정: " + ", ".join(f"{k}={aroma[k]}" for k in confirmed_aroma))
    return "\n".join(f"- {x}" for x in items) if items else "(없음)"


def _pending_intensity_description(slots: dict) -> str:
    """초기 태그 pending 시드 축 — 강도만 물어봐야 함 (선호 여부는 이미 확정)."""
    items: list[str] = []
    taste = slots.get("taste_profile") or {}
    pending_taste = [k for k, v in taste.items() if v == INTENSITY_PENDING]
    if pending_taste:
        items.append("맛: " + ", ".join(pending_taste) + " (선호는 확정, 강도만 확/은은/적당 조정)")
    aroma = slots.get("aroma_profile") or {}
    pending_aroma = [k for k, v in aroma.items() if v == INTENSITY_PENDING]
    if pending_aroma:
        items.append("향: " + ", ".join(pending_aroma) + " (선호는 확정, 강도만 확/은은/적당 조정)")
    return "\n".join(f"- {x}" for x in items) if items else "(없음)"


_MISSING_SLOT_LABEL = {
    "party_purpose": "자리/상황",
    "current_mood": "기분",
    "strength_preference": "도수",
    "taste_profile": "맛 방향 + 강도",
    "aroma_profile": "향 방향 + 강도",
}


def _missing_slots_description(slots: dict) -> str:
    miss = _missing_slots(slots)
    if not miss:
        return "(없음 — 충분함)"
    return ", ".join(f"{_MISSING_SLOT_LABEL.get(k, k)}({k})" for k in miss)


def _format_extracted_this_turn(extracted: dict) -> str:
    """Pass 1 에서 뽑힌 이번 턴 슬롯을 bartender prompt 에 주입할 형태로 요약."""
    if not extracted:
        return "(이번 턴엔 새로 추출된 축 없음)"
    lines: list[str] = []
    for k in ("current_mood", "party_purpose", "strength_preference"):
        if k in extracted:
            lines.append(f"- {k} = {extracted[k]!r}")
    for k in ("taste_profile", "aroma_profile"):
        d = extracted.get(k)
        if isinstance(d, dict) and d:
            parts = ", ".join(f"{sk}={sv}" for sk, sv in d.items())
            lines.append(f"- {k}: {parts}")
    if extracted.get("disliked_bases"):
        lines.append(f"- disliked_bases = {extracted['disliked_bases']}")
    if extracted.get("favorite_drinks"):
        lines.append(f"- favorite_drinks = {extracted['favorite_drinks']}")
    return "\n".join(lines) if lines else "(이번 턴엔 새로 추출된 축 없음)"


_FAMILIARITY_HINT = {
    "처음": "처음 (novice) — 칵테일 용어(드라이/스피릿/베이스) 금지. 일상 취향(음식/향수/카페/여행) 으로 우회해서 물어라.",
    "가끔": "가끔 (occasional) — 과거 경험으로 풀어라 (모히토/마가리타 드셔보셨어요?). 용어는 진토닉·사워 정도까지 OK.",
    "자주": "자주 (regular) — 전문 용어 자유. 베이스/스타일/스피릿 직접 물어도 OK.",
}


def _build_bartender_user_prompt(
    history: list[dict],
    slots: dict,
    user_msg: str,
    familiarity: Optional[str],
    remaining_turns: int,
    extracted_this_turn: Optional[dict] = None,
) -> str:
    """Bartender Pass2 프롬프트 — 메타 섹션 최소화.

    과거 10 섹션 중 아래 4개만 유지:
      - 친숙도 (대화 톤 결정)
      - 지금까지 파악한 취향 (초기 태그 포함 전체 상태)
      - 이번 턴 새로 파악된 것 (LLM 이 reply 에 짚어줄 것)
      - 최근 대화 + 손님 발화
    [완성도]/[강도 미확정]/[부족한 슬롯] 섹션은 제거 — LLM 이 메타 숫자 보고
    슬롯 질문으로 직행하던 문제 차단.
    """
    this_turn = _format_extracted_this_turn(extracted_this_turn or {})
    familiarity_line = _FAMILIARITY_HINT.get(familiarity or "", f"{familiarity or '알 수 없음'}")
    # 이번 턴 추출 반영한 preview 로 summary 뽑기 — 한 턴 지연 방지.
    slots_preview = merge_slots(slots, extracted_this_turn or {})
    summary = _summarize_slots_for_prompt(slots_preview)
    return (
        f"[손님 친숙도] {familiarity_line}\n\n"
        f"[지금까지 파악한 취향 — 이미 확정된 정보, 다시 묻지 마라]\n{summary}\n\n"
        f"[이번 턴 새로 파악된 것]\n{this_turn}\n\n"
        f"[최근 대화]\n{_format_history(history)}\n\n"
        f"[손님의 방금 발화]\n{user_msg}\n\n"
        f"위 발화에 바텐더로서 자연스럽게 반응해라. "
        f"[지금까지 파악한 취향] 에 이미 있는 축은 **절대 다시 묻지 마라**. "
        f"공감 한 마디 + (필요시) 아직 모르는 축 중 하나만 짧게 질문. "
        f"extracted_slots 는 {{}} 로 비우고 reply/action/user_intent 만 채워라. JSON 한 덩어리만 출력."
    )


def _build_relaxed_dialogue_user_prompt(
    history: list[dict],
    slots: dict,
    user_msg: str,
    familiarity: Optional[str],
    remaining_turns: int,
    extracted_this_turn: Optional[dict] = None,
) -> str:
    last_q = _last_llm_question(history)
    familiarity_line = _FAMILIARITY_HINT.get(familiarity or "", f"{familiarity or '알 수 없음'}")
    pending = _pending_intensity_description(merge_slots(slots, extracted_this_turn or {}))
    this_turn = _format_extracted_this_turn(extracted_this_turn or {})
    return (
        f"[손님 친숙도] {familiarity_line}\n"
        f"[남은 대화 턴] {remaining_turns} (relaxed 모드 상한 {RELAXED_MAX_USER_TURNS})\n\n"
        f"[현재까지 파악한 취향]\n{_summarize_slots_for_prompt(slots)}\n\n"
        f"[이번 턴 파악된 취향]\n{this_turn}\n\n"
        f"[아직 강도를 더 보면 좋은 축]\n{pending}\n\n"
        f"[직전 질문]\n{last_q or '(없음)'}\n\n"
        f"[최근 대화]\n{_format_history(history)}\n\n"
        f"[손님의 방금 발화]\n{user_msg}\n\n"
        "자연스럽게 대답하고, 파악된 취향만 extracted_slots 에 업데이트해라. JSON 한 객체만 출력."
    )


def _run_slot_extraction_pipeline(
    history: list[dict],
    slots: dict,
    user_msg: str,
    trace: dict[str, Any],
) -> tuple[dict, dict, str]:
    """슬롯 추출 — LLM 신뢰 + 대화형 UX 를 위한 최소 보정만 유지.

    과한 rule-chain 은 제거했지만, 아래 두 가지는 대화 UX 에 직접 중요해서 유지한다.
      - pending 축에 대한 긍정 응답 salvage
      - 직전 질문 축에 대한 짧은 답변 앵커링

    유지:
      - validate_extracted_slots: enum 검증 (downstream 안전)
      - _salvage_affirmed_pending_axes: "좋아/괜찮아" 류 긍정 보정
      - _anchor_last_asked_axis_reply: "은은한 쪽/강한게 좋아/응" 축 앵커링
      - _suppress_repeat_ack_slot_changes: "아까 말했어" 류 ghost 방어
      - _drop_unchanged_slots: 중복 재처리 방지
    """
    extracted_raw, extract_raw_text = _extract_slots_llm(history, user_msg)
    trace["extract_raw"] = extract_raw_text
    trace["extract_json"] = _trace_clone(extracted_raw)

    extracted = validate_extracted_slots(extracted_raw)
    trace["validated"] = _trace_clone(extracted)

    extracted = _apply_rule_based_slot_guards(history, user_msg, extracted)
    trace["rule_guarded"] = _trace_clone(extracted)

    extracted = _suppress_unanchored_scalar_overrides(slots, user_msg, extracted)
    trace["scalar_guarded"] = _trace_clone(extracted)

    extracted = _salvage_affirmed_pending_axes(history, slots, user_msg, extracted)
    trace["affirmed_pending_salvaged"] = _trace_clone(extracted)

    extracted = _anchor_last_asked_axis_reply(history, slots, user_msg, extracted)
    trace["anchored_last_axis"] = _trace_clone(extracted)

    extracted = _suppress_repeat_ack_slot_changes(history, user_msg, extracted)
    trace["repeat_suppressed"] = _trace_clone(extracted)

    extracted = _drop_unchanged_slots(extracted, slots)
    trace["final"] = _trace_clone(extracted)
    return extracted, extracted_raw, extract_raw_text


def _analyze_user_turn_relaxed(
    history: list[dict],
    slots: dict,
    user_msg: str,
    max_new_tokens: int,
    generate_next_question: bool,
    familiarity: Optional[str],
    user_turn_count: int,
) -> dict:
    trace: dict[str, Any] = {"mode": "relaxed"}
    user_msg = _sanitize_user_text(user_msg)
    try:
        import torch

        extracted, extracted_raw, extract_raw_text = _run_slot_extraction_pipeline(
            history=history,
            slots=slots,
            user_msg=user_msg,
            trace=trace,
        )

        if not generate_next_question:
            _log_dialogue_trace(user_msg, trace)
            return {
                "extracted_slots": extracted,
                "extracted_raw": extracted_raw,
                "should_stop": False,
                "stop_reason": "",
                "next_question": "",
                "reply": "",
                "action": "ASK",
                "user_intent": "SLOT",
                "source": "llm_relaxed_extract_only",
                "raw": "",
                "extract_raw": extract_raw_text,
                "trace": trace,
            }

        tokenizer, model, render_chat = _load_dialogue_llm_resources()
        remaining = max(RELAXED_MAX_USER_TURNS - user_turn_count, 0)
        rendered = render_chat(
            tokenizer,
            _RELAXED_DIALOGUE_SYSTEM_PROMPT,
            _build_relaxed_dialogue_user_prompt(
                history=history,
                slots=slots,
                user_msg=user_msg,
                familiarity=familiarity,
                remaining_turns=remaining,
                extracted_this_turn=extracted,
            ),
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                repetition_penalty=1.03,
                pad_token_id=tokenizer.eos_token_id,
            )

        raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        del inputs, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        parsed = _extract_json_object(raw) or {}
        trace["relaxed_raw"] = raw
        trace["relaxed_parsed"] = _trace_clone(parsed)

        action = str(parsed.get("action") or "ASK").upper()
        if action not in ("ASK", "RECOMMEND"):
            action = "ASK"

        user_intent = str(parsed.get("user_intent") or "OTHER").upper()
        if user_intent not in ("SLOT", "QUESTION", "UNKNOWN", "CORRECTION", "STOP", "OTHER"):
            user_intent = "OTHER"

        reply = str(parsed.get("reply") or "").strip()
        reply = _strip_non_korean_tokens(reply)
        if not generate_next_question:
            reply = ""
        elif not reply:
            reply = "좋아요. 지금 느낌을 바탕으로 한 가지만 더 여쭤볼게요."

        should_stop = action == "RECOMMEND"
        stop_reason = "llm_recommend" if should_stop else ""
        if _explicit_user_stop(user_msg):
            should_stop = True
            stop_reason = "user_requested"
            action = "RECOMMEND"

        trace["final"] = {
            "extracted_slots": _trace_clone(extracted),
            "reply": reply,
            "action": action,
            "user_intent": user_intent,
            "should_stop": should_stop,
            "stop_reason": stop_reason,
        }
        _log_dialogue_trace(user_msg, trace)

        return {
            "extracted_slots": extracted,
            "extracted_raw": extracted_raw,
            "should_stop": should_stop,
            "stop_reason": stop_reason,
            "next_question": reply,
            "reply": reply,
            "action": action,
            "user_intent": user_intent,
            "source": "llm_relaxed",
            "raw": raw,
            "extract_raw": extract_raw_text,
            "trace": trace,
        }
    except Exception as e:
        logger.warning("LLM relaxed analyze_user_turn failed: %r", e, exc_info=True)
        return {
            "extracted_slots": {},
            "extracted_raw": {},
            "should_stop": False,
            "stop_reason": "",
            "next_question": "좋아요. 너무 딱딱하게 안 갈게요. 오늘은 어떤 느낌의 한 잔이 끌리세요?",
            "reply": "좋아요. 너무 딱딱하게 안 갈게요. 오늘은 어떤 느낌의 한 잔이 끌리세요?",
            "action": "ASK",
            "user_intent": "OTHER",
            "source": "llm_relaxed_fallback",
            "raw": "",
            "extract_raw": "",
            "trace": trace,
        }


def analyze_user_turn(
    history: list[dict],
    slots: dict,
    user_msg: str,
    max_new_tokens: int = 280,
    generate_next_question: bool = True,
    skip_slots: Optional[set[str]] = None,  # legacy, 무시됨
    alt_slots: Optional[set[str]] = None,  # legacy, 무시됨
    familiarity: Optional[str] = None,
    user_turn_count: int = 0,
) -> dict:
    user_msg = _sanitize_user_text(user_msg)
    trace: dict[str, Any] = {}
    try:
        import torch

        # ─── Pass 1 : 슬롯 추출 전용 LLM 호출 ───────────────────────
        extracted, extracted_raw, extract_raw_text = _run_slot_extraction_pipeline(
            history=history,
            slots=slots,
            user_msg=user_msg,
            trace=trace,
        )

        if not generate_next_question:
            _log_dialogue_trace(user_msg, trace)
            return {
                "extracted_slots": extracted,
                "extracted_raw": extracted_raw,
                "should_stop": False,
                "stop_reason": "",
                "next_question": "",
                "reply": "",
                "action": "ASK",
                "user_intent": "SLOT",
                "source": "llm_extract_only",
                "raw": "",
                "extract_raw": extract_raw_text,
                "trace": trace,
            }

        # ─── Pass 2 : reply/action 생성 (추출 결과 주입) ─────────────
        tokenizer, model, render_chat = _load_dialogue_llm_resources()

        remaining = max(MAX_USER_TURNS - user_turn_count, 0)
        rendered = render_chat(
            tokenizer,
            _BARTENDER_SYSTEM_PROMPT,
            _build_bartender_user_prompt(
                history, slots, user_msg, familiarity, remaining,
                extracted_this_turn=extracted,
            ),
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )

        raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        del inputs, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        parsed = _extract_json_object(raw) or {}
        trace["bartender_raw"] = raw
        trace["bartender_parsed"] = _trace_clone(parsed)

        # Pass 2 는 이제 추출 담당이 아니다 — bartender 가 뽑은 extracted_slots 는 무시.
        # 단, reply/action/user_intent 를 extracted_slots 안에 중첩 출력한 legacy 포맷은 un-nest.
        es = parsed.get("extracted_slots")
        if isinstance(es, dict):
            for top_key in ("reply", "action", "user_intent"):
                if top_key in es and top_key not in parsed:
                    parsed[top_key] = es.pop(top_key)

        action = str(parsed.get("action") or "ASK").upper()
        if action not in ("ASK", "RECOMMEND"):
            action = "ASK"

        user_intent = str(parsed.get("user_intent") or "").upper()
        if user_intent not in ("SLOT", "QUESTION", "UNKNOWN", "CORRECTION", "STOP", "OTHER"):
            user_intent = "OTHER"

        # 기본 모드는 추출은 LLM 에 맡기되, ASK 단계의 실제 질문은
        # "이미 파악한 취향 + 남은 필수 축 1개" 흐름으로 정돈한다.
        # 그렇지 않으면 LLM 이 스키마 밖 질문(초콜릿/커피 등)으로 새면서
        # 대화가 슬롯 수집과 추천 입력 사이에서 붕 뜨기 쉽다.
        reply = str(parsed.get("reply") or "").strip()
        reply = _strip_non_korean_tokens(reply)
        if not reply and generate_next_question:
            reply = "네, 어떤 느낌으로 드시고 싶으세요?"

        if action == "ASK" and generate_next_question:
            merged_after_extract = merge_slots(slots, extracted or {})
            pending_axes = _pending_intensity_keys(merged_after_extract)
            target_axis = pending_axes[0] if pending_axes else None
            guided_reply = _build_guided_reply(
                history=history,
                current_slots=slots,
                extracted=extracted,
                user_intent=user_intent,
                fallback_reply=reply,
            )

            should_override = False
            if user_intent in ("QUESTION", "UNKNOWN"):
                should_override = True
            elif target_axis:
                should_override = not _reply_is_specific_axis_followup(reply, target_axis)
            elif not reply:
                should_override = True

            if guided_reply and should_override:
                reply = guided_reply

        should_stop = action == "RECOMMEND"
        stop_reason = "llm_recommend" if should_stop else ""

        if _explicit_user_stop(user_msg):
            should_stop = True
            stop_reason = "user_requested"
            action = "RECOMMEND"  # LLM 이 ASK 로 돌려놔도 사용자 STOP 이면 강제 RECOMMEND

        trace["final_action"] = action
        trace["final_user_intent"] = user_intent
        trace["final_reply"] = reply
        trace["final_should_stop"] = should_stop
        trace["final_stop_reason"] = stop_reason
        _log_dialogue_trace(user_msg, trace)

        return {
            "extracted_slots": extracted,
            "extracted_raw": extracted_raw,
            "should_stop": should_stop,
            "stop_reason": stop_reason,
            "next_question": reply,
            "reply": reply,
            "action": action,
            "user_intent": user_intent,
            "source": "llm",
            "raw": raw,
            "extract_raw": extract_raw_text,
            "trace": trace,
        }

    except Exception as e:
        logger.warning("LLM analyze_user_turn failed: %r", e, exc_info=True)
        trace["error"] = repr(e)
        _log_dialogue_trace(user_msg, trace)
        fallback_reply = "좋아요. 흐름을 이어가게 한 가지만 더 여쭤볼게요."
        return {
            "extracted_slots": {},
            "should_stop": False,
            "stop_reason": "",
            "next_question": fallback_reply,
            "reply": fallback_reply,
            "action": "ASK",
            "user_intent": "OTHER",
            "source": "fallback",
            "raw": "",
            "trace": trace,
        }

def generate_opening_question() -> str:
    """대화 첫 질문 — 열린 인사말. (LLM 호출 비용 아끼려고 고정)

    슬롯 질문으로 시작하지 않고 가볍게 연다 — 바텐더 톤.
    """
    return (
        "안녕하세요, 오늘 한 잔 같이 골라봐요. "
        "어떤 자리에서 드시는지, 아니면 오늘 어떤 기분이신지 편하게 말씀해 주세요."
    )

# ============================================================
# 슬롯 병합
# ============================================================

def merge_slots(current: dict, extracted: dict) -> dict:
    merged = dict(current) if current else {}

    for key in ("current_mood", "party_purpose", "strength_preference"):
        if key in extracted:
            val = extracted[key]
            if val in (None, "", "null"):
                merged.pop(key, None)
            else:
                merged[key] = val

    for key in ("taste_profile", "aroma_profile"):
        if key in extracted and isinstance(extracted[key], dict):
            base = dict(merged.get(key) or {})
            for sub_k, sub_v in extracted[key].items():
                if sub_v is None:
                    base.pop(sub_k, None)
                else:
                    base[sub_k] = sub_v
            merged[key] = base

    for key in ("disliked_bases", "favorite_drinks"):
        if key in extracted and isinstance(extracted[key], list):
            if len(extracted[key]) == 0:
                merged[key] = []
                continue
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

def _seed_slots_from_initial_tags(tag_row) -> dict:
    if not tag_row:
        return {}

    seeded: dict[str, Any] = {}

    strength_map = {
        "무알콜": "zero",
        "논알콜": "zero",
        "알코올 없음": "zero",
        "제로": "zero",
        "약함": "light",
        "가볍게": "light",
        "중간": "medium",
        "적당히": "medium",
        "보통": "medium",
        "강함": "strong",
        "센 거": "strong",
        "강하게": "strong",
    }
    taste_map = {
        "단맛": "sweet",
        "신맛": "sour",
        "쓴맛": "bitter",
        "청량함": "freshness",
        "청량감": "freshness",
        "바디감": "body",
        "크리미함": "creamy",
    }
    aroma_map = {
        "과일향": "fruity",
        "허브향": "herbal",
        "민트향": "minty",
        "시트러스향": "citrus",
        "우디향": "woody",
        "커피향": "coffee",
        "꽃향": "floral",
    }

    if getattr(tag_row, "strength_tag", None) in strength_map:
        seeded["strength_preference"] = strength_map[tag_row.strength_tag]

    # 초기 태그는 손님이 관심 있다고 체크한 축. 기본 강도 "medium" 으로 시드한다.
    # 대화에서 high/low/zero 로 덮어쓰면 그 값 우선. pending 으로 두면 대화 내내
    # "이미 고른 건데 왜 또 강도 물어?" 하게 되어 UX 가 깨진다.
    taste_profile: dict[str, str] = {}
    for tag in getattr(tag_row, "taste_tags_json", None) or []:
        mapped = taste_map.get(tag)
        if mapped:
            taste_profile[mapped] = "medium"
    if taste_profile:
        seeded["taste_profile"] = taste_profile

    aroma_profile: dict[str, str] = {}
    for tag in getattr(tag_row, "aroma_tags_json", None) or []:
        mapped = aroma_map.get(tag)
        if mapped:
            aroma_profile[mapped] = "medium"
    if aroma_profile:
        seeded["aroma_profile"] = aroma_profile

    return seeded


_EFFECTIVE_VECTOR_DEFAULTS = {
    "sweetness_score": 3.0,
    "bitterness_score": 3.0,
    "sourness_score": 3.0,
    "freshness_score": 3.0,
    "body_score": 3.0,
    "herbal_score": 2.0,
    "citrus_score": 2.0,
    "alcohol_score": 3.0,
}
_TASTE_TO_VECTOR_FIELD = {
    "sweet": "sweetness_score",
    "bitter": "bitterness_score",
    "sour": "sourness_score",
    "freshness": "freshness_score",
    "body": "body_score",
}
_AROMA_TO_VECTOR_FIELD = {
    "herbal": "herbal_score",
    "citrus": "citrus_score",
}
_TASTE_INTENSITY_TO_VECTOR = {
    "zero": 1.0,
    "low": 2.2,
    "medium": 3.5,
    "high": 4.4,
    INTENSITY_PENDING: 3.7,
}
_AROMA_INTENSITY_TO_VECTOR = {
    "zero": 0.5,
    "low": 1.3,
    "medium": 2.6,
    "high": 3.6,
    INTENSITY_PENDING: 2.8,
}
_STRENGTH_TO_VECTOR = {
    "zero": 0.0,
    "light": 1.0,
    "medium": 2.5,
    "strong": 4.0,
}


def build_effective_vector(vector_source: Any, merged_slots: dict) -> SimpleNamespace:
    """DB vector 위에 초기 태그/확정 슬롯 prior 를 얹은 추천용 벡터."""
    raw: dict[str, float] = dict(_EFFECTIVE_VECTOR_DEFAULTS)
    if vector_source is not None:
        for field, default in _EFFECTIVE_VECTOR_DEFAULTS.items():
            value = vector_source.get(field) if isinstance(vector_source, dict) else getattr(vector_source, field, None)
            try:
                raw[field] = float(value)
            except (TypeError, ValueError):
                raw[field] = default

    effective = dict(raw)

    def _apply_prior(field: str, prior: Optional[float]) -> None:
        if prior is None:
            return
        default = _EFFECTIVE_VECTOR_DEFAULTS[field]
        # feedback 등으로 실제 vector 가 이미 바뀐 축은 raw 값을 우선한다.
        if abs(raw.get(field, default) - default) <= 0.05:
            effective[field] = float(prior)

    taste_profile = (merged_slots or {}).get("taste_profile") or {}
    for key, intensity in taste_profile.items():
        field = _TASTE_TO_VECTOR_FIELD.get(key)
        if field:
            _apply_prior(field, _TASTE_INTENSITY_TO_VECTOR.get(intensity))

    aroma_profile = (merged_slots or {}).get("aroma_profile") or {}
    for key, intensity in aroma_profile.items():
        field = _AROMA_TO_VECTOR_FIELD.get(key)
        if field:
            _apply_prior(field, _AROMA_INTENSITY_TO_VECTOR.get(intensity))

    strength_pref = (merged_slots or {}).get("strength_preference")
    _apply_prior("alcohol_score", _STRENGTH_TO_VECTOR.get(strength_pref))

    return SimpleNamespace(**effective)


def build_user_profile(db: Session, guest_session_id: str) -> dict:
    guest = get_guest_session(db, guest_session_id)
    tags = get_initial_tag_response(db, guest_session_id)
    slot = get_preference_slot(db, guest_session_id)
    vector = get_preference_vector(db, guest_session_id)
    space = get_latest_space_analysis_by_party(db, guest.party_session_id) if guest else None

    merged_slots = _empty_slot_dict()
    merged_slots = merge_slots(merged_slots, _seed_slots_from_initial_tags(tags))

    if slot:
        merged_slots = merge_slots(merged_slots, {
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
        "effective_vector": build_effective_vector(vector, merged_slots),
        "space": space,
        "merged_slots": merged_slots,
        "effective_completion": _calc_effective_completion(merged_slots),
    }
