"""
preference_agent.py
역할: 슬롯 추출, 대화 로직, 피드백 인텐트 분류, 벡터 업데이트 해석, 사용자 프로필 빌드
"""
from __future__ import annotations

from typing import Any, Optional
from sqlalchemy.orm import Session

from app.db.crud import (
    get_guest_session,
    get_initial_tag_response,
    get_preference_slot,
    get_preference_vector,
    get_latest_space_analysis_by_party,
)


# ============================================================
# 피드백 인텐트 분류 (rule-based MVP)
# ============================================================

def classify_intent(feedback_text: str) -> str:
    """피드백 텍스트를 ACCEPT / ADJUST / REJECT 중 하나로 분류."""
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


# ============================================================
# 선호 벡터 조정 (rule-based MVP)
# ============================================================

def update_vector(before_vec: dict[str, float], feedback_text: str) -> dict[str, float]:
    """피드백 키워드에 따라 선호 벡터를 소폭 조정."""
    updated = dict(before_vec)
    text = feedback_text.lower()

    def clamp(v: float) -> float:
        return max(0.0, min(5.0, v))

    adjustments: list[tuple[list[str], str, float]] = [
        (["더 달", "달게", "달았으면"],          "sweetness_score",  +0.5),
        (["덜 달", "덜달", "달지 않게"],          "sweetness_score",  -0.5),
        (["더 쓰", "쓰게", "썼으면"],             "bitterness_score", +0.5),
        (["덜 쓰", "덜쓰"],                       "bitterness_score", -0.5),
        (["더 시", "시게"],                       "sourness_score",   +0.5),
        (["덜 시", "덜시"],                       "sourness_score",   -0.5),
        (["더 강", "강하게"],                     "alcohol_score",    +0.5),
        (["약하게", "더 약", "가볍게"],            "alcohol_score",    -0.5),
        (["청량", "상큼"],                        "freshness_score",  +0.5),
    ]

    for keywords, field, delta in adjustments:
        if any(kw in text for kw in keywords):
            updated[field] = clamp(updated.get(field, 2.5) + delta)

    return updated


# ============================================================
# 대화 에이전트 — 슬롯 추출 + 다음 질문 생성 (rule-based MVP)
# ============================================================

_SLOT_QUESTIONS: list[tuple[str, str]] = [
    ("party_purpose",       "오늘 어떤 자리인가요? (생일, 데이트, 모임 등)"),
    ("strength_preference", "알코올은 어느 정도 선호하세요? (약하게, 중간, 강하게)"),
    ("current_mood",        "지금 기분이 어떠세요?"),
    ("preferred_tastes",    "어떤 맛을 좋아하세요? (단맛, 쓴맛, 신맛 등)"),
    ("preferred_aromas",    "어떤 향을 좋아하세요? (과일향, 민트향, 커피향 등)"),
    ("disliked_tastes",     "싫어하는 맛이 있나요?"),
    ("disliked_bases",      "못 마시는 술 베이스가 있나요? (위스키, 럼, 보드카 등)"),
    ("finish_preference",   "끝맛은 어떤 게 좋으세요? (깔끔하게, 여운 있게)"),
    ("favorite_drinks",     "평소에 즐겨 마시는 술이 있나요?"),
    ("disliked_aromas",     "싫어하는 향이 있나요?"),
]

_TASTE_KEYWORDS: list[tuple[str, str]] = [
    ("달", "단맛"), ("단맛", "단맛"),
    ("쓴", "쓴맛"), ("쓴맛", "쓴맛"),
    ("신", "신맛"), ("상큼", "신맛"),
    ("청량", "청량함"),
    ("바디", "바디감"),
    ("크리미", "크리미함"),
    ("스파이시", "스파이시"),
    ("고소", "고소함"),
]

_AROMA_KEYWORDS: list[tuple[str, str]] = [
    ("민트", "민트향"), ("과일", "과일향"), ("시트러스", "시트러스향"),
    ("허브", "허브향"), ("커피", "커피향"), ("우디", "우디향"), ("꽃", "꽃향"),
]

_DISLIKED_TASTE_KEYWORDS: list[tuple[str, str]] = [
    ("쓴 거 싫", "쓴맛"), ("단 거 싫", "단맛"), ("신 거 싫", "신맛"),
    ("쓴맛 싫", "쓴맛"), ("단맛 싫", "단맛"), ("신맛 싫", "신맛"),
]

_BASE_KEYWORDS: list[str] = ["위스키", "럼", "보드카", "진", "데킬라", "사케", "소주", "맥주", "와인"]

#대화 중단 의사 감지
STOP_KEYWORDS = [
    "그만", "됐어", "충분해", "이제 추천해줘", "추천해",
    "그냥 해줘", "바로 해줘", "넘어가자", "skip", "그정도면 돼",
]

def wants_to_skip(user_msg: str) -> bool:
    """사용자가 대화 중단 의사를 표현했는지 확인"""
    text = user_msg.lower()
    return any(kw in text for kw in STOP_KEYWORDS)

def _is_slot_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    if isinstance(value, dict) and len(value) == 0:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False

# 필수 슬롯 — 추천 품질 보장을 위해 최소한 채워져야 하는 필드 (FR-06)
REQUIRED_SLOTS: list[str] = ["strength_preference", "disliked_bases"]


def missing_required_slots(merged_slots: dict) -> list[str]:
    """필수 슬롯 중 비어있는 키 목록 반환. disliked_bases는 빈 리스트도 '응답함'으로 간주."""
    missing: list[str] = []
    for key in REQUIRED_SLOTS:
        value = merged_slots.get(key)
        # disliked_bases는 '없음'이라는 빈 리스트도 유효 응답으로 인정
        if key == "disliked_bases":
            if value is None:
                missing.append(key)
            continue
        if _is_slot_empty(value):
            missing.append(key)
    return missing


def _calc_effective_completion(merged_slots: dict) -> float:
    fields = [
        merged_slots.get("party_purpose"),
        merged_slots.get("current_mood"),
        merged_slots.get("preferred_tastes"),
        merged_slots.get("disliked_tastes"),
        merged_slots.get("preferred_aromas"),
        merged_slots.get("disliked_aromas"),
        merged_slots.get("strength_preference"),
        merged_slots.get("favorite_drinks"),
        merged_slots.get("disliked_bases"),
        merged_slots.get("finish_preference"),
    ]
    filled = sum(1 for v in fields if not _is_slot_empty(v))
    return round((filled / 10) * 100, 2)

def choose_next_question(slots: dict) -> str:
    # 필수 슬롯을 최우선으로 질문 (FR-06)
    missing_required = set(missing_required_slots(slots))
    if missing_required:
        for slot_key, q in _SLOT_QUESTIONS:
            if slot_key in missing_required:
                return q
    for slot_key, q in _SLOT_QUESTIONS:
        if _is_slot_empty(slots.get(slot_key)):
            return q
    return "네, 알겠습니다! 더 말씀해 주실 내용이 있나요?"

def should_move_to_recommendation(
    merged_slots: dict,
    user_turn_count: int,
    user_msg: str,
) -> tuple[bool, str]:
    completion = _calc_effective_completion(merged_slots)
    required_missing = missing_required_slots(merged_slots)

    if completion >= 80:
        return True, "slot_completion"
    if user_turn_count >= 7:
        # 턴 한도 도달 시엔 필수 누락이어도 진행 (무한 루프 방지)
        return True, "turn_limit"
    if wants_to_skip(user_msg):
        # 사용자가 중단 의사 표현해도 필수 슬롯 누락 상태면 한 번 더 묻는다
        if required_missing:
            return False, "need_required_slots"
        return True, "user_skip"

    return False, "continue"

def run_dialogue(
    history: list[dict],
    slots: dict,
    user_msg: str,
) -> dict:
    """
    대화 한 턴 처리.
    slots: 현재까지 채워진 슬롯 (초기 태그 포함 병합 상태)
    Returns: { extracted_slots, question, completion, should_proceed }
    """
    extracted: dict[str, Any] = {}
    text = user_msg.lower()

    # ── 파티 목적 추출 ──────────────────────────────────────────
    purpose_map = [
        ("생일", "생일 파티"), ("기념일", "기념일"), ("데이트", "데이트"),
        ("회식", "회식"), ("모임", "친목 모임"), ("파티", "파티"),
    ]
    for kw, val in purpose_map:
        if kw in text:
            extracted["party_purpose"] = val
            break

    # ── 도수 선호 추출 ──────────────────────────────────────────
    strength_map = [
        ("못 마셔", "약함"), ("약하게", "약함"), ("약한", "약함"), ("가볍게", "약함"),
        ("강하게", "강함"), ("강한", "강함"), ("잘 마셔", "강함"),
        ("중간", "중간"), ("보통", "중간"), ("적당히", "중간"),
    ]
    for kw, val in strength_map:
        if kw in text:
            extracted["strength_preference"] = val
            break

    # ── 선호 맛 추출 ────────────────────────────────────────────
    prev_tastes: list[str] = list(slots.get("preferred_tastes") or [])
    new_tastes = list(prev_tastes)
    for kw, tag in _TASTE_KEYWORDS:
        if kw in text and tag not in new_tastes:
            new_tastes.append(tag)
    if new_tastes != prev_tastes:
        extracted["preferred_tastes"] = new_tastes

    # ── 선호 향 추출 ────────────────────────────────────────────
    prev_aromas: list[str] = list(slots.get("preferred_aromas") or [])
    new_aromas = list(prev_aromas)
    for kw, tag in _AROMA_KEYWORDS:
        if kw in text and tag not in new_aromas:
            new_aromas.append(tag)
    if new_aromas != prev_aromas:
        extracted["preferred_aromas"] = new_aromas

    # ── 비선호 맛 추출 ──────────────────────────────────────────
    prev_disliked: list[str] = list(slots.get("disliked_tastes") or [])
    new_disliked = list(prev_disliked)
    for kw, tag in _DISLIKED_TASTE_KEYWORDS:
        if kw in text and tag not in new_disliked:
            new_disliked.append(tag)
    if new_disliked != prev_disliked:
        extracted["disliked_tastes"] = new_disliked

    # ── 비선호 베이스 추출 ──────────────────────────────────────
    neg_markers = ["싫어", "못 마셔", "안 마셔", "안마셔", "싫"]
    is_negative_context = any(m in text for m in neg_markers)
    if is_negative_context:
        prev_bases: list[str] = list(slots.get("disliked_bases") or [])
        new_bases = list(prev_bases)
        for base in _BASE_KEYWORDS:
            if base in text and base not in new_bases:
                new_bases.append(base)
        if new_bases != prev_bases:
            extracted["disliked_bases"] = new_bases

    # ── 기분 추출 ────────────────────────────────────────────────
    mood_map = [
        ("신나", "신남"), ("설레", "설렘"), ("피곤", "피곤함"),
        ("편안", "편안함"), ("슬프", "슬픔"), ("기뻐", "기쁨"), ("즐거", "즐거움"),
    ]
    for kw, val in mood_map:
        if kw in text:
            extracted["current_mood"] = val
            break

    # ── 끝맛 선호 ────────────────────────────────────────────────
    if "깔끔" in text:
        extracted["finish_preference"] = "깔끔하게"
    elif "여운" in text or "진하게" in text:
        extracted["finish_preference"] = "여운 있게"

    # ── 좋아하는 음료 추출 ───────────────────────────────────────
    favorite_drink_keywords = ["모히토", "하이볼", "마가리타", "진토닉", "위스키 사워"]
    for drink in favorite_drink_keywords:
        if drink in text:
            extracted["favorite_drinks"] = drink
            break

    # ── 완성도 재계산 ────────────────────────────────────────────
    merged = dict(slots)
    merged.update(extracted)
    completion = _calc_effective_completion(merged)

    # ── 다음 질문 결정 ───────────────────────────────────────────
    question = "네, 알겠습니다! 더 말씀해 주실 내용이 있나요?"
    for slot_key, q in _SLOT_QUESTIONS:
        if _is_slot_empty(merged.get(slot_key)):
            question = q
            break

    return {
        "extracted_slots": extracted,
        "question": question,
        "completion": completion,
        "should_proceed": completion >= 80,
    }


# ============================================================
# 사용자 프로필 빌드 (초기 태그 + 슬롯 병합)
# ============================================================

def _merge_unique_list(*values) -> list[str]:
    merged: list[str] = []
    for v in values:
        if not v:
            continue
        for item in v:
            if item not in merged:
                merged.append(item)
    return merged


def _normalize_strength_tag(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    mapping = {
        "약함": "약함", "약한": "약함", "술 못 마셔요": "약함", "가볍게": "약함",
        "중간": "중간", "적당히": "중간", "보통": "중간", "무난하게": "중간",
        "강함": "강함", "센 거": "강함", "센걸 원해요": "강함", "술 잘 마셔요": "강함",
    }
    return mapping.get(value, value)


def build_user_profile(db: Session, guest_session_id: str) -> dict:
    """초기 태그 + preference_slot + preference_vector + space_analysis 통합."""
    guest = get_guest_session(db, guest_session_id)
    tags  = get_initial_tag_response(db, guest_session_id)
    slot  = get_preference_slot(db, guest_session_id)
    vector = get_preference_vector(db, guest_session_id)
    space  = get_latest_space_analysis_by_party(db, guest.party_session_id) if guest else None

    merged_slots = {
        "party_purpose": slot.party_purpose if slot else None,
        "current_mood":  slot.current_mood  if slot else None,
        "preferred_tastes": _merge_unique_list(
            tags.taste_tags_json  if tags else [],
            slot.preferred_tastes_json if slot else [],
        ),
        "disliked_tastes": _merge_unique_list(
            slot.disliked_tastes_json if slot else [],
        ),
        "preferred_aromas": _merge_unique_list(
            tags.aroma_tags_json   if tags else [],
            slot.preferred_aromas_json if slot else [],
        ),
        "disliked_aromas": _merge_unique_list(
            slot.disliked_aromas_json if slot else [],
        ),
        "strength_preference": (
            slot.strength_preference
            if slot and slot.strength_preference
            else _normalize_strength_tag(tags.strength_tag if tags else None)
        ),
        "favorite_drinks": slot.favorite_drinks_text if slot else None,
        "disliked_bases":  _merge_unique_list(
            slot.disliked_bases_json if slot else [],
        ),
        "finish_preference": slot.finish_preference if slot else None,
    }

    return {
        "guest":              guest,
        "initial_tags":       tags,
        "slot":               slot,
        "vector":             vector,
        "space":              space,
        "merged_slots":       merged_slots,
        "effective_completion": _calc_effective_completion(merged_slots),
    }
