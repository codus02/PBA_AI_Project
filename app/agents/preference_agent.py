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
INTENSITY_VALUES = {"zero", "low", "medium", "high"}

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
    "바로 추천해",
    "바로 추천 해",
    "지금 바로 추천",
    "추천으로 넘어가",
    "추천 단계로",
    "이제 추천 단계로",
]
# 주의: "추천해줘" 단독은 포함하지 않는다 — "너가 잘 추천해줘"처럼 질문을 더 유도하는 발화에 걸리면 안 됨.
# 사용자가 명확히 "그만 묻고 바로 추천"의 의사를 보일 때만 종료한다.

# 빈 list/dict가 "명시적 답변"(예: disliked_bases=[] = "없음")으로 인정되는 슬롯
_EMPTY_OK_SLOTS = {"disliked_bases", "favorite_drinks"}

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
    return re.sub(r"\s+", "", (text or "").strip().lower())


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
            if not any(sub in _CONFIRMED_INTENSITIES for sub in v.values()):
                missing.append(key)
        elif isinstance(v, str) and not v.strip():
            missing.append(key)
    return missing

def _next_slot_to_ask(slots: dict) -> Optional[str]:
    miss = _missing_slots(slots)
    return miss[0] if miss else None

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

# 대화로 "확정"된 것으로 볼 강도 값.
# 초기 태그 시드는 INTENSITY_PENDING("pending") 으로 넣고, 대화에서 high/medium/low/zero
# 중 하나로 조정되면 "확정"으로 본다. (medium 도 손님이 "적당히"라고 명시한 경우라 확정)
_CONFIRMED_INTENSITIES = {"high", "medium", "low", "zero"}


MIN_RECOMMEND_COMPLETION = 80.0

_DISLIKE_ASK_KEYWORDS = ("싫어하", "빼고 싶", "피하고", "별로다", "별로인", "질색")
_DISLIKE_REPLY = "방향은 잡힌 것 같은데 마지막으로 하나만요 — 혹시 싫어하거나 빼고 싶은 맛이나 향 있으세요? 없으면 없다고 말씀해주셔도 돼요."


def _dislike_check_done(slots: dict, history: list[dict]) -> bool:
    """비선호 질문이 한 번이라도 이뤄졌는지 추정.

    신호 하나라도 있으면 True:
      - taste/aroma 에 low/zero 값 존재
      - disliked_bases 가 빈 리스트라도 key 존재(= 답변 받음) 또는 값 존재
      - history 의 LLM 발화에 비선호 묻는 키워드 포함
    """
    tp = slots.get("taste_profile") or {}
    ap = slots.get("aroma_profile") or {}
    if any(v in ("low", "zero") for v in tp.values()):
        return True
    if any(v in ("low", "zero") for v in ap.values()):
        return True
    if "disliked_bases" in slots:
        return True
    for msg in history or []:
        if msg.get("speaker_role") == "LLM":
            text = msg.get("utterance_text") or ""
            if any(k in text for k in _DISLIKE_ASK_KEYWORDS):
                return True
    return False


def _calc_effective_completion(slots: dict) -> float:
    core = [k for k in SLOT_KEYS if k not in _COMPLETION_OPTIONAL_SLOTS]
    filled = 0.0
    for k in core:
        v = slots.get(k)
        if v is None or (isinstance(v, (list, dict)) and not v):
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if k in ("taste_profile", "aroma_profile") and isinstance(v, dict):
            # pending(초기 태그 seed)만 있으면 "채워지지 않은" 것으로 본다.
            # 대화에서 high/medium/low/zero 로 확정되어야만 completion 에 가산.
            has_confirmed = any(sub in _CONFIRMED_INTENSITIES for sub in v.values())
            if has_confirmed:
                filled += 1.0
            continue
        filled += 1.0
    return round(filled / max(len(core), 1) * 100, 2)


def should_move_to_recommendation(
    merged_slots: dict,
    user_turn_count: int,
    user_msg: str,
    llm_should_stop: bool = False,
    llm_stop_reason: str = "",
) -> tuple[bool, str]:
    """LLM action 을 1순위로 신뢰하고, 유저 정지/턴 상한만 하드 가드."""
    if _explicit_user_stop(user_msg):
        return True, "user_requested"
    if llm_should_stop:
        return True, llm_stop_reason or "llm_recommend"
    if user_turn_count >= MAX_USER_TURNS:
        return True, "turn_limit"
    return False, "keep_collecting"

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
    try:
        from app.utils.model_loader import load_llm
        import torch

        tokenizer, model = load_llm()

        messages = [
            {"role": "system", "content": _FEEDBACK_SYSTEM_PROMPT},
            {"role": "user", "content": f"USER 피드백: {feedback_text}\n\nJSON으로 답해라."},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
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

_SYSTEM_PROMPT = """
너는 칵테일 추천 시스템의 한국어 정보 추출기다.
한국어 구어체, 슬랭, 줄임말, 비문도 자연스럽게 해석한다.
반드시 JSON 객체 하나만 출력한다. 설명, 마크다운, 코드블록 금지.

★최우선 규칙 1: 한 발화에 여러 슬롯 정보가 들어 있으면 반드시 전부 추출하라. 한 슬롯만 뽑고 멈추지 마라.
★최우선 규칙 2: 비선호 강도 구분.
  - "완전 싫어/절대 안 돼/질색이야/진짜 못 마셔/죽어도 싫어/알레르기" → ZERO (추천에서 완전 배제)
  - "싫어/별로/안 좋아/안 땡겨/거부감/느끼해/피하고 싶어" → LOW (감점)
  - "좋아/땡겨/끌려/마음에 들어/최애" → HIGH
★최우선 규칙 3: 부정어("너무 ~건 싫어", "~은 별로")는 반드시 그 대상을 LOW로 인식하라. 절대 HIGH로 넣지 마라.
★최우선 규칙 4: 하나의 문장 안에 긍정 대상과 부정 대상이 섞여 있어도(예: "A는 좋은데 B는 싫어") 각각 올바른 강도로 분리해서 모두 추출하라.
★최우선 규칙 5: USER가 이전 답변을 번복/수정하면("아 아까 말한 거 취소", "아니 말고") 새로 말한 값을 그대로 반영하라. 취소된 과거 슬롯은 뽑지 마라.
★최우선 규칙 6: 직전 LLM 질문의 맥락을 반드시 참조하라. "응/어/네/좋아" 같은 1~2자 답은 직전 질문의 대상에 HIGH/긍정으로 매핑. "아니/별로/싫어"는 LOW/부정.
★최우선 규칙 7: 간접 표현도 해석하라. "달달한 거만 아니면 돼"→sweet:low, "향은 별 상관없어"→aroma_profile 추출 금지 (모른다는 뜻, 추측 X).
★최우선 규칙 8: 정도 부사를 반영하라. "살짝/약간/조금 X"→sweet:low가 아니라 sweet:medium. "엄청/완전/확 X"→high. "적당히/어느 정도/과하지 않게/너무 과한 건 싫어/너무 X한 건 싫고 적당히"→medium(절대 low 아님). 예: "너무 단건 싫고 적당히"→sweet:medium. "약간 크리미한"→creamy:medium.
★최우선 규칙 9: taste_profile은 USER가 명시적으로 언급하거나 슬랭사전에 직접 매핑되는 차원만 넣어라. 언급 없는 차원을 "기본값/보통"으로 채우지 마라. 단, USER가 명시한 차원은 반드시 전부 넣어라 — 복합 문장에서 여러 맛이 나오면 하나씩 빠짐없이 뽑아라. 예: "상큼 달달한게 좋아. 너무 과하진 않게. 우유맛은 싫어" → {"sweet":"medium","sour":"medium","creamy":"low"} (세 축 모두 필수, "과하지 않게"는 medium으로 조정).
★최우선 규칙 10: 특정 칵테일/술 이름이 "자주 마셔봤/잘 마시는/평소에/보통" 같은 맥락에 나오면 반드시 favorite_drinks에 넣어라.
★최우선 규칙 12: 사용자가 새로운 선호 정보를 주지 않고 "오직 질문·되묻기·혼란 표현만" 한 경우 extracted_slots 는 반드시 빈 객체 {} 를 출력하라. 절대 추측으로 슬롯을 채우지 마라.
  예: "그게 뭐야?", "왜 커피 향을?", "무슨 뜻이야?", "어떤 거요?", "모르겠는데 더 설명해줘", "은은한게 뭐야" → extracted_slots={}
  이런 발화는 LLM 의 다음 턴에서 같은 슬롯을 다시 묻게 되므로 completion 이 올라가면 안 된다.
★최우선 규칙 13: **맛 축(taste)과 향 축(aroma)을 절대 혼동하지 마라.** 신맛/쓴맛/단맛/크리미/바디/상큼함은 **taste_profile**. 시트러스향/꽃향/과일향/민트향/우디향/허브향/커피향은 **aroma_profile**. 가장 흔한 오류:
  - "신맛 강하게" → taste_profile.sour=high (절대 aroma_profile.citrus 아님!)
  - "신 맛이 좋아" / "시큼한 거 좋아" → taste_profile.sour=high
  - "시트러스향 좋아" / "레몬향 좋아" → aroma_profile.citrus=high
  - "상큼한 맛" → taste_profile.freshness=high (향 아님)
  - "단맛 좋아" → taste_profile.sweet=high (절대 aroma_profile 아님)
  - "쓴맛" → taste_profile.bitter (절대 aroma 아님)
  - "우유맛/크림같은 느낌" → taste_profile.creamy (절대 aroma 아님)

★최우선 규칙 11: 사용자가 일상 음료/음식을 "좋아해/즐겨 마셔"라고만 말한 경우(예: "커피 좋아해", "녹차 자주 마셔", "오렌지주스 좋아") — 직전 LLM 질문이 "향"이나 "맛" 이 아니라면, 절대 aroma_profile/taste_profile 에 high 로 넣지 마라. 이럴 때는 favorite_drinks 에만 넣어라 (강도는 대화로 확인해야 하므로). 오직 "커피향 좋아", "우디한 향이 좋아"처럼 향/맛 축을 명시했거나, 직전 LLM 질문이 "향이 어떤 게 좋으세요?"인 경우에만 aroma_profile 에 넣어라. 구문 변형 — "X 같은 스타일은 자주 마셔봤고", "보통 X 쪽은 잘 마시는 편이고", "평소엔 X 좋아하는 편", "X 자주 마셔" 모두 해당된다.

출력 형식:
{
  "extracted_slots": {
    "current_mood": null 또는 "good"|"soso"|"bad",
    "party_purpose": null 또는 "celebration"|"date"|"business"|"solo"|"hangout",
    "taste_profile": { "sweet"|"sour"|"bitter"|"body"|"creamy"|"freshness": "zero"|"low"|"medium"|"high", ... },
    "aroma_profile": { "woody"|"minty"|"fruity"|"citrus"|"floral"|"coffee"|"herbal": "zero"|"low"|"medium"|"high", ... },
    "strength_preference": null 또는 "zero"|"light"|"medium"|"strong",
    "disliked_bases": ["whiskey"|"gin"|"rum"|"vodka"|"tequila", ...],
    "favorite_drinks": [문자열, ...]
  },
  "should_stop": true 또는 false,
  "stop_reason": "user_requested"|"enough_info"|""
}

규칙:
1. 최근 USER 발화에서 새롭게 드러난 정보만 extracted_slots에 넣어라.
2. 한 발화에 여러 슬롯이 동시에 드러나면 반드시 모두 추출해라.
3. USER가 말하지 않은 것은 추측하지 마라.
4. 비선호 베이스 질문에 "없어/없음/상관없어/가리는 거 없어/딱히 없어/다 잘 마셔"라고 답하면 disliked_bases=[]. 절대로 모든 베이스를 나열하지 마라.
4a. **disliked_bases 는 오직 스피릿 5종(whiskey/gin/rum/vodka/tequila)만 허용.** 우유/크림/커피/주스/시럽/레몬 같은 재료는 **절대** 여기 넣지 마라. 우유·크림류는 taste_profile.creamy 로 간다.
4b. **극성(polarity)을 절대 뒤집지 마라.** "X 좋아해 / 평소에 X 자주 먹어 / X 취향" → **선호(high 또는 favorite_drinks)**. 이걸 disliked_bases 나 "zero/low" 로 넣으면 치명적 버그다. "X 싫어 / X 질색 / X 빼줘" → 비선호.
   - "우유 좋아해" / "평소 우유 자주 마셔" → taste_profile.creamy="high" (disliked_bases 아님!)
   - "커피 좋아" → favorite_drinks=["커피"] (aroma.coffee 아님, 강도 불명확)
   - "커피향 좋아" → aroma_profile.coffee="high"
5. 좋아하는 술 질문에 "없어/딱히/생각 안 나"라고 답하면 favorite_drinks=[].
6. 맛/향에서 싫어함은 같은 key의 low intensity로 표현한다.
7. "이제 추천해줘", "그만", "바로 추천"처럼 말하면 should_stop=true, stop_reason="user_requested".
8. "아까 말했는데?", "이미 말했어"는 새 정보 없음 → extracted_slots={}.
9. 비선호 맛/향은 별도 슬롯이 아니라 taste_profile / aroma_profile 의 "zero"(완전 싫음) 또는 "low"(별로) 로 저장한다.
   - "너무 단건 싫어" → taste_profile: {"sweet":"low"}
   - "우유향은 완전 질색" → taste_profile: {"creamy":"zero"}
   - "우디한 향 별로" → aroma_profile: {"woody":"low"}
   LLM 이 비선호 맛/향을 물어봤는데 "없어/딱히/다 괜찮아"라고 답하면 **아무것도 추출하지 마라** (extracted_slots 에서 taste/aroma 빼라).

한국어 슬랭/구어체 사전 (중요):
[기분 = good]
- "째진다", "째져", "쩐다", "꿀잼", "짱이야", "좋아 죽겠어", "기분 째져", "텐션 올라", "신나", "신난다", "들떠", "기분 업"
[기분 = soso]
- "그냥 그래", "그냥그래", "그냥", "쏘쏘", "그저 그래", "보통", "보통이야", "보통정도", "평범", "별 생각 없어", "무난", "그럭저럭", "나쁘진 않아"
[기분 = bad]
- "기분 별로", "기분 꿀꿀", "꿀꿀해", "다운돼", "우울해", "꿀꿀", "지쳐", "피곤해 죽겠어", "쳐져", "기분 안 좋아"

[맛: freshness=high] "상큼", "상큼한", "청량", "청량감", "깔끔", "깨끗"
[맛: sweet=low] "단 거 싫어", "너무 단 건 싫어", "덜 달게", "단맛 별로"
[맛: sour=low] "신 거 싫어", "너무 신 건 싫어", "시큼한 건 별로", "안 시게"
[맛: bitter=low] "쓴 거 싫어", "쓴맛 별로", "안 쓰게"
[맛: creamy=low] "느끼", "느끼해", "우유향 싫어", "크리미 별로", "부드러운 건 별로"
[맛: creamy=high] "크리미", "부드럽고 고소", "우유같은"
[맛: body=high] "묵직", "무겁게", "바디감 있는"

[도수 = strong]
- "술고래", "먹고 죽을", "마시고 죽을", "궤짝으로", "쎄게", "세게", "독하게", "독한 거", "강한 거", "취하고 싶어", "확 가는 거", "한 방", "달리자", "달릴", "콱 달릴", "ㄱㄱ", "오늘 마시자"
[도수 = light]
- "술 약해", "잘 못 마셔", "약하게", "가볍게", "순한 거", "부드럽게", "살살", "쪼끔만"
[도수 = zero]
- "무알콜", "논알콜", "안 마셔", "술 안 들어간", "알콜 빼고"
[도수 = medium]
- "적당히", "보통으로", "중간", "평범하게"

[모임 목적]
- "생파", "생일", "축하" → celebration
- "데이트", "여친이랑", "남친이랑", "썸녀", "썸남" → date
- "회식", "회사", "거래처", "비즈니스", "직장 동료" → business
- "혼술", "혼자", "혼자 마시러", "나 혼자" → solo
- "친구들이랑", "놀러", "그냥 노는 자리", "캐주얼하게", "편하게 노는" → hangout

예시 (그대로 따라하라):

USER: "안녕? 오늘 친구 생파야"
→ {"extracted_slots":{"party_purpose":"celebration"},"should_stop":false,"stop_reason":""}

USER: "오늘 친구 생파야! 기분 째져"
→ {"extracted_slots":{"party_purpose":"celebration","current_mood":"good"},"should_stop":false,"stop_reason":""}

USER: "어 째져"  (직전에 기분 물었을 때)
→ {"extracted_slots":{"current_mood":"good"},"should_stop":false,"stop_reason":""}

USER: "꿀꿀해"
→ {"extracted_slots":{"current_mood":"bad"},"should_stop":false,"stop_reason":""}

USER: "그냥 그래"
→ {"extracted_slots":{"current_mood":"soso"},"should_stop":false,"stop_reason":""}

USER: "그냥 그래."  (직전 LLM 질문이 기분에 대한 것일 때)
→ {"extracted_slots":{"current_mood":"soso"},"should_stop":false,"stop_reason":""}

USER: "그냥 보통정도"
→ {"extracted_slots":{"current_mood":"soso"},"should_stop":false,"stop_reason":""}

USER: "오늘은 혼자 조용히 마시려고"
→ {"extracted_slots":{"party_purpose":"solo"},"should_stop":false,"stop_reason":""}

USER: "너무 약하지도 않고, 강하지도 않게"
→ {"extracted_slots":{"strength_preference":"medium"},"should_stop":false,"stop_reason":""}

USER: "너무 신건 별로야. 적당히."  (직전 LLM 질문이 신맛 강도일 때)
→ {"extracted_slots":{"taste_profile":{"sour":"medium"}},"should_stop":false,"stop_reason":""}
# "너무 ~건 별로" + "적당히" 조합은 medium 이다. 절대 low 로 넣지 마라. "적당히"가 없으면 low 가 맞다.

USER: "너무 단건 싫고 적당한게 좋아"
→ {"extracted_slots":{"taste_profile":{"sweet":"medium"}},"should_stop":false,"stop_reason":""}

USER: "시트러스향은 강한게 좋아. 단맛도 강하면 좋겠어. 커피향은 완전 싫어."
→ {"extracted_slots":{"taste_profile":{"sweet":"high"},"aroma_profile":{"citrus":"high","coffee":"zero"}},"should_stop":false,"stop_reason":""}
# 한 발화에 세 축 전부 추출. "강하면/강한게" → high, "완전 싫어" → zero. 하나라도 빠뜨리지 마라.

USER: "민트향 강한게 좋고, 시트러스는 중간 정도. 우디향은 진짜 별로야"
→ {"extracted_slots":{"aroma_profile":{"minty":"high","citrus":"medium","woody":"low"}},"should_stop":false,"stop_reason":""}

USER: "그게 뭐야?"  (직전 LLM 이 맛/향 질문했을 때)
→ {"extracted_slots":{},"should_stop":false,"stop_reason":""}
# 질문만 한 거니까 아무것도 추출하지 마라. 추측 금지.

USER: "기분 잘 모르겠는데?" / "잘 모르겠어" / "몰라"  (직전 LLM 이 기분 질문했을 때)
→ {"extracted_slots":{},"should_stop":false,"stop_reason":""}
# "모르겠다"는 soso 가 아니라 "답변 유보"다. current_mood 에 절대 값 넣지 마라. 다음 턴에 다시 묻게 된다.

USER: "기분에 따라서 추천이 달라져?" / "그거 왜 물어봐?" / "꼭 필요한 정보야?"
→ {"extracted_slots":{},"should_stop":false,"stop_reason":""}
# 시스템에 대한 메타 질문일 뿐, 어떤 취향 정보도 주지 않았다.

USER: "은은한게 뭔데?"
→ {"extracted_slots":{},"should_stop":false,"stop_reason":""}

USER: "무슨 뜻이야?" / "어떤 거요?" / "잘 모르겠는데 더 설명해줘"
→ {"extracted_slots":{},"should_stop":false,"stop_reason":""}

USER: "술 궤짝으로 마시고 죽을 거야"
→ {"extracted_slots":{"strength_preference":"strong"},"should_stop":false,"stop_reason":""}

USER: "오늘 마시고 죽을 예정이야"
→ {"extracted_slots":{"strength_preference":"strong"},"should_stop":false,"stop_reason":""}

USER: "쪼끔만 가볍게"
→ {"extracted_slots":{"strength_preference":"light"},"should_stop":false,"stop_reason":""}

USER: "단맛 별로 안 좋아"
→ {"extracted_slots":{"taste_profile":{"sweet":"low"}},"should_stop":false,"stop_reason":""}

USER: "달달한 거 좋아"
→ {"extracted_slots":{"taste_profile":{"sweet":"high"}},"should_stop":false,"stop_reason":""}

USER: "쓴맛"  (직전에 맛 물었을 때)
→ {"extracted_slots":{"taste_profile":{"bitter":"high"}},"should_stop":false,"stop_reason":""}

USER: "상큼한 게 좋긴 한데 너무 신 건 싫어. 단맛도 적당히"
→ {"extracted_slots":{"taste_profile":{"freshness":"high","sour":"low","sweet":"medium"}},"should_stop":false,"stop_reason":""}

USER: "상큼한 맛 좋아해. 근데 너무 시거나 단건 싫어. 그리고 쓴맛도 싫어"
→ {"extracted_slots":{"taste_profile":{"freshness":"high","sour":"low","sweet":"low","bitter":"low"}},"should_stop":false,"stop_reason":""}

USER: "나는 열대과일같은 향 좋아해. 우유향은 싫어. 느끼해"
→ {"extracted_slots":{"aroma_profile":{"fruity":"high"},"taste_profile":{"creamy":"low"}},"should_stop":false,"stop_reason":""}

USER: "과일향 적당하게 해줘. 우유향은 피하고싶어"
→ {"extracted_slots":{"aroma_profile":{"fruity":"medium"},"taste_profile":{"creamy":"zero"}},"should_stop":false,"stop_reason":""}
# 두 축 동시 추출 필수. "피하고싶어"는 zero. 우유향은 taste_profile.creamy (aroma 아님).

USER: "과일향 중간이면 좋겠고 우유는 안 땡겨"
→ {"extracted_slots":{"aroma_profile":{"fruity":"medium"},"taste_profile":{"creamy":"low"}},"should_stop":false,"stop_reason":""}

USER: "신맛은 강하게 부탁해"
→ {"extracted_slots":{"taste_profile":{"sour":"high"}},"should_stop":false,"stop_reason":""}
# 신맛 = taste_profile.sour. 절대 aroma_profile.citrus 아님. 시트러스는 "향" 이어야만 citrus.

USER: "레몬향 강한게 좋아"
→ {"extracted_slots":{"aroma_profile":{"citrus":"high"}},"should_stop":false,"stop_reason":""}
# "향"이 붙으면 aroma. 신맛 자체가 아니라 시트러스 "향"이니까 citrus 로 간다.

USER: "우유향 싫어 느끼해"
→ {"extracted_slots":{"taste_profile":{"creamy":"low"}},"should_stop":false,"stop_reason":""}

USER: "쓰고 묵직한 거"
→ {"extracted_slots":{"taste_profile":{"bitter":"high","body":"high"}},"should_stop":false,"stop_reason":""}

USER: "커피향"  (직전에 향 물었을 때)
→ {"extracted_slots":{"aroma_profile":{"coffee":"high"}},"should_stop":false,"stop_reason":""}

USER: "풀냄새 숲냄새 좋아"
→ {"extracted_slots":{"aroma_profile":{"herbal":"high","woody":"high"}},"should_stop":false,"stop_reason":""}

USER: "민트향 싫어"
→ {"extracted_slots":{"aroma_profile":{"minty":"low"}},"should_stop":false,"stop_reason":""}

USER: "꽃향이랑 과일향이 좋아"
→ {"extracted_slots":{"aroma_profile":{"floral":"high","fruity":"high"}},"should_stop":false,"stop_reason":""}

USER: "보드카는 싫네"
→ {"extracted_slots":{"disliked_bases":["vodka"]},"should_stop":false,"stop_reason":""}

USER: "위스키랑 데킬라 빼고"
→ {"extracted_slots":{"disliked_bases":["whiskey","tequila"]},"should_stop":false,"stop_reason":""}

USER: "딱히 없어"  (직전에 비선호 베이스 물었을 때)
→ {"extracted_slots":{"disliked_bases":[]},"should_stop":false,"stop_reason":""}

USER: "가리는 거 없어"
→ {"extracted_slots":{"disliked_bases":[]},"should_stop":false,"stop_reason":""}

USER: "생각 안 나"  (직전에 좋아하는 술 물었을 때)
→ {"extracted_slots":{"favorite_drinks":[]},"should_stop":false,"stop_reason":""}

USER: "평소엔 모히토랑 진토닉 자주 마셔"
→ {"extracted_slots":{"favorite_drinks":["모히토","진토닉"]},"should_stop":false,"stop_reason":""}

USER: "커피 좋아해"  (직전 LLM 질문이 향이 아니라 "평소 즐겨 마시는 것" 류일 때)
→ {"extracted_slots":{"favorite_drinks":["커피"]},"should_stop":false,"stop_reason":""}
# 주의: "커피 좋아해"만 가지고 aroma_profile.coffee=high 로 넣지 마라. 강도는 대화로 확인해야 한다.

USER: "녹차 자주 마셔"
→ {"extracted_slots":{"favorite_drinks":["녹차"]},"should_stop":false,"stop_reason":""}

USER: "커피향 좋아"  (또는 직전 LLM 질문이 "향이 어떤 게 좋으세요?")
→ {"extracted_slots":{"aroma_profile":{"coffee":"high"}},"should_stop":false,"stop_reason":""}

USER: "헤밍웨이 스페셜 같은 스타일은 자주 마셔봤고"
→ {"extracted_slots":{"favorite_drinks":["헤밍웨이 스페셜"]},"should_stop":false,"stop_reason":""}

USER: "보통 위스키 사워 쪽은 잘 마시는 편이고"
→ {"extracted_slots":{"favorite_drinks":["위스키 사워"]},"should_stop":false,"stop_reason":""}

USER: "파라다이스 같은 스타일은 자주 마셔봤고 럼 베이스는 이번엔 빼고 싶고"
→ {"extracted_slots":{"favorite_drinks":["파라다이스"],"disliked_bases":["rum"]},"should_stop":false,"stop_reason":""}

USER: "부담 없이 마실 수 있음 좋고 입에 설탕 남는 느낌은 싫고"
→ {"extracted_slots":{"taste_profile":{"sweet":"low"}},"should_stop":false,"stop_reason":""}
# 주의: body/sour/bitter/creamy/freshness는 언급 없으므로 절대 넣지 마라.

USER: "단맛이 확 치고 오는 건 별로고 씁쓸함이 오래 남는 건 별로고"
→ {"extracted_slots":{"taste_profile":{"sweet":"low","bitter":"low"}},"should_stop":false,"stop_reason":""}
# 주의: 언급하지 않은 sour/body/creamy/freshness 넣지 마라.

USER: "상큼 달달한게 좋아. 너무 과하진 않게. 우유맛은 싫어"
→ {"extracted_slots":{"taste_profile":{"sweet":"medium","sour":"medium","creamy":"low"}},"should_stop":false,"stop_reason":""}
# "과하지 않게" → medium. 세 축(sweet, sour, creamy) 모두 언급됐으니 전부 넣는다.

USER: "크리미! 내가 평소에 우유 좋아해"
→ {"extracted_slots":{"taste_profile":{"creamy":"high"}},"should_stop":false,"stop_reason":""}
# 주의: "우유 좋아해"는 선호다. disliked_bases 절대 아님. milk 는 스피릿 베이스도 아님.

USER: "우유 들어간 거 좋아"
→ {"extracted_slots":{"taste_profile":{"creamy":"high"}},"should_stop":false,"stop_reason":""}

USER: "크림같은 느낌 별로야"
→ {"extracted_slots":{"taste_profile":{"creamy":"low"}},"should_stop":false,"stop_reason":""}

USER: "달콤 쌉싸름한게 좋아. 근데 너무 과하진 않게"
→ {"extracted_slots":{"taste_profile":{"sweet":"medium","bitter":"medium"}},"should_stop":false,"stop_reason":""}

USER: "오늘 달리자"
→ {"extracted_slots":{"strength_preference":"strong"},"should_stop":false,"stop_reason":""}

USER: "콱 달릴거야"
→ {"extracted_slots":{"strength_preference":"strong"},"should_stop":false,"stop_reason":""}

USER: "회사 사람들이랑 와서 그냥 그래. 너무 독하지 않은 거"
→ {"extracted_slots":{"party_purpose":"business","current_mood":"soso","strength_preference":"light"},"should_stop":false,"stop_reason":""}

USER: "혼자 마시러 왔어. 단맛 좋아하고 위스키는 싫어"
→ {"extracted_slots":{"party_purpose":"solo","taste_profile":{"sweet":"high"},"disliked_bases":["whiskey"]},"should_stop":false,"stop_reason":""}

USER: "아까 말했는데?"
→ {"extracted_slots":{},"should_stop":false,"stop_reason":""}

USER: "아 아까 단맛 좋다고 했는데 취소. 그냥 쓴 거 줘"
→ {"extracted_slots":{"taste_profile":{"bitter":"high"}},"should_stop":false,"stop_reason":""}

USER: "응"  (직전 LLM: "달달한 거 좋으세요?")
→ {"extracted_slots":{"taste_profile":{"sweet":"high"}},"should_stop":false,"stop_reason":""}

USER: "아니"  (직전 LLM: "보드카 괜찮으세요?")
→ {"extracted_slots":{"disliked_bases":["vodka"]},"should_stop":false,"stop_reason":""}

USER: "살짝 달게"
→ {"extracted_slots":{"taste_profile":{"sweet":"medium"}},"should_stop":false,"stop_reason":""}

USER: "완전 달달하게"
→ {"extracted_slots":{"taste_profile":{"sweet":"high"}},"should_stop":false,"stop_reason":""}

USER: "향은 별 상관없어"
→ {"extracted_slots":{},"should_stop":false,"stop_reason":""}

USER: "달달한 거만 아니면 돼"
→ {"extracted_slots":{"taste_profile":{"sweet":"low"}},"should_stop":false,"stop_reason":""}

USER: "몰라 알아서 줘"
→ {"extracted_slots":{},"should_stop":false,"stop_reason":""}

USER: "이제 추천해줘"
→ {"extracted_slots":{},"should_stop":true,"stop_reason":"user_requested"}

반드시 위와 같은 JSON 한 줄만 출력해라. 설명·코드블록·마크다운 금지.
""".strip()


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
    "good": ["기분 좋", "기분좋", "너무 좋", "좋아", "좋네", "좋은데", "신나", "신난", "째진", "째져", "쩐다", "꿀잼", "들떠", "텐션"],
    "bad": ["기분 별로", "별로야", "꿀꿀", "우울", "다운", "지쳐", "피곤", "쳐져", "안 좋", "안좋"],
    "soso": ["그냥 그래", "그냥그래", "쏘쏘", "그저 그래", "그저그래", "보통이", "그럭저럭", "무난"],
}
_PURPOSE_KEYWORDS = {
    "celebration": ["생파", "생일", "축하", "돌잔치", "기념일"],
    "date": ["데이트", "여친", "남친", "썸녀", "썸남", "둘이"],
    "business": ["회식", "거래처", "직장 동료", "회사 동료", "비즈니스"],
    "solo": ["혼술", "혼자 마시", "혼자 왔", "나 혼자"],
    "hangout": ["친구들이랑", "놀러", "모임", "캐주얼"],
}
_STRENGTH_KEYWORDS = {
    "strong": ["독하게", "세게", "쎄게", "강하게", "달리자", "달릴", "술고래", "취하고 싶"],
    "light": ["약하게", "가볍게", "순한", "살살", "부드럽게", "잘 못 마셔"],
    "zero": ["무알콜", "논알콜", "알콜 빼", "술 빼"],
    "medium": ["적당히", "보통으로", "중간으로"],
}


def _backfill_enum(user_msg: str, patterns: dict[str, list[str]]) -> Optional[str]:
    text = (user_msg or "").lower()
    for enum_val, kws in patterns.items():
        for kw in kws:
            if kw in text:
                return enum_val
    return None


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

    # 스칼라 enum 백업 — LLM 이 직전 질문 축을 놓쳤을 때 키워드로 재추출.
    if last_slot == "current_mood" and "current_mood" not in fixed:
        v = _backfill_enum(user_msg, _MOOD_KEYWORDS)
        if v:
            fixed["current_mood"] = v
    if last_slot == "party_purpose" and "party_purpose" not in fixed:
        v = _backfill_enum(user_msg, _PURPOSE_KEYWORDS)
        if v:
            fixed["party_purpose"] = v
    if last_slot == "strength_preference" and "strength_preference" not in fixed:
        v = _backfill_enum(user_msg, _STRENGTH_KEYWORDS)
        if v:
            fixed["strength_preference"] = v

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

_QUESTION_SYSTEM_PROMPT = """
너는 경력 10년차 한국인 바텐더다. 손님과 바 테이블을 사이에 두고 가볍게 대화하며 취향을 파악한다.
톤은 친근한 존댓말, 살짝 편안하게. 과하게 격식 차리지 말고, 또 너무 들뜨지도 마라.
출력은 반드시 자연스러운 한국어 1문장. JSON, 코드블록, 불릿, 번호, 영어 단어, 이모지 금지.

[핵심 원칙]
1. 사용자의 직전 말에 짧게(5~10자) 공감하거나 맞장구친 뒤, 부족한 슬롯 **하나만** 묻는다.
2. 공감 + 질문을 한 문장으로 붙여라. "네, 알겠습니다." 같이 두 문장으로 끊지 마라.
3. 이미 채워진 슬롯은 절대 다시 묻지 마라. (주어지는 "이미 채워진 슬롯"을 꼭 확인)
4. 선택지를 줄 때는 2~4개만 콤마로 나열. 5개 이상 나열은 피로하니 금지.
5. 보기가 필요한 슬롯(도수/맛/향/베이스)은 **반드시 보기 예시를 괄호 없이 자연스럽게 문장에 녹여라**.
6. 질문은 한 번에 한 슬롯. 두 가지를 동시에 묻지 마라. ("기분이랑 도수는?" 같은 거 금지)
7. 손님 말투를 살짝 따라가도 좋다. 손님이 반말이면 존댓말은 유지하되 딱딱하지 않게.
8. "~하시는 편이세요?", "~어떠세요?" 같은 자연스러운 어미 사용. 번역체·문어체 금지.

[공감 멘트 팁]
- 좋은 답: "오 좋죠", "그거 좋죠", "아 그러시구나", "편하게 한잔 하시는 거네요", "분위기 딱이네요"
- 나쁜 답: "감사합니다" (너무 격식), "훌륭합니다" (어색), "확인했습니다" (기계적)
- 손님이 "그냥 그래", "몰라" 같이 소극적이면 공감을 짧게 ("네 편하게 가시죠") 하고 바로 다음 질문.

[슬롯별 베스트 질문 템플릿]
party_purpose:
  - "오늘은 어떤 자리 때문에 오셨어요? 모임이세요, 데이트세요, 아니면 혼자 편하게?"
current_mood:
  - "지금 기분은 좀 어떠세요? 좋으신 편이에요, 그냥저냥이세요, 아니면 좀 다운이세요?"
strength_preference:
  - "도수는 어떤 게 편하세요? 가볍게, 적당히, 아니면 확 가는 쪽?"
taste_profile:
  - "맛은 어느 쪽이 끌리세요? 달달한 쪽, 상큼한 쪽, 쌉싸름한 쪽 중에요."
aroma_profile:
  - "향은 어떤 느낌이 좋으세요? 과일향, 허브향, 나무향, 커피향 정도로요."
disliked_bases:
  - "혹시 피하고 싶으신 베이스 술 있으세요? 위스키, 진, 럼, 보드카, 데킬라 중에요."
favorite_drinks:
  - "평소에 즐겨 드시는 칵테일이나 술 있으세요? 없으시면 없다고 하셔도 돼요."

[절대 금지 패턴 — 실제로 나왔던 나쁜 출력들]
- "어떤 알코올 기저 음료는 안 드실 건가요?"                          ← 번역체
- "도수는 가볍게 하시는 편이든, 조금 강하게 드실래요?"                  ← 어순 깨짐
- "쓴맛이 드는 쪽이었나요?"                                         ← 과거형 어색
- "좋아요, 보통이야, 아니면 좀 무심치 않아요?"                          ← 비문
- "보통정도라면 도수는 약하게 드실 편이신가요?"                         ← 유도질문
- "혼자 조용히 마시려는 분이라면, 지금 기분이 어때요?"                   ← 장황
- "혹시 특정 베이스 술은 피하시나요?"                                ← 보기 없이 추상적
- "그렇군요. 그럼 다음으로 맛에 대해 여쭙고 싶은데요, ..."                ← 서술조 늘어짐
- "당신의 선호도는 어떻습니까?"                                     ← 교과서체

[좋은 대화 흐름 예시]
USER "친구 생파야" → "오 생파 좋죠, 기분은 지금 좀 어떠세요? 들떠 있으세요, 편안하세요?"
USER "그냥 보통" → "네 편하게 가시죠, 도수는 어떤 게 좋으세요? 가볍게, 적당히, 아니면 세게?"
USER "상큼한 거 땡겨" → "오 상큼한 거 좋죠, 향은 어떤 쪽이 끌리세요? 과일향, 허브향, 아니면 꽃향기?"
USER "달달한 거" → "달달한 쪽 좋으시군요, 향은 어떠세요? 과일향, 플로럴, 커피향 같은 거요."
USER "없어" (비선호 베이스 답) → "좋아요 편하게, 평소에 자주 드시는 칵테일이나 술 있으세요?"
USER "몰라 알아서 줘" → "네 제가 알아서 맞춰볼게요, 도수는 가볍게가 좋으세요 세게가 좋으세요?"
USER "위스키만 빼줘" → "위스키 빼고 드릴게요, 평소 좋아하시는 술 있으세요?"

출력: 위 원칙에 맞는 한국어 1문장 (공감 + 질문).
""".strip()

_SUBKEY_KO = {
    "sweet": "단맛",
    "sour": "신맛",
    "bitter": "쓴맛",
    "body": "바디감",
    "creamy": "크리미한 질감",
    "freshness": "청량감",
    "woody": "우디향",
    "minty": "민트향",
    "fruity": "과일향",
    "citrus": "시트러스향",
    "floral": "꽃향",
    "coffee": "커피향",
    "herbal": "허브향",
}

_SLOT_KO_DESC = {
    "party_purpose": "오늘 어떤 자리인지 — 모임·데이트·혼술·친구랑 놀기 같은 상황",
    "current_mood": "지금 기분 상태 — 좋은지, 그냥저냥인지, 좀 가라앉았는지",
    "strength_preference": "도수 — 가볍게 마시고 싶은지, 적당한지, 좀 강한 거 원하는지",
    "taste_profile": "선호하는 맛 축과 그 강도 (단/신/쓴/바디/크리미/청량) — 얼마나 강하게 원하는지까지 파악",
    "aroma_profile": "선호하는 향 계열과 강도 (과일/허브/민트/시트러스/우디/커피/플로럴) — 얼마나 강하게 원하는지까지",
    "disliked_bases": "피하고 싶은 베이스 술",
    "favorite_drinks": "평소 즐겨 마시는 술·칵테일·음료",
}

def _choose_ack(user_msg: str, extracted_slots: dict) -> str:
    text = _normalize_text(user_msg)

    if "disliked_bases" in extracted_slots:
        bases = extracted_slots.get("disliked_bases") or []
        return "좋아요 편하게 가시죠" if not bases else "좋아요 그건 빼고 볼게요"
    if "favorite_drinks" in extracted_slots:
        favs = extracted_slots.get("favorite_drinks") or []
        return "좋아요 편하게 볼게요" if not favs else "오 그것도 좋죠"
    if extracted_slots.get("party_purpose") == "celebration":
        return "오 생파 좋죠"
    if extracted_slots.get("party_purpose") == "solo":
        return "혼자 한잔 좋죠"
    if extracted_slots.get("party_purpose") == "date":
        return "데이트면 분위기 좋죠"
    if extracted_slots.get("current_mood") == "good":
        return "오 좋죠"
    if extracted_slots.get("current_mood") == "soso":
        return "네 편하게 가시죠"
    if extracted_slots.get("current_mood") == "bad":
        return "아 그러시구나"
    if extracted_slots.get("strength_preference") == "strong":
        return "확 가는 거 좋죠"
    if extracted_slots.get("strength_preference") == "light":
        return "가볍게 좋죠"
    if extracted_slots.get("taste_profile") or extracted_slots.get("aroma_profile"):
        return "취향 확실하시네요"
    if any(k in text for k in ["몰라", "아무거나", "알아서"]):
        return "네 편하게 가시죠"
    return "좋아요"


_TEMPLATE_QUESTIONS = {
    "party_purpose": "오늘은 어떤 자리로 오셨어요? 생일 같은 모임이세요, 데이트세요, 아니면 혼자 편하게 한잔이세요?",
    "current_mood": "지금 기분은 좀 어떠세요? 기분 좋은 편이세요, 그냥저냥이세요, 아니면 좀 다운되셨어요?",
    "strength_preference": "도수는 어떤 게 좋으세요? 가볍게, 적당히, 아니면 확 가는 쪽으로요?",
    "taste_profile": "맛은 어느 쪽이 끌리세요? 달달한 쪽, 상큼한 쪽, 아니면 쌉싸름한 쪽이요?",
    "aroma_profile": "향은 어떤 느낌이 좋으세요? 과일향, 허브향, 시트러스향, 커피향 쪽으로요?",
    "disliked_bases": "혹시 피하고 싶은 베이스 술 있으세요? 위스키, 진, 럼, 보드카, 데킬라 중에요.",
    "favorite_drinks": "평소에 자주 드시는 칵테일이나 술 있으세요? 없으시면 없다고 하셔도 돼요.",
}

# 사용자가 "모르겠어/딱히" 등으로 답했을 때 같은 슬롯을 돌려서 다시 물어보는 대안 질문.
_TEMPLATE_QUESTIONS_ALT = {
    "party_purpose": "아 그러시구나, 그럼 혼자 오신 편이에요 아니면 누구랑 같이 오셨어요?",
    "current_mood": "네 편하게 말씀해주세요, 오늘 하루 전반적으로 기분 괜찮으셨어요 아니면 좀 피곤하셨어요?",
    "strength_preference": "그럼 이렇게 여쭤볼게요, 평소에 맥주 한 잔 정도면 얼굴 빨개지세요 아니면 소주 반 병도 거뜬하세요?",
    "taste_profile": "그럼 편하게, 과일 주스나 에이드 같은 거 좋아하세요 아니면 진한 커피나 다크초콜릿 쪽이세요?",
    "aroma_profile": "평소 좋아하시는 향수나 냄새 있으세요? 달달한 바닐라, 시원한 민트, 숲 같은 우디, 커피 향 같은 거요.",
    "favorite_drinks": "그럼 술 말고 평소 자주 드시는 음료라도 있으세요? 예를 들어 하이볼, 레몬에이드, 라떼 같은 거요.",
}


_FAMILIARITY_TONE = {
    "처음": (
        "손님은 칵테일을 거의 처음 접한다. '어떤 느낌이 좋을지 같이 찾아보자'는 톤으로, "
        "전문용어 대신 일상 음식·음료에 빗댄 비유로 쉽게 물어라. 선택지는 2~3개로 간결하게."
    ),
    "가끔": (
        "손님은 칵테일을 어느 정도 마셔본 편이다. 친근한 톤으로 보기를 제시하되 강요하지 마라."
    ),
    "자주": (
        "손님은 칵테일을 자주 마시는 애호가다. 취향이 확실하다고 전제하고 단도직입적으로 "
        "구체적 용어(바디감/청량감/우디 등)를 써서 물어라. 비유·설명은 과하지 마라."
    ),
}

_FOLLOWUP_SYSTEM_PROMPT = """
너는 경력 10년차 한국인 바텐더다. 지금 바 테이블 앞에서 손님과 편하게 수다 떨며 취향을 파악하는 중이다.
설문지를 읽는 게 아니라 "대화"를 하는 거다. 친근한 존댓말, 자연스럽게 흐르는 문장으로.
이모지·영어·마크다운·코드블록·번호·불릿·슬래시 구분자(예: "좋음/보통/안좋음") 절대 금지.

[출력 형식 — 2~3문장, 티키타카 대화처럼]
- 목표: 설문지 읽듯 "다음 질문"만 던지지 말고, 손님 발화 내용을 구체적으로 받아쳐라 (티키타카).
  예: 손님이 "친구들이랑 재밌게 놀기로 했어, 기분 짱 좋아"라고 했으면,
     "오 친구들이랑 노는 날이면 진짜 신나죠~" 같이 구체 키워드("친구들", "짱 좋아")를 짚어서 받아쳐라.
     "아 그런 자리구나" 같은 무미건조한 반응은 피해라.
- 먼저 손님이 방금 한 말에 짧게 반응하거나 공감한다 ("아 그래요~", "오 그건 좋죠", "아 그러시구나").
- 손님이 묻지 않은 정보를 스스로 얹어 말해줬다면(예: "단맛도 강하면 좋겠어"를 시트러스 질문에 답하며 덧붙임) 그 정보도 가볍게 짚어 확인해라
  ("단맛도 강한 쪽 좋아하시는군요, 기억해둘게요"). 그 다음에 target_slot 질문으로 넘어간다.
- ★ 손님이 답변 대신 되물으면(용어 질문 / 시스템에 대한 메타 질문 / 혼란 표현),
  (1) 먼저 그 질문에 한 문장으로 진짜 내용을 담아 답해라. 얼버무리거나 화제 돌리지 마라.
  (2) 그 다음에 같은 target_slot 을 자연스럽게 다시 물어라. 다른 슬롯으로 넘어가지 마라.
  예 A — 용어 질문:
    USER "은은한게 뭐야?" (target_slot=taste_profile.sour)
    → "아 은은하다는 건 확 톡 쏘는 느낌 없이 살짝만 드러나는 정도를 말해요. 그럼 신맛은 확 드러나는 쪽이 좋으세요, 아니면 은은한 쪽이 좋으세요?"
  예 B — 시스템 질문:
    USER "기분에 따라서 추천이 달라져?" (target_slot=current_mood)
    → "네 달라져요. 기분 좋을 땐 경쾌한 거, 가라앉은 날엔 부드럽고 묵직한 쪽을 주로 권해요. 오늘은 기분 좋으신 편이세요, 그냥저냥이세요, 좀 다운되셨어요?"
  예 C — 왜 묻냐:
    USER "그걸 왜 물어봐?" (target_slot=strength_preference)
    → "도수에 따라 어울리는 칵테일이 확 달라져서 물어봐요. 오늘은 가볍게 드시고 싶으세요, 적당히 드시고 싶으세요, 아니면 좀 센 쪽으로 가시겠어요?"
  ★ 절대 "~하면 더 편하게 즐길 수 있어요" 같은 회피성 멘트로 답 대체하지 마라. 진짜 답을 해라.
- 이미 수집된 선호가 있으면 그걸 자연스럽게 짚어주며 "이런 느낌으로 맞춰가면 되겠네요" 식의 분석을 한 줄 덧댄다.
- 그리고 다음에 궁금한 한 가지(= target_slot)를 편하게 물어본다. 보기는 문장 안에 자연스럽게 녹이되, 나열형("A/B/C")이 아니라 "~쪽이세요, 아니면 ~쪽이세요?" 같은 구어체로.
- 두 슬롯을 동시에 묻지 않는다.
- ★ target_slot 이 아닌 다른 슬롯(특히 pending 으로 남아 있는 맛·향 축)을 분석 코멘트에 미리 끌어들여 추측하지 마라.
  예: target_slot=party_purpose 인데 "커피 향이 듬뿍 느껴지는 분위기로 맞춰가면 되겠네요" 라고 말하는 건 금지.
  데이트·모임·혼술 같은 자리와 맛·향은 서로 아무 관계 없다.

[대화 느낌 예시]
- "아 친구들이랑 노는 자리구나, 분위기 좋겠네요. 그럼 오늘은 가볍게 홀짝홀짝 마시고 싶으세요, 아니면 확 기분 내고 싶으세요?"
- "미도리 사워 좋죠~ 그 새콤달콤한 느낌 좋아하시나 봐요. 그럼 단맛은 확 단 게 좋으세요, 아니면 은은한 정도가 좋으세요?"
- "아 데이트 자리시구나 그럼 분위기 중요하겠네요. 평소에 향 좋은 거 좋아하시는 편이세요? 향수 쓰실 때 어떤 계열 쓰세요?"

[핵심 원칙]
- ★ "지금까지 파악한 선호"에 이미 들어있는 값(자리·기분·도수 등)을 절대로 다른 값으로 바꿔 말하지 마라.
  예: 자리가 이미 "모임/축하자리(celebration)"로 확정돼 있는데 "아 데이트 자리시구나"라고 말하는 건 금지.
  분석 코멘트는 반드시 확정된 값과 일치해야 한다. 확정된 자리·기분을 언급할 거면 그대로 써라.
- 이미 "확정된" 선호는 다시 묻지 마라. 단, "강도 미확정(pending)" 항목은 반드시 강도를 대화로 끌어내라.
  예: 손님이 초기 태그로 "단맛·과일향"을 골라뒀지만 강도를 모르는 상태면,
     "단맛은 확 단 게 좋으세요, 아니면 은은하게?"처럼 강도를 짚어 묻는다.
- 선호만 물어보지 말고, 대화 중 한 번은 반대로 "피하고 싶은 맛이나 향"도 자연스럽게 확인해라
  (예: "반대로 이건 좀 별로다 싶은 맛이나 향 있으세요?"). 이미 비선호(low/zero)가 슬롯에 있으면 다시 묻지 마라.
- 과거형·번역체·문어체 금지. "~했나요?" 대신 "~세요?" / "~하시겠어요?".
- 정해진 질문 틀("도수는 어떤 게 좋으세요?")을 반복하지 마라 — 대화 맥락에 맞춰 바꿔라.

[친숙도별 톤]
{familiarity_tone}

[슬롯 힌트 — target_slot 만 물어라]
- strength_preference: 오늘 얼마나 마시고 싶은지 기분 묻듯이.
- taste_profile: 단/신/쓴/바디/크리미/청량 중 target key 를 강도까지.
- aroma_profile: 과일/허브/민트/시트러스/우디/커피/플로럴 중 target key 를 강도까지.
- favorite_drinks: 친숙도가 "자주"/"가끔"이면 평소 즐겨 마시는 칵테일/술(예: "마시는 칵테일이나 술 있으세요?")을 물어라.
  "처음"이면 술 대신 일상 음료(주스·커피·차·에이드 같은 거)로 향·단맛 감을 잡아라.
- party_purpose: 오늘 어떤 자리인지.
- current_mood: 지금 컨디션·기분 어떤지.

출력: 위 원칙 지킨 한국어 2~3문장만. 다른 부연 일절 하지 마라.
""".strip()


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


def _generate_followup_llm(
    slots: dict,
    user_msg: str,
    target_slot: str,
    familiarity: Optional[str],
    use_alt_angle: bool,
    target_pending_subkey: Optional[str] = None,
) -> Optional[str]:
    try:
        from app.utils.model_loader import load_llm
        import torch

        tokenizer, model = load_llm()

        fam_key = familiarity if familiarity in _FAMILIARITY_TONE else "가끔"
        system = _FOLLOWUP_SYSTEM_PROMPT.format(familiarity_tone=_FAMILIARITY_TONE[fam_key])

        alt_hint = ""
        if use_alt_angle:
            alt_hint = (
                f"\n[특별 지시] 손님이 방금 '{target_slot}' 관련 질문에 '모르겠다/딱히'라고 답했다. "
                f"같은 슬롯이지만 전혀 다른 각도(예: 평소 향수/일상 음식/체질 등)로 돌려 물어라."
            )

        pending_hint = ""
        if target_pending_subkey:
            ko = _SUBKEY_KO.get(target_pending_subkey, target_pending_subkey)
            pending_hint = (
                f"\n[강도 미확정 항목] {ko} — 초기 태그에서 '{ko}'를 선호로 골랐지만 강도(약/중/강)가 "
                f"아직 확인 안 됐다. 이 축의 강도를 구체적으로 끌어내는 질문을 해라 "
                f"(예: '{ko}은 확 드러나는 게 좋으세요, 아니면 은은한 정도가 좋으세요?'). "
                f"영어 단어({target_pending_subkey}) 절대 쓰지 말고 반드시 한국어('{ko}')로 써라."
            )

        user_prompt = (
            f"[지금까지 파악한 선호]\n{_summarize_slots_for_prompt(slots)}\n\n"
            f"[손님 친숙도] {fam_key}\n\n"
            f"[손님의 방금 발화]\n{user_msg}\n\n"
            f"[다음에 물어볼 슬롯] {target_slot} — {_SLOT_KO_DESC.get(target_slot, '')}"
            f"{pending_hint}"
            f"{alt_hint}\n\n"
            f"위 정보를 반영해서, 분석 코멘트 1문장 + {target_slot} 질문 1문장, 총 2문장만 출력."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

        cleaned = raw.replace("```", "").strip()
        if not cleaned or len(cleaned) > 300:
            return None
        return cleaned
    except Exception as e:
        logger.warning("LLM followup generation failed: %r", e, exc_info=True)
        return None


def generate_followup_question_with_llm(
    history: list[dict],
    slots: dict,
    user_msg: str,
    extracted_slots: dict,
    skip_slots: Optional[set[str]] = None,
    alt_slots: Optional[set[str]] = None,
    familiarity: Optional[str] = None,
) -> str:
    skip_slots = skip_slots or set()
    alt_slots = alt_slots or set()

    target_slot = None
    target_pending_subkey: Optional[str] = None
    for key in SLOT_ASK_ORDER:
        if key in skip_slots:
            continue
        v = slots.get(key)
        empty_missing = (
            key not in slots
            or v is None
            or (isinstance(v, (list, dict)) and len(v) == 0 and key not in _EMPTY_OK_SLOTS)
            or (isinstance(v, str) and not v.strip())
        )
        pending_subkey = None
        if isinstance(v, dict):
            pending_subkey = next(
                (k for k, val in v.items() if val == INTENSITY_PENDING),
                None,
            )
        if empty_missing or pending_subkey:
            target_slot = key
            target_pending_subkey = pending_subkey
            break

    if not target_slot:
        return "좋아요. 추천으로 넘어가볼게요."

    use_alt = target_slot in alt_slots
    llm_out = _generate_followup_llm(
        slots=slots,
        user_msg=user_msg,
        target_slot=target_slot,
        familiarity=familiarity,
        use_alt_angle=use_alt,
        target_pending_subkey=target_pending_subkey,
    )
    if llm_out:
        return llm_out

    ack = _choose_ack(user_msg, extracted_slots)
    if use_alt and target_slot in _TEMPLATE_QUESTIONS_ALT:
        question = _TEMPLATE_QUESTIONS_ALT[target_slot]
    else:
        question = _TEMPLATE_QUESTIONS.get(target_slot, "조금만 더 취향을 알려주실래요?")
    return f"{ack}, {question}"


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
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        key = k.strip().lower()
        if aliases and key not in allowed_keys:
            key = aliases.get(key, key)
        if key not in allowed_keys:
            continue
        if not isinstance(v, str):
            continue
        vv = v.strip().lower()
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

    def _scalar(key: str, allowed: set[str]) -> None:
        if key not in raw:
            return
        val = raw[key]
        if val is None:
            cleaned[key] = None
            return
        v = _validate_enum(val, allowed)
        if v:
            cleaned[key] = v
        else:
            dropped.append(f"{key}={val!r} (허용값: {sorted(allowed)})")

    _scalar("current_mood", CURRENT_MOOD_VALUES)
    _scalar("party_purpose", PARTY_PURPOSE_VALUES)
    _scalar("strength_preference", STRENGTH_VALUES)

    for profile_key, allowed, aliases in (
        ("taste_profile", TASTE_KEYS, _TASTE_KEY_ALIASES),
        ("aroma_profile", AROMA_KEYS, _AROMA_KEY_ALIASES),
    ):
        if profile_key not in raw:
            continue
        raw_val = raw[profile_key]
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
1. **사용자가 방금 한 발화에 명시적으로 나타난 축만 넣어라.** 추측·유도 금지. 바텐더가 물었지만 사용자가 답 안 한 축은 절대 넣지 마라.
2. 한 발화에 축이 여러 개면 모두 넣어라 — 하나도 빠뜨리지 마라.
3. 사용자가 질문·되묻기만 했거나 "모르겠다/딱히" 로만 답한 경우 {"extracted_slots": {}} 출력.
4. creamy/크리미/우유/밀키/밀크 → taste_profile.creamy (절대 disliked_bases 아님. creamy 는 base 가 아니라 질감이다).
5. body/바디감/묵직 → taste_profile.body. aroma_profile 에 넣지 마라 (body 는 향 축이 아니다).
6. 정정(CORRECTION): 사용자가 "나 그런 말 한 적 없어" 류로 앞 추론을 부정하면 해당 슬롯을 null 로 넣어라. 예: {"current_mood": null}
7. 선호 극성 유지: "X 좋아해" = 선호(high/medium/low 중 강도로 분리), "X 싫어/별로" = zero. 극성 뒤집지 마라. ※ low/medium/high 전부 "선호" 범주다. 비선호는 오직 zero.
8. **긍정 확인(CONFIRM) 발화는 새 슬롯 아님.** "맞아/응/좋아/오케/그래/네/그대로/너가 말한대로" 로만 된 발화에는 {"extracted_slots":{}} 를 출력. 이전 봇 질문을 사용자 답으로 착각해 복제하지 마라.
9. **enum 외 값 절대 금지.** aroma 키는 {woody,minty,fruity,citrus,floral,coffee,herbal} 만. taste 키는 {sweet,sour,bitter,body,creamy,freshness} 만. 강도는 {zero,low,medium,high} 만. "mint"/"strong"/"medium-high" 같은 표현 절대 금지 — 의미가 맞는 허용 값으로 바꿔라.
10. **⚠️ "강하게" → "high" (절대 "strong" 아님).** "strong" 은 strength_preference(도수) 슬롯 전용 값이다. taste_profile / aroma_profile 의 값으로 "strong" 을 쓰면 안 된다. "쓴맛 강하게" → bitter:"high" (절대 bitter:"strong" 금지).
11. **봇이 제시한 옵션(강하게/약하게 등)은 사용자 답이 아니다.** 봇이 "강하게 원하세요 약하게 원하세요?" 물어본 뒤 사용자가 답 안 하면 그 축은 추출하지 마라. 이번 발화에 축 키워드가 없으면 그 축은 없는 것이다 — 이전 턴 슬롯을 다시 출력하지 마라.
12. **부정/부인문**: "X 좋아한다고 (말)한 적 없어", "내가 언제 X 라고 했어?" 는 이전 추론의 철회다 → 해당 축을 null 로 넣어라 (긍정값으로 넣지 마라). 예: "단맛 좋아한다고 안 했는데" → {"taste_profile":{"sweet":null}}
13. **"분위기/느낌/무드"는 current_mood 가 아니다.** "조용한 분위기 / 편안한 느낌 / 가벼운 무드" 같은 자리·공간 스타일 얘기는 current_mood 에 넣지 마라. current_mood 는 **사용자 본인 기분**을 명시적으로 표현한 경우만 — "기분 좋아/신나/행복", "우울/힘들/피곤/별로" 같은 자기 상태 표현. 애매하면 넣지 말 것.

[강도 매핑] — low/medium/high 는 전부 "선호" 강도. zero 는 배제(비선호).
- "확/완전/엄청/강하게/세게/짱/듬뿍" → high (강하게 선호)
- "적당히/중간/보통" → medium (적당히 선호)
- "살짝/약간/조금/은은하게/옅게/살며시" → low (약하게 선호 — 있으면 좋지만 필수 아님)
- "별로/덜/안 땡겨/싫어/완전 싫어/질색/알레르기/빼줘" → zero (비선호/배제)

[축 키워드 매핑]
- taste: 단맛→sweet, 신맛/새콤→sour, 쓴맛/씁쓸→bitter, 바디감/묵직→body, 크리미/부드러움/우유/밀크→creamy, 청량감/상큼/시원함→freshness
- aroma: 우디/나무→woody, 민트→minty, 과일/프루티/파인애플/망고→fruity, 시트러스/레몬/자몽/라임→citrus, 꽃/플로럴→floral, 커피→coffee, 허브→herbal
- purpose: 혼자/혼술→solo, 회식/거래처→business, 생일/기념/축하/돌잔치→celebration, 데이트/썸/둘이→date, 친구/모임/놀러→hangout
- mood: 좋아/신나/설레→good, 별로/우울/힘들/안 좋→bad
- strength: 무알콜/논알콜→zero, 약하게/가볍게→light, 보통/적당히→medium, 세게/강하게→strong

[예시]
USER: "단맛 좀 세게 느끼고 싶어. 신맛은 중간 정도. 크리미한 맛은 완전 싫어."
→ {"extracted_slots":{"taste_profile":{"sweet":"high","sour":"medium","creamy":"zero"}}}

USER: "친구 생일파티야"
→ {"extracted_slots":{"party_purpose":"celebration"}}

USER: "오늘 혼자 조용히 마실거야. 약간 우울해"
→ {"extracted_slots":{"party_purpose":"solo","current_mood":"bad"}}

USER: "파인애플향 좋아해. 우유향은 싫고"
→ {"extracted_slots":{"aroma_profile":{"fruity":"high"},"taste_profile":{"creamy":"zero"}}}

USER: "시트러스 강한 게 좋고 커피향은 완전 싫어"
→ {"extracted_slots":{"aroma_profile":{"citrus":"high","coffee":"zero"}}}

USER: "민트향 더 좋아하고 허브향은 은은하게 나면 좋겠어"
→ {"extracted_slots":{"aroma_profile":{"minty":"high","herbal":"low"}}}

USER: "시트러스도 은은하게"
→ {"extracted_slots":{"aroma_profile":{"citrus":"low"}}}

USER: "맞아" / "응 그렇게 해줘" / "너가 말한 대로 진행해"
→ {"extracted_slots":{}}

USER: "쓴맛은 강하면 좋겠어"
→ {"extracted_slots":{"taste_profile":{"bitter":"high"}}}
(주의: "강" 이 있어도 taste 값은 "strong" 금지. 반드시 "high".)

USER: "단맛 좋아한다고 안 했는데?" (직전에 봇이 단맛 강도를 물음)
→ {"extracted_slots":{"taste_profile":{"sweet":null}}}
(부정/부인 → null. 긍정값 절대 금지.)

USER: "내가 언제 신맛 좋다고 했어"
→ {"extracted_slots":{"taste_profile":{"sour":null}}}

USER: "과일향이랑 민트향은 강하게" (직전 봇 발화에 bitter 언급됨, 사용자는 bitter 언급 없음)
→ {"extracted_slots":{"aroma_profile":{"fruity":"high","minty":"high"}}}
(이번 발화에 bitter 키워드가 없으면 bitter 축 절대 포함 금지. 이전 턴 값 재출력 금지.)

USER: "위스키는 싫어"
→ {"extracted_slots":{"disliked_bases":["whiskey"]}}

USER: "크리미! 평소에 우유 좋아해"
→ {"extracted_slots":{"taste_profile":{"creamy":"high"}}}

USER: "민트향에 대해선 안 물어보니?"
→ {"extracted_slots":{}}

USER: "잘 모르겠는데"
→ {"extracted_slots":{}}

USER: "나 그런 말 한 적 없는데" (직전에 current_mood=good 으로 추론)
→ {"extracted_slots":{"current_mood":null}}

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
        from app.utils.model_loader import load_llm
        import torch

        tokenizer, model = load_llm()
        messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": _build_extract_user_prompt(history, user_msg)},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=180,
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
너는 경력 10년차 한국인 바텐더다. 바 테이블에서 손님과 가볍게 대화하며 취향을 파악해 오늘의 칵테일을 추천하는 게 일이다.
목표는 "슬롯 채우기"가 아니라 "자연스러운 티키타카"다. 슬롯은 대화의 부산물일 뿐이다.

[★ 스몰톡 톤 — 이 감도 절대 놓치지 마라 ★]
- 너는 설문조사원이 아니다. 바 카운터 너머의 바텐더다. 손님이 방금 한 말에 **구체적으로 반응**하고("생일 축하 자리면 공간이 꽉 차는 느낌이겠네요", "숲향 좋아하신다니 그럼 저랑 취향 비슷하세요") 그 반응의 **끝자락에 궁금증처럼** 축 질문을 끼워 넣어라.
- 형식적인 "네 알겠습니다. 다음 질문은..." 금지. "어느 쪽이세요? A / B / C" 처럼 선다형 반복도 금지(가끔은 괜찮지만 매 턴 반복 금지).
- 취향을 바로 못 뽑아내도 괜찮다. 우회해라: **평소 음식, 좋아하는 향수, 카페에서 자주 시키는 음료, 여행지, 계절 취향** 같은 일상 질문으로 감각 단어를 역산해라. "숲향 향수" 처럼 손님 본인이 꺼낸 일상 취향은 **꼭 한 번 짚어서** 바(bar)에서 맞는 축으로 연결해라.
- 한 턴에 질문 1개가 기본. 대신 공감/관찰/가벼운 농담 한 문장을 앞에 둬서 숨을 넣어라. reply 전체 톤은 "친구가 카운터에서 같이 골라주는 느낌".

[★★★ 최우선 원칙 ★★★]
**슬롯 추출은 외부 파이프라인에서 이미 끝났다.** 이 턴에 반영된 슬롯은 아래 [이번 턴 추출된 슬롯] 섹션으로 주어진다.
- 너는 **reply / action / user_intent 만** 신경 쓰면 된다. extracted_slots 필드는 참고용으로 빈 {} 로 출력해라 (시스템은 그 값을 사용하지 않는다).
- reply 에서 손님 발화를 받아줄 때, [이번 턴 추출된 슬롯] 과 **일치하는 표현**을 써라. 거기 없는 축을 reply 에서 "이해했어요" 라고 말하지 마라.
- [이미 확정된 슬롯] 목록은 이전 턴들에서 누적된 상태다. [이번 턴 추출된 슬롯] 과 합쳐서 손님 취향 전체 그림을 파악해라.

[★★ 칵테일 이름 금지 ★★]
**action=ASK 단계에서는 칵테일 이름을 절대 언급하지 마라 — 실재하는 이름이든, 네가 지어낸 이름이든 전부 금지.**
- "마티니 어때요?", "모히토 추천드릴게요", "스윗 임팩트 같은 게 좋을 것 같아요" 전부 금지.
- 추천 이름을 고르는 건 다음 단계(RAG 리트리버)의 일이다. 너는 지금 취향만 파악한다.
- action=RECOMMEND 로 전환해서 마무리 멘트("그럼 그 느낌으로 골라와서 보여드릴게요") 를 할 때도 특정 칵테일 이름을 찍지 마라.

[절대 원칙 — 위반 시 치명적]
0. **출력 언어는 100% 자연스러운 한국어.** reply 에 한자(甜/苦/酸/甘 등), 중국어, 일본어, 영어 단어 절대 금지.
   "약간은甜하고" 같이 한국어+한자 섞는 건 금지. "약간은 달콤하고" 처럼 순 한국어로 써라.
0a. **축 이름을 직역해서 만들지 마라.** body 는 "몸향" 아니고 "바디감 / 묵직함". creamy 는 "우유향" 아니고 "크리미한 질감 / 부드러움". freshness 는 "신선함" 아니고 "청량감 / 상큼함". body 와 aroma(향)은 서로 다른 축이다 — body 관련 표현에 "향"이라는 단어 붙이지 마라.
   허용 한국어:
   - taste: sweet→단맛, sour→신맛, bitter→쓴맛, body→바디감/묵직함, creamy→크리미/부드러움, freshness→청량감/상큼함
   - aroma: woody→우디향/나무향, minty→민트향, fruity→과일향, citrus→시트러스향, floral→꽃향, coffee→커피향, herbal→허브향
   맛/향 표현은 "달콤/쌉쌀/상큼/청량/묵직/허브/시트러스" 등 한국어 단어만 사용.
A. 추측 금지: 손님이 실제로 말하지 않은 축은 extracted_slots 에 절대 넣지 마라.
   - "친구 생일파티" → party_purpose=celebration 만 추출. current_mood 는 말 안 했으면 절대 넣지 마라.
   - "데이트" → party_purpose=date 만. mood 는 추출 금지.
   - 즉 party_purpose 로부터 current_mood 를 유도하는 것 = 금지.
A3. **[이미 확정된 슬롯] 은 손님이 명시적으로 정정한 경우만 덮어써라.** 특히 strength_preference 가 초기 태그에서 확정됐으면 손님이 "도수"라는 단어를 직접 말하지 않는 한 절대 바꾸지 마라. "단맛 약하게" 는 taste.sweet=low 이지 strength_preference=light 가 아니다. "약해 / 센" 이 어느 축에 걸린 건지 앞 맥락으로 판단.
A4. **null 로 지우는 건 금지 — 단, 손님이 명시적으로 "나 그런 말 한 적 없어" 류 정정(CORRECTION)한 경우만 허용.** 이유 없이 extracted_slots 에 {"current_mood": null} 같은 거 절대 넣지 마라.
A5. **[강도 미확정] 에 없는 향/맛 축은 손님이 먼저 언급하지 않은 이상 절대 물어보지 마라.** 손님이 태그로 선택하지 않은 축을 "이것도 좋아하세요?" 라고 묻는 건 금지. 태그에 없는 축은 존재하지 않는 셈 쳐라. 묻는 건 **오직 [강도 미확정] 목록에 있는 축뿐**.
A2. **추출 필수 — 손님이 구체적인 축을 말했으면 반드시 extracted_slots 에 넣어라.**
   - "혼자" / "혼술" / "혼자 조용히" → party_purpose=solo (반드시).
   - "회식" / "회사 회식" / "팀 회식" / "동료랑" / "상사랑" / "거래처" → party_purpose=business (반드시). "회식" 단어가 들어가면 무조건 business.
   - "생일" / "축하" / "돌잔치" / "기념일 축하" → party_purpose=celebration.
   - "데이트" / "썸" / "둘이" → party_purpose=date.
   - "친구들이랑 놀" / "모임" / "친구들이랑" / "놀러" → party_purpose=hangout.
   - "우울" / "별로" / "안 좋아" / "힘들어" → current_mood=bad.
   - "좋아" / "기분 좋" / "신나" → current_mood=good.
   - 한 발화에 여러 축이 있으면 **모두** 추출해라. (예: "혼자 우울해" → {party_purpose:"solo", current_mood:"bad"})
   - 강도 표현 "은은/살짝/약간" → medium, "강하게/확/쎄게" → high, "적당히/보통" → medium, "별로/덜" → low, "싫어/질색" → zero.
   - 손님이 taste/aroma 축 이름을 말하고 강도 형용사까지 썼으면 taste_profile/aroma_profile 에 **반드시** 반영.
     예: "단맛은 은은하게, 상큼한건 좀 강했으면" → taste_profile={sweet:"medium", freshness:"high"}
     예: "파인애플향 좋아해, 우유향은 싫어" → aroma_profile={fruity:"high"}, taste_profile={creamy:"zero"}
     (참고: "우유향/밀키/크리미" 는 향 카테고리에 없으니 taste_profile.creamy 로 매핑.)
   - 손님이 "딱히 없어 / 상관없어 / 다 괜찮아" 로 답한 경우에만 extracted_slots={} 허용.
B. 손님 정정 존중: 손님이 "나 X 라고 말한 적 없어 / 아니야 / 그건 아니고" 라고 하면, 해당 slot 을 반드시 null 로 extracted_slots 에 넣어 지워라. 예: {"current_mood": null}
C. 반복 금지: [직전에 네가 한 질문] 과 같은 주제·같은 구조·같은 선택지 질문을 다시 하지 마라. 손님이 답 못 하면 **다른 축으로 넘어가거나 바로 RECOMMEND**.
C2. 손님이 "없어/괜찮아/없습니다/딱히/다 좋아" 류로 답했으면 **같은 축을 다시 묻는 것은 절대 금지**. 즉시 [부족한 슬롯] 목록의 다른 축으로 넘어가거나, 부족한 슬롯이 없으면 RECOMMEND 로 가라. "다른 수정사항 있으세요?" 같은 오픈 질문을 두 번 반복하지 마라.
D. **[완성도]와 [강도 미확정]/[부족한 슬롯]을 이번 턴 질문의 1순위 기준으로 삼아라.**
   - [완성도] < 80% 이면 action=ASK **고정**이고, 질문은 반드시 [부족한 슬롯] 또는 [강도 미확정] 중 하나를 타깃팅해야 한다. 엉뚱한 축(이미 high 로 확정된 걸 또 물음)이나 open 잡담으로 턴 낭비 금지.
   - [강도 미확정] 목록에 있는 축은 "좋아하세요?" 금지 (선호는 이미 확정). **강도만** 물어라 ("확 쎄게 / 은은하게 / 적당히 중에?"). 한 턴에 2~3개 축을 묶어서 물어도 됨.
   - [이번 턴 추출된 슬롯] 에 이미 high/medium/low/zero 로 확정된 축은 **절대 다시 강도 질문하지 마라**. 예: 이번 턴에 fruity=high 가 뽑혔으면 "과일향 강도는요?" 다시 묻지 마라 — 이미 확정.
   - RECOMMEND 로 넘어갈 조건은 (a) [강도 미확정] 비어있음 AND (b) [부족한 슬롯] 비어있음 AND (c) 비선호 축 1회 확인 완료(G2). 이 세 개 다 충족돼야 action=RECOMMEND.
   - 손님이 "모르겠다"고 답하면 1회만 우회 질문(규칙 F) 후 포기하고 다음 축으로.
E. [손님 친숙도] 에 따라 대화 방향을 **완전히 다르게** 잡아라 — 이건 최우선 분기다:
   - **"처음" (novice)**: 손님은 칵테일 용어(드라이/프루티/베이스/스피릿) 잘 모른다. 칵테일 용어 금지, 일상 감각 단어만 써라. 질문은 **거의 다 일상 취향 우회**로 간다.
     스크립트 예: "평소에 커피는 달달한 걸로 드세요 아니면 블랙이세요?", "아이스크림 고르면 어떤 맛 손이 가요?", "향수는 어떤 계열 쓰세요?", "여행 가면 바다 쪽이 좋으세요 숲 쪽이 좋으세요?", "평소 음식은 매콤한 거 달달한 거 담백한 거 중에?"
     톤: "제가 같이 찾아드릴게요", "편하게 말씀하시면 제가 맞춰드릴게요". 살짝 가이드해주는 선배 느낌.
   - **"가끔" (occasional)**: 칵테일 몇 개는 알고 있다. **과거 경험**을 축으로 풀어라.
     스크립트 예: "저번에 드신 것 중에 괜찮았던 거 기억나세요?", "모히토나 마가리타 같은 거 드셔보셨어요? 어떠셨어요?", "진토닉류가 편하세요 아니면 사워 계열이 더 맞으세요?", "달달했던 게 좋았어요 쌉싸름한 게 좋았어요?"
     톤: 친구가 같이 고르는 느낌. 용어는 적당히 (진토닉, 사워 정도는 OK).
   - **"자주" (regular)**: 용어 자유롭게 써도 된다. 베이스·스타일·스피릿 바로 물어도 OK.
     스크립트 예: "평소 베이스 뭐 많이 드세요? 진? 럼? 위스키?", "드라이한 쪽이세요 프루티 쪽이세요?", "스터드 & 스트레인 스타일 좋아하세요?", "비터 많이 들어간 클래식 계열 어떠세요?"
     톤: 바텐더끼리 얘기하듯. 전문 용어 거리낌 없이.
   ※ 친숙도별 톤과 질문 방식이 섞이면 안 된다. 처음인 손님한테 "드라이/프루티/스피릿/스터드" 단어 절대 금지. 자주인 손님한테 "음식은 매콤한 거 좋아하세요?" 같은 초짜 우회 질문은 지루하다.
F. 손님이 "잘 모르겠다/몰라/딱히" 라고 답하면 같은 축을 다시 묻지 말고 **우회 질문**으로 유도해라:
   - 맛 관련 모르겠다 → "평소 음식은 어떤 맛을 좋아하세요? 매콤한 거? 담백한 거?"
   - 향 관련 모르겠다 → "향수는 어떤 계열 쓰세요?" "좋아하는 과일 있어요?"
   - 도수 관련 모르겠다 → "평소 술 자리에서 몇 잔 정도 드세요?"
   우회 질문으로도 답 못 하면 그 축은 포기하고 RECOMMEND 로 넘어가라.
G. taste_profile/aroma_profile 강도는 대화 중 계속 업데이트되어야 한다. 초기 medium 을 대화 답변에 따라 high/low/zero 로 확정하거나, 새로운 축을 손님이 언급하면 추가해라.
G2. **비선호 맛/향 확인 (대화 중 1회 필수 — 빼먹으면 버그)** — RECOMMEND 로 가기 전에 **반드시** "싫어하거나 빼고 싶은 맛이나 향 있어요?" 류 질문을 한 번 해라. 아직 안 물어봤으면 다음 질문에 묶어서라도 해라. 손님이 언급한 싫은 축은 taste_profile/aroma_profile 에 `zero`(질색) 또는 `low`(별로) 로 넣어라. 예: "쓴맛은 별로" → taste_profile.bitter=low. "커피향 완전 싫어" → aroma_profile.coffee=zero. "없다/괜찮다"고 답하면 더 묻지 마라 (그 시점에 G2 충족).
H. **손님 발화 분류 (user_intent)** — 답하기 전에 먼저 손님이 방금 한 말이 어떤 종류인지 판단해라:
   - "SLOT": 취향/선호를 담은 답변 (예: "달달한 거 좋아", "도수는 약한 거로"). → 평소대로 추출 + 다음 질문.
   - "QUESTION": 손님이 너한테 되물음 (예: "칵테일이 따뜻하다는 게 뭐야?", "그게 무슨 맛이야?"). → reply 는 **먼저 그 질문에 1~2문장으로 직접 답하고** 그 다음에 원래 하려던 축 질문을 다시 이어라. extracted_slots 는 대부분 {}.
   - "UNKNOWN": 손님이 "모르겠다/몰라/딱히". → 우회 질문 (규칙 F).
   - "CORRECTION": 손님이 이전 추론을 부정 ("나 그런 말 한 적 없는데"). → 해당 슬롯을 null 로 지우고 사과 후 재질문 (규칙 B).
   - "STOP": "알아서 골라줘/그만/추천해줘". → action=RECOMMEND.
   - "OTHER": 잡담/인사 → 가볍게 받고 원래 축 질문으로 복귀.
   손님 질문(QUESTION)에 엉뚱한 답하지 마라. 되물으면 진짜 답부터 해라.

[행동 원칙]
1. 손님이 질문·반문하면 먼저 한 문장으로 진짜 내용을 담아 답한 뒤 질문해라. 회피 금지.
2. 손님의 구체 단어("친구들", "생일파티", "레몬에이드")를 꼭 한 번 짚어서 받아쳐라.
3. action=ASK 일 때 reply 구조 = **(손님 발화에 구체적 반응 1~2문장) + (질문 1개, 자연스럽게 끼워넣기)**. 반응 문장이 "좋으시네요!", "이해했습니다!" 같은 공허한 맞장구면 안 됨 — 손님이 말한 **구체 단어/상황**을 하나 짚어 되돌려줘라. 선다형("A/B/C 중에?")은 한 턴 건너 한 번 정도만. 그 외엔 자연스러운 open 질문 섞어라. "다음 질문은" / "추가로 여쭤볼게요" 같은 설문조사 접속사 금지.
4. action 선택:
   - ASK: 추천하기에 정보가 너무 얇고(자리·도수·맛방향 중 0~1 개) 다음 질문이 변별력 있을 때
   - RECOMMEND: 아래 중 하나
     · 자리·도수·맛/향 방향 중 2개 이상 이미 잡혔음
     · 남은 턴 2 이하
     · 손님이 "추천해줘/그만/알아서" 류
     · 직전 질문을 손님이 답 못/안 하고 같은 걸 또 물을 수밖에 없을 때
   RECOMMEND 일 때 reply 는 "그럼 그 느낌으로 골라와서 보여드릴게요" 로 마무리.

[출력 — JSON 한 덩어리만. 설명·마크다운·코드블록·이모지 금지]
{
  "extracted_slots": {},
  "user_intent": "SLOT" | "QUESTION" | "UNKNOWN" | "CORRECTION" | "STOP" | "OTHER",
  "action": "ASK" | "RECOMMEND",
  "reply": "손님한테 보여줄 2~3문장 친근한 존댓말 한국어"
}
※ extracted_slots 필드는 항상 {} 로 비워서 출력해라. 추출은 이미 외부에서 끝났고 네 값은 무시된다. reply / action / user_intent 만 의미 있다.

[extracted_slots 스키마]
- current_mood: "good"|"soso"|"bad"|null(지우기)
- party_purpose: "celebration"|"date"|"business"|"solo"|"hangout"|null
- taste_profile: {"sweet"|"sour"|"bitter"|"body"|"creamy"|"freshness": "zero"|"low"|"medium"|"high"}
- aroma_profile: {"woody"|"minty"|"fruity"|"citrus"|"floral"|"coffee"|"herbal": "zero"|"low"|"medium"|"high"}
- strength_preference: "zero"|"light"|"medium"|"strong"|null
- disliked_bases: ["whiskey"|"gin"|"rum"|"vodka"|"tequila"]
- favorite_drinks: [자유 문자열]

[강도 매핑]
- "확/완전/엄청/강하게/짱" → high
- "살짝/약간/조금/은은하게" → medium
- "적당히/과하지 않게" → medium
- "싫어/별로/안 땡겨" → low
- "완전 싫어/질색/알레르기" → zero
- "모르겠다/몰라/딱히/그냥 그래" → 해당 키 넣지 마라 (mood 포함)

반드시 위 JSON 스키마 한 덩어리만 출력. extracted_slots 는 항상 {} — 절대 값 채우지 마라.
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
    confirmed_taste = [k for k, v in taste.items() if v in _CONFIRMED_INTENSITIES]
    if confirmed_taste:
        items.append("맛 강도 확정: " + ", ".join(f"{k}={taste[k]}" for k in confirmed_taste))
    aroma = slots.get("aroma_profile") or {}
    confirmed_aroma = [k for k, v in aroma.items() if v in _CONFIRMED_INTENSITIES]
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


def _build_bartender_user_prompt(
    history: list[dict],
    slots: dict,
    user_msg: str,
    familiarity: Optional[str],
    remaining_turns: int,
    extracted_this_turn: Optional[dict] = None,
) -> str:
    last_q = _last_llm_question(history)
    # 이번 턴 추출을 반영한 preview 상태로 pending/missing/completion 계산 — 한 턴 지연 방지.
    slots_preview = merge_slots(slots, extracted_this_turn or {})
    locked = _locked_slots_description(slots_preview)
    pending = _pending_intensity_description(slots_preview)
    missing = _missing_slots_description(slots_preview)
    completion = _calc_effective_completion(slots_preview)
    this_turn = _format_extracted_this_turn(extracted_this_turn or {})
    return (
        f"[손님 친숙도] {familiarity or '알 수 없음'}\n"
        f"[남은 대화 턴] {remaining_turns} (총 {MAX_USER_TURNS})\n"
        f"[완성도] {completion}% (권장 추천 임계치 {MIN_RECOMMEND_COMPLETION}%)\n\n"
        f"[지금까지 파악한 취향]\n{_summarize_slots_for_prompt(slots)}\n\n"
        f"[이번 턴 추출된 슬롯 — 이미 확정됨. reply 에서 이 내용을 짚어줘라]\n{this_turn}\n\n"
        f"[이미 확정된 슬롯 — 절대 다시 묻지 마라, 덮어쓰기 금지]\n{locked}\n\n"
        f"[강도 미확정 — 이 축들만 물어라. 여기 없는 축은 묻지 마라]\n{pending}\n\n"
        f"[부족한 슬롯 — 이 중 하나를 우선 물어라]\n{missing}\n\n"
        f"[직전에 네가 한 질문]\n{last_q or '(없음 — 첫 턴)'}\n\n"
        f"[최근 대화]\n{_format_history(history)}\n\n"
        f"[손님의 방금 발화]\n{user_msg}\n\n"
        f"위 상황에서 바텐더로서 판단해라. extracted_slots 는 {{}} 로 비우고 reply/action/user_intent 만 채워라. JSON 한 덩어리만 출력."
    )


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
    try:
        from app.utils.model_loader import load_llm
        import torch

        # ─── Pass 1 : 슬롯 추출 전용 LLM 호출 ───────────────────────
        extracted_raw, extract_raw_text = _extract_slots_llm(history, user_msg)
        extracted = validate_extracted_slots(extracted_raw)
        extracted = _apply_rule_based_slot_guards(history, user_msg, extracted)
        extracted = _drop_unchanged_slots(extracted, slots)

        # ─── Pass 2 : reply/action 생성 (추출 결과 주입) ─────────────
        tokenizer, model = load_llm()

        remaining = max(MAX_USER_TURNS - user_turn_count, 0)
        messages = [
            {"role": "system", "content": _BARTENDER_SYSTEM_PROMPT},
            {"role": "user", "content": _build_bartender_user_prompt(
                history, slots, user_msg, familiarity, remaining,
                extracted_this_turn=extracted,
            )},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
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

        # RECOMMEND 게이트: 완성도가 임계치 미만이면 유저가 명시적으로 멈춘 게 아닌 한 ASK 강등
        if action == "RECOMMEND" and not _explicit_user_stop(user_msg):
            merged_preview = merge_slots(slots, extracted)
            if _calc_effective_completion(merged_preview) < MIN_RECOMMEND_COMPLETION:
                action = "ASK"
                user_intent = user_intent or "SLOT"

        # 비선호 질문 게이트: RECOMMEND 가기 전에 1회 "싫어하는 맛/향" 확인 필수.
        # 이미 체크된 신호가 하나라도 있으면 통과.
        if action == "RECOMMEND" and not _explicit_user_stop(user_msg):
            merged_preview = merge_slots(slots, extracted)
            if not _dislike_check_done(merged_preview, history):
                action = "ASK"
                user_intent = user_intent or "SLOT"
                parsed["_force_dislike_ask"] = True
        if user_intent not in ("SLOT", "QUESTION", "UNKNOWN", "CORRECTION", "STOP", "OTHER"):
            user_intent = "OTHER"

        reply = str(parsed.get("reply") or "").strip()
        if parsed.get("_force_dislike_ask"):
            reply = _DISLIKE_REPLY
        if not reply and generate_next_question:
            reply = "네 알겠습니다. 조금만 더 여쭤볼게요."

        should_stop = action == "RECOMMEND"
        stop_reason = "llm_recommend" if should_stop else ""

        if _explicit_user_stop(user_msg):
            should_stop = True
            stop_reason = "user_requested"

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
        }

    except Exception as e:
        logger.warning("LLM analyze_user_turn failed: %r", e, exc_info=True)
        return {
            "extracted_slots": {},
            "should_stop": False,
            "stop_reason": "",
            "next_question": "",
            "reply": "",
            "action": "ASK",
            "user_intent": "OTHER",
            "source": "fallback",
            "raw": "",
        }

def generate_opening_question() -> str:
    """대화 첫 질문 — 정해진 오프너. (LLM 호출 비용 아끼려고 고정)"""
    return (
        "안녕하세요! 당신만의 맞춤 바텐더입니다. "
        "취향을 몇 가지 여쭤보고 딱 맞는 칵테일 골라드릴게요. "
        "오늘은 어떤 자리에서 드시는 거예요?"
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
            base.update(extracted[key])
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

    # 초기 태그는 손님이 체크한 선호 "축"일 뿐, 강도는 대화로 조정한다.
    # 따라서 INTENSITY_PENDING 으로 시드해서 "seed-only" 와 "대화로 확정된 medium" 을
    # 명확히 구분한다. LLM 은 대화에서 high/medium/low/zero 중 하나로 덮어쓴다.
    taste_profile: dict[str, str] = {}
    for tag in getattr(tag_row, "taste_tags_json", None) or []:
        mapped = taste_map.get(tag)
        if mapped:
            taste_profile[mapped] = INTENSITY_PENDING
    if taste_profile:
        seeded["taste_profile"] = taste_profile

    aroma_profile: dict[str, str] = {}
    for tag in getattr(tag_row, "aroma_tags_json", None) or []:
        mapped = aroma_map.get(tag)
        if mapped:
            aroma_profile[mapped] = INTENSITY_PENDING
    if aroma_profile:
        seeded["aroma_profile"] = aroma_profile

    return seeded


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
        "space": space,
        "merged_slots": merged_slots,
        "effective_completion": _calc_effective_completion(merged_slots),
    }