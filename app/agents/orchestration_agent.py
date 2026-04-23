from __future__ import annotations

import json
import logging
import re
from typing import Optional
from sqlalchemy.orm import Session

from app.db.models import Cocktail, Ingredient
from app.db.crud import (
    create_sample_recommendation,
    get_all_recipes_with_ingredients,
    get_available_ingredient_ids,
    get_sample_recommendation,
    create_sample_feedback,
    create_final_recommendation,
    update_preference_vector,
    list_feedbacks_by_sample_recommendation,
    list_recommended_cocktail_ids_by_guest,
)
from app.agents.preference_agent import (
    analyze_feedback,
    build_user_profile,
)
from app.agents.output_agent import generate_recipe_snapshot
from app.utils.model_loader import embed_texts

logger = logging.getLogger(__name__)

# ============================================================
# 매핑 테이블
# ============================================================

# taste_profile 키 → Cocktail 컬럼
TASTE_TO_COCKTAIL = {
    "sweet": "sweet_level",
    "sour": "sour_level",
    "bitter": "bitter_level",
    "body": "body_level",
    "creamy": "creamy_level",
    "freshness": "freshness_level",
}

# aroma_profile 키 → Ingredient 컬럼
AROMA_TO_INGREDIENT = {
    "minty": "minty_score",
    "fruity": "fruity_score",
    "citrus": "citrus_score",
    "herbal": "herbal_score",
    "coffee": "coffee_score",
    "woody": "woody_score",
    "floral": "floral_score",
}

STRENGTH_RANGE = {
    "zero": (0.0, 0.0),
    "light": (0.0, 2.0),
    "medium": (1.5, 3.5),
    "strong": (3.0, 5.0),
}

# 도수 선호별 이상치 (0~5 스케일). score_cocktail 에서 대칭 가중치 계산에 사용.
# (zero 는 is_non_alcoholic 하드필터가 잡으므로 여기서는 비-zero 만 정의.)
STRENGTH_TARGET = {"light": 1.0, "medium": 2.5, "strong": 4.0}

# intensity → 점수 스케일
INTENSITY_WEIGHT = {"low": 0.2, "medium": 0.5, "high": 1.0}
# low/medium/high 모두 "선호"로 취급(가점). zero만 하드 필터로 제외.
HIGH_THRESHOLD_COL_VALUE = 3.5
AROMA_ING_THRESHOLD = 3.0

# ============================================================
# helper
# ============================================================

def _normalize(user: float, cocktail: float, max_range: float = 4.0) -> float:
    return 1.0 - abs(user - cocktail) / max_range


def _ingredient_name(ingredient: Ingredient) -> str:
    return getattr(ingredient, "ingredients_name", None) or getattr(ingredient, "name_kr", "")


def _vector_row_to_dict(vector_row) -> dict[str, float]:
    return {
        "sweetness_score": float(vector_row.sweetness_score),
        "bitterness_score": float(vector_row.bitterness_score),
        "sourness_score": float(vector_row.sourness_score),
        "freshness_score": float(vector_row.freshness_score),
        "body_score": float(vector_row.body_score),
        "herbal_score": float(vector_row.herbal_score),
        "citrus_score": float(vector_row.citrus_score),
        "alcohol_score": float(vector_row.alcohol_score),
    }

def _get_cocktail_strength_value(cocktail: Cocktail) -> Optional[float]:
    """칵테일 자체의 도수/강도 점수(0~5 스케일) 추출.

    모델 스키마가 프로젝트마다 다를 수 있어서 후보 컬럼을 순서대로 확인한다.
    0~5 범위를 벗어나는 값(예: raw ABV 18, 40 등)은 현재 score 체계와 다르므로 사용하지 않는다.
    """
    for attr in ("alcohol_score", "strength_score", "strength_level", "alcohol_level"):
        raw = getattr(cocktail, attr, None)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if 0.0 <= value <= 5.0:
            return value
    return None

def _build_reason_parts(
    cocktail: Cocktail,
    profile: dict,
    recipe_ingredients: list[tuple],
) -> list[str]:
    merged = profile["merged_slots"]
    space = profile["space"]
    reasons: list[str] = []

    taste_profile: dict = merged.get("taste_profile") or {}
    matched_tastes: list[str] = []
    for tag, intensity in taste_profile.items():
        if intensity not in ("low", "medium", "high"):
            continue
        col = TASTE_TO_COCKTAIL.get(tag)
        if col and getattr(cocktail, col) is not None:
            if float(getattr(cocktail, col)) >= HIGH_THRESHOLD_COL_VALUE:
                matched_tastes.append(tag)
    if matched_tastes:
        reasons.append(f"선호 맛과 일치: {', '.join(matched_tastes)}")

    aroma_profile: dict = merged.get("aroma_profile") or {}
    matched_aromas: list[str] = []
    for _, ingredient in recipe_ingredients:
        for tag, intensity in aroma_profile.items():
            if intensity not in ("low", "medium", "high"):
                continue
            col = AROMA_TO_INGREDIENT.get(tag)
            if col and getattr(ingredient, col, 0) >= AROMA_ING_THRESHOLD and tag not in matched_aromas:
                matched_aromas.append(tag)
    if matched_aromas:
        reasons.append(f"선호 향과 일치: {', '.join(matched_aromas)}")

    if space and cocktail.mood_tag:
        mood_prob = (space.mood_tags_json or {}).get(cocktail.mood_tag, 0.0)
        if mood_prob >= 0.5:
            reasons.append(f"공간 무드와 어울림: {cocktail.mood_tag}")

    strength_pref = merged.get("strength_preference")
    cocktail_strength = _get_cocktail_strength_value(cocktail)
    if strength_pref in STRENGTH_RANGE and cocktail_strength is not None:
        low, high = STRENGTH_RANGE[strength_pref]
        if low <= cocktail_strength <= high:
            reasons.append(f"도수 선호 반영: {strength_pref}")

    if merged.get("favorite_drinks"):
        reasons.append(f"선호 음료 참고: {', '.join(merged['favorite_drinks'][:3])}")

    if not reasons:
        reasons.append("기본 취향 점수 기반 추천")

    return reasons


# ============================================================
# RAG: 쿼리 합성 + pgvector 검색 + LLM 리랭크
# ============================================================

_RAG_TASTE_KR = {
    "sweet": "단맛", "sour": "신맛", "bitter": "쓴맛",
    "body": "바디감", "creamy": "크리미", "freshness": "청량감",
}
_RAG_AROMA_KR = {
    "minty": "민트", "fruity": "과일 향", "citrus": "시트러스",
    "herbal": "허브", "coffee": "커피", "woody": "우디", "floral": "플로럴",
}
_RAG_INTENSITY_KR = {
    "high": "강하게 선호",
    "medium": "적당히 선호",
    "low": "약하게 선호",
    "zero": "완전 비선호(제외)",
}

# 칵테일의 해당 축 level >= 이 임계값이면 "두드러진다"고 본다.
_ZERO_EXCLUDE_THRESHOLD = 3.5

# 향 축은 Cocktail 컬럼이 없어 재료 기반 판단 → 일단 description/레시피 감각노트 스캔으로 근사.
# 현재 구조에서 aroma zero 는 rerank LLM 단계에서 강하게 감점 처리하게 둔다.
_RAG_STRENGTH_KR = {"zero": "논알콜(무알콜)", "light": "가벼운 도수", "medium": "중간 도수", "strong": "강한 도수"}
_RAG_PURPOSE_KR = {
    "celebration": "축하 자리",
    "date": "데이트 자리",
    "business": "비즈니스 자리",
    "solo": "혼자 마시는 자리",
    "hangout": "친구들과 가볍게 마시는 자리",
}
_RAG_MOOD_KR = {
    "good": "기분 좋음",
    "soso": "그냥저냥",
    "bad": "기분이 가라앉음",
}

# 피드백 ADJUST 로 갱신된 vector 의 편차를 쿼리에 반영.
_VEC_AXIS_KR = {
    "sweetness_score": "단맛",
    "bitterness_score": "쓴맛",
    "sourness_score": "신맛",
    "freshness_score": "청량감",
    "body_score": "바디감",
    "herbal_score": "허브",
    "citrus_score": "시트러스",
    "alcohol_score": "도수",
}
_VEC_AXIS_DEFAULT = {
    "sweetness_score": 3.0, "bitterness_score": 3.0, "sourness_score": 3.0,
    "freshness_score": 3.0, "body_score": 3.0,
    "herbal_score": 2.0, "citrus_score": 2.0, "alcohol_score": 3.0,
}


def _vec_deviation_bits(vector) -> list[str]:
    """vector 가 기본값에서 얼마나 벌어졌는지 자연어 조각으로 변환.

    ADJUST 피드백 이후 RAG 쿼리가 조정 방향을 반영하게 만들기 위함.
    """
    if vector is None:
        return []
    out: list[str] = []
    for field, label in _VEC_AXIS_KR.items():
        val = getattr(vector, field, None)
        if val is None:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        delta = v - _VEC_AXIS_DEFAULT[field]
        if abs(delta) < 0.3:
            continue
        mag = abs(delta)
        strength = "크게" if mag >= 0.7 else ("꽤" if mag >= 0.4 else "살짝")
        direction = "강조" if delta > 0 else "약화"
        out.append(f"{label} {strength} {direction}")
    return out


def _rag_collect_profile(profile_dict: dict, label_map: dict[str, str]) -> list[str]:
    """RAG 쿼리에 넣을 선호만 추림.

    - pending: 미확정 → 제외
    - zero: 하드필터가 잡으므로 자연어 쿼리엔 불필요 → 제외
    - low: 약한 선호를 자연어로 강조하면 retrieval 노이즈만 됨 → 제외
    - medium / high 만 쿼리에 반영
    """
    out = []
    for tag, intensity in (profile_dict or {}).items():
        if intensity not in ("medium", "high"):
            continue
        label = label_map.get(tag, tag)
        lvl = _RAG_INTENSITY_KR.get(intensity, intensity)
        out.append(f"{label} {lvl}")
    return out


def synthesize_query(profile: dict) -> str:
    """build_user_profile 결과 → 자연어 쿼리 한 덩어리."""
    merged = profile.get("merged_slots") or {}
    space = profile.get("space")
    parts: list[str] = []

    purpose = merged.get("party_purpose")
    mood = merged.get("current_mood")
    if purpose or mood:
        ctx = ", ".join(
            v for v in [
                _RAG_PURPOSE_KR.get(purpose, purpose),
                _RAG_MOOD_KR.get(mood, mood),
            ] if v
        )
        parts.append(f"상황/무드: {ctx}")

    taste_bits = _rag_collect_profile(merged.get("taste_profile") or {}, _RAG_TASTE_KR)
    if taste_bits:
        parts.append(f"맛 선호: {', '.join(taste_bits)}")

    aroma_bits = _rag_collect_profile(merged.get("aroma_profile") or {}, _RAG_AROMA_KR)
    if aroma_bits:
        parts.append(f"향 선호: {', '.join(aroma_bits)}")

    strength = merged.get("strength_preference")
    if strength:
        parts.append(f"도수: {_RAG_STRENGTH_KR.get(strength, strength)}")

    disliked_bases = merged.get("disliked_bases") or []
    if disliked_bases:
        # dense query에는 부정 조건을 넣지 않고 하드필터 단계에만 맡긴다.
        pass

    favs = merged.get("favorite_drinks") or []
    if favs:
        parts.append(f"유사 선호 음료: {', '.join(favs[:3])}")

    vec_bits = _vec_deviation_bits(profile.get("vector"))
    if vec_bits:
        parts.append(f"피드백 반영 조정: {', '.join(vec_bits)}")

    if space and getattr(space, "mood_tags_json", None):
        top_moods = sorted(space.mood_tags_json.items(), key=lambda x: x[1], reverse=True)[:3]
        mood_str = ", ".join(k for k, _ in top_moods if k)
        if mood_str:
            parts.append(f"공간 무드: {mood_str}")

    return "\n".join(parts) if parts else "일반적인 칵테일 추천"


def retrieve_candidates(
    db: Session,
    query_text: str,
    top_k: int = 20,
    exclude_ids: Optional[list[int]] = None,
    strength_preference: Optional[str] = None,
) -> list[tuple[Cocktail, float]]:
    """쿼리 텍스트 임베딩 → cocktails.embedding 코사인 거리로 top_k.

    strength_preference="zero"면 논알콜만, 그 외(또는 None)이면 알콜만 반환.
    """
    # 코퍼스(build_cocktail_embeddings.py)가 instruction prefix 없이 인코딩되므로
    # 쿼리도 동일하게 인코딩해서 임베딩 공간 대칭성 유지 (R2 fix).
    vec = embed_texts(
        [query_text],
        batch_size=1,
        max_length=512,
    )[0].tolist()

    distance = Cocktail.embedding.cosine_distance(vec).label("distance")
    q = (
        db.query(Cocktail, distance)
        .filter(Cocktail.is_active.is_(True))
        .filter(Cocktail.embedding.isnot(None))
    )
    if strength_preference == "zero":
        q = q.filter(Cocktail.is_non_alcoholic.is_(True))
    else:
        q = q.filter(Cocktail.is_non_alcoholic.is_not(True))
    if exclude_ids:
        q = q.filter(Cocktail.cocktail_id.notin_(exclude_ids))

    rows = q.order_by(distance.asc()).limit(top_k).all()
    return [(c, float(d)) for c, d in rows]


_RERANK_SYSTEM_PROMPT = """
너는 칵테일 추천 시스템의 리랭커다.
사용자 선호 프로파일과 후보 칵테일 목록이 주어진다.
후보 중 사용자에게 가장 잘 맞는 칵테일을 최대 K개 선택하고, 각 선택에 대해 한 줄 한국어 이유를 써라.
반드시 JSON 객체 하나만 출력. 설명·마크다운·코드블록 금지.

★판단 기준 (우선순위 순):
1) 맛 프로파일 일치
   - "zero"가 붙은 축은 해당 축이 두드러진 칵테일을 **절대 포함하지 마라**(이미 하드필터되지만 ranked에도 넣지 말 것).
   - "low"는 비선호가 아니라 **약하게만 느껴지면 좋다**는 뜻이다.
     해당 축이 과하게 두드러지는 후보는 감점하고, 은은하거나 약한 수준이면 허용하라.
     예: creamy=low이면 creamy_level이 높은 칵테일(아이리시 커피·포르토 플립 등)을 top에 넣지 마라.
   - "high"는 해당 축이 3.5 이상, "medium"은 2~3.5, "low"는 1~2.5 정도가 바람직.
   - 향 축도 동일: aroma_profile 에 zero 가 있으면 해당 향 계열(description·카테고리 기반) 칵테일은 배제.
2) 향 프로파일 일치 — 향 키는 서로 다른 개념이니 혼동하지 마라.
   - fruity(열대/베리/복숭아/사과 등 과일) ≠ citrus(레몬/라임/자몽 등 시트러스)
   - herbal(허브/약초) ≠ minty(민트 특유 청량)
   - woody(오크/스모키) ≠ coffee(커피/로스팅)
   fruity=high를 요구했으면 레몬·라임 중심 "시트러스 사워"류보다는 "과일계/티키/트로피컬"계를 우선하라.
3) 도수 선호 — strength_preference=strong이면 alcohol_score 3.5 이상(스피릿포워드·강한 칵테일) 우선.
   strong 요구에 medium 도수(알콜 2~3)를 top에 넣지 마라.
4) 공간 무드와의 조화.
5) favorite_drinks와 계열/베이스/향이 유사하면 가점.

절대 기준:
- disliked_bases는 이미 필터된 상태라 고려할 필요 없음.
- reason에 거짓을 쓰지 마라. 후보 데이터에 없는 속성(예: 사용자가 low라 했는데 "크리미한 질감이 적당")을 근거로 들지 마라. 실제 후보의 taste/카테고리/description에 근거한 이유만 써라.

출력 형식:
{
  "ranked": [
    {"cocktail_id": <int>, "reason": "<한국어 2~3문장>"},
    ...
  ]
}
ranked는 사용자 적합도 내림차순. 최대 K개.

[reason 작성 규칙 — 중요]
- 2~3문장, 80~180자 **순수 한국어**.
- ★ 영어 단어·슬롯키·enum 절대 금지. 따옴표로 감싼 영어도 금지.
  금지 예: "sweet", "citrus", "good", "strong", 'fruity', 'mood', 'taste'
  허용 예: 단맛, 시트러스향, 기분 좋을 때, 강한 도수, 과일향
  (알파벳이 섞이면 즉시 실패로 간주하고 처음부터 한국어로만 다시 써라.)
- 단답형("~해서 적합합니다") 금지.
- 반드시 다음 3요소를 모두 녹여라:
  (1) 이 칵테일이 어떤 맛/향/도수 캐릭터인지 짧은 묘사 (예: "라임의 산미와 진의 허브향이 깔끔하게 맞물리는 드라이한 스타일")
  (2) 도수·강도 느낌 — "가볍게 마시기 좋음 / 중간 정도 바디 / 스피릿포워드로 묵직함" 같은 표현을 넣어라.
  (3) 사용자의 선호(맛/향/도수/분위기) 중 어느 부분과 맞는지 연결.
- 후보 설명(description)에 있는 풍미·질감 단어를 적극 활용하되, 없는 속성은 지어내지 마라.
- reason 은 **네가 선택한 그 칵테일**에 대한 설명이다. 다른 칵테일 이름(사이드카·모히토 등)을 reason 안에 쓰지 마라.
- "사용자의 선호에 부합합니다" 같은 상투어로 문장 끝내지 마라. 실제 칵테일 캐릭터를 설명하는 톤으로 써라.
""".strip()


_PURPOSE_KO = {
    "celebration": "축하 자리", "date": "데이트", "business": "비즈니스",
    "solo": "혼술", "hangout": "친구들과 편하게",
}
_MOOD_KO = {"good": "기분 좋음", "soso": "그냥저냥", "bad": "다운"}
_STRENGTH_KO = {"zero": "무알콜", "light": "가볍게", "medium": "중간", "strong": "강한 도수"}
_INTENSITY_KO = {"zero": "배제", "low": "은은하게", "medium": "적당히", "high": "강하게"}
_TASTE_KO = {"sweet": "단맛", "sour": "신맛", "bitter": "쓴맛",
             "body": "바디감", "creamy": "크리미", "freshness": "청량감"}
_AROMA_KO = {"woody": "우디향", "minty": "민트향", "fruity": "과일향",
             "citrus": "시트러스향", "floral": "꽃향", "coffee": "커피향", "herbal": "허브향"}

# vibe/mood tag 영어→한국어 — DB mood_tag / space.mood_tags_json 방어용.
# 이미 한국어면 mapping 없음 → 그대로 통과. 미지의 영어 토큰은 비노출(drop).
_MOOD_TAG_KO = {
    "modern": "모던한", "dark": "어두운", "bright": "밝은", "neon": "네온",
    "romantic": "로맨틱한", "casual": "캐주얼", "energetic": "신나는",
    "classic": "클래식", "cozy": "아늑한", "chic": "시크한",
    "tropical": "트로피컬", "summer": "여름", "elegant": "세련된",
    "sophisticated": "세련된", "light": "가벼운", "subtle": "은은한",
    "vivid": "화려한", "calm": "차분한", "warm": "따뜻한",
    "celebration": "축하",
}


def _ko_mood_tag(tag: str) -> Optional[str]:
    """영어면 매핑 시도, 실패 시 None (비노출). 한국어(비-ASCII 포함)는 그대로 반환."""
    if not tag:
        return None
    s = str(tag).strip()
    if not s:
        return None
    # 한글/비ASCII 문자가 하나라도 있으면 한국어로 간주 → 그대로
    if any(ord(ch) > 127 for ch in s):
        return s
    mapped = _MOOD_TAG_KO.get(s.lower())
    return mapped  # 알 수 없는 영어면 None → 노출 안 함


def _ko_taste_line(tp: dict) -> str:
    if not isinstance(tp, dict) or not tp:
        return "(없음)"
    parts = [f"{_TASTE_KO.get(k, k)} {_INTENSITY_KO.get(v, v)}"
             for k, v in tp.items() if v in _INTENSITY_KO]
    return ", ".join(parts) if parts else "(없음)"


def _ko_aroma_line(ap: dict) -> str:
    if not isinstance(ap, dict) or not ap:
        return "(없음)"
    parts = [f"{_AROMA_KO.get(k, k)} {_INTENSITY_KO.get(v, v)}"
             for k, v in ap.items() if v in _INTENSITY_KO]
    return ", ".join(parts) if parts else "(없음)"


def _format_profile_for_rerank(profile: dict) -> str:
    merged = profile.get("merged_slots") or {}
    lines = [
        f"자리: {_PURPOSE_KO.get(merged.get('party_purpose'), '—')}",
        f"기분: {_MOOD_KO.get(merged.get('current_mood'), '—')}",
        f"맛 선호: {_ko_taste_line(merged.get('taste_profile') or {})}",
        f"향 선호: {_ko_aroma_line(merged.get('aroma_profile') or {})}",
        f"도수: {_STRENGTH_KO.get(merged.get('strength_preference'), '—')}",
    ]
    favs = merged.get("favorite_drinks") or []
    if favs:
        lines.append(f"즐겨 마시는 술: {', '.join(favs)}")
    space = profile.get("space")
    if space and getattr(space, "mood_tags_json", None):
        top_moods = sorted(space.mood_tags_json.items(), key=lambda x: x[1], reverse=True)[:3]
        ko_moods = [ko for ko in (_ko_mood_tag(k) for k, _ in top_moods) if ko]
        if ko_moods:
            lines.append(f"공간 무드: {', '.join(ko_moods)}")
    return "\n".join(lines)


def _aggregate_aroma_from_ingredients(recipe_items: list[tuple]) -> dict[str, float]:
    """cocktail의 recipe ingredient들에서 aroma 축별 최댓값 집계.

    recipe_items: [(Recipe, Ingredient), ...] — crud.get_all_recipes_with_ingredients 반환값의 값.
    반환: {"fruity": 3.0, "citrus": 2.0, ...} — AROMA_TO_INGREDIENT 키 중 >0 인 것만.
    """
    agg: dict[str, float] = {}
    for _recipe, ing in recipe_items or []:
        for aroma_key, col in AROMA_TO_INGREDIENT.items():
            raw = getattr(ing, col, None)
            if raw is None:
                continue
            v = float(raw)
            if v > agg.get(aroma_key, 0.0):
                agg[aroma_key] = v
    return agg


def _format_candidates_for_rerank(
    candidates: list[Cocktail],
    recipe_ingredients: Optional[dict[int, list[tuple]]] = None,
) -> str:
    rows = []
    for c in candidates:
        taste = []
        for ko, col in [("단맛", "sweet_level"), ("신맛", "sour_level"),
                        ("쓴맛", "bitter_level"), ("바디감", "body_level"),
                        ("청량감", "freshness_level"), ("크리미", "creamy_level")]:
            v = getattr(c, col, None)
            if v is not None:
                taste.append(f"{ko} {float(v):.1f}")

        strength_val = _get_cocktail_strength_value(c)
        strength_str = f"도수 {strength_val:.1f}/5" if strength_val is not None else "도수 미기재"

        aroma_str = ""
        if recipe_ingredients is not None:
            agg = _aggregate_aroma_from_ingredients(recipe_ingredients.get(c.cocktail_id, []))
            if agg:
                parts = [f"{_AROMA_KO.get(k, k)} {v:.1f}"
                         for k, v in sorted(agg.items(), key=lambda x: -x[1]) if v >= 2.0]
                if parts:
                    aroma_str = f" | 향(재료기준): {', '.join(parts)}"

        mood_str = _ko_mood_tag(c.mood_tag) or "—"
        desc = (c.description or "").replace("\n", " ")[:220]
        rows.append(
            f"- id={c.cocktail_id} | {c.name_kr} | {c.category} | "
            f"무드: {mood_str} | {strength_str} | 맛: {', '.join(taste)}{aroma_str}\n"
            f"    설명: {desc}"
        )
    return "\n".join(rows)


def _extract_json_object(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# rerank reason 검증 ────────────────────────────────────────────
# LLM 이 프롬프트 지시를 어기고 영어 토큰을 흘리거나, 다른 후보의 이름을
# reason 안에 써넣는 케이스를 잡아낸다. 실패 시 fallback 으로 대체.

_REASON_ENGLISH_RE = re.compile(r"[A-Za-z]{3,}")
_REASON_ENGLISH_ALLOW = {"pm", "am"}  # 시간 표기 등 예외. (3자 미만은 정규식이 이미 통과)


def _build_fallback_reason(cocktail: Cocktail, profile: dict) -> str:
    merged = (profile or {}).get("merged_slots") or {}
    parts: list[str] = []
    tp = merged.get("taste_profile") or {}
    if isinstance(tp, dict):
        confirmed = [_TASTE_KO[k] for k, v in tp.items()
                     if k in _TASTE_KO and v in ("high", "medium")]
        if confirmed:
            parts.append(", ".join(confirmed[:3]) + " 선호")
    ap = merged.get("aroma_profile") or {}
    if isinstance(ap, dict):
        confirmed = [_AROMA_KO[k] for k, v in ap.items()
                     if k in _AROMA_KO and v in ("high", "medium")]
        if confirmed:
            parts.append(", ".join(confirmed[:2]) + " 취향")
    strength = _STRENGTH_KO.get(merged.get("strength_preference"))
    if strength and strength != "—":
        parts.append(f"{strength} 도수")
    lead = ", ".join(parts) if parts else "선호 프로파일"
    return f"{cocktail.name_kr} 는 {lead} 와 잘 맞습니다."


def _validate_rerank_reason(
    reason: str,
    chosen: Cocktail,
    all_candidates: list[Cocktail],
) -> tuple[bool, str]:
    """reason 검증. (ok, fail_reason) 반환. ok=False 면 호출측이 fallback 대체."""
    r = (reason or "").strip()
    if len(r) < 10:
        return False, "too_short"

    # 영어 토큰 누출
    for m in _REASON_ENGLISH_RE.finditer(r):
        tok = m.group(0).lower()
        if tok in _REASON_ENGLISH_ALLOW:
            continue
        return False, f"english_token:{tok}"

    # 다른 후보 이름 누출 (자기 자신 제외)
    chosen_id = chosen.cocktail_id
    chosen_name_kr = (chosen.name_kr or "").strip()
    for c in all_candidates:
        if c.cocktail_id == chosen_id:
            continue
        other_kr = (c.name_kr or "").strip()
        other_en = (getattr(c, "name_en", None) or "").strip()
        if other_kr and len(other_kr) >= 3 and other_kr != chosen_name_kr and other_kr in r:
            return False, f"other_name_kr:{other_kr}"
        if other_en and len(other_en) >= 4 and other_en.lower() in r.lower():
            return False, f"other_name_en:{other_en}"

    return True, ""


def rerank_with_llm(
    profile: dict,
    candidates: list[Cocktail],
    k: int = 3,
    max_new_tokens: int = 512,
    recipe_ingredients: Optional[dict[int, list[tuple]]] = None,
) -> Optional[list[dict]]:
    """LLM 후보 리랭킹. 실패 시 None → 호출측 fallback.

    recipe_ingredients 가 주어지면 각 후보에 ingredient 기반 aroma max 집계를 포함해
    LLM 이 향 매칭을 수치로 판단할 수 있게 한다. None 이면 description 텍스트만으로 리랭크.
    """
    if not candidates:
        return []
    try:
        from app.utils.model_loader import load_llm, build_chat_prompt
        import torch

        tokenizer, model = load_llm()
        user_content = (
            f"사용자 프로파일:\n{_format_profile_for_rerank(profile)}\n\n"
            f"후보 칵테일 (N={len(candidates)}):\n"
            f"{_format_candidates_for_rerank(candidates, recipe_ingredients)}\n\n"
            f"위 후보 중에서 사용자에게 가장 잘 맞는 상위 {k}개를 ranked로 JSON 반환해라."
        )
        messages = [
            {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        rendered = build_chat_prompt(
            tokenizer, messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
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

        ranked_raw = parsed.get("ranked") or []
        if not isinstance(ranked_raw, list):
            return None

        valid_ids = {c.cocktail_id for c in candidates}
        id_to_cocktail = {c.cocktail_id: c for c in candidates}
        out_rows: list[dict] = []
        seen: set[int] = set()
        for item in ranked_raw:
            if not isinstance(item, dict):
                continue
            try:
                cid = int(item.get("cocktail_id"))
            except (TypeError, ValueError):
                continue
            if cid not in valid_ids or cid in seen:
                continue
            reason = str(item.get("reason") or "").strip()
            chosen = id_to_cocktail[cid]
            ok, fail_reason = _validate_rerank_reason(reason, chosen, candidates)
            if not ok:
                logger.warning(
                    "rerank reason rejected cid=%s fail=%s raw=%r",
                    cid, fail_reason, reason[:120],
                )
                reason = _build_fallback_reason(chosen, profile)
            out_rows.append({"cocktail_id": cid, "reason": reason})
            seen.add(cid)
            if len(out_rows) >= k:
                break

        return out_rows or None

    except Exception as e:
        logger.warning("LLM rerank failed: %r", e, exc_info=True)
        return None


# ============================================================
# 1. 후보 칵테일 조회
# ============================================================

def get_candidate_cocktails(
    db: Session,
    exclude_ids: Optional[list[int]] = None,
) -> list[Cocktail]:
    query = db.query(Cocktail).filter(Cocktail.is_active.is_(True))
    if exclude_ids:
        query = query.filter(Cocktail.cocktail_id.notin_(exclude_ids))
    return query.all()


# ============================================================
# 3. 비선호 베이스
# ============================================================

# 영문 베이스 enum → 한국어 재료명 후보
_BASE_EN_TO_KR_NAMES = {
    "whiskey": ("위스키", "whiskey", "whisky", "wh-"),
    "gin": ("진", "gin"),
    "rum": ("럼", "rum"),
    "vodka": ("보드카", "vodka"),
    "tequila": ("테킬라", "tequila"),
}


def _ingredient_matches_base(ingredient_name: str, base_en: str) -> bool:
    if not ingredient_name:
        return False
    name = ingredient_name.lower()
    for kw in _BASE_EN_TO_KR_NAMES.get(base_en, ()):
        if kw.lower() in name:
            return True
    return False


def _has_zero_taste_conflict(merged_slots: dict, cocktail) -> bool:
    """taste_profile 에 zero 로 마킹된 축이 칵테일에서 두드러지면 배제."""
    taste = (merged_slots or {}).get("taste_profile") or {}
    for axis, level in taste.items():
        if level != "zero":
            continue
        col = TASTE_TO_COCKTAIL.get(axis)
        if not col:
            continue
        val = getattr(cocktail, col, None)
        if val is None:
            continue
        if float(val) >= _ZERO_EXCLUDE_THRESHOLD:
            return True
    return False


def _has_disliked_base(merged_slots: dict, recipe_ingredients: list[tuple]) -> bool:
    disliked_bases = merged_slots.get("disliked_bases") or []
    if not disliked_bases:
        return False

    for _, ingredient in recipe_ingredients:
        if ingredient.ingredient_type != "BASE":
            continue
        name = _ingredient_name(ingredient)
        for base_en in disliked_bases:
            if _ingredient_matches_base(name, base_en):
                return True
    return False


def _is_unstockable(
    recipe_ingredients: list[tuple],
    available_ingredient_ids: Optional[set[int]],
) -> bool:
    """필수(비선택) 재료 중 하나라도 재고 부족이면 제조 불가.

    available_ingredient_ids=None (재고 데이터 미세팅) → 필터 비활성화.
    """
    if available_ingredient_ids is None:
        return False
    if not recipe_ingredients:
        return True
    for recipe, ingredient in recipe_ingredients:
        if recipe.is_optional:
            continue
        if ingredient.ingredient_id not in available_ingredient_ids:
            return True
    return False


# ============================================================
# 4. 칵테일 점수 계산
# ============================================================

def score_cocktail(
    cocktail: Cocktail,
    profile: dict,
    recipe_ingredients: list[tuple],
) -> float:
    merged = profile["merged_slots"]
    vector = profile["vector"]
    space = profile["space"]
    score = 0.0

    # 1. 벡터 유사도
    pairs = [
        (vector.sweetness_score, cocktail.sweet_level, 4.0, 20),
        (vector.sourness_score, cocktail.sour_level, 4.0, 15),
        (vector.bitterness_score, cocktail.bitter_level, 4.0, 10),
        (vector.freshness_score, cocktail.freshness_level, 4.0, 10),
        (vector.body_score, cocktail.body_level, 4.0, 8),
    ]
    for user_s, cocktail_s, max_r, weight in pairs:
        if cocktail_s is None:
            continue
        score += _normalize(float(user_s), float(cocktail_s), max_r) * weight

    # 2. 맛 프로파일 (intensity 가중)
    taste_profile: dict = merged.get("taste_profile") or {}
    for tag, intensity in taste_profile.items():
        col = TASTE_TO_COCKTAIL.get(tag)
        if not col:
            continue
        val = getattr(cocktail, col)
        if val is None:
            continue
        val = float(val)
        if intensity == "high":
            if val >= 3.5:
                score += 10
            elif val <= 1.5:
                score -= 12
        elif intensity == "medium":
            if val >= 3.0:
                score += 5
        elif intensity == "low":
            # 약하게 선호 → 은은하면 좋지만, 강하게 두드러지면 오히려 감점.
            if 1.0 <= val <= 2.5:
                score += 1
            elif val >= 3.5:
                score -= 4

    # 3. 향 프로파일 (intensity 가중)
    aroma_profile: dict = merged.get("aroma_profile") or {}
    for _, ingredient in recipe_ingredients:
        for tag, intensity in aroma_profile.items():
            col = AROMA_TO_INGREDIENT.get(tag)
            if not col:
                continue
            ing_val = getattr(ingredient, col, 0) or 0
            ing_val = float(ing_val)
            if intensity == "high" and ing_val >= AROMA_ING_THRESHOLD:
                score += 8
            elif intensity == "medium" and ing_val >= AROMA_ING_THRESHOLD:
                score += 3
            elif intensity == "low" and ing_val >= (AROMA_ING_THRESHOLD + 1.0):
                score -= 1

    # 4. 공간 무드 보너스
    if space and cocktail.mood_tag:
        mood_prob = (space.mood_tags_json or {}).get(cocktail.mood_tag, 0.0)
        score += mood_prob * 15

    # 7. 도수 선호 — 이상치(STRENGTH_TARGET) 와의 편차로 대칭 가중.
    strength_pref = merged.get("strength_preference")
    cocktail_strength = _get_cocktail_strength_value(cocktail)
    if strength_pref in STRENGTH_TARGET and cocktail_strength is not None:
        dev = abs(cocktail_strength - STRENGTH_TARGET[strength_pref])
        if dev <= 0.5:
            score += 15
        elif dev <= 1.0:
            score += 8
        elif dev <= 1.5:
            score += 0
        elif dev <= 2.0:
            score -= 8
        else:
            score -= 15

    return round(score, 2)


# ============================================================
# 5. Top-K 추천
# ============================================================

RAG_RETRIEVE_N = 20


def recommend_top_k(
    db: Session,
    guest_session_id: str,
    k: int = 3,
    exclude_ids: Optional[list[int]] = None,
) -> list[dict]:
    """RAG 파이프라인: 임베딩 검색 → 하드 필터 → LLM 리랭크.

    LLM 리랭크 실패 시 score_cocktail 기반으로 fallback.
    """
    profile = build_user_profile(db, guest_session_id)
    merged_slots = profile["merged_slots"]

    all_ri = get_all_recipes_with_ingredients(db)
    available_ids = get_available_ingredient_ids(db)

    # 1) 임베딩 검색으로 상위 N개
    query_text = synthesize_query(profile)
    retrieved = retrieve_candidates(
        db, query_text, top_k=RAG_RETRIEVE_N, exclude_ids=exclude_ids,
        strength_preference=merged_slots.get("strength_preference"),
    )

    # 2) 하드 필터 (비선호 베이스 / 재고 / 맛 축 zero)
    survivors: list[tuple[Cocktail, float]] = []
    for cocktail, dist in retrieved:
        ri = all_ri.get(cocktail.cocktail_id, [])
        if _has_disliked_base(merged_slots, ri):
            continue
        if _is_unstockable(ri, available_ids):
            continue
        if _has_zero_taste_conflict(merged_slots, cocktail):
            continue
        survivors.append((cocktail, dist))

    if not survivors:
        return []

    survivor_cocktails = [c for c, _ in survivors]
    dist_map = {c.cocktail_id: d for c, d in survivors}

    # 3) LLM 리랭크 (실패 시 score_cocktail fallback)
    reranked = rerank_with_llm(profile, survivor_cocktails, k=k, recipe_ingredients=all_ri)

    if reranked:
        id_to_cocktail = {c.cocktail_id: c for c in survivor_cocktails}
        results = []
        for item in reranked:
            c = id_to_cocktail.get(item["cocktail_id"])
            if c is None:
                continue
            ri = all_ri.get(c.cocktail_id, [])
            score = score_cocktail(c, profile, ri)
            results.append({
                "cocktail_id": c.cocktail_id,
                "name_kr": c.name_kr,
                "score": score,
                "retrieval_distance": dist_map.get(c.cocktail_id),
                "reason_parts": [item["reason"]],
                "source": "rag_llm",
            })
        if results:
            return results[:k]

    # Fallback: score_cocktail로 재정렬
    results = []
    for c, dist in survivors:
        ri = all_ri.get(c.cocktail_id, [])
        score = score_cocktail(c, profile, ri)
        results.append({
            "cocktail_id": c.cocktail_id,
            "name_kr": c.name_kr,
            "score": score,
            "retrieval_distance": dist,
            "reason_parts": _build_reason_parts(c, profile, ri),
            "source": "rag_fallback",
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]


# ============================================================
# 6. 추천 실행 + DB 저장
# ============================================================

def run_recommendation(
    db: Session,
    guest_session_id: str,
    k: int = 3,
    exclude_ids: Optional[list[int]] = None,
    force: bool = False,          # ← 추가: 강제 진행 플래그
) -> dict:
    profile = build_user_profile(db, guest_session_id)
    space   = profile["space"]

    if not space:
        return {"status": "need_space_image",
                "message": "공간 분석 결과가 없습니다. 이미지를 먼저 업로드해주세요."}

    if not profile["vector"]:
        return {"status": "need_vector",
                "message": "선호 벡터가 없습니다. 초기 태그를 먼저 저장해주세요."}

    effective_completion = profile["effective_completion"]

    # force=True이면 80% 미만이어도 진행
    if effective_completion < 80 and not force:
        return {
            "status": "need_more_info",
            "completion": effective_completion,
            "message": "아직 추천에 필요한 정보가 부족합니다.",
        }
    top_k = recommend_top_k(db, guest_session_id, k=k, exclude_ids=exclude_ids)

    if not top_k:
        return {
            "status": "no_candidates",
            "top_k": [],
            "sample_recommendation_id": None,
        }

    best = top_k[0]
    reason_text = " / ".join(best["reason_parts"])

    row = create_sample_recommendation(
        db=db,
        guest_session_id=guest_session_id,
        space_analysis_id=space.space_analysis_id,
        recommended_cocktail_id=best["cocktail_id"],
        recommendation_reason=reason_text,
        rag_retrieved_ids_json=[r["cocktail_id"] for r in top_k],
        recipe_snapshot_json={"top_k": top_k},
    )

    return {
        "status": "ok",
        "completion": effective_completion,
        "top_k": top_k,
        "sample_recommendation_id": str(row.sample_recommendation_id),
    }


# ============================================================
# 6.5 최종 추천 확정 (ACCEPT 또는 3회 초과 강제)
# ============================================================

def finalize_sample(
    db: Session,
    guest_session_id: str,
    sample_recommendation_id: str,
    forced: bool = False,
) -> dict:
    sample_row = get_sample_recommendation(db, sample_recommendation_id)
    if not sample_row:
        raise ValueError("sample recommendation not found")
    if str(sample_row.guest_session_id) != str(guest_session_id):
        raise ValueError("sample recommendation does not belong to this guest")

    all_feedbacks = list_feedbacks_by_sample_recommendation(db, sample_recommendation_id)
    feedback_ids = [str(row.sample_feedback_id) for row in all_feedbacks]

    aggregated_deltas = {
        "sweetness_delta":  sum(float(fb.sweetness_delta  or 0) for fb in all_feedbacks),
        "sourness_delta":   sum(float(fb.sourness_delta   or 0) for fb in all_feedbacks),
        "bitterness_delta": sum(float(fb.bitterness_delta or 0) for fb in all_feedbacks),
        "body_delta":       sum(float(fb.body_delta       or 0) for fb in all_feedbacks),
        "freshness_delta":  sum(float(fb.freshness_delta  or 0) for fb in all_feedbacks),
    }

    final_snapshot = generate_recipe_snapshot(
        db=db,
        cocktail_id=sample_row.recommended_cocktail_id,
        volume_ml=90,
        feedback_deltas=aggregated_deltas,
    )
    is_adjusted = bool(final_snapshot.get("is_adjusted"))

    if forced:
        reason_parts = ["피드백 3회 한도 도달하여 현재 추천을 최종 확정했습니다."]
    else:
        reason_parts = ["사용자가 시음 후 현재 추천을 최종 선택했습니다."]

    if is_adjusted:
        adjusted_names = [
            item["ingredient_name"] for item in final_snapshot["recipe"]
            if item.get("adjusted")
        ]
        if adjusted_names:
            reason_parts.append(
                f"누적 피드백 반영하여 재료량 조정: {', '.join(adjusted_names)}"
            )

    final_row = create_final_recommendation(
        db=db,
        guest_session_id=guest_session_id,
        sample_recommendation_id=sample_recommendation_id,
        final_cocktail_id=sample_row.recommended_cocktail_id,
        used_feedback_ids_json=feedback_ids,
        is_adjusted_recipe=is_adjusted,
        final_recipe_snapshot_json=final_snapshot,
        final_reason_text=" ".join(reason_parts),
    )

    return {
        "status": "accepted" if not forced else "force_finalized",
        "intent": "ACCEPT" if not forced else "FORCED",
        "final_recommendation_id": str(final_row.final_recommendation_id),
        "final_cocktail_id": sample_row.recommended_cocktail_id,
    }


# ============================================================
# 7. 피드백 처리
# ============================================================

def process_feedback(
    db: Session,
    guest_session_id: str,
    sample_recommendation_id: str,
    feedback_text: str,
    feedback_round: int = 1,
) -> dict:
    sample_row = get_sample_recommendation(db, sample_recommendation_id)
    if not sample_row:
        raise ValueError("sample recommendation not found")

    if str(sample_row.guest_session_id) != str(guest_session_id):
        raise ValueError("sample recommendation does not belong to this guest")

    profile = build_user_profile(db, guest_session_id)
    vector_row = profile["vector"]
    if not vector_row:
        raise ValueError("preference vector not found")

    before_vec = _vector_row_to_dict(vector_row)
    fb = analyze_feedback(before_vec, feedback_text)
    intent = fb["intent"]
    updated_vec = fb["updated_vec"]

    if intent == "ADJUST" and fb["deltas"]:
        update_preference_vector(
            db=db,
            guest_session_id=guest_session_id,
            updates=updated_vec,
            increment_version=True,
        )

    feedback_row = create_sample_feedback(
        db=db,
        sample_recommendation_id=sample_recommendation_id,
        feedback_text=feedback_text,
        feedback_intent=intent,
        sweetness_delta=(
            updated_vec["sweetness_score"] - before_vec["sweetness_score"]
            if intent == "ADJUST" else None
        ),
        sourness_delta=(
            updated_vec["sourness_score"] - before_vec["sourness_score"]
            if intent == "ADJUST" else None
        ),
        bitterness_delta=(
            updated_vec["bitterness_score"] - before_vec["bitterness_score"]
            if intent == "ADJUST" else None
        ),
        body_delta=(
            updated_vec["body_score"] - before_vec["body_score"]
            if intent == "ADJUST" else None
        ),
        freshness_delta=(
            updated_vec["freshness_score"] - before_vec["freshness_score"]
            if intent == "ADJUST" else None
        ),
        aroma_delta_json=None,
        parsed_summary=feedback_text,
    )

    # ACCEPT → 최종 추천 확정
    if intent == "ACCEPT":
        return finalize_sample(db, guest_session_id, sample_recommendation_id, forced=False)

    # 3회차 ADJUST/REJECT → 재추천 없이 현재 샘플을 강제 확정 (세션 전체 3회 한도).
    # 벡터 델타는 이미 위에서 반영됐으므로, finalize 시 누적 델타가 적용된다.
    if feedback_round >= 3 and intent in ("ADJUST", "REJECT"):
        return finalize_sample(db, guest_session_id, sample_recommendation_id, forced=True)

# ADJUST → 벡터 업데이트 후 재추천
    if intent == "ADJUST":
        # ADJUST는 "현재 방향은 맞지만 조금 수정"의 의미이므로,
        # 현재 샘플 칵테일은 제외하지 않고 과거 다른 추천들만 배제한다.
        # force=True: 이미 FEEDBACK_LOOP 단계 = 80% 게이트 한번 통과한 상태.
        # 피드백으로 슬롯이 바뀌는 건 아니니 다시 게이트 걸 필요 없음.
        current_cocktail_id = sample_row.recommended_cocktail_id
        historical_ids = list_recommended_cocktail_ids_by_guest(db, guest_session_id)
        adjusted_excluded = [cid for cid in historical_ids if cid != current_cocktail_id]

        rerun = run_recommendation(
            db=db,
            guest_session_id=guest_session_id,
            k=3,
            exclude_ids=adjusted_excluded or None,
            force=True,
        )

        if rerun.get("status") == "no_candidates":
            rerun = run_recommendation(
                db=db,
                guest_session_id=guest_session_id,
                k=3,
                exclude_ids=None,
                force=True,
            )
            rerun["message"] = "새로운 후보가 없어 전체 후보에서 다시 추천합니다."

        if rerun.get("status") != "ok":
            return {
                "status": "adjust_processed",
                "intent": "ADJUST",
                "updated_vector": updated_vec,
                "sample_feedback_id": str(feedback_row.sample_feedback_id),
                "next": rerun,
            }

        return {
            "status": "re_recommended",
            "intent": "ADJUST",
            "updated_vector": updated_vec,
            "top_k": rerun["top_k"],
            "sample_recommendation_id": rerun["sample_recommendation_id"],
            "sample_feedback_id": str(feedback_row.sample_feedback_id),
        }

    # REJECT → 현재 추천 제외 후 새 후보 추천
    all_excluded = list_recommended_cocktail_ids_by_guest(db, guest_session_id)
    rerun = run_recommendation(
        db=db,
        guest_session_id=guest_session_id,
        k=3,
        exclude_ids=all_excluded,
        force=True,  # ← 추가
    )

    # 후보 없으면 exclude 해제하고 전체에서 재추천
    if rerun.get("status") == "no_candidates":
        rerun = run_recommendation(db, guest_session_id, k=3,
                                   exclude_ids=None, force=True)
        rerun["message"] = "새로운 후보가 없어 전체 후보에서 다시 추천합니다."

    if rerun.get("status") != "ok":
        return {
            "status": "reject_processed",
            "intent": "REJECT",
            "sample_feedback_id": str(feedback_row.sample_feedback_id),
            "next": rerun,
        }

    return {
        "status": "re_recommended",
        "intent": "REJECT",
        "top_k": rerun["top_k"],
        "sample_recommendation_id": rerun["sample_recommendation_id"],
        "sample_feedback_id": str(feedback_row.sample_feedback_id),
    }
