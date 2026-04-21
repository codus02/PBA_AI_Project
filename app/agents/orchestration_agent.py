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

# intensity → 점수 스케일
INTENSITY_WEIGHT = {"low": -1.0, "medium": 0.3, "high": 1.0}
# high/medium만 "선호"로 취급하고 점수에 가점, low는 감점
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
        if intensity not in ("medium", "high"):
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
            if intensity not in ("medium", "high"):
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
# RAG: 쿼리 합성 + pgvector 검색 + Qwen 리랭크
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
    "low": "약하게 비선호",
    "zero": "완전 비선호(제외)",
}

# 칵테일의 해당 축 level >= 이 임계값이면 "두드러진다"고 본다.
_ZERO_EXCLUDE_THRESHOLD = 3.5

# taste 축 → Cocktail 컬럼명 매핑
_TASTE_LEVEL_COL = {
    "sweet": "sweet_level",
    "sour": "sour_level",
    "bitter": "bitter_level",
    "body": "body_level",
    "creamy": "creamy_level",
    "freshness": "freshness_level",
}
# 향 축은 Cocktail 컬럼이 없어 재료 기반 판단 → 일단 description/레시피 감각노트 스캔으로 근사.
# 현재 구조에서 aroma zero 는 rerank LLM 단계에서 강하게 감점 처리하게 둔다.
_RAG_STRENGTH_KR = {"zero": "논알콜(무알콜)", "light": "가벼운 도수", "medium": "중간 도수", "strong": "강한 도수"}


def _rag_collect_profile(profile_dict: dict, label_map: dict[str, str]) -> list[str]:
    out = []
    for tag, intensity in (profile_dict or {}).items():
        if intensity == "pending":
            # 강도 미확정은 RAG 쿼리에 넣지 않는다 (대화로 확정되면 반영).
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
        ctx = ", ".join([v for v in [purpose, mood] if v])
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
        parts.append(f"베이스 회피: {', '.join(disliked_bases)}")

    favs = merged.get("favorite_drinks") or []
    if favs:
        parts.append(f"유사 선호 음료: {', '.join(favs[:3])}")

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
    vec = embed_texts(
        [query_text],
        batch_size=1,
        max_length=512,
        instruction="주어진 사용자 선호 설명에 가장 잘 맞는 칵테일을 찾는다",
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
   - "low"가 붙은 축은 해당 축이 "강한" 칵테일을 반드시 배제하거나 강하게 감점하라.
     예: creamy=low이면 creamy_level이 높은 칵테일(아이리시 커피·포르토 플립 등)을 top에 넣지 마라.
   - "high"는 해당 축이 3.5 이상, "medium"은 2~3.5, "low"는 2 이하가 바람직.
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
- 2~3문장, 80~180자 한국어. 단답형("~해서 적합합니다") 금지.
- 반드시 다음 3요소를 모두 녹여라:
  (1) 이 칵테일이 어떤 맛/향/도수 캐릭터인지 짧은 묘사 (예: "라임의 산미와 진의 허브향이 깔끔하게 맞물리는 드라이한 스타일")
  (2) 도수·강도 느낌 — "가볍게 마시기 좋음 / 중간 정도 바디 / 스피릿포워드로 묵직함" 같은 표현을 넣어라.
  (3) 사용자의 선호(맛/향/도수/분위기) 중 어느 부분과 맞는지 연결.
- 후보 description 에 있는 풍미·질감 단어를 적극 활용하되, 없는 속성은 지어내지 마라.
- "사용자의 선호에 부합합니다" 같은 상투어로 문장 끝내지 마라. 실제 칵테일 캐릭터를 설명하는 톤으로 써라.
""".strip()


def _format_profile_for_rerank(profile: dict) -> str:
    merged = profile.get("merged_slots") or {}
    lines = [
        f"party_purpose: {merged.get('party_purpose')}",
        f"current_mood: {merged.get('current_mood')}",
        f"taste_profile: {json.dumps(merged.get('taste_profile') or {}, ensure_ascii=False)}",
        f"aroma_profile: {json.dumps(merged.get('aroma_profile') or {}, ensure_ascii=False)}",
        f"strength_preference: {merged.get('strength_preference')}",
        f"favorite_drinks: {merged.get('favorite_drinks') or []}",
    ]
    space = profile.get("space")
    if space and getattr(space, "mood_tags_json", None):
        top_moods = sorted(space.mood_tags_json.items(), key=lambda x: x[1], reverse=True)[:3]
        lines.append(f"space_mood_top: {[k for k, _ in top_moods]}")
    return "\n".join(lines)


def _format_candidates_for_rerank(candidates: list[Cocktail]) -> str:
    rows = []
    for c in candidates:
        taste = []
        for label, col in [("sweet", "sweet_level"), ("sour", "sour_level"),
                           ("bitter", "bitter_level"), ("body", "body_level"),
                           ("freshness", "freshness_level"), ("creamy", "creamy_level")]:
            v = getattr(c, col, None)
            if v is not None:
                taste.append(f"{label}={float(v):.1f}")
        desc = (c.description or "").replace("\n", " ")[:260]
        rows.append(
            f"- id={c.cocktail_id} | {c.name_kr} | {c.category} | "
            f"mood={c.mood_tag or '-'} | {', '.join(taste)}\n"
            f"    desc: {desc}"
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


def rerank_with_qwen(
    profile: dict,
    candidates: list[Cocktail],
    k: int = 3,
    max_new_tokens: int = 512,
) -> Optional[list[dict]]:
    """Qwen3-8B로 후보 리랭킹. 실패 시 None → 호출측 fallback."""
    if not candidates:
        return []
    try:
        from app.utils.model_loader import load_qwen3
        import torch

        tokenizer, model = load_qwen3()
        user_content = (
            f"사용자 프로파일:\n{_format_profile_for_rerank(profile)}\n\n"
            f"후보 칵테일 (N={len(candidates)}):\n{_format_candidates_for_rerank(candidates)}\n\n"
            f"위 후보 중에서 사용자에게 가장 잘 맞는 상위 {k}개를 ranked로 JSON 반환해라."
        )
        messages = [
            {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
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
            reason = str(item.get("reason") or "").strip() or "취향 적합도 기반 추천"
            out_rows.append({"cocktail_id": cid, "reason": reason})
            seen.add(cid)
            if len(out_rows) >= k:
                break

        return out_rows or None

    except Exception as e:
        logger.warning("Qwen rerank failed: %r", e, exc_info=True)
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
        col = _TASTE_LEVEL_COL.get(axis)
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
            # 사용자가 "이 맛 안 좋아함" → 강하게 있으면 감점
            if val >= 4.0:
                score -= 25
            elif val >= 3.0:
                score -= 10

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
            elif intensity == "low" and ing_val >= AROMA_ING_THRESHOLD:
                score -= 20

    # 4. 공간 무드 보너스
    if space and cocktail.mood_tag:
        mood_prob = (space.mood_tags_json or {}).get(cocktail.mood_tag, 0.0)
        score += mood_prob * 15

    # 7. 도수 선호 보너스
# 7. 도수 선호 보너스
    strength_pref = merged.get("strength_preference")
    cocktail_strength = _get_cocktail_strength_value(cocktail)
    if strength_pref in STRENGTH_RANGE and cocktail_strength is not None:
        low, high = STRENGTH_RANGE[strength_pref]
        if low <= cocktail_strength <= high:
            score += 10

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
    """RAG 파이프라인: 임베딩 검색 → 하드 필터 → Qwen 리랭크.

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

    # 3) Qwen 리랭크 (실패 시 score_cocktail fallback)
    reranked = rerank_with_qwen(profile, survivor_cocktails, k=k)

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
                "source": "rag_qwen",
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

# ADJUST → 벡터 업데이트 후 재추천
    if intent == "ADJUST":
        # ADJUST는 "현재 방향은 맞지만 조금 수정"의 의미이므로,
        # 현재 샘플 칵테일은 제외하지 않고 과거 다른 추천들만 배제한다.
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