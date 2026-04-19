from __future__ import annotations

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
    classify_intent,
    update_vector,
    build_user_profile,
)
from app.agents.output_agent import generate_recipe_snapshot

# ============================================================
# 매핑 테이블
# ============================================================

TASTE_TO_COCKTAIL = {
    "단맛": "sweet_level",
    "달콤함": "sweet_level",
    "쓴맛": "bitter_level",
    "쌉쌀함": "bitter_level",
    "신맛": "sour_level",
    "상큼함": "sour_level",
    "청량함": "freshness_level",
    "바디감": "body_level",
    "크리미함": "creamy_level",
    "스파이시": "spicy_level",
    "고소함": "nutty_level",
}

AROMA_TO_INGREDIENT = {
    "민트향": "minty_score",
    "과일향": "fruity_score",
    "시트러스향": "citrus_score",
    "허브향": "herbal_score",
    "커피향": "coffee_score",
    "우디향": "woody_score",
    "꽃향": "floral_score",
}

STRENGTH_RANGE = {
    "약함": (0.0, 2.0),
    "중간": (1.5, 3.5),
    "강함": (3.0, 5.0),
}

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


def _build_reason_parts(
    cocktail: Cocktail,
    profile: dict,
    recipe_ingredients: list[tuple],
) -> list[str]:
    merged = profile["merged_slots"]
    space = profile["space"]
    reasons: list[str] = []

    matched_tastes: list[str] = []
    for tag in (merged.get("preferred_tastes") or []):
        col = TASTE_TO_COCKTAIL.get(tag)
        if col and getattr(cocktail, col) is not None:
            if float(getattr(cocktail, col)) >= 3.5:
                matched_tastes.append(tag)

    if matched_tastes:
        reasons.append(f"선호 맛과 일치: {', '.join(matched_tastes)}")

    matched_aromas: list[str] = []
    for _, ingredient in recipe_ingredients:
        for tag in (merged.get("preferred_aromas") or []):
            col = AROMA_TO_INGREDIENT.get(tag)
            if col and getattr(ingredient, col, 0) >= 3.0 and tag not in matched_aromas:
                matched_aromas.append(tag)

    if matched_aromas:
        reasons.append(f"선호 향과 일치: {', '.join(matched_aromas)}")

    if space and cocktail.mood_tag:
        mood_prob = (space.mood_tags_json or {}).get(cocktail.mood_tag, 0.0)
        if mood_prob >= 0.5:
            reasons.append(f"공간 무드와 어울림: {cocktail.mood_tag}")

    strength_pref = merged.get("strength_preference")
    if strength_pref:
        reasons.append(f"도수 선호 반영: {strength_pref}")

    finish_pref = merged.get("finish_preference")
    if finish_pref:
        reasons.append(f"끝맛 선호 반영: {finish_pref}")

    if not reasons:
        reasons.append("기본 취향 점수 기반 추천")

    return reasons


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

def _has_disliked_base(merged_slots: dict, recipe_ingredients: list[tuple]) -> bool:
    disliked_bases = merged_slots.get("disliked_bases") or []
    if not disliked_bases:
        return False

    for _, ingredient in recipe_ingredients:
        if ingredient.ingredient_type == "BASE" and _ingredient_name(ingredient) in disliked_bases:
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

    # 2. 선호 맛 보너스
    for tag in (merged.get("preferred_tastes") or []):
        col = TASTE_TO_COCKTAIL.get(tag)
        if col and getattr(cocktail, col) is not None:
            if float(getattr(cocktail, col)) >= 3.5:
                score += 10

    # 3. 선호 향 보너스
    for _, ingredient in recipe_ingredients:
        for tag in (merged.get("preferred_aromas") or []):
            col = AROMA_TO_INGREDIENT.get(tag)
            if col and getattr(ingredient, col, 0) >= 3.0:
                score += 8

    # 4. 공간 무드 보너스
    if space and cocktail.mood_tag:
        mood_prob = (space.mood_tags_json or {}).get(cocktail.mood_tag, 0.0)
        score += mood_prob * 15

    # 5. 비선호 맛 패널티
    for tag in (merged.get("disliked_tastes") or []):
        col = TASTE_TO_COCKTAIL.get(tag)
        if col and getattr(cocktail, col) is not None:
            val = float(getattr(cocktail, col))
            if val >= 4.0:
                score -= 25
            elif val >= 3.0:
                score -= 10

    # 6. 비선호 향 패널티
    for _, ingredient in recipe_ingredients:
        for tag in (merged.get("disliked_aromas") or []):
            col = AROMA_TO_INGREDIENT.get(tag)
            if col and getattr(ingredient, col, 0) >= 3.0:
                score -= 20

    # 7. 도수 선호 보너스
    strength_pref = merged.get("strength_preference")
    if strength_pref in STRENGTH_RANGE:
        low, high = STRENGTH_RANGE[strength_pref]
        if low <= float(vector.alcohol_score) <= high:
            score += 10

    return round(score, 2)


# ============================================================
# 5. Top-K 추천
# ============================================================

def recommend_top_k(
    db: Session,
    guest_session_id: str,
    k: int = 3,
    exclude_ids: Optional[list[int]] = None,
) -> list[dict]:
    profile = build_user_profile(db, guest_session_id)
    merged_slots = profile["merged_slots"]
    candidates = get_candidate_cocktails(db, exclude_ids=exclude_ids)

    all_ri = get_all_recipes_with_ingredients(db)
    available_ids = get_available_ingredient_ids(db)

    results = []
    for cocktail in candidates:
        ri = all_ri.get(cocktail.cocktail_id, [])

        if _has_disliked_base(merged_slots, ri):
            continue

        if _is_unstockable(ri, available_ids):
            continue

        score = score_cocktail(cocktail, profile, ri)
        reason_parts = _build_reason_parts(cocktail, profile, ri)

        results.append(
            {
                "cocktail_id": cocktail.cocktail_id,
                "name_kr": cocktail.name_kr,
                "score": score,
                "reason_parts": reason_parts,
            }
        )

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

    intent = classify_intent(feedback_text)

    before_vec = _vector_row_to_dict(vector_row)
    updated_vec = dict(before_vec)

    if intent == "ADJUST":
        updated_vec = update_vector(before_vec, feedback_text)

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
        all_feedbacks = list_feedbacks_by_sample_recommendation(db, sample_recommendation_id)
        feedback_ids = [str(row.sample_feedback_id) for row in all_feedbacks]

        # 누적 피드백 델타 합산 → 최종 레시피에 반영 (FR-16)
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
            "status": "accepted",
            "intent": "ACCEPT",
            "final_recommendation_id": str(final_row.final_recommendation_id),
            "final_cocktail_id": sample_row.recommended_cocktail_id,
        }

# ADJUST → 벡터 업데이트 후 재추천
    if intent == "ADJUST":
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