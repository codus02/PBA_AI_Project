from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import Cocktail, Ingredient
from app.db.crud import get_all_recipes_with_ingredients


def _ingredient_name(ingredient: Ingredient) -> str:
    return getattr(ingredient, "ingredients_name", None) or getattr(ingredient, "name_kr", "")


# 피드백 델타 1.0당 해당 재료 유형의 비율 변화 (±25%)
_ADJUST_SENSITIVITY = 0.25


def _ratio_from_delta(delta: float) -> float:
    """델타값(-5~+5) → 비율 배수 (0.5~1.5 내)."""
    if delta is None:
        return 1.0
    return max(0.5, min(1.5, 1.0 + float(delta) * _ADJUST_SENSITIVITY))


def _apply_feedback_adjustment(
    ingredient,
    base_ratio: float,
    deltas: dict,
) -> float:
    """재료 유형 + 감각 점수 기반으로 델타에 따른 배수를 곱해서 반환."""
    itype = getattr(ingredient, "ingredient_type", None)
    ratio = base_ratio

    sweet_d = deltas.get("sweetness_delta") or 0.0
    sour_d = deltas.get("sourness_delta") or 0.0
    bitter_d = deltas.get("bitterness_delta") or 0.0

    # SYRUP: 단맛
    if itype == "SYRUP" and sweet_d:
        ratio *= _ratio_from_delta(sweet_d)
    # JUICE (특히 시트러스): 신맛
    if itype == "JUICE" and sour_d:
        # 신맛 점수 3 이상인 주스만 조정 (lemon, lime 등)
        if float(getattr(ingredient, "sour_score", 0) or 0) >= 3.0:
            ratio *= _ratio_from_delta(sour_d)
    # MIXER/BASE: 쓴맛 (비터스나 쓴맛 높은 베이스)
    if itype in ("MIXER", "BASE") and bitter_d:
        if float(getattr(ingredient, "bitter_score", 0) or 0) >= 3.0:
            ratio *= _ratio_from_delta(bitter_d)

    return ratio


def generate_recipe_snapshot(
    db: Session,
    cocktail_id: int,
    volume_ml: int = 90,
    feedback_deltas: dict | None = None,
) -> dict:
    cocktail = db.query(Cocktail).filter(Cocktail.cocktail_id == cocktail_id).first()
    if not cocktail:
        raise ValueError("cocktail not found")

    all_ri = get_all_recipes_with_ingredients(db)
    rows = all_ri.get(cocktail_id, [])

    total_original = sum(float(recipe.amount_ml or 0) for recipe, _ in rows)
    scale = (volume_ml / total_original) if total_original > 0 else 1.0

    deltas = feedback_deltas or {}
    is_adjusted = any(abs(float(v or 0)) > 0.01 for v in deltas.values())

    recipe_items = []
    for recipe, ingredient in sorted(rows, key=lambda x: x[0].step_order):
        base_ml = float(recipe.amount_ml or 0) * scale
        ratio = _apply_feedback_adjustment(ingredient, 1.0, deltas) if is_adjusted else 1.0
        scaled_ml = round(base_ml * ratio, 1)
        recipe_items.append(
            {
                "ingredient_id": ingredient.ingredient_id,
                "ingredient_name": _ingredient_name(ingredient),
                "amount_ml": scaled_ml,
                "step_order": recipe.step_order,
                "is_optional": bool(recipe.is_optional),
                "adjusted": is_adjusted and abs(ratio - 1.0) > 0.01,
            }
        )

    return {
        "cocktail_id": cocktail.cocktail_id,
        "cocktail_name": cocktail.name_kr,
        "total_volume_ml": volume_ml,
        "recipe": recipe_items,
        "is_adjusted": is_adjusted,
        "applied_deltas": {k: float(v) for k, v in deltas.items() if v},
    }


def generate_output_json(
    db: Session,
    cocktail_id: int,
    volume_ml: int = 90,
) -> dict:
    snapshot = generate_recipe_snapshot(db, cocktail_id, volume_ml)

    return {
        "cocktail_id": snapshot["cocktail_id"],
        "cocktail_name": snapshot["cocktail_name"],
        "total_volume_ml": snapshot["total_volume_ml"],
        "steps": snapshot["recipe"],
    }