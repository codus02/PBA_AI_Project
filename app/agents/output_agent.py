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
    """재료 유형 + 감각 점수 기반으로 델타에 따른 배수를 곱해서 반환.

    여러 축이 동시에 해당되는 재료(예: 허브 + 알코올) 는 곱연산으로 누적된다.
    """
    itype = getattr(ingredient, "ingredient_type", None)
    ratio = base_ratio

    sweet_d     = deltas.get("sweetness_delta")  or 0.0
    sour_d      = deltas.get("sourness_delta")   or 0.0
    bitter_d    = deltas.get("bitterness_delta") or 0.0
    body_d      = deltas.get("body_delta")       or 0.0
    freshness_d = deltas.get("freshness_delta")  or 0.0
    herbal_d    = deltas.get("herbal_delta")     or 0.0
    citrus_d    = deltas.get("citrus_delta")     or 0.0
    alcohol_d   = deltas.get("alcohol_delta")    or 0.0

    def _score(attr: str) -> float:
        return float(getattr(ingredient, attr, 0) or 0)

    # SYRUP: 단맛
    if itype == "SYRUP" and sweet_d:
        ratio *= _ratio_from_delta(sweet_d)
    # JUICE (특히 시트러스): 신맛 — sour_score 높은 주스만 (lemon/lime 등)
    if itype == "JUICE" and sour_d and _score("sour_score") >= 3.0:
        ratio *= _ratio_from_delta(sour_d)
    # MIXER/BASE: 쓴맛 (비터스나 쓴맛 높은 베이스)
    if itype in ("MIXER", "BASE") and bitter_d and _score("bitter_score") >= 3.0:
        ratio *= _ratio_from_delta(bitter_d)
    # 바디감: 바디 점수 높은 재료 전반
    if body_d and _score("body_score") >= 3.0:
        ratio *= _ratio_from_delta(body_d)
    # 상큼함: 신선도 점수 높은 재료 (민트, 토닉, 시트러스)
    if freshness_d and _score("freshness_score") >= 3.0:
        ratio *= _ratio_from_delta(freshness_d)
    # 허브향: 허브 점수 높은 재료 (아마로, 샤르트뢰즈, 베르무트)
    if herbal_d and _score("herbal_score") >= 3.0:
        ratio *= _ratio_from_delta(herbal_d)
    # 시트러스향: 시트러스 점수 높은 재료
    if citrus_d and _score("citrus_score") >= 3.0:
        ratio *= _ratio_from_delta(citrus_d)
    # 도수: BASE 재료 전체 (스피릿)
    if alcohol_d and itype == "BASE":
        ratio *= _ratio_from_delta(alcohol_d)

    return ratio


def _rebalance_recipe_amounts(recipe_items: list[dict], target_volume_ml: int) -> float:
    """피드백 비율 조정 후 총량이 목표 용량과 다시 맞도록 재정규화한다."""
    if not recipe_items:
        return 0.0

    total_raw = sum(float(item.get("_raw_amount_ml") or 0.0) for item in recipe_items)
    if total_raw <= 0:
        for item in recipe_items:
            item["amount_ml"] = 0.0
            item.pop("_raw_amount_ml", None)
        return 0.0

    normalize = float(target_volume_ml) / total_raw
    for item in recipe_items:
        item["amount_ml"] = round(float(item.get("_raw_amount_ml") or 0.0) * normalize, 1)

    rounded_total = round(sum(float(item["amount_ml"]) for item in recipe_items), 1)
    diff = round(float(target_volume_ml) - rounded_total, 1)
    if abs(diff) >= 0.1:
        for item in reversed(recipe_items):
            candidate = round(float(item["amount_ml"]) + diff, 1)
            if candidate >= 0:
                item["amount_ml"] = candidate
                rounded_total = round(sum(float(r["amount_ml"]) for r in recipe_items), 1)
                break

    for item in recipe_items:
        item.pop("_raw_amount_ml", None)
    return rounded_total


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
        recipe_items.append(
            {
                "ingredient_id": ingredient.ingredient_id,
                "ingredient_name": _ingredient_name(ingredient),
                "_raw_amount_ml": base_ml * ratio,
                "step_order": recipe.step_order,
                "is_optional": bool(recipe.is_optional),
                "adjusted": is_adjusted and abs(ratio - 1.0) > 0.01,
            }
        )

    final_total_volume = _rebalance_recipe_amounts(recipe_items, volume_ml)

    return {
        "cocktail_id": cocktail.cocktail_id,
        "cocktail_name": cocktail.name_kr,
        "total_volume_ml": final_total_volume,
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
