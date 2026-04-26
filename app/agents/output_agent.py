from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import Cocktail
from app.db.crud import get_all_recipes_with_ingredients


def _ingredient_name(ingredient) -> str:
    return (
        getattr(ingredient, "ingredients_name", None)
        or getattr(ingredient, "ingredient_name", None)
        or getattr(ingredient, "name_kr", "")
    )


# ------------------------------------------------------------
# 공통 설정
# ------------------------------------------------------------
_SCORE_THRESHOLD = 3.0
_WORKING_VOLUME_ML = 90.0  # 내부 계산용 기준 볼륨


_AXIS_SENSITIVITY = {
    "sweetness_delta": 0.22,
    "sourness_delta": 0.22,
    "bitterness_delta": 0.22,
    "body_delta": 0.24,
    "freshness_delta": 0.33,
    # 아래 3개는 팀원 v2 에 누락돼 있던 axis 보강.
    # orchestration_agent 와 DB schema 가 8 axis 다 처리하므로 통합 시 필수.
    "herbal_delta": 0.22,    # 허브 향 (Fernet, Chartreuse, Vermouth 등)
    "citrus_delta": 0.22,    # 시트러스 향 (lemon/lime juice, peel)
    "alcohol_delta": 0.20,   # 도수 (BASE 스피릿 비율)
}

_BODY_INVERSE_MULTIPLIER = 0.70
_FRESHNESS_INVERSE_MULTIPLIER = 0.80


# ------------------------------------------------------------
# 기본 유틸
# ------------------------------------------------------------
def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _score(ingredient, attr: str) -> float:
    return _safe_float(getattr(ingredient, attr, 0.0), 0.0)


def _ingredient_type(ingredient) -> str:
    return str(getattr(ingredient, "ingredient_type", "") or "").upper().strip()


def _clamp_ratio(x: float) -> float:
    return max(0.35, min(1.65, x))


def _ratio_from_delta(delta: float, sensitivity: float) -> float:
    if abs(delta) < 1e-12:
        return 1.0
    return _clamp_ratio(1.0 + float(delta) * sensitivity)


def _sum_amount(recipe_items: list[dict]) -> float:
    return sum(_safe_float(item["amount_ml"]) for item in recipe_items)


# ------------------------------------------------------------
# 축별 관련 재료 판정
# ------------------------------------------------------------
def _is_sweet_related(ingredient) -> bool:
    it = _ingredient_type(ingredient)
    return (
        it == "SYRUP"
        or _score(ingredient, "sweet_score") >= _SCORE_THRESHOLD
        or (_score(ingredient, "fruity_score") >= 4.0 and it in {"MIXER", "JUICE"})
    )


def _is_sour_related(ingredient) -> bool:
    it = _ingredient_type(ingredient)
    return (
        it == "JUICE"
        or _score(ingredient, "sour_score") >= _SCORE_THRESHOLD
        or _score(ingredient, "citrus_score") >= _SCORE_THRESHOLD
    )


def _is_bitter_related(ingredient) -> bool:
    it = _ingredient_type(ingredient)
    return (
        (_score(ingredient, "bitter_score") >= _SCORE_THRESHOLD and it in {"BASE", "MIXER"})
        or (_score(ingredient, "herbal_score") >= 4.0 and it in {"BASE", "MIXER"})
        or (_score(ingredient, "woody_score") >= 4.0 and it == "BASE")
    )


def _is_body_positive(ingredient) -> bool:
    it = _ingredient_type(ingredient)
    return (
        _score(ingredient, "body_score") >= _SCORE_THRESHOLD
        or _score(ingredient, "creamy_score") >= _SCORE_THRESHOLD
        or _score(ingredient, "nutty_score") >= _SCORE_THRESHOLD
        or it in {"BASE", "SYRUP", "TOPPING"}
    )


def _is_body_negative(ingredient) -> bool:
    it = _ingredient_type(ingredient)
    return (
        _score(ingredient, "freshness_score") >= _SCORE_THRESHOLD
        or _score(ingredient, "citrus_score") >= _SCORE_THRESHOLD
        or _score(ingredient, "minty_score") >= _SCORE_THRESHOLD
        or it in {"JUICE", "GARNISH"}
    )


def _is_fresh_positive(ingredient) -> bool:
    it = _ingredient_type(ingredient)

    if it == "JUICE":
        return True

    if it == "GARNISH":
        return (
            _score(ingredient, "freshness_score") >= 2.5
            or _score(ingredient, "minty_score") >= 2.5
            or _score(ingredient, "citrus_score") >= 2.5
        )

    return (
        _score(ingredient, "freshness_score") >= _SCORE_THRESHOLD
        or _score(ingredient, "minty_score") >= _SCORE_THRESHOLD
        or _score(ingredient, "citrus_score") >= _SCORE_THRESHOLD
    )


def _is_fresh_negative(ingredient) -> bool:
    it = _ingredient_type(ingredient)

    if _is_fresh_positive(ingredient):
        return False

    return (
        _score(ingredient, "creamy_score") >= 2.8
        or _score(ingredient, "nutty_score") >= 3.0
        or it in {"SYRUP", "TOPPING"}
    )


# 보강된 3 axis 판정 — 이전 output_agent (v1) 에 있던 로직 유지.

def _is_herbal_related(ingredient) -> bool:
    """Fernet, Chartreuse, Vermouth 등 허브 향 강한 재료.
    이전 v1: herbal_score >= 3.0 만 체크. type 무관 (모든 재료 카테고리 가능).
    """
    return _score(ingredient, "herbal_score") >= _SCORE_THRESHOLD


def _is_citrus_related(ingredient) -> bool:
    """레몬/라임 juice, citrus zest 등 시트러스 향 재료.
    citrus_score 또는 JUICE type 으로 판정.
    """
    it = _ingredient_type(ingredient)
    return (
        _score(ingredient, "citrus_score") >= _SCORE_THRESHOLD
        or (it == "JUICE" and _score(ingredient, "citrus_score") >= 2.0)
    )


def _is_alcohol_related(ingredient) -> bool:
    """BASE 재료 = 메인 스피릿 (보드카/위스키/진/럼/데킬라/브랜디).
    이전 v1: ingredient_type == "BASE" 단순 매칭.
    """
    return _ingredient_type(ingredient) == "BASE"


# ------------------------------------------------------------
# 재료별 피드백 반영
# ------------------------------------------------------------
def _apply_feedback_adjustment(
    ingredient,
    base_ratio: float,
    deltas: dict,
) -> float:
    ratio = base_ratio

    sweet_d = _safe_float(deltas.get("sweetness_delta"), 0.0)
    sour_d = _safe_float(deltas.get("sourness_delta"), 0.0)
    bitter_d = _safe_float(deltas.get("bitterness_delta"), 0.0)
    body_d = _safe_float(deltas.get("body_delta"), 0.0)
    fresh_d = _safe_float(deltas.get("freshness_delta"), 0.0)
    herbal_d = _safe_float(deltas.get("herbal_delta"), 0.0)
    citrus_d = _safe_float(deltas.get("citrus_delta"), 0.0)
    alcohol_d = _safe_float(deltas.get("alcohol_delta"), 0.0)

    if abs(sweet_d) > 1e-12 and _is_sweet_related(ingredient):
        ratio *= _ratio_from_delta(sweet_d, _AXIS_SENSITIVITY["sweetness_delta"])

    if abs(sour_d) > 1e-12 and _is_sour_related(ingredient):
        ratio *= _ratio_from_delta(sour_d, _AXIS_SENSITIVITY["sourness_delta"])

    if abs(bitter_d) > 1e-12 and _is_bitter_related(ingredient):
        ratio *= _ratio_from_delta(bitter_d, _AXIS_SENSITIVITY["bitterness_delta"])

    if abs(body_d) > 1e-12:
        if _is_body_positive(ingredient):
            ratio *= _ratio_from_delta(body_d, _AXIS_SENSITIVITY["body_delta"])
        if _is_body_negative(ingredient):
            ratio *= _ratio_from_delta(
                -body_d,
                _AXIS_SENSITIVITY["body_delta"] * _BODY_INVERSE_MULTIPLIER,
            )

    if abs(fresh_d) > 1e-12:
        if _is_fresh_positive(ingredient):
            ratio *= _ratio_from_delta(fresh_d, _AXIS_SENSITIVITY["freshness_delta"])
        if _is_fresh_negative(ingredient):
            ratio *= _ratio_from_delta(
                -fresh_d,
                _AXIS_SENSITIVITY["freshness_delta"] * _FRESHNESS_INVERSE_MULTIPLIER,
            )

    # 보강된 3 axis — orchestration_agent 가 보내는 axis 모두 처리.
    if abs(herbal_d) > 1e-12 and _is_herbal_related(ingredient):
        ratio *= _ratio_from_delta(herbal_d, _AXIS_SENSITIVITY["herbal_delta"])

    if abs(citrus_d) > 1e-12 and _is_citrus_related(ingredient):
        ratio *= _ratio_from_delta(citrus_d, _AXIS_SENSITIVITY["citrus_delta"])

    if abs(alcohol_d) > 1e-12 and _is_alcohol_related(ingredient):
        ratio *= _ratio_from_delta(alcohol_d, _AXIS_SENSITIVITY["alcohol_delta"])

    return max(0.10, round(ratio, 6))


# ------------------------------------------------------------
# renormalization 보호 점수
# ------------------------------------------------------------
def _renorm_protection_score(ingredient, deltas: dict) -> float:
    score = 0.0

    sweet_d = abs(_safe_float(deltas.get("sweetness_delta"), 0.0))
    sour_d = abs(_safe_float(deltas.get("sourness_delta"), 0.0))
    bitter_d = abs(_safe_float(deltas.get("bitterness_delta"), 0.0))
    body_d = abs(_safe_float(deltas.get("body_delta"), 0.0))
    fresh_d = abs(_safe_float(deltas.get("freshness_delta"), 0.0))
    herbal_d = abs(_safe_float(deltas.get("herbal_delta"), 0.0))
    citrus_d = abs(_safe_float(deltas.get("citrus_delta"), 0.0))
    alcohol_d = abs(_safe_float(deltas.get("alcohol_delta"), 0.0))

    if sweet_d > 0 and _is_sweet_related(ingredient):
        score += sweet_d

    if sour_d > 0 and _is_sour_related(ingredient):
        score += sour_d

    if bitter_d > 0 and _is_bitter_related(ingredient):
        score += bitter_d

    if body_d > 0 and (_is_body_positive(ingredient) or _is_body_negative(ingredient)):
        score += body_d

    if fresh_d > 0 and (_is_fresh_positive(ingredient) or _is_fresh_negative(ingredient)):
        score += fresh_d

    # 보강된 3 axis 보호 점수.
    if herbal_d > 0 and _is_herbal_related(ingredient):
        score += herbal_d

    if citrus_d > 0 and _is_citrus_related(ingredient):
        score += citrus_d

    if alcohol_d > 0 and _is_alcohol_related(ingredient):
        score += alcohol_d

    return round(score, 6)


# ------------------------------------------------------------
# 스케일/정규화
# ------------------------------------------------------------
def _apply_diff_to_least_protected(
    recipe_items: list[dict],
    diff: float,
) -> list[dict]:
    """
    diff > 0  : 총량이 모자라므로 재료를 늘려야 함
    diff < 0  : 총량이 초과하므로 재료를 줄여야 함

    원칙:
    - delta와 덜 관련된 재료(_protection_score 낮음)부터 보정
    - amount가 충분한 재료를 우선 활용
    """
    if not recipe_items or abs(diff) < 0.1:
        return recipe_items

    if diff > 0:
        candidates = sorted(
            range(len(recipe_items)),
            key=lambda i: (
                _safe_float(recipe_items[i].get("_protection_score"), 0.0),
                -_safe_float(recipe_items[i]["amount_ml"]),
            ),
        )
        remain = round(diff, 1)
        for idx in candidates:
            if abs(remain) < 0.1:
                break
            recipe_items[idx]["amount_ml"] = round(
                _safe_float(recipe_items[idx]["amount_ml"]) + remain,
                1,
            )
            remain = 0.0
        return recipe_items

    to_remove = round(-diff, 1)
    candidates = sorted(
        range(len(recipe_items)),
        key=lambda i: (
            _safe_float(recipe_items[i].get("_protection_score"), 0.0),
            -_safe_float(recipe_items[i]["amount_ml"]),
        ),
    )

    for idx in candidates:
        if to_remove < 0.1:
            break

        cur = _safe_float(recipe_items[idx]["amount_ml"])
        removable = round(max(0.0, cur - 0.1), 1)
        if removable < 0.1:
            continue

        take = round(min(removable, to_remove), 1)
        recipe_items[idx]["amount_ml"] = round(cur - take, 1)
        to_remove = round(to_remove - take, 1)

    if to_remove >= 0.1:
        fallback = sorted(
            range(len(recipe_items)),
            key=lambda i: -_safe_float(recipe_items[i]["amount_ml"]),
        )
        for idx in fallback:
            if to_remove < 0.1:
                break
            cur = _safe_float(recipe_items[idx]["amount_ml"])
            removable = round(max(0.0, cur - 0.1), 1)
            if removable < 0.1:
                continue
            take = round(min(removable, to_remove), 1)
            recipe_items[idx]["amount_ml"] = round(cur - take, 1)
            to_remove = round(to_remove - take, 1)

    return recipe_items


def _renormalize_amounts(recipe_items: list[dict], target_volume_ml: float) -> list[dict]:
    if not recipe_items:
        return recipe_items

    total = _sum_amount(recipe_items)
    if total <= 0:
        equal_amt = round(float(target_volume_ml) / len(recipe_items), 1)
        for item in recipe_items:
            item["amount_ml"] = max(0.1, equal_amt)
        return recipe_items

    scale = float(target_volume_ml) / total
    for item in recipe_items:
        scaled = _safe_float(item["amount_ml"]) * scale
        item["amount_ml"] = round(max(0.1, scaled), 1)

    rounded_total = round(_sum_amount(recipe_items), 1)
    diff = round(float(target_volume_ml) - rounded_total, 1)
    recipe_items = _apply_diff_to_least_protected(recipe_items, diff)

    final_total = round(_sum_amount(recipe_items), 1)
    remain = round(float(target_volume_ml) - final_total, 1)
    recipe_items = _apply_diff_to_least_protected(recipe_items, remain)

    return recipe_items


def _rescale_amounts(recipe_items: list[dict], target_volume_ml: float) -> list[dict]:
    if not recipe_items:
        return recipe_items

    total = _sum_amount(recipe_items)
    if total <= 0:
        return recipe_items

    scale = float(target_volume_ml) / total
    for item in recipe_items:
        item["amount_ml"] = round(max(0.1, _safe_float(item["amount_ml"]) * scale), 1)

    rounded_total = round(_sum_amount(recipe_items), 1)
    diff = round(float(target_volume_ml) - rounded_total, 1)
    recipe_items = _apply_diff_to_least_protected(recipe_items, diff)

    return recipe_items


def _round_amounts_to_int(recipe_items: list[dict], target_volume_ml: int) -> list[dict]:
    """각 amount_ml 을 정수 ml 로 반올림 + 합계 = target_volume_ml 유지.

    반올림 후 합계 차이는 보호 점수 낮은 항목 (delta 와 관련 적은 재료) 부터 1ml 씩 +/-.
    재료가 0 이 되지 않도록 최소 1ml 보장.
    """
    if not recipe_items:
        return recipe_items

    target = int(round(target_volume_ml))

    for item in recipe_items:
        rounded = int(round(_safe_float(item["amount_ml"])))
        item["amount_ml"] = max(1, rounded)

    total = sum(int(item["amount_ml"]) for item in recipe_items)
    diff = target - total
    if diff == 0:
        return recipe_items

    candidates = sorted(
        range(len(recipe_items)),
        key=lambda i: (
            _safe_float(recipe_items[i].get("_protection_score"), 0.0),
            -int(recipe_items[i]["amount_ml"]),
        ),
    )

    max_rounds = abs(diff) * 2 + 2
    rounds = 0
    while diff != 0 and rounds < max_rounds:
        progressed = False
        for idx in candidates:
            if diff == 0:
                break
            cur = int(recipe_items[idx]["amount_ml"])
            if diff > 0:
                recipe_items[idx]["amount_ml"] = cur + 1
                diff -= 1
                progressed = True
            elif diff < 0 and cur > 1:
                recipe_items[idx]["amount_ml"] = cur - 1
                diff += 1
                progressed = True
        if not progressed:
            break
        rounds += 1

    return recipe_items


# ------------------------------------------------------------
# 메인 로직
# ------------------------------------------------------------
def generate_recipe_snapshot(
    db: Session,
    cocktail_id: int,
    volume_ml: int | None = None,
    feedback_deltas: dict | None = None,
) -> dict:
    """
    내부 계산은 _WORKING_VOLUME_ML(기본 90ml)에서 수행하고,
    최종 반환은 volume_ml이 주어지면 그 값으로,
    아니면 원본 레시피 총량으로 되돌려 반환한다.
    """
    cocktail = db.query(Cocktail).filter(Cocktail.cocktail_id == cocktail_id).first()
    if not cocktail:
        raise ValueError("cocktail not found")

    all_recipe_rows = get_all_recipes_with_ingredients(db)
    rows = all_recipe_rows.get(cocktail_id, [])
    if not rows:
        raise ValueError("recipe not found")

    original_total = sum(_safe_float(recipe.amount_ml) for recipe, _ingredient in rows)
    if original_total <= 0:
        raise ValueError("invalid original recipe total")

    final_target_volume = float(volume_ml) if volume_ml is not None else float(original_total)
    working_target_volume = float(_WORKING_VOLUME_ML)

    working_scale = working_target_volume / original_total

    deltas = feedback_deltas or {}
    is_adjusted = any(abs(_safe_float(v)) > 1e-12 for v in deltas.values())

    recipe_items: list[dict] = []
    for recipe, ingredient in sorted(rows, key=lambda x: x[0].step_order):
        base_ml_working = _safe_float(recipe.amount_ml) * working_scale
        ratio = _apply_feedback_adjustment(ingredient, 1.0, deltas) if is_adjusted else 1.0
        adjusted_ml_working = round(max(0.1, base_ml_working * ratio), 1)

        recipe_items.append(
            {
                "ingredient_id": int(ingredient.ingredient_id),
                "ingredient_name": _ingredient_name(ingredient),
                "amount_ml": adjusted_ml_working,
                "step_order": int(recipe.step_order),
                "is_optional": bool(recipe.is_optional),
                "adjusted": bool(is_adjusted and abs(ratio - 1.0) > 1e-9),
                # 모터 제어용 메타 (DB ingredient 컬럼).
                "pump_no": int(ingredient.pump_no) if getattr(ingredient, "pump_no", None) is not None else None,
                "is_pumpable": bool(getattr(ingredient, "is_pumpable", False)),
                "_protection_score": _renorm_protection_score(ingredient, deltas),
            }
        )

    # 내부 계산은 90ml 기준으로 정규화
    recipe_items = _renormalize_amounts(recipe_items, working_target_volume)

    # 최종 반환 직전 원래 총량(또는 volume_ml)로 복원
    recipe_items = _rescale_amounts(recipe_items, final_target_volume)

    # 정수 ml 로 반올림 (사용자 표시 + 모터 제어 단순화. 합계 = final_target_volume 유지)
    final_target_int = int(round(final_target_volume))
    recipe_items = _round_amounts_to_int(recipe_items, final_target_int)

    # 내부용 키 제거 (단, 모터 메타는 유지)
    cleaned_items = []
    for item in recipe_items:
        cleaned_items.append(
            {
                "ingredient_id": item["ingredient_id"],
                "ingredient_name": item["ingredient_name"],
                "amount_ml": item["amount_ml"],
                "step_order": item["step_order"],
                "is_optional": item["is_optional"],
                "adjusted": item["adjusted"],
                "pump_no": item.get("pump_no"),
                "is_pumpable": item.get("is_pumpable", False),
            }
        )

    return {
        "cocktail_id": int(cocktail.cocktail_id),
        "cocktail_name": getattr(cocktail, "name_kr", None) or getattr(cocktail, "name_en", ""),
        "total_volume_ml": final_target_int,
        "recipe": cleaned_items,
        "is_adjusted": is_adjusted,
        "original_total_volume_ml": int(round(original_total)),
        "working_total_volume_ml": int(round(working_target_volume)),
        "applied_deltas": {
            k: float(v)
            for k, v in deltas.items()
            if abs(_safe_float(v)) > 1e-12
        },
    }


def generate_output_json(
    db: Session,
    cocktail_id: int,
    volume_ml: int | None = None,
    feedback_deltas: dict | None = None,
) -> dict:
    snapshot = generate_recipe_snapshot(
        db=db,
        cocktail_id=cocktail_id,
        volume_ml=volume_ml,
        feedback_deltas=feedback_deltas,
    )

    return {
        "cocktail_id": snapshot["cocktail_id"],
        "cocktail_name": snapshot["cocktail_name"],
        "total_volume_ml": snapshot["total_volume_ml"],
        "steps": snapshot["recipe"],
    }


# ============================================================
# 모터 제어용 출력 (라즈베리파이 5 + 8채널 MOSFET + 페리스탈틱 펌프)
# ============================================================
#
# 하드웨어:
#   - Pi 5 (4GB) → GPIO → 8채널 MOSFET 보드 → 12V 페리스탈틱 펌프 × 8
#   - 12V 10A 어댑터 가 펌프 전원
#   - DB.Ingredient.pump_no = 모터 채널 (0-7 또는 1-8, DB 라벨 따라)
#   - DB.Ingredient.is_pumpable = 액체 자동 펌핑 가능 여부 (False = mint, peel 등 사람 토핑)
#
# 펌프 유속 calibration:
#   페리스탈틱 펌프 typical 50~100 ml/min = 0.83~1.67 ml/sec.
#   실제 펌프로 측정 후 DEFAULT_FLOW_RATE 또는 채널별 dict 로 지정.

DEFAULT_FLOW_RATE_ML_PER_SEC = 1.0  # 보수적 기본 (60 ml/min). 실제 펌프 측정 후 교체 필요.

# 단계별 표준 볼륨 — Pi 펌프가 따라줄 양.
DEFAULT_SAMPLE_VOLUME_ML = 30  # 시음 단계 (사용자가 맛보고 피드백 줄 만큼)
DEFAULT_FINAL_VOLUME_ML = 90   # 최종 확정 후 풀 사이즈 칵테일


def generate_motor_commands(
    snapshot: dict,
    flow_rate_ml_per_sec: float = DEFAULT_FLOW_RATE_ML_PER_SEC,
    flow_rates_per_channel: dict[int, float] | None = None,
) -> dict:
    """레시피 snapshot → 라즈베리파이 모터 제어 명령.

    Args:
        snapshot: generate_recipe_snapshot 결과 (steps 안에 pump_no, is_pumpable 필요)
        flow_rate_ml_per_sec: 모든 채널 공통 유속 기본값 (ml/sec)
        flow_rates_per_channel: 채널별 유속 override {channel: ml_per_sec}
                                펌프마다 유속 다르면 calibration 결과 주입.

    Returns:
        {
            "cocktail_name": str,
            "total_volume_ml": float,
            "pump_commands": [
                {
                    "channel": int,           # MOSFET 채널 번호 (= pump_no)
                    "ingredient_name": str,
                    "amount_ml": float,
                    "duration_ms": int,       # 펌프 ON 유지 시간
                },
                ...
            ],
            "manual_steps": [                 # 사람 토핑 (mint, peel 등)
                {
                    "ingredient_name": str,
                    "amount_ml": float,
                    "reason": str,            # "not_pumpable" 또는 "no_pump_assigned"
                },
                ...
            ],
            "warnings": [str, ...],           # 매핑 누락 등
        }

    동작:
      - is_pumpable=True 이고 pump_no 있으면 → pump_commands 에 추가
      - is_pumpable=False 또는 pump_no=None 이면 → manual_steps (사람 토핑)
      - duration_ms = (amount_ml / flow_rate) * 1000

    참고:
      - 순서 (step_order) 는 무시 (사용자 요청). pump_commands 는 채널 번호순 정렬.
      - 동시 다채널 ON 가능 (MOSFET 보드 8채널 독립). 단 12V 10A 한도 고려해
        Pi 펌웨어에서 동시 ON 채널 수 제한할 수도 있음 (이건 Pi 쪽 책임).
    """
    pump_commands: list[dict] = []
    manual_steps: list[dict] = []
    warnings: list[str] = []
    flow_overrides = flow_rates_per_channel or {}

    for step in snapshot.get("recipe", []) or snapshot.get("steps", []):
        amount = float(step.get("amount_ml") or 0.0)
        if amount < 0.1:
            continue  # 0 에 가까운 양은 skip

        pump_no = step.get("pump_no")
        is_pumpable = step.get("is_pumpable", False)
        name = step.get("ingredient_name", "")

        amount_int = int(round(amount))

        # 펌프 처리 가능한지 판정.
        if not is_pumpable:
            manual_steps.append({
                "ingredient_name": name,
                "amount_ml": amount_int,
                "reason": "not_pumpable",  # 가니쉬 / 토핑 류
            })
            continue
        if pump_no is None:
            manual_steps.append({
                "ingredient_name": name,
                "amount_ml": amount_int,
                "reason": "no_pump_assigned",  # DB 에 채널 매핑 안 된 재료
            })
            warnings.append(f"ingredient '{name}' is_pumpable=True but pump_no=None")
            continue

        # 채널별 유속 override 우선, 없으면 default.
        flow = float(flow_overrides.get(pump_no, flow_rate_ml_per_sec))
        if flow <= 0:
            warnings.append(f"channel {pump_no} flow_rate <= 0, using default")
            flow = flow_rate_ml_per_sec

        duration_ms = int(round((amount / flow) * 1000))

        pump_commands.append({
            "channel": int(pump_no),
            "ingredient_name": name,
            "amount_ml": amount_int,
            "duration_ms": duration_ms,
        })

    # 채널 번호 순으로 정렬 (Pi 측 ordering 단순화).
    pump_commands.sort(key=lambda x: x["channel"])

    return {
        "cocktail_id": snapshot.get("cocktail_id"),
        "cocktail_name": snapshot.get("cocktail_name"),
        "total_volume_ml": snapshot.get("total_volume_ml"),
        "pump_commands": pump_commands,
        "manual_steps": manual_steps,
        "warnings": warnings,
    }


def generate_motor_recipe(
    db: Session,
    cocktail_id: int,
    volume_ml: int | None = None,
    feedback_deltas: dict | None = None,
    flow_rate_ml_per_sec: float = DEFAULT_FLOW_RATE_ML_PER_SEC,
    flow_rates_per_channel: dict[int, float] | None = None,
) -> dict:
    """추천 확정 → 레시피 비율 결정 → 모터 명령 변환 통합 entry.

    이 함수 하나로 라즈베리파이가 받을 최종 JSON 생성. Pi 측은 받아서:
      1. pump_commands 순회
      2. 각 명령에 대해 GPIO.output(channel, HIGH) → sleep(duration_ms/1000) → LOW
      3. manual_steps 는 화면/음성으로 사용자에 안내
    """
    snapshot = generate_recipe_snapshot(
        db=db,
        cocktail_id=cocktail_id,
        volume_ml=volume_ml,
        feedback_deltas=feedback_deltas,
    )
    motor = generate_motor_commands(
        snapshot,
        flow_rate_ml_per_sec=flow_rate_ml_per_sec,
        flow_rates_per_channel=flow_rates_per_channel,
    )
    return motor
