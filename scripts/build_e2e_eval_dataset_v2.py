"""E2E eval dataset V2 생성.

V1 과 달리 score top-3 를 그대로 gold 로 쓰지 않는다. 먼저 사용자가
명시한 핵심 high 슬롯을 hard-check 로 만족하는 후보만 남긴 뒤, 그 안에서
score_cocktail 로 정렬해 exact gold 를 0~3개로 만든다.

입력:
  data/eval/slot_extraction_eval_v2_500.csv

출력:
  data/eval/e2e_recommendation_eval_v2_500.csv
  eval_results/audit/e2e_gold_audit_v2_50.txt
  eval_results/audit/e2e_gold_audit_v2_50.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.orchestration_agent import (
    AROMA_TO_INGREDIENT,
    TASTE_TO_COCKTAIL,
    _has_disliked_base,
    _has_zero_taste_conflict,
    _is_unstockable,
    score_cocktail,
)
from app.db.crud import get_all_recipes_with_ingredients, get_available_ingredient_ids
from app.db.database import SessionLocal
from app.db.models import Cocktail


SRC_CSV = Path("data/eval/slot_extraction_eval_v2_500.csv")
OUT_CSV = Path("data/eval/e2e_recommendation_eval_v2_2_500.csv")
AUDIT_TXT = Path("eval_results/audit/e2e_gold_audit_v2_2_50.txt")
AUDIT_JSONL = Path("eval_results/audit/e2e_gold_audit_v2_2_50.jsonl")

SAMPLE_N = 50
SEED = 42

# Gold 생성 전용 threshold. Production score_cocktail 의 threshold 와 의도적으로 분리한다.
GOLD_AROMA_HIGH_MIN = {
    "coffee": 3.0,
    "minty": 2.5,
    "floral": 2.5,
    "woody": 3.0,
    "fruity": 3.0,
    "citrus": 3.0,
    "herbal": 3.0,
}

GOLD_TASTE_HIGH_MIN = {
    "sweet": 3.1,
    "sour": 3.1,
    "bitter": 2.7,
    "body": 2.5,
    "creamy": 3.0,
    "freshness": 3.7,
}

# 무알콜 pool(5개) 전용 완화 임계값. 무알콜 DB 구조적 한계 반영.
# None = 구조적으로 불가능(실제 값이 거의 없음) → high 요청 시 zero_high_impossible 로 라벨.
GOLD_TASTE_HIGH_MIN_ZERO = {
    "sweet": 3.0,      # 셜리(3.6), 피냐콜라다(3.4)
    "sour": 2.8,       # 노지토(3.0), 버진모히토(2.8)
    "bitter": None,    # 무알콜에 bitter 없음 (최대 0.4)
    "body": 2.0,       # 피냐콜라다(2.2) 만 통과
    "creamy": 2.5,     # 피냐콜라다(3.1) 만
    "freshness": 3.0,  # 노지토/모히토(4.0)
}

GOLD_AROMA_HIGH_MIN_ZERO = {
    "coffee": None,    # 무알콜에 coffee aroma 없음 (전부 1.0)
    "minty": 3.0,      # 노지토/모히토(5.0)
    "floral": None,    # 무알콜에 floral 없음 (전부 1.0)
    "woody": None,     # 무알콜에 woody 없음
    "fruity": 3.0,     # 피냐(5.0), 셜리(4.0)
    "citrus": 3.0,     # 4개 통과
    "herbal": 3.0,     # 노지토/모히토(4.0), 메리(3.0)
}

# strength hard filter — gold 생성 시 alcohol_score 범위 강제. V2.1 추가.
# orchestration_agent.STRENGTH_RANGE 와 의도적으로 분리: light 를 2.5 까지 완만하게 허용.
GOLD_STRENGTH_RANGE = {
    "light": (0.0, 2.5),
    "medium": (1.5, 3.5),
    "strong": (3.0, 5.0),
}

SLOT_TO_CATEGORY = {
    ("aroma", "coffee", "high"): "커피/디저트",
    ("aroma", "minty", "high"): "리프레싱 하이볼",
    ("taste", "bitter", "high"): "비터/허벌",
    ("taste", "creamy", "high"): "크리미",
    ("taste", "sour", "high"): "시트러스 사워",
}


class MockVector:
    sweetness_score = Decimal("3.0")
    bitterness_score = Decimal("3.0")
    sourness_score = Decimal("3.0")
    freshness_score = Decimal("3.0")
    body_score = Decimal("3.0")
    herbal_score = Decimal("2.0")
    citrus_score = Decimal("2.0")
    alcohol_score = Decimal("3.0")


def _parse_json(v: Any, default: Any):
    if pd.isna(v) or str(v).strip() == "":
        return default
    try:
        parsed = json.loads(v)
    except Exception:
        return default
    return parsed if parsed is not None else default


def _str_or_none(v: Any) -> str | None:
    if pd.isna(v) or str(v).strip() == "":
        return None
    return str(v).strip()


def build_profile(row) -> dict:
    merged = {
        "party_purpose": _str_or_none(row.gold_party_purpose),
        "current_mood": _str_or_none(row.gold_current_mood),
        "taste_profile": _parse_json(row.gold_taste_profile, {}),
        "aroma_profile": _parse_json(row.gold_aroma_profile, {}),
        "strength_preference": _str_or_none(row.gold_strength_preference),
        "disliked_bases": _parse_json(row.gold_disliked_bases, []),
        "favorite_drinks": _parse_json(row.gold_favorite_drinks, []),
    }
    return {"merged_slots": merged, "vector": MockVector(), "space": None}


def _max_recipe_aroma(recipe_ingredients: list[tuple], aroma_key: str) -> float:
    col = AROMA_TO_INGREDIENT.get(aroma_key)
    if not col:
        return 0.0
    max_score = 0.0
    for _, ingredient in recipe_ingredients:
        value = getattr(ingredient, col, 0) or 0
        try:
            max_score = max(max_score, float(value))
        except (TypeError, ValueError):
            continue
    return max_score


def _hard_check_slots(
    cocktail: Cocktail,
    recipe_ingredients: list[tuple],
    merged_slots: dict,
) -> tuple[bool, list[str]]:
    """
    사용자가 명시한 high 슬롯은 exact gold 조건으로 강제.
    low 슬롯은 '강한 충돌'만 제외 — 해당 축 값이 threshold 이상이면 탈락.
    strength=zero 면 무알콜 pool 완화 threshold 사용. None 이면 구조적 불가능 → 실패.
    """
    failures: list[str] = []
    is_zero = merged_slots.get("strength_preference") == "zero"
    taste_table = GOLD_TASTE_HIGH_MIN_ZERO if is_zero else GOLD_TASTE_HIGH_MIN
    aroma_table = GOLD_AROMA_HIGH_MIN_ZERO if is_zero else GOLD_AROMA_HIGH_MIN

    taste_profile: dict = merged_slots.get("taste_profile") or {}
    for axis, intensity in taste_profile.items():
        col = TASTE_TO_COCKTAIL.get(axis)
        if not col:
            continue
        threshold = taste_table.get(axis)
        raw = getattr(cocktail, col, None)
        val = float(raw) if raw is not None else 0.0
        if intensity == "high":
            if threshold is None:
                failures.append(f"taste.{axis}: zero pool 에 해당 축 없음")
            elif val < threshold:
                failures.append(f"taste.{axis}: {val:.1f} < {threshold:.1f}")
        elif intensity == "low":
            # low 충돌 체크는 기본 테이블 기준(zero 여부 상관 없이 '강한 충돌' 기준)
            base_threshold = GOLD_TASTE_HIGH_MIN.get(axis)
            if base_threshold is not None and val >= base_threshold:
                failures.append(f"taste.{axis} low conflict: {val:.1f} >= {base_threshold:.1f}")

    aroma_profile: dict = merged_slots.get("aroma_profile") or {}
    for axis, intensity in aroma_profile.items():
        val = _max_recipe_aroma(recipe_ingredients, axis)
        if intensity == "high":
            threshold = aroma_table.get(axis)
            if threshold is None:
                failures.append(f"aroma.{axis}: zero pool 에 해당 향 없음")
            elif val < threshold:
                failures.append(f"aroma.{axis}: {val:.1f} < {threshold:.1f}")
        elif intensity == "low":
            base_threshold = GOLD_AROMA_HIGH_MIN.get(axis)
            if base_threshold is not None and val >= base_threshold:
                failures.append(f"aroma.{axis} low conflict: {val:.1f} >= {base_threshold:.1f}")

    return not failures, failures


def _slot_categories(merged_slots: dict) -> set[str]:
    out: set[str] = set()
    taste_profile: dict = merged_slots.get("taste_profile") or {}
    aroma_profile: dict = merged_slots.get("aroma_profile") or {}

    for axis, intensity in taste_profile.items():
        category = SLOT_TO_CATEGORY.get(("taste", axis, intensity))
        if category:
            out.add(category)
    for axis, intensity in aroma_profile.items():
        category = SLOT_TO_CATEGORY.get(("aroma", axis, intensity))
        if category:
            out.add(category)
    return out


def _base_pool(profile: dict, all_ri: dict, available_ids, all_cocktails: list[Cocktail]):
    merged = profile["merged_slots"]
    strength = merged.get("strength_preference")
    strength_range = GOLD_STRENGTH_RANGE.get(strength) if strength != "zero" else None

    pool: list[Cocktail] = []
    for cocktail in all_cocktails:
        if strength == "zero":
            if not cocktail.is_non_alcoholic:
                continue
        else:
            if cocktail.is_non_alcoholic:
                continue
            # strength hard filter: 명시된 선호(light/medium/strong)만 허용
            if strength_range is not None:
                raw = getattr(cocktail, "alcohol_score", None)
                val = float(raw) if raw is not None else 2.5
                lo, hi = strength_range
                if not (lo <= val <= hi):
                    continue

        recipe_ingredients = all_ri.get(cocktail.cocktail_id, [])
        if _has_disliked_base(merged, recipe_ingredients):
            continue
        if _is_unstockable(recipe_ingredients, available_ids):
            continue
        if _has_zero_taste_conflict(merged, cocktail):
            continue
        pool.append(cocktail)
    return pool


def _compute_unlabelable_reason(
    merged: dict, base_size: int, passed_size: int, exact_count: int
) -> str | None:
    """gold 가 비었을 때 '왜 비었는지' 분류. 가득차면 None."""
    if exact_count > 0:
        return None
    has_high = False
    for v in (merged.get("taste_profile") or {}).values():
        if v == "high":
            has_high = True
            break
    if not has_high:
        for v in (merged.get("aroma_profile") or {}).values():
            if v == "high":
                has_high = True
                break
    strength = merged.get("strength_preference")
    if strength == "zero" and has_high:
        return "zero_high_impossible"
    if base_size == 0:
        return "base_pool_empty"
    if passed_size == 0:
        return "no_candidates_passed_hard_check"
    return "unknown"


def generate_gold(row, all_ri: dict, available_ids, all_cocktails: list[Cocktail]) -> dict:
    profile = build_profile(row)
    merged = profile["merged_slots"]
    base_pool = _base_pool(profile, all_ri, available_ids, all_cocktails)

    checked: list[dict] = []
    passed: list[tuple[Cocktail, float]] = []
    for cocktail in base_pool:
        recipe_ingredients = all_ri.get(cocktail.cocktail_id, [])
        ok, failures = _hard_check_slots(cocktail, recipe_ingredients, merged)
        score = score_cocktail(cocktail, profile, recipe_ingredients)
        checked.append({
            "cocktail": cocktail,
            "score": score,
            "passed": ok,
            "failures": failures,
        })
        if ok:
            passed.append((cocktail, score))

    passed.sort(key=lambda x: -x[1])
    checked.sort(key=lambda x: (-int(x["passed"]), -float(x["score"])))

    exact = passed[:3]
    gold_cocktails = [cocktail.name_kr for cocktail, _ in exact]
    gold_categories = {cocktail.category for cocktail, _ in exact}
    gold_categories |= _slot_categories(merged)

    reason = _compute_unlabelable_reason(
        merged, len(base_pool), len(passed), len(exact)
    )

    return {
        "profile": profile,
        "base_pool_size": len(base_pool),
        "passed_pool_size": len(passed),
        "gold_cocktails": gold_cocktails,
        "gold_categories": sorted(gold_categories),
        "exact": exact,
        "checked": checked,
        "unlabelable_reason": reason,
    }


def _row_to_output(row, gold: dict) -> dict:
    return {
        "case_id": int(row.case_id),
        "user_text": row.user_text,
        "gold_current_mood": _str_or_none(row.gold_current_mood),
        "gold_party_purpose": _str_or_none(row.gold_party_purpose),
        "gold_taste_profile": json.dumps(_parse_json(row.gold_taste_profile, {}), ensure_ascii=False),
        "gold_aroma_profile": json.dumps(_parse_json(row.gold_aroma_profile, {}), ensure_ascii=False),
        "gold_strength_preference": _str_or_none(row.gold_strength_preference),
        "gold_disliked_bases": json.dumps(_parse_json(row.gold_disliked_bases, []), ensure_ascii=False),
        "gold_favorite_drinks": json.dumps(_parse_json(row.gold_favorite_drinks, []), ensure_ascii=False),
        "gold_expected_cocktails": json.dumps(gold["gold_cocktails"], ensure_ascii=False),
        "gold_allowed_categories": json.dumps(gold["gold_categories"], ensure_ascii=False),
        "unlabelable_reason": gold["unlabelable_reason"] or "",
    }


def _audit_block(row, gold: dict) -> tuple[str, dict]:
    lines: list[str] = []
    lines.append("=" * 76)
    lines.append(f"[case_id={row.case_id}]")
    lines.append(f"  user_text  : {str(row.user_text)[:260]}")
    lines.append(
        "  slots      : "
        f"strength={row.gold_strength_preference}  "
        f"taste={row.gold_taste_profile}  "
        f"aroma={row.gold_aroma_profile}  "
        f"bases={row.gold_disliked_bases}  "
        f"favs={row.gold_favorite_drinks}  "
        f"mood={row.gold_current_mood}  purpose={row.gold_party_purpose}"
    )
    lines.append(
        f"  pool       : base={gold['base_pool_size']}  "
        f"hard_check_pass={gold['passed_pool_size']}"
    )
    lines.append(f"  gold       : {gold['gold_cocktails']}")
    lines.append(f"  categories : {gold['gold_categories']}")
    if gold["unlabelable_reason"]:
        lines.append(f"  unlabelable: {gold['unlabelable_reason']}")

    lines.append("  hard-check 통과 후보 top:")
    if not gold["exact"]:
        lines.append("    [없음]")
    else:
        for idx, (cocktail, score) in enumerate(gold["exact"], 1):
            alc = "무알콜" if cocktail.is_non_alcoholic else "알콜"
            lines.append(f"    {idx}. {cocktail.name_kr}  score={score}  cat={cocktail.category}  {alc}")

    failed_preview = [item for item in gold["checked"] if not item["passed"]][:5]
    if failed_preview:
        lines.append("  탈락 후보 예시:")
        for item in failed_preview:
            cocktail = item["cocktail"]
            fail_text = "; ".join(item["failures"][:3])
            lines.append(f"    - {cocktail.name_kr}  score={item['score']}  why={fail_text}")

    lines.append("  판정: [ OK / NG / 애매 ]  코멘트:")
    lines.append("")

    payload = {
        "case_id": int(row.case_id),
        "user_text": str(row.user_text),
        "gold_slots": {
            "strength": _str_or_none(row.gold_strength_preference),
            "taste": _parse_json(row.gold_taste_profile, {}),
            "aroma": _parse_json(row.gold_aroma_profile, {}),
            "disliked_bases": _parse_json(row.gold_disliked_bases, []),
            "favorite_drinks": _parse_json(row.gold_favorite_drinks, []),
            "current_mood": _str_or_none(row.gold_current_mood),
            "party_purpose": _str_or_none(row.gold_party_purpose),
        },
        "base_pool_size": gold["base_pool_size"],
        "passed_pool_size": gold["passed_pool_size"],
        "gold_cocktails": gold["gold_cocktails"],
        "gold_categories": gold["gold_categories"],
        "unlabelable_reason": gold["unlabelable_reason"],
        "top_passed": [
            {
                "rank": idx,
                "name_kr": cocktail.name_kr,
                "category": cocktail.category,
                "score": float(score),
                "is_non_alcoholic": bool(cocktail.is_non_alcoholic),
            }
            for idx, (cocktail, score) in enumerate(gold["exact"], 1)
        ],
        "failed_preview": [
            {
                "name_kr": item["cocktail"].name_kr,
                "category": item["cocktail"].category,
                "score": float(item["score"]),
                "failures": item["failures"],
            }
            for item in failed_preview
        ],
    }
    return "\n".join(lines), payload


def main() -> None:
    df = pd.read_csv(SRC_CSV)
    print(f"source: {SRC_CSV}  n={len(df)}")

    db = SessionLocal()
    rows_out: list[dict] = []
    all_gold: list[dict] = []

    try:
        all_ri = get_all_recipes_with_ingredients(db)
        available_ids = get_available_ingredient_ids(db)
        all_cocktails = db.query(Cocktail).filter(Cocktail.is_active == True).all()
        print(f"active cocktails: {len(all_cocktails)}")

        for idx, row in enumerate(df.itertuples(index=False), 1):
            gold = generate_gold(row, all_ri, available_ids, all_cocktails)
            rows_out.append(_row_to_output(row, gold))
            all_gold.append({"row": row, "gold": gold})

            if idx % 50 == 0:
                print(f"  progress {idx}/{len(df)}")
    finally:
        db.close()

    out_df = pd.DataFrame(rows_out)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    random.seed(SEED)
    sample_idx = sorted(random.sample(range(len(all_gold)), min(SAMPLE_N, len(all_gold))))
    audit_text: list[str] = []
    audit_jsonl: list[dict] = []
    for idx in sample_idx:
        block, payload = _audit_block(all_gold[idx]["row"], all_gold[idx]["gold"])
        audit_text.append(block)
        audit_jsonl.append(payload)

    AUDIT_TXT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_TXT.write_text("\n".join(audit_text), encoding="utf-8")
    AUDIT_JSONL.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in audit_jsonl),
        encoding="utf-8",
    )

    gold_counts = out_df["gold_expected_cocktails"].apply(lambda s: len(json.loads(s)))
    passed_counts = [item["gold"]["passed_pool_size"] for item in all_gold]
    base_counts = [item["gold"]["base_pool_size"] for item in all_gold]

    print(f"\n[saved] {OUT_CSV}  rows={len(out_df)}")
    print(f"[saved] {AUDIT_TXT}")
    print(f"[saved] {AUDIT_JSONL}")
    print("\n[통계]")
    print(f"  base_pool min/med/max = {min(base_counts)}/{sorted(base_counts)[len(base_counts)//2]}/{max(base_counts)}")
    print(f"  pass_pool min/med/max = {min(passed_counts)}/{sorted(passed_counts)[len(passed_counts)//2]}/{max(passed_counts)}")
    print("  gold_count distribution:")
    for count, n in gold_counts.value_counts().sort_index().items():
        print(f"    {count}: {n}")
    print(f"  exact_gold_coverage = {(gold_counts.gt(0).mean() * 100):.1f}%")

    reason_counts = out_df["unlabelable_reason"].value_counts(dropna=False)
    print("  unlabelable_reason distribution:")
    for reason, n in reason_counts.items():
        label = reason if reason else "<labeled>"
        print(f"    {label}: {n}")


if __name__ == "__main__":
    main()
