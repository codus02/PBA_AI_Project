"""E2E eval dataset 생성 — user_text + gold_slots + gold_cocktails 통합.

입력: data/eval/slot_extraction_eval_v2_500.csv (user_text + gold slots)
출력: data/eval/e2e_recommendation_eval_v1_500.csv

생성 규칙:
  - slots 로 mock profile 조립 (space=None, MockVector 중립값)
  - 하드필터:
      * strength=zero → non_alcoholic only / else → alcohol only
      * disliked_bases 포함 칵테일 제외
      * taste=zero 축의 level 높은 칵테일 제외
      * 재고 없는 재료 포함 제외
  - score_cocktail 로 랭킹 → 항상 top-3 (pool <3 이면 pool 전체)
  - gold_allowed_categories = top-3 의 category set
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.orchestration_agent import (
    _has_disliked_base,
    _has_zero_taste_conflict,
    _is_unstockable,
    score_cocktail,
)
from app.db.crud import get_all_recipes_with_ingredients, get_available_ingredient_ids
from app.db.database import SessionLocal
from app.db.models import Cocktail

SRC_CSV = Path("data/eval/slot_extraction_eval_v2_500.csv")
OUT_CSV = Path("data/eval/e2e_recommendation_eval_v1_500.csv")


class MockVector:
    sweetness_score = Decimal("3.0")
    bitterness_score = Decimal("3.0")
    sourness_score = Decimal("3.0")
    freshness_score = Decimal("3.0")
    body_score = Decimal("3.0")
    herbal_score = Decimal("2.0")
    citrus_score = Decimal("2.0")
    alcohol_score = Decimal("3.0")


def _parse_json(v, default):
    if pd.isna(v) or str(v).strip() == "":
        return default
    try:
        return json.loads(v)
    except Exception:
        return default


def _str_or_none(v):
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


def generate_top3(row, all_ri, available_ids, all_cocktails):
    profile = build_profile(row)
    merged = profile["merged_slots"]
    strength = merged.get("strength_preference")

    pool = []
    for c in all_cocktails:
        if strength == "zero":
            if not c.is_non_alcoholic:
                continue
        else:
            if c.is_non_alcoholic:
                continue
        ri = all_ri.get(c.cocktail_id, [])
        if _has_disliked_base(merged, ri):
            continue
        if _is_unstockable(ri, available_ids):
            continue
        if _has_zero_taste_conflict(merged, c):
            continue
        pool.append(c)

    scored = []
    for c in pool:
        ri = all_ri.get(c.cocktail_id, [])
        s = score_cocktail(c, profile, ri)
        scored.append((c, s))
    scored.sort(key=lambda x: -x[1])
    return scored[:3], len(pool)


def main():
    df = pd.read_csv(SRC_CSV)
    print(f"source: {SRC_CSV}  n={len(df)}")

    db = SessionLocal()
    rows_out: list[dict] = []
    pool_sizes: list[int] = []
    top1_scores: list[float] = []
    empty_count = 0
    short_gold_count = 0

    try:
        all_ri = get_all_recipes_with_ingredients(db)
        available_ids = get_available_ingredient_ids(db)
        all_cocktails = db.query(Cocktail).filter(Cocktail.is_active == True).all()
        print(f"active cocktails: {len(all_cocktails)}")

        for idx, row in enumerate(df.itertuples(index=False), 1):
            top3, pool_size = generate_top3(row, all_ri, available_ids, all_cocktails)
            pool_sizes.append(pool_size)

            if not top3:
                empty_count += 1
                gold_cocktails = []
                gold_categories = []
            else:
                top1_scores.append(float(top3[0][1]))
                gold_cocktails = [c.name_kr for c, _ in top3]
                gold_categories = sorted({c.category for c, _ in top3})
                if len(top3) < 3:
                    short_gold_count += 1

            rows_out.append({
                "case_id": int(row.case_id),
                "user_text": row.user_text,
                "gold_current_mood": _str_or_none(row.gold_current_mood),
                "gold_party_purpose": _str_or_none(row.gold_party_purpose),
                "gold_taste_profile": json.dumps(
                    _parse_json(row.gold_taste_profile, {}), ensure_ascii=False
                ),
                "gold_aroma_profile": json.dumps(
                    _parse_json(row.gold_aroma_profile, {}), ensure_ascii=False
                ),
                "gold_strength_preference": _str_or_none(row.gold_strength_preference),
                "gold_disliked_bases": json.dumps(
                    _parse_json(row.gold_disliked_bases, []), ensure_ascii=False
                ),
                "gold_favorite_drinks": json.dumps(
                    _parse_json(row.gold_favorite_drinks, []), ensure_ascii=False
                ),
                "gold_expected_cocktails": json.dumps(gold_cocktails, ensure_ascii=False),
                "gold_allowed_categories": json.dumps(gold_categories, ensure_ascii=False),
            })

            if idx % 50 == 0:
                print(f"  progress {idx}/{len(df)}")

    finally:
        db.close()

    out_df = pd.DataFrame(rows_out)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"\n[saved] {OUT_CSV}  rows={len(out_df)}")
    print(f"columns: {list(out_df.columns)}")
    print(f"\n[통계]")
    print(f"  pool_size min/med/max = {min(pool_sizes)}/{sorted(pool_sizes)[len(pool_sizes)//2]}/{max(pool_sizes)}")
    print(f"  top-1 score min/med/max = {min(top1_scores):.1f}/{sorted(top1_scores)[len(top1_scores)//2]:.1f}/{max(top1_scores):.1f}")
    print(f"  empty gold (pool=0): {empty_count}/{len(df)}")
    print(f"  short gold (1~2개): {short_gold_count}/{len(df)}")

    # pool<3 인 case 에 대한 strength 분포
    zero_n = sum(1 for r in rows_out if r["gold_strength_preference"] == "zero")
    zero_short = sum(
        1
        for r in rows_out
        if r["gold_strength_preference"] == "zero"
        and 1 <= len(json.loads(r["gold_expected_cocktails"])) < 3
    )
    print(f"  zero-strength rows: {zero_n}  (1~2개 gold: {zero_short})")


if __name__ == "__main__":
    main()
