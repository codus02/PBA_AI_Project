# scripts/eval_recommendation.py
import sys, json, ast
sys.path.insert(0, ".")

import pandas as pd
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.agents.orchestration_agent import build_user_profile, recommend_top_k

def _parse_list(val):
    if pd.isna(val) or str(val).strip() == "":
        return []
    try:
        return json.loads(val)
    except:
        return [str(val).strip()]

# eval_recommendation.py - _make_mock_profile 함수 수정

from decimal import Decimal

class MockVector:
    sweetness_score  = Decimal("3.0")
    bitterness_score = Decimal("3.0")
    sourness_score   = Decimal("3.0")
    freshness_score  = Decimal("3.0")
    body_score       = Decimal("3.0")
    herbal_score     = Decimal("2.0")
    citrus_score     = Decimal("2.0")
    alcohol_score    = Decimal("3.0")

def _make_mock_profile(row) -> dict:
    return {
        "merged_slots": {
            "party_purpose":       row.get("party_purpose"),
            "current_mood":        row.get("current_mood"),
            "preferred_tastes":    _parse_list(row.get("preferred_tastes")),
            "disliked_tastes":     _parse_list(row.get("disliked_tastes")),
            "preferred_aromas":    _parse_list(row.get("preferred_aromas")),
            "disliked_aromas":     _parse_list(row.get("disliked_aromas")),
            "strength_preference": row.get("strength_preference"),
            "disliked_bases":      _parse_list(row.get("disliked_bases")),
            "finish_preference":   row.get("finish_preference"),
        },
        "vector": MockVector(),  # None → 기본값으로 교체
        "space":  None,
    }

def eval_recommendation():
    df  = pd.read_csv("data/eval/recommendation_eval_v2_500.csv")
    db: Session = SessionLocal()

    hit_k1  = 0   # top-1 정답 포함
    hit_k3  = 0   # top-3 정답 포함
    cat_hit = 0   # 카테고리 적중
    total   = 0

    from app.db.crud import get_all_recipes_with_ingredients
    from app.db.models import Cocktail
    from app.agents.orchestration_agent import score_cocktail, _has_disliked_base

    all_ri = get_all_recipes_with_ingredients(db)
    candidates = db.query(Cocktail).filter(Cocktail.is_active == True).all()

    for _, row in df.iterrows():
        gold_cocktails = _parse_list(row.get("gold_expected_cocktails", ""))
        gold_categories = _parse_list(row.get("gold_allowed_categories", ""))

        if not gold_cocktails and not gold_categories:
            continue

        profile = _make_mock_profile(row)
        merged  = profile["merged_slots"]

        # 점수 계산
        scored = []
        for c in candidates:
            ri = all_ri.get(c.cocktail_id, [])
            if _has_disliked_base(merged, ri):
                continue
            s = score_cocktail(c, profile, ri)
            scored.append({"name_kr": c.name_kr, "category": c.category, "score": s})

        scored.sort(key=lambda x: x["score"], reverse=True)
        top3 = scored[:3]
        top1 = scored[:1]

        total += 1

        # Hit@1
        if gold_cocktails and any(t["name_kr"] in gold_cocktails for t in top1):
            hit_k1 += 1

        # Hit@3
        if gold_cocktails and any(t["name_kr"] in gold_cocktails for t in top3):
            hit_k3 += 1

        # 카테고리 Hit@3
        if gold_categories and any(t["category"] in gold_categories for t in top3):
            cat_hit += 1

    db.close()

    print(f"\n[추천 적합도 평가] 총 {total}건")
    print(f"  Hit@1  (top1 정답 포함): {hit_k1}/{total} = {hit_k1/total*100:.1f}%")
    print(f"  Hit@3  (top3 정답 포함): {hit_k3}/{total} = {hit_k3/total*100:.1f}%")
    print(f"  카테고리 Hit@3:           {cat_hit}/{total} = {cat_hit/total*100:.1f}%")

    return {"hit@1": hit_k1/total, "hit@3": hit_k3/total, "cat_hit@3": cat_hit/total}

if __name__ == "__main__":
    eval_recommendation()