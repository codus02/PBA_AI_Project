"""특정 case 의 두 칵테일 score breakdown 비교.

용도: 정글 버드가 gold 를 누른 case 1개 골라서 어느 axis 에서 차이 나는지 확인.

사용:
  python scripts/inspect_score_breakdown.py --case 159 --vs "정글 버드,페이퍼 플레인"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app.agents.orchestration_agent import score_cocktail_breakdown
from app.db.database import SessionLocal
from app.db.models import Cocktail
from app.db.crud import get_all_recipes_with_ingredients


from decimal import Decimal


class MockVector:
    # eval_e2e_recommendation.py 의 MockVector 와 동일 — 모두 medium 기준값.
    sweetness_score = Decimal("3.0")
    bitterness_score = Decimal("3.0")
    sourness_score = Decimal("3.0")
    freshness_score = Decimal("3.0")
    body_score = Decimal("3.0")
    herbal_score = Decimal("2.0")
    citrus_score = Decimal("2.0")
    alcohol_score = Decimal("3.0")


def _parse(s):
    if pd.isna(s) or not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/eval/e2e_recommendation_eval_v2_2_500.csv")
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--vs", required=True, help="콤마구분 칵테일 한국어 이름")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    row = df[df["case_id"] == args.case]
    if row.empty:
        raise SystemExit(f"case {args.case} not found")
    row = row.iloc[0]

    merged = {
        "party_purpose": row.get("gold_party_purpose") or None,
        "current_mood": row.get("gold_current_mood") or None,
        "taste_profile": _parse(row.get("gold_taste_profile")) or {},
        "aroma_profile": _parse(row.get("gold_aroma_profile")) or {},
        "strength_preference": row.get("gold_strength_preference") or None,
        "disliked_bases": _parse(row.get("gold_disliked_bases")) or [],
        "favorite_drinks": _parse(row.get("gold_favorite_drinks")) or [],
    }
    profile = {"merged_slots": merged, "vector": MockVector(), "space": None}

    print(f"=== case {args.case} ===")
    print(f"user_text: {row.get('user_text','')}")
    print(f"gold slots: mood={merged['current_mood']}  party={merged['party_purpose']}  strength={merged['strength_preference']}")
    print(f"  taste : {merged['taste_profile']}")
    print(f"  aroma : {merged['aroma_profile']}")
    print(f"  bases!={merged['disliked_bases']}")
    print()

    targets = [n.strip() for n in args.vs.split(",") if n.strip()]
    db: Session = SessionLocal()
    try:
        all_ri = get_all_recipes_with_ingredients(db)
        cocktails = db.query(Cocktail).filter(Cocktail.name_kr.in_(targets)).all()
        by_name = {c.name_kr: c for c in cocktails}
        missing = [n for n in targets if n not in by_name]
        if missing:
            print(f"[warn] not found in DB: {missing}")

        breakdowns = {}
        for name in targets:
            c = by_name.get(name)
            if c is None:
                continue
            ri = all_ri.get(c.cocktail_id, [])
            bd = score_cocktail_breakdown(c, profile, ri)
            breakdowns[name] = bd

        if not breakdowns:
            return

        keys = ["vector_similarity", "taste_profile", "aroma_profile",
                "space_mood", "context", "favorite_drinks", "strength", "total"]
        # 파일에 따라 키가 다를 수 있어 실제로 채워진 키만 출력
        present = sorted({k for bd in breakdowns.values() for k in bd.keys()},
                         key=lambda k: keys.index(k) if k in keys else 99)

        # cocktail metadata
        print(f"{'metadata':<22}", *(f"{n:>16}" for n in breakdowns.keys()))
        for attr in ("category", "alcohol_volume", "sweet_level", "sour_level",
                     "bitter_level", "freshness_level", "body_level"):
            vals = []
            for n in breakdowns.keys():
                c = by_name[n]
                v = getattr(c, attr, None)
                vals.append(str(v) if v is not None else "-")
            print(f"  {attr:<20}", *(f"{v:>16}" for v in vals))
        print()

        print(f"{'breakdown':<22}", *(f"{n:>16}" for n in breakdowns.keys()))
        for k in present:
            row_vals = [breakdowns[n].get(k, 0.0) for n in breakdowns.keys()]
            print(f"  {k:<20}", *(f"{v:>16.2f}" for v in row_vals))
        # delta
        if len(breakdowns) == 2:
            a, b = list(breakdowns.keys())
            print()
            print(f"  {'delta ('+a+'-'+b+')':<20}")
            for k in present:
                d = breakdowns[a].get(k, 0.0) - breakdowns[b].get(k, 0.0)
                marker = " <<<" if abs(d) >= 3 else ""
                print(f"  {k:<20}  {d:>16.2f}{marker}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
