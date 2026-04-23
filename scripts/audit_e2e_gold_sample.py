"""E2E gold 생성 규칙 감사 — 50건 랜덤 샘플 dump.

slot_extraction_eval_v2_500.csv 의 gold_slots 를 입력으로,
현재 추천 로직 (score_cocktail) 으로 top-5 칵테일 생성 → 사람이 눈으로 판정.

출력:
  eval_results/audit/e2e_gold_audit_50.txt   (사람용 dump)
  eval_results/audit/e2e_gold_audit_50.jsonl (구조화)
"""
from __future__ import annotations

import json
import random
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.crud import get_all_recipes_with_ingredients, get_available_ingredient_ids
from app.db.database import SessionLocal
from app.db.models import Cocktail
from app.agents.orchestration_agent import (
    _has_disliked_base,
    _has_zero_taste_conflict,
    _is_unstockable,
    score_cocktail,
)

SAMPLE_N = 50
SEED = 42


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


def generate_top5(db, row, all_ri, available_ids, all_cocktails):
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
    return scored[:5], len(pool)


def main():
    df = pd.read_csv("data/eval/slot_extraction_eval_v2_500.csv")
    random.seed(SEED)
    sample_idx = sorted(random.sample(range(len(df)), SAMPLE_N))

    db = SessionLocal()
    out_txt: list[str] = []
    out_jsonl: list[dict] = []

    try:
        all_ri = get_all_recipes_with_ingredients(db)
        available_ids = get_available_ingredient_ids(db)
        all_cocktails = db.query(Cocktail).filter(Cocktail.is_active == True).all()

        for idx in sample_idx:
            row = df.iloc[idx]
            top5, pool_size = generate_top5(db, row, all_ri, available_ids, all_cocktails)

            block: list[str] = []
            block.append("=" * 70)
            block.append(f"[case_id={row.case_id}]")
            block.append(f"  user_text  : {str(row.user_text)[:200]}")
            block.append(
                f"  slots      : strength={row.gold_strength_preference}  "
                f"taste={row.gold_taste_profile}  aroma={row.gold_aroma_profile}  "
                f"bases={row.gold_disliked_bases}  favs={row.gold_favorite_drinks}  "
                f"mood={row.gold_current_mood}  purpose={row.gold_party_purpose}"
            )
            block.append(f"  pool_size  : {pool_size} (하드필터 통과)")

            if not top5:
                block.append("  생성 top-5 : [없음 — 하드필터 통과한 칵테일이 0개]")
            else:
                block.append("  생성 top-5 :")
                for i, (c, s) in enumerate(top5, 1):
                    mark = "**" if i <= 3 else "  "
                    alc = "알콜" if not c.is_non_alcoholic else "무알콜"
                    block.append(
                        f"    {mark}{i}. {c.name_kr}  (score={s}, cat={c.category}, {alc})"
                    )

            block.append("  판정: [ OK / NG / 애매 ]  코멘트:")
            block.append("")

            msg = "\n".join(block)
            print(msg)
            out_txt.extend(block)

            out_jsonl.append({
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
                "pool_size": pool_size,
                "top5": [
                    {
                        "rank": i,
                        "name_kr": c.name_kr,
                        "score": float(s),
                        "category": c.category,
                        "is_non_alcoholic": bool(c.is_non_alcoholic),
                    }
                    for i, (c, s) in enumerate(top5, 1)
                ],
            })
    finally:
        db.close()

    out_dir = Path("eval_results/audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e2e_gold_audit_50.txt").write_text("\n".join(out_txt))
    (out_dir / "e2e_gold_audit_50.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in out_jsonl)
    )
    print(f"\n[saved] {out_dir}/e2e_gold_audit_50.txt")
    print(f"[saved] {out_dir}/e2e_gold_audit_50.jsonl")

    # 요약 통계
    pool_sizes = [r["pool_size"] for r in out_jsonl]
    top1_scores = [r["top5"][0]["score"] for r in out_jsonl if r["top5"]]
    empty = sum(1 for r in out_jsonl if not r["top5"])
    print(f"\n[요약] pool_size min/med/max = {min(pool_sizes)}/{sorted(pool_sizes)[len(pool_sizes)//2]}/{max(pool_sizes)}")
    print(f"       top-1 score min/med/max = {min(top1_scores):.1f}/{sorted(top1_scores)[len(top1_scores)//2]:.1f}/{max(top1_scores):.1f}")
    print(f"       빈 gold (pool=0): {empty}/{SAMPLE_N}")


if __name__ == "__main__":
    main()
