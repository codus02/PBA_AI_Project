from __future__ import annotations

import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.database import SessionLocal
from app.db.crud import get_all_recipes_with_ingredients, get_available_ingredient_ids
from app.agents.orchestration_agent import (
    synthesize_query,
    retrieve_candidates,
    rerank_with_qwen,
    score_cocktail,
    _has_disliked_base,
    _is_unstockable,
)

RAG_RETRIEVE_N = 20
CSV_PATH = Path("data/eval/recommendation_eval_v2_500.csv")


def _parse_list(val):
    if pd.isna(val) or str(val).strip() == "":
        return []
    try:
        loaded = json.loads(val)
        return loaded if isinstance(loaded, list) else [str(val).strip()]
    except Exception:
        return [str(val).strip()]


class MockVector:
    sweetness_score = Decimal("3.0")
    bitterness_score = Decimal("3.0")
    sourness_score = Decimal("3.0")
    freshness_score = Decimal("3.0")
    body_score = Decimal("3.0")
    herbal_score = Decimal("2.0")
    citrus_score = Decimal("2.0")
    alcohol_score = Decimal("3.0")


def _list_to_profile(preferred, disliked) -> dict[str, str]:
    profile: dict[str, str] = {}
    for tag in preferred or []:
        if tag:
            profile[str(tag)] = "high"
    for tag in disliked or []:
        if tag:
            profile[str(tag)] = "low"
    return profile


def _make_mock_profile(row) -> dict:
    return {
        "merged_slots": {
            "party_purpose": row.get("party_purpose"),
            "current_mood": row.get("current_mood"),
            "taste_profile": _list_to_profile(
                _parse_list(row.get("preferred_tastes")),
                _parse_list(row.get("disliked_tastes")),
            ),
            "aroma_profile": _list_to_profile(
                _parse_list(row.get("preferred_aromas")),
                _parse_list(row.get("disliked_aromas")),
            ),
            "strength_preference": row.get("strength_preference"),
            "disliked_bases": _parse_list(row.get("disliked_bases")),
            "favorite_drinks": _parse_list(row.get("favorite_drinks")),
        },
        "vector": MockVector(),
        "space": None,
    }


def _llm_top3(
    db: Session,
    profile: dict,
    all_ri: dict,
    available_ids,
) -> list[dict]:
    merged = profile["merged_slots"]

    query_text = synthesize_query(profile)
    retrieved = retrieve_candidates(
        db=db,
        query_text=query_text,
        top_k=RAG_RETRIEVE_N,
        exclude_ids=None,
        strength_preference=merged.get("strength_preference"),
    )

    survivors: list[tuple] = []
    for cocktail, dist in retrieved:
        ri = all_ri.get(cocktail.cocktail_id, [])
        if _has_disliked_base(merged, ri):
            continue
        if _is_unstockable(ri, available_ids):
            continue
        survivors.append((cocktail, dist))

    if not survivors:
        return []

    survivor_cocktails = [c for c, _ in survivors]
    reranked = rerank_with_qwen(profile, survivor_cocktails, k=3)

    if reranked:
        id_to_cocktail = {c.cocktail_id: c for c in survivor_cocktails}
        top3 = []
        for item in reranked:
            c = id_to_cocktail.get(item["cocktail_id"])
            if c is None:
                continue
            top3.append({
                "name_kr": c.name_kr,
                "category": c.category,
                "source": "rag_qwen",
            })
        if top3:
            return top3[:3]

    scored = []
    for c, _dist in survivors:
        ri = all_ri.get(c.cocktail_id, [])
        s = score_cocktail(c, profile, ri)
        scored.append({
            "name_kr": c.name_kr,
            "category": c.category,
            "score": s,
            "source": "rag_fallback",
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:3]


def eval_recommendation(limit: int | None = None):
    df = pd.read_csv(CSV_PATH)
    if limit:
        df = df.head(limit)

    db: Session = SessionLocal()
    hit_k1 = 0
    hit_k3 = 0
    cat_hit = 0
    total = 0
    no_candidate = 0

    t0 = time.perf_counter()

    try:
        all_ri = get_all_recipes_with_ingredients(db)
        available_ids = get_available_ingredient_ids(db)

        for i, (_, row) in enumerate(df.iterrows(), 1):
            gold_cocktails = _parse_list(row.get("gold_expected_cocktails", ""))
            gold_categories = _parse_list(row.get("gold_allowed_categories", ""))

            if not gold_cocktails and not gold_categories:
                continue

            profile = _make_mock_profile(row)
            top3 = _llm_top3(db, profile, all_ri, available_ids)
            if not top3:
                no_candidate += 1
                total += 1
                continue

            top1 = top3[:1]
            total += 1

            if gold_cocktails and any(t["name_kr"] in gold_cocktails for t in top1):
                hit_k1 += 1

            if gold_cocktails and any(t["name_kr"] in gold_cocktails for t in top3):
                hit_k3 += 1

            if gold_categories and any(t["category"] in gold_categories for t in top3):
                cat_hit += 1

            if i % 25 == 0:
                dt = time.perf_counter() - t0
                print(f"  progress {i}/{len(df)}  elapsed={dt:.1f}s")

    finally:
        db.close()

    elapsed = time.perf_counter() - t0
    denom = max(total, 1)

    print(f"\n[Qwen 추천 적합도 평가] 총 {total}건")
    print(f"  Hit@1  (top1 정답 포함): {hit_k1}/{total} = {hit_k1/denom*100:.1f}%")
    print(f"  Hit@3  (top3 정답 포함): {hit_k3}/{total} = {hit_k3/denom*100:.1f}%")
    print(f"  카테고리 Hit@3:           {cat_hit}/{total} = {cat_hit/denom*100:.1f}%")
    print(f"  후보 없음:                {no_candidate}/{total} = {no_candidate/denom*100:.1f}%")
    print(f"  소요 시간:                {elapsed:.1f}s ({elapsed/denom:.2f}s/case)")

    return {
        "hit@1": hit_k1 / denom,
        "hit@3": hit_k3 / denom,
        "cat_hit@3": cat_hit / denom,
        "no_candidate_rate": no_candidate / denom,
        "n": total,
        "elapsed_sec": elapsed,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    eval_recommendation(limit=args.limit)
