from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
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
    rerank_with_llm,
    RERANK_REASON_POOL_N,
    score_cocktail,
    _expand_retrieved_candidates,
    _has_disliked_base,
    _is_unstockable,
    _has_zero_taste_conflict,
)
from scripts._eval_save import save_eval_result, short_model_name

RAG_RETRIEVE_N = 50
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
            profile[str(tag)] = "zero"
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
    use_rerank: bool = True,
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
    retrieved = _expand_retrieved_candidates(
        db,
        retrieved,
        merged,
        all_ri,
        exclude_ids=None,
    )

    survivors: list[tuple] = []
    for cocktail, dist in retrieved:
        ri = all_ri.get(cocktail.cocktail_id, [])
        if _has_disliked_base(merged, ri):
            continue
        if _is_unstockable(ri, available_ids):
            continue
        if _has_zero_taste_conflict(merged, cocktail):
            continue
        survivors.append((cocktail, dist))

    if not survivors:
        return []

    scored = []
    for c, _dist in survivors:
        ri = all_ri.get(c.cocktail_id, [])
        s = score_cocktail(c, profile, ri)
        scored.append({
            "cocktail_id": c.cocktail_id,
            "name_kr": c.name_kr,
            "category": c.category,
            "score": s,
            "source": "score_primary",
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    if use_rerank:
        survivor_cocktails = [c for c, _ in survivors]
        rerank_pool_ids = [item["cocktail_id"] for item in scored[:max(3, RERANK_REASON_POOL_N)]]
        id_to_cocktail = {c.cocktail_id: c for c in survivor_cocktails}
        rerank_pool = [id_to_cocktail[cid] for cid in rerank_pool_ids if cid in id_to_cocktail]
        reranked = rerank_with_llm(profile, rerank_pool, k=len(rerank_pool), recipe_ingredients=all_ri)
        if reranked:
            llm_ids = {int(item["cocktail_id"]) for item in reranked if item.get("cocktail_id")}
            for item in scored[:3]:
                if item["cocktail_id"] in llm_ids:
                    item["source"] = "score_llm_reason"
    return scored[:3]


def eval_recommendation(
    limit: int | None = None,
    tag: str | None = None,
    use_rerank: bool = True,
):
    df = pd.read_csv(CSV_PATH)
    if limit:
        df = df.head(limit)

    db: Session = SessionLocal()
    hit_k1 = 0
    hit_k3 = 0
    cat_hit = 0
    total = 0
    no_candidate = 0
    per_item: list[dict] = []

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
            top3 = _llm_top3(db, profile, all_ri, available_ids, use_rerank=use_rerank)
            case_id = row.get("case_id", i)

            if not top3:
                no_candidate += 1
                total += 1
                per_item.append({
                    "case_id": case_id,
                    "gold_cocktails": json.dumps(gold_cocktails, ensure_ascii=False),
                    "gold_categories": json.dumps(gold_categories, ensure_ascii=False),
                    "top3_names": "",
                    "top3_categories": "",
                    "top3_sources": "",
                    "hit_1": 0,
                    "hit_3": 0,
                    "cat_hit_3": 0,
                    "no_candidate": 1,
                })
                continue

            top1 = top3[:1]
            total += 1

            h1 = int(bool(gold_cocktails and any(t["name_kr"] in gold_cocktails for t in top1)))
            h3 = int(bool(gold_cocktails and any(t["name_kr"] in gold_cocktails for t in top3)))
            ch = int(bool(gold_categories and any(t["category"] in gold_categories for t in top3)))
            hit_k1 += h1
            hit_k3 += h3
            cat_hit += ch

            per_item.append({
                "case_id": case_id,
                "gold_cocktails": json.dumps(gold_cocktails, ensure_ascii=False),
                "gold_categories": json.dumps(gold_categories, ensure_ascii=False),
                "top3_names": json.dumps([t["name_kr"] for t in top3], ensure_ascii=False),
                "top3_categories": json.dumps([t["category"] for t in top3], ensure_ascii=False),
                "top3_sources": json.dumps([t.get("source", "") for t in top3], ensure_ascii=False),
                "hit_1": h1,
                "hit_3": h3,
                "cat_hit_3": ch,
                "no_candidate": 0,
            })

            if i % 25 == 0:
                dt = time.perf_counter() - t0
                print(f"  progress {i}/{len(df)}  elapsed={dt:.1f}s")

    finally:
        db.close()

    elapsed = time.perf_counter() - t0
    denom = max(total, 1)

    lines: list[str] = []

    def _log(msg: str = ""):
        print(msg)
        lines.append(msg)

    _log(f"\n[LLM 추천 적합도 평가] 총 {total}건")
    _log(f"  설정:                    rerank={'on' if use_rerank else 'off'}  retrieve_n={RAG_RETRIEVE_N}  rerank_pool={RERANK_REASON_POOL_N}")
    _log(f"  Hit@1  (top1 정답 포함): {hit_k1}/{total} = {hit_k1/denom*100:.1f}%")
    _log(f"  Hit@3  (top3 정답 포함): {hit_k3}/{total} = {hit_k3/denom*100:.1f}%")
    _log(f"  카테고리 Hit@3:           {cat_hit}/{total} = {cat_hit/denom*100:.1f}%")
    _log(f"  후보 없음:                {no_candidate}/{total} = {no_candidate/denom*100:.1f}%")
    _log(f"  소요 시간:                {elapsed:.1f}s ({elapsed/denom:.2f}s/case)")

    if tag:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_name = os.getenv("LLM_MODEL", "")
        per_dir = Path("eval_results/cases/rec")
        per_dir.mkdir(parents=True, exist_ok=True)
        per_csv = per_dir / f"{short_model_name(model_name)}_rec_{tag}_{stamp}.csv"
        pd.DataFrame(per_item).to_csv(per_csv, index=False)
        _log(f"  [per-item] {len(per_item)} cases → {per_csv}  (model={model_name})")

    return {
        "hit@1": hit_k1 / denom,
        "hit@3": hit_k3 / denom,
        "cat_hit@3": cat_hit / denom,
        "no_candidate_rate": no_candidate / denom,
        "n": total,
        "elapsed_sec": elapsed,
        "_summary_lines": lines,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", type=str, default=None,
                    help="저장 라벨. 지정 시 eval_results/summary/rec/{model}_rec_{tag}_{stamp}.{json,txt} 저장.")
    ap.add_argument("--model", type=str, default=None,
                    help="결과 메타에 기록할 모델명 (미지정 시 env LLM_MODEL)")
    ap.add_argument("--no-rerank", action="store_true",
                    help="LLM rerank preview/보강 없이 score 기반 추천만 사용")
    args = ap.parse_args()

    result = eval_recommendation(limit=args.limit, tag=args.tag, use_rerank=not args.no_rerank)
    if args.tag:
        summary_lines = result.pop("_summary_lines", [])
        json_path, txt_path = save_eval_result(
            kind="rec",
            tag=args.tag,
            limit=args.limit,
            payload=result,
            summary_lines=summary_lines,
            model=args.model,
        )
        print(f"\n[saved] {json_path}")
        print(f"[saved] {txt_path}")
