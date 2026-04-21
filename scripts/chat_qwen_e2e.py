"""Qwen 슬롯 추출 + RAG 추천 end-to-end 인터랙티브 테스트."""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.preference_agent import (
    analyze_user_turn,
    generate_opening_question,
    merge_slots,
    should_move_to_recommendation,
    _calc_effective_completion,
)
from app.agents.orchestration_agent import (
    synthesize_query,
    retrieve_candidates,
    rerank_with_qwen,
    score_cocktail,
    _has_disliked_base,
    _is_unstockable,
)
from app.db.database import SessionLocal
from app.db.crud import get_all_recipes_with_ingredients, get_available_ingredient_ids

MAX_USER_TURNS = 7
RAG_TOP_K = 20


class MockVector:
    sweetness_score = Decimal("3.0")
    bitterness_score = Decimal("3.0")
    sourness_score = Decimal("3.0")
    freshness_score = Decimal("3.0")
    body_score = Decimal("3.0")
    herbal_score = Decimal("2.0")
    citrus_score = Decimal("2.0")
    alcohol_score = Decimal("3.0")


def run_recommendation(slots: dict) -> None:
    profile = {"merged_slots": slots, "vector": MockVector(), "space": None}
    query = synthesize_query(profile)

    print("\n=== 생성된 RAG 쿼리 ===")
    print(query)

    db = SessionLocal()
    try:
        all_ri = get_all_recipes_with_ingredients(db)
        available_ids = get_available_ingredient_ids(db)

        retrieved = retrieve_candidates(
            db, query, top_k=RAG_TOP_K,
            strength_preference=slots.get("strength_preference"),
        )
        print(f"\n=== RAG 검색 top {len(retrieved)} (하드필터 전) ===")
        for c, dist in retrieved[:10]:
            print(f"  {c.name_kr:25s}  {c.category:15s}  dist={dist:.3f}")

        survivors = []
        for cocktail, dist in retrieved:
            ri = all_ri.get(cocktail.cocktail_id, [])
            if _has_disliked_base(slots, ri):
                continue
            if _is_unstockable(ri, available_ids):
                continue
            survivors.append((cocktail, dist))

        print(f"\n=== 하드필터 후 생존 {len(survivors)}개 ===")

        if not survivors:
            print("추천 불가 — 후보 0")
            return

        survivor_cocktails = [c for c, _ in survivors]
        reranked = rerank_with_qwen(profile, survivor_cocktails, k=3)

        if reranked:
            id_to_c = {c.cocktail_id: c for c in survivor_cocktails}
            print("\n=== Qwen rerank TOP 3 ===")
            for i, item in enumerate(reranked[:3], 1):
                c = id_to_c.get(item["cocktail_id"])
                if c is None:
                    continue
                print(f"  [{i}] {c.name_kr} ({c.category})")
                print(f"      이유: {item.get('reason', '')}")
        else:
            print("\n(rerank 실패 → score_cocktail fallback)")
            scored = []
            for c, _d in survivors:
                ri = all_ri.get(c.cocktail_id, [])
                scored.append((c, score_cocktail(c, profile, ri)))
            scored.sort(key=lambda x: x[1], reverse=True)
            for i, (c, s) in enumerate(scored[:3], 1):
                print(f"  [{i}] {c.name_kr} ({c.category})  score={s:.2f}")
    finally:
        db.close()


def main() -> None:
    history: list[dict] = []
    slots: dict = {}
    turn = 0

    print("대화 테스트 시작. 종료하려면 quit")
    print(f"(최대 {MAX_USER_TURNS}턴, LLM should_stop 판단 시 자동 종료 → 추천 실행)\n")

    first_q = generate_opening_question()
    print(f"LLM: {first_q}")
    history.append({"speaker_role": "LLM", "utterance_text": first_q})

    while turn < MAX_USER_TURNS:
        try:
            user_msg = input("\n너: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료")
            return
        if not user_msg:
            continue
        if user_msg.lower() in {"quit", "exit"}:
            print("종료")
            return

        turn += 1
        result = analyze_user_turn(history=history, slots=slots, user_msg=user_msg)
        extracted = result["extracted_slots"]
        slots = merge_slots(slots, extracted)
        history.append({"speaker_role": "USER", "utterance_text": user_msg})

        completion = _calc_effective_completion(slots)
        print(f"source: {result['source']}")
        print(f"extracted: {json.dumps(extracted, ensure_ascii=False)}")
        print(f"slots: {json.dumps(slots, ensure_ascii=False)}")
        print(f"completion: {completion}%  turn: {turn}/{MAX_USER_TURNS}")

        proceed, reason = should_move_to_recommendation(
            merged_slots=slots,
            user_turn_count=turn,
            user_msg=user_msg,
            llm_should_stop=result["should_stop"],
            llm_stop_reason=result["stop_reason"],
        )
        if proceed:
            print(f"\n[TERMINATE] reason={reason}")
            break

        next_q = result["next_question"]
        print(f"\nLLM: {next_q}")
        history.append({"speaker_role": "LLM", "utterance_text": next_q})
    else:
        print(f"\n[TERMINATE] reason=turn_limit")

    print("\n=== 최종 슬롯 ===")
    print(json.dumps(slots, ensure_ascii=False, indent=2))
    print(f"최종 completion: {_calc_effective_completion(slots)}%")

    run_recommendation(slots)


if __name__ == "__main__":
    main()
