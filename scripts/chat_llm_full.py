"""LLM 전체 플로우 메모리-only 시뮬레이터.

초기 태그는 DB InitialTagResponse 스키마와 동일 (familiarity / strength / tastes / aromas).
party_purpose, current_mood 는 초기 태그에 없으므로 대화 루프에서 채운다.

흐름:
  1) 초기 태그 입력 (familiarity, strength_tag, taste_tags, aroma_tags) → _seed_slots_from_initial_tags
  2) 슬롯 추출 대화 루프 (analyze_user_turn / should_move_to_recommendation)
     - 사용자가 "몰라/모르겠/딱히" → 해당 슬롯 1회 alt 각도로 재질문, 그래도 모르면 skip
     - disliked_bases는 사용자가 먼저 말하지 않으면 절대 묻지 않음
  3) 1차 RAG+LLM 추천 top3
  4) 사용자가 하나 고르고 피드백
  5) analyze_feedback → intent+deltas → 벡터 갱신 → 재추천 (이전 제외)
  6) ACCEPT / 최대 3 라운드 종료
  7) 만족도 1~5 (메모리 only, DB 저장 X)
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.preference_agent import (
    analyze_user_turn,
    analyze_feedback,
    generate_opening_question,
    merge_slots,
    should_move_to_recommendation,
    _calc_effective_completion,
    _seed_slots_from_initial_tags,
    SLOT_ASK_ORDER,
)
from app.agents.orchestration_agent import (
    synthesize_query,
    retrieve_candidates,
    rerank_with_llm,
    score_cocktail,
    _has_disliked_base,
    _is_unstockable,
    _has_zero_taste_conflict,
    _build_reason_parts,
)
from app.db.database import SessionLocal
from app.db.crud import get_all_recipes_with_ingredients, get_available_ingredient_ids

MAX_USER_TURNS = 10
RAG_TOP_K = 20
MAX_FEEDBACK_ROUNDS = 3

# 초기 태그 선택지 (DB 컬럼값과 _seed_slots_from_initial_tags 매핑에 맞춘 한국어)
FAMILIARITY_CHOICES = {
    "1": "처음",
    "2": "가끔",
    "3": "자주",
}
STRENGTH_CHOICES = {
    "1": "무알콜",
    "2": "약함",
    "3": "중간",
    "4": "강함",
}
TASTE_TAG_CHOICES = {
    "1": "단맛",
    "2": "신맛",
    "3": "쓴맛",
    "4": "청량함",
    "5": "바디감",
    "6": "크리미함",
}
AROMA_TAG_CHOICES = {
    "1": "과일향",
    "2": "허브향",
    "3": "민트향",
    "4": "시트러스향",
    "5": "우디향",
    "6": "커피향",
    "7": "꽃향",
}

UNKNOWN_PATTERNS = ["몰라", "모르겠", "모름", "딱히", "잘 모르", "잘모르", "글쎄"]

DEFAULT_VEC = {
    "sweetness_score": 3.0,
    "bitterness_score": 3.0,
    "sourness_score": 3.0,
    "freshness_score": 3.0,
    "body_score": 3.0,
    "herbal_score": 2.0,
    "citrus_score": 2.0,
    "alcohol_score": 3.0,
}


def vec_to_profile_obj(vec: dict) -> SimpleNamespace:
    return SimpleNamespace(**{k: Decimal(str(v)) for k, v in vec.items()})


def make_profile(slots: dict, vec: dict) -> dict:
    return {"merged_slots": slots, "vector": vec_to_profile_obj(vec), "space": None}


def _pick_one(prompt: str, choices: dict, required: bool = False) -> str | None:
    while True:
        print(f"\n{prompt}")
        for k, v in choices.items():
            print(f"  {k}) {v}")
        suffix = "" if required else " (엔터=건너뛰기)"
        raw = input(f"번호 입력{suffix}: ").strip()
        if not raw and not required:
            return None
        if raw in choices:
            return choices[raw]
        print("  (잘못된 입력)")


def _pick_multi(prompt: str, choices: dict) -> list[str]:
    print(f"\n{prompt} (콤마로 여러 개, 엔터=건너뛰기)")
    for k, v in choices.items():
        print(f"  {k}) {v}")
    raw = input("번호들: ").strip()
    if not raw:
        return []
    result = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok in choices and choices[tok] not in result:
            result.append(choices[tok])
    return result


def ask_initial_tags() -> SimpleNamespace:
    print("=== 초기 태그 입력 (DB InitialTagResponse 스키마) ===")
    familiarity = _pick_one("[친숙도] 칵테일 얼마나 드셔보셨어요?", FAMILIARITY_CHOICES)
    strength = _pick_one("[선호 도수]", STRENGTH_CHOICES, required=True)
    tastes = _pick_multi("[선호 맛 태그]", TASTE_TAG_CHOICES)
    aromas = _pick_multi("[선호 향 태그]", AROMA_TAG_CHOICES)

    return SimpleNamespace(
        familiarity_tag=familiarity,
        strength_tag=strength,
        taste_tags_json=tastes,
        aroma_tags_json=aromas,
    )


def _is_unknown_reply(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(p in t for p in UNKNOWN_PATTERNS)


def _find_asked_slot(last_llm_msg: str) -> str | None:
    """직전 LLM 질문 텍스트에서 어떤 슬롯을 물었는지 추정."""
    if not last_llm_msg:
        return None
    m = last_llm_msg
    if any(k in m for k in ["자리", "모임", "데이트", "혼자"]):
        return "party_purpose"
    if any(k in m for k in ["기분", "다운"]):
        return "current_mood"
    if any(k in m for k in ["도수", "가볍게", "확 가", "맥주", "소주"]):
        return "strength_preference"
    if any(k in m for k in ["맛", "달달", "상큼", "쌉싸름", "과일 주스", "에이드"]):
        return "taste_profile"
    if any(k in m for k in ["향", "향수", "냄새", "민트", "우디", "바닐라"]):
        return "aroma_profile"
    if any(k in m for k in ["자주 드시", "즐겨", "평소", "음료"]):
        return "favorite_drinks"
    return None


def dialogue_loop(initial_slots: dict, familiarity: str | None = None) -> dict:
    history: list[dict] = []
    slots = dict(initial_slots)
    turn = 0

    skip_slots: set[str] = set()
    alt_slots: set[str] = set()
    giveup_count: dict[str, int] = {}
    last_asked_slot: str | None = None

    first_q = generate_opening_question()
    print(f"\nLLM: {first_q}")
    history.append({"speaker_role": "LLM", "utterance_text": first_q})

    while turn < MAX_USER_TURNS:
        try:
            user_msg = input("\n너: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료")
            sys.exit(0)
        if not user_msg:
            continue
        if user_msg.lower() in {"quit", "exit"}:
            sys.exit(0)

        turn += 1

        # 직전 질문 슬롯에 대해 "모르겠어"면 해당 슬롯의 giveup 카운트 증가
        if last_asked_slot and _is_unknown_reply(user_msg):
            giveup_count[last_asked_slot] = giveup_count.get(last_asked_slot, 0) + 1
            if giveup_count[last_asked_slot] == 1:
                alt_slots.add(last_asked_slot)
            else:
                skip_slots.add(last_asked_slot)
                alt_slots.discard(last_asked_slot)

        result = analyze_user_turn(
            history=history,
            slots=slots,
            user_msg=user_msg,
            familiarity=familiarity,
            user_turn_count=turn,
        )
        extracted = result["extracted_slots"]
        slots = merge_slots(slots, extracted)
        history.append({"speaker_role": "USER", "utterance_text": user_msg})

        comp = _calc_effective_completion(slots)
        print(f"  source={result['source']}  intent={result.get('user_intent','?')}  action={result['action']}  completion={comp}%  turn={turn}/{MAX_USER_TURNS}")
        print(f"  llm_emitted: {json.dumps(result.get('extracted_raw', {}), ensure_ascii=False)}")
        print(f"  kept_after_diff: {json.dumps(extracted, ensure_ascii=False)}")
        print(f"  slots: {json.dumps(slots, ensure_ascii=False)}")

        reply = result["reply"]
        print(f"\nLLM: {reply}")
        history.append({"speaker_role": "LLM", "utterance_text": reply})

        proceed, reason = should_move_to_recommendation(
            merged_slots=slots,
            user_turn_count=turn,
            user_msg=user_msg,
            llm_should_stop=result["should_stop"],
            llm_stop_reason=result["stop_reason"],
        )
        if proceed:
            print(f"\n[TERMINATE] {reason}")
            break
        last_asked_slot = _find_asked_slot(reply)
    else:
        print("\n[TERMINATE] turn_limit")

    return slots


def recommend_once(
    db, slots: dict, vec: dict, all_ri, available_ids, exclude_ids: list[int]
) -> list[dict]:
    profile = make_profile(slots, vec)
    query = synthesize_query(profile)
    print("\n=== RAG 쿼리 ===")
    print(query)

    retrieved = retrieve_candidates(
        db,
        query,
        top_k=RAG_TOP_K,
        exclude_ids=exclude_ids or None,
        strength_preference=slots.get("strength_preference"),
    )

    survivors = []
    for c, dist in retrieved:
        ri = all_ri.get(c.cocktail_id, [])
        if _has_disliked_base(slots, ri):
            continue
        if _is_unstockable(ri, available_ids):
            continue
        if _has_zero_taste_conflict(slots, c):
            continue
        survivors.append((c, dist))

    print(f"하드필터 후 생존 {len(survivors)}개")
    if not survivors:
        return []

    survivor_cocktails = [c for c, _ in survivors]
    reranked = rerank_with_llm(profile, survivor_cocktails, k=3)

    results: list[dict] = []
    if reranked:
        id_to_c = {c.cocktail_id: c for c in survivor_cocktails}
        for item in reranked[:3]:
            c = id_to_c.get(item["cocktail_id"])
            if c is None:
                continue
            results.append({
                "cocktail_id": c.cocktail_id,
                "name_kr": c.name_kr,
                "category": c.category,
                "reason": item.get("reason", ""),
                "source": "rag_llm",
            })

    if not results:
        scored = []
        for c, _d in survivors:
            ri = all_ri.get(c.cocktail_id, [])
            scored.append((c, score_cocktail(c, profile, ri), ri))
        scored.sort(key=lambda x: x[1], reverse=True)
        for c, s, ri in scored[:3]:
            reason_parts = _build_reason_parts(c, profile, ri)
            results.append({
                "cocktail_id": c.cocktail_id,
                "name_kr": c.name_kr,
                "category": c.category,
                "reason": " · ".join(reason_parts),
                "score": s,
                "source": "rag_fallback",
            })

    print("\n=== 추천 TOP3 ===")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['name_kr']} ({r['category']})  ← {r['source']}")
        if r["reason"]:
            print(f"      이유: {r['reason']}")
    return results


def _apply_feedback_on_drink(
    vec: dict, picked: dict,
) -> tuple[str, dict, dict | None]:
    """이미 선택된 picked 에 대해 피드백만 받아서 분석. ACCEPT/ADJUST/REJECT 반환."""
    fb = input(f"피드백 (예: 좋아 이걸로 / 좀 달아 / 별로야 다른거 줘): ").strip()
    if not fb:
        print("(피드백 비어있음 → ACCEPT 취급)")
        return "ACCEPT", vec, picked

    res = analyze_feedback(before_vec=vec, feedback_text=fb)
    intent = res["intent"]
    deltas = res["deltas"]
    new_vec = res["updated_vec"] if res["updated_vec"] else dict(vec)

    print(f"\n[피드백 분석] intent={intent}")
    if deltas:
        print(f"  deltas: {json.dumps(deltas, ensure_ascii=False)}")
    print(f"  vec: {json.dumps({k: round(v,2) for k,v in new_vec.items()}, ensure_ascii=False)}")

    if intent == "ADJUST":
        msg = _describe_adjustments(deltas)
        print(f"\n🛠  {picked['name_kr']} 를 {msg} 다시 제조해드릴게요.")

    return intent, new_vec, picked


def feedback_round(
    db, slots: dict, vec: dict, all_ri, available_ids,
    top3: list[dict], excluded: list[int],
    preselected: dict | None = None,
) -> tuple[str, dict, list[int], list[dict], dict | None]:
    """preselected 가 있으면 그 음료에 대한 피드백만 받는다 (ADJUST 연속 라운드)."""
    if preselected is not None:
        picked = preselected
        print(f"→ 같은 음료({picked['name_kr']}) 재제조 후 피드백")
    else:
        picked, _ = pick_and_feedback_pick_only(top3)
        if picked is None:
            return "SKIP", vec, excluded, top3, None

    intent, new_vec, picked = _apply_feedback_on_drink(vec, picked)

    if intent == "ACCEPT":
        excluded = list(set(excluded + [picked["cocktail_id"]]))
        return "ACCEPT", new_vec, excluded, top3, picked

    if intent == "ADJUST":
        # 같은 음료를 조정 — top3/excluded 유지, picked 유지
        return "ADJUST", new_vec, excluded, top3, picked

    # REJECT: 지금 음료 제외 + 새로 검색
    excluded = list(set(excluded + [picked["cocktail_id"]]))
    next_top3 = recommend_once(db, slots, new_vec, all_ri, available_ids, excluded)
    return intent, new_vec, excluded, next_top3, None


def pick_and_feedback_pick_only(top3: list[dict]) -> tuple[dict | None, str]:
    raw = input("\n시음할 번호 (1/2/3, 엔터=1번, skip=건너뛰기): ").strip().lower()
    if raw == "skip":
        return None, ""
    idx = 0
    if raw in {"1", "2", "3"}:
        idx = int(raw) - 1
    if idx >= len(top3):
        idx = 0
    picked = top3[idx]
    print(f"→ 선택: {picked['name_kr']}")
    return picked, ""


_AXIS_KO = {
    "sweetness_score": "단맛",
    "bitterness_score": "쓴맛",
    "sourness_score": "신맛",
    "freshness_score": "청량감",
    "body_score": "바디감",
    "herbal_score": "허브향",
    "citrus_score": "시트러스향",
    "alcohol_score": "도수",
}


def _describe_adjustments(deltas: dict) -> str:
    if not deltas:
        return "살짝 느낌만 다듬어서"
    parts = []
    for k, v in deltas.items():
        name = _AXIS_KO.get(k, k)
        mag = abs(v)
        strength = "살짝" if mag <= 0.3 else ("꽤" if mag <= 0.5 else "확")
        direction = "줄여서" if v < 0 else "올려서"
        parts.append(f"{name} {strength} {direction}")
    return ", ".join(parts)


def print_final_recommendation(db, picked: dict) -> None:
    from app.db.models import Cocktail
    c = db.query(Cocktail).filter(Cocktail.cocktail_id == picked["cocktail_id"]).first()
    print("\n" + "=" * 50)
    print("🍸  최종 추천")
    print("=" * 50)
    name = f"{picked['name_kr']} ({picked['category']})"
    if c and c.name_en:
        name = f"{picked['name_kr']} / {c.name_en} — {picked['category']}"
    print(f"\n  {name}\n")
    if picked.get("reason"):
        print(f"[추천 이유]\n  {picked['reason']}\n")
    if c and c.description:
        print(f"[칵테일 설명]\n  {c.description.strip()}\n")


def main() -> None:
    print("=== LLM 전체 플로우 시뮬레이터 (메모리 only) ===")
    tag_row = ask_initial_tags()
    seeded = _seed_slots_from_initial_tags(tag_row)
    print(f"\n초기 태그 seed 결과: {json.dumps(seeded, ensure_ascii=False)}")

    slots = dialogue_loop(seeded, familiarity=tag_row.familiarity_tag)
    print("\n=== 최종 슬롯 ===")
    print(json.dumps(slots, ensure_ascii=False, indent=2))

    db = SessionLocal()
    try:
        all_ri = get_all_recipes_with_ingredients(db)
        available_ids = get_available_ingredient_ids(db)

        vec = dict(DEFAULT_VEC)
        excluded: list[int] = []

        top3 = recommend_once(db, slots, vec, all_ri, available_ids, excluded)
        if not top3:
            print("\n추천 불가 — 후보 없음")
            return

        rounds = 0
        final_pick: dict | None = None
        preselected: dict | None = None
        while rounds < MAX_FEEDBACK_ROUNDS:
            rounds += 1
            print(f"\n--- 시음 라운드 {rounds}/{MAX_FEEDBACK_ROUNDS} ---")
            intent, vec, excluded, top3, picked = feedback_round(
                db, slots, vec, all_ri, available_ids, top3, excluded,
                preselected=preselected,
            )
            if intent == "ACCEPT":
                final_pick = picked
                break
            if intent == "SKIP":
                break
            if intent == "ADJUST":
                preselected = picked  # 같은 음료로 다음 라운드
                continue
            preselected = None  # REJECT → 새 top3 에서 다시 고르기
            if not top3:
                print("더 이상 후보 없음 — 종료")
                break

        if final_pick is not None:
            print_final_recommendation(db, final_pick)
    finally:
        db.close()

    try:
        sat_raw = input("\n최종 만족도 1~5 (엔터=건너뛰기): ").strip()
        if sat_raw:
            sat = max(1, min(5, int(sat_raw)))
            print(f"만족도: {sat}/5  (메모리 only — DB 저장 안 함)")
    except ValueError:
        print("만족도 파싱 실패 — 무시")


if __name__ == "__main__":
    main()
