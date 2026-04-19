"""Qwen 슬롯 추출 인터랙티브 테스트 (단일 LLM 호출 구조).

- 최대 7턴 제한
- LLM이 should_stop=true 반환하면 종료
- quit/exit 입력 시 종료
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.preference_agent import (
    analyze_user_turn,
    generate_opening_question,
    merge_slots,
    should_move_to_recommendation,
    _calc_effective_completion,
)

MAX_USER_TURNS = 7


def main() -> None:
    history: list[dict] = []
    slots: dict = {}
    turn = 0

    print("대화 테스트 시작. 종료하려면 quit 입력")
    print(f"(최대 {MAX_USER_TURNS}턴, LLM이 should_stop 판단 시 자동 종료)")

    first_q = generate_opening_question()
    print(f"\nLLM: {first_q}")
    history.append({"speaker_role": "LLM", "utterance_text": first_q})

    while turn < MAX_USER_TURNS:
        try:
            user_msg = input("\n너: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료")
            break
        if not user_msg:
            continue
        if user_msg.lower() in {"quit", "exit"}:
            print("종료")
            break

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
        print(f"llm.should_stop={result['should_stop']} stop_reason={result['stop_reason']!r}")

        proceed, reason = should_move_to_recommendation(
            merged_slots=slots,
            user_turn_count=turn,
            user_msg=user_msg,
            llm_should_stop=result["should_stop"],
            llm_stop_reason=result["stop_reason"],
        )
        if proceed:
            print(f"\n[TERMINATE] reason={reason} → 추천 단계로 이동")
            break

        next_q = result["next_question"]
        print(f"\nLLM: {next_q}")
        history.append({"speaker_role": "LLM", "utterance_text": next_q})
    else:
        print(f"\n[TERMINATE] reason=turn_limit ({MAX_USER_TURNS}턴 소진) → 추천 단계로 이동")

    print("\n=== 최종 슬롯 ===")
    print(json.dumps(slots, ensure_ascii=False, indent=2))
    print(f"최종 completion: {_calc_effective_completion(slots)}%")


if __name__ == "__main__":
    main()
