"""Qwen3 단일 호출 스모크 테스트.

실행: python -m scripts.smoke_qwen_slots
요구: GPU + Qwen3-8B 모델 캐시 (bnb 4bit)
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from app.agents.preference_agent import analyze_user_turn

SAMPLES: list[tuple[list[dict], dict, str]] = [
    ([], {}, "오늘 친구 생일이라 기분 좋고, 달달하고 상큼한 거 마시고 싶어. 위스키는 별로야."),
    ([], {}, "혼자 조용히 마실 건데 너무 독한 건 싫고 허브 향 나는 걸로 부탁해."),
    (
        [{"speaker_role": "USER", "utterance_text": "데이트 중이야"}],
        {"party_purpose": "date"},
        "진한 여운 있는 게 좋고, 럼이나 테킬라는 피해줘.",
    ),
    ([], {}, "무난하게 중간 정도 도수로, 과일향 나는 거 추천해줘."),
    ([], {}, "이제 추천해줘 바로."),
]


def main() -> None:
    for i, (history, slots, user_msg) in enumerate(SAMPLES, 1):
        print(f"\n===== 샘플 {i} =====")
        print(f"user_msg: {user_msg}")
        print(f"existing slots: {json.dumps(slots, ensure_ascii=False)}")
        t0 = time.perf_counter()
        result = analyze_user_turn(history, slots, user_msg)
        dt = time.perf_counter() - t0
        print(f"source={result['source']} elapsed={dt:.2f}s")
        print(f"extracted: {json.dumps(result['extracted_slots'], ensure_ascii=False)}")
        print(f"should_stop={result['should_stop']} stop_reason={result['stop_reason']!r}")
        print(f"next_question: {result['next_question']}")


if __name__ == "__main__":
    main()
