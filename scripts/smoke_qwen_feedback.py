"""Qwen analyze_feedback 스모크 테스트.

실행: python -m scripts.smoke_qwen_feedback
요구: GPU + Qwen3-8B (bnb 4bit)
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from app.agents.preference_agent import analyze_feedback

BEFORE_VEC = {
    "sweetness_score":  3.0,
    "bitterness_score": 3.0,
    "sourness_score":   3.0,
    "freshness_score":  3.0,
    "body_score":       3.0,
    "herbal_score":     2.0,
    "citrus_score":     2.0,
    "alcohol_score":    3.0,
}

# (label, expected_intent, feedback_text)
SAMPLES: list[tuple[str, str, str]] = [
    ("ACCEPT-기본",     "ACCEPT", "좋아 이걸로 갈게"),
    ("ACCEPT-우회",     "ACCEPT", "지금 결과는 딱 내가 원하던 느낌이라 이대로 받아도 될 것 같아"),
    ("ACCEPT-우회2",    "ACCEPT", "굳이 더 손볼 필요는 없어 보여서 이대로 마무리해도 괜찮을 것 같아"),
    ("ADJUST-단일축",   "ADJUST", "조금만 더 달게 해줘"),
    ("ADJUST-강조",     "ADJUST", "훨씬 강하게 가줘"),
    ("ADJUST-복합",     "ADJUST", "덜 시고 좀 더 묵직하게"),
    ("ADJUST-향",       "ADJUST", "허브향 좀 빼고 시트러스 살려줘"),
    ("ADJUST-감각어",   "ADJUST", "쨍하고 상큼한 느낌으로 가줘"),
    ("ADJUST-약하게",   "ADJUST", "조금 가볍게 부탁해"),
    ("REJECT-직설",     "REJECT", "별로야 다른 거 줘"),
    ("REJECT-우회",     "REJECT", "이건 내 취향이랑 너무 달라 처음부터 다시"),
]


def main() -> None:
    ok = 0
    for i, (label, expected, text) in enumerate(SAMPLES, 1):
        print(f"\n===== 샘플 {i} | {label} =====")
        print(f"  expected intent: {expected}")
        print(f"  text           : {text}")
        t0 = time.perf_counter()
        result = analyze_feedback(BEFORE_VEC, text)
        dt = time.perf_counter() - t0
        match = "✓" if result["intent"] == expected else "✗"
        if result["intent"] == expected:
            ok += 1
        print(f"  → intent       : {result['intent']} {match}")
        print(f"  → deltas       : {json.dumps(result['deltas'], ensure_ascii=False)}")
        if result["deltas"]:
            changed = {k: v for k, v in result["updated_vec"].items()
                       if v != BEFORE_VEC.get(k)}
            print(f"  → updated_vec  : {json.dumps(changed, ensure_ascii=False)}")
        print(f"  raw            : {result['raw'][:200]}")
        print(f"  elapsed        : {dt:.2f}s")

    print(f"\n[총평] intent 일치 {ok}/{len(SAMPLES)} = {ok/len(SAMPLES)*100:.1f}%")


if __name__ == "__main__":
    main()
