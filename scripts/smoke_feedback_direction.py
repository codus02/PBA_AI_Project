"""analyze_feedback 부호(방향) 검증 스모크.

ADJUST 문장에서 델타 '부호'가 프롬프트 예시대로 나오는지만 검사.
최종 추천 이유 서술을 붙이기 전 **차단 요건** — 부호가 반대로 나오면
나레이션이 사용자 피드백을 정반대로 설명하게 된다.

실행: python -m scripts.smoke_feedback_direction
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

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

# (label, feedback_text, axis, expected_sign)  expected_sign ∈ {"-", "+"}
SAMPLES: list[tuple[str, str, str, str]] = [
    # 단맛 — 감소 (코덱스 차단 요건)
    ("너무 달아",       "너무 달아",              "sweetness_score", "-"),
    ("좀 달아",         "좀 달아",                "sweetness_score", "-"),
    ("조금 달아",       "조금 달아",              "sweetness_score", "-"),
    ("덜 달게",         "덜 달게 해줘",           "sweetness_score", "-"),
    # 단맛 — 증가
    ("조금 더 달게",    "조금 더 달게 해줘",      "sweetness_score", "+"),
    ("더 달게",         "더 달게",                "sweetness_score", "+"),
    # 신맛
    ("너무 시다",       "너무 시다",              "sourness_score",  "-"),
    ("덜 시게",         "덜 시게",                "sourness_score",  "-"),
    ("더 상큼하게",     "더 상큼하게 가줘",       "freshness_score", "+"),
    # 도수
    ("너무 세다",       "너무 세다",              "alcohol_score",   "-"),
    ("더 강하게",       "더 강하게",              "alcohol_score",   "+"),
    ("좀 가볍게",       "조금 가볍게 부탁해",     "alcohol_score",   "-"),
]


def _sign(v: float) -> str:
    return "+" if v > 0 else ("-" if v < 0 else "0")


def main() -> int:
    ok = 0
    fail_rows: list[str] = []
    for i, (label, text, axis, expected) in enumerate(SAMPLES, 1):
        t0 = time.perf_counter()
        result = analyze_feedback(BEFORE_VEC, text)
        dt = time.perf_counter() - t0

        intent = result["intent"]
        deltas = result["deltas"] or {}
        delta_v = float(deltas.get(axis, 0.0))
        got_sign = _sign(delta_v)

        match = (intent == "ADJUST" and got_sign == expected)
        mark = "OK" if match else "FAIL"
        if match:
            ok += 1
        else:
            fail_rows.append(f"  - {label}: intent={intent} {axis}={delta_v} (expected {expected})")

        print(f"[{i:2d}/{len(SAMPLES)}] {mark:<4} {label:<14} | intent={intent:<7} "
              f"{axis}={delta_v:+.2f} (expect {expected}) {dt:.2f}s")
        if deltas:
            print(f"           all_deltas: {json.dumps(deltas, ensure_ascii=False)}")
        if result.get("raw"):
            print(f"           raw: {result['raw'][:160]!r}")

    print(f"\n[총평] 부호 일치 {ok}/{len(SAMPLES)} = {ok/len(SAMPLES)*100:.1f}%")
    if fail_rows:
        print("[실패 케이스]")
        print("\n".join(fail_rows))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
