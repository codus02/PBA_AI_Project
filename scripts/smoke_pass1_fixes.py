"""Pass 1 (슬롯 추출) 핀포인트 스모크 — 최근 고친 4개 버그만 빠르게 검증.

chat_llm_full.py 로그에서 드러난 bug 재현 발화를 Pass 1 + validator 까지만
돌려서 실제로 원하는 slot 이 뽑히는지 본다. Pass 2 (bartender reply) 는 건너뛰므로
GPU 부담·시간 절반 수준.

실행:
  python -m scripts.smoke_pass1_fixes

환경변수:
  LLM_MODEL 로 모델 스왑 가능 (기본: config.py 값).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.preference_agent import (
    _extract_slots_llm,
    validate_extracted_slots,
    _apply_rule_based_slot_guards,
)


# (라벨, 직전 LLM 질문, 사용자 발화, 기대 포함 키, 기대 불포함 키)
CASES = [
    (
        "bug1_sour_not_citrus",
        "신맛 강도는 어느 정도가 좋으세요?",
        "신맛은 강하게 부탁해",
        {"taste_profile": {"sour": "high"}},
        {"aroma_profile": ["citrus"]},  # citrus가 들어가면 실패
    ),
    (
        "bug2_multi_axis",
        "향은 어떤 느낌이면 좋으실까요?",
        "과일향 적당하게 해줘. 우유향은 피하고싶어",
        {"aroma_profile": {"fruity": "medium"}, "taste_profile": {"creamy": "zero"}},
        {},
    ),
    (
        "bug3_milk_to_creamy",
        "혹시 싫어하거나 빼고 싶은 맛이나 향 있으세요?",
        "우유향은 피하고싶다고",
        {"taste_profile": {"creamy": "zero"}},
        {"disliked_bases": ["milk", "cream", "dairy"]},
    ),
    (
        "multi_axis_sweet_sour_creamy",
        "맛은 어떤 걸 좋아하세요?",
        "상큼 달달한게 좋아. 너무 과하진 않게. 우유맛은 싫어",
        {"taste_profile": {"sweet": "medium", "sour": "medium", "creamy": "low"}},
        {},
    ),
    (
        "sour_vs_citrus_aroma",
        "어떤 향이 좋으세요?",
        "레몬향 강한게 좋아",
        {"aroma_profile": {"citrus": "high"}},
        {"taste_profile": ["sour"]},
    ),
]


def _check_includes(extracted: dict, expect: dict) -> list[str]:
    """expect 내용이 extracted 에 다 들어있는지 확인. 없으면 문제 리스트 반환."""
    problems: list[str] = []
    for k, v in expect.items():
        got = extracted.get(k)
        if isinstance(v, dict):
            if not isinstance(got, dict):
                problems.append(f"{k} missing (got {got!r})")
                continue
            for sk, sv in v.items():
                if got.get(sk) != sv:
                    problems.append(f"{k}.{sk} expected {sv!r}, got {got.get(sk)!r}")
        else:
            if got != v:
                problems.append(f"{k} expected {v!r}, got {got!r}")
    return problems


def _check_excludes(extracted: dict, forbid: dict) -> list[str]:
    """forbid 에 들어있는 키는 절대 extracted 에 없어야 함 (list 면 그 서브키)."""
    problems: list[str] = []
    for k, v in forbid.items():
        got = extracted.get(k)
        if v == [] or v is None:
            if got:
                problems.append(f"{k} should be empty, got {got!r}")
            continue
        if isinstance(v, list) and isinstance(got, dict):
            hit = [sk for sk in v if sk in got]
            if hit:
                problems.append(f"{k} contains forbidden subkeys {hit} (got {got!r})")
        elif isinstance(v, list) and isinstance(got, list):
            hit = [x for x in got if x in v]
            if hit:
                problems.append(f"{k} contains forbidden items {hit}")
    return problems


def main() -> None:
    pass_count = 0
    fail_count = 0
    results: list[dict] = []

    for label, last_q, user_msg, expect_include, expect_exclude in CASES:
        history = [{"speaker_role": "LLM", "utterance_text": last_q}] if last_q else []
        print("\n" + "=" * 72)
        print(f"[{label}]")
        print(f"  직전 LLM: {last_q}")
        print(f"  USER    : {user_msg}")

        extracted_raw, raw_text = _extract_slots_llm(history, user_msg)
        validated = validate_extracted_slots(extracted_raw)
        guarded = _apply_rule_based_slot_guards(history, user_msg, dict(validated))

        print(f"  raw LLM : {json.dumps(extracted_raw, ensure_ascii=False)}")
        print(f"  validated: {json.dumps(validated, ensure_ascii=False)}")
        if guarded != validated:
            print(f"  guarded : {json.dumps(guarded, ensure_ascii=False)}")

        problems = _check_includes(guarded, expect_include) + _check_excludes(guarded, expect_exclude)
        if problems:
            fail_count += 1
            print(f"  ✗ FAIL:")
            for p in problems:
                print(f"     - {p}")
        else:
            pass_count += 1
            print(f"  ✓ PASS")

        results.append({
            "label": label,
            "user_msg": user_msg,
            "raw": extracted_raw,
            "validated": validated,
            "guarded": guarded,
            "problems": problems,
        })

    print("\n" + "=" * 72)
    print(f"  결과: PASS {pass_count} / FAIL {fail_count} / 총 {len(CASES)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
