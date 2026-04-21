"""피드백 분류 평가 — Qwen analyze_feedback 기반.

메트릭:
  - intent 정확도 (전체 + 클래스별)
  - ADJUST 케이스에서 vector_deltas 비어있지 않은 비율 (LLM이 실제로 축을 짚어내는지)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.agents.preference_agent import analyze_feedback


def eval_feedback(limit: int | None = None) -> float:
    df = pd.read_csv("data/eval/feedback_eval_v2_500.csv")
    if limit:
        df = df.head(limit)

    correct = 0
    errors = []
    adjust_with_deltas = 0
    adjust_total = 0
    per_class = {"ACCEPT": [0, 0], "ADJUST": [0, 0], "REJECT": [0, 0]}  # [correct, total]

    t0 = time.perf_counter()
    n = len(df)
    for i, row in enumerate(df.itertuples(index=False), 1):
        result = analyze_feedback({}, row.feedback_text)
        pred = result["intent"]
        gold = row.gold_intent

        per_class[gold][1] += 1
        if pred == gold:
            correct += 1
            per_class[gold][0] += 1
        else:
            if len(errors) < 10:
                errors.append({
                    "text": row.feedback_text,
                    "expected": gold,
                    "got": pred,
                    "deltas": result["deltas"],
                })

        if gold == "ADJUST":
            adjust_total += 1
            if result["deltas"]:
                adjust_with_deltas += 1

        if i % 25 == 0:
            dt = time.perf_counter() - t0
            print(f"  progress {i}/{n}  elapsed={dt:.1f}s")

    total = len(df)
    acc = correct / total * 100
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print(f"FEEDBACK EVAL — {total} cases ({elapsed:.1f}s, {elapsed/max(total,1):.2f}s/case)")
    print("=" * 60)

    for intent, (c, t) in per_class.items():
        if t:
            print(f"  {intent}: {c}/{t} = {c/t*100:.1f}%")

    print(f"\n[전체 intent 정확도] {correct}/{total} = {acc:.1f}%")
    if adjust_total:
        print(f"[ADJUST 델타 검출률] {adjust_with_deltas}/{adjust_total} = "
              f"{adjust_with_deltas/adjust_total*100:.1f}%  (LLM이 축을 짚어낸 비율)")

    print("\n오답 샘플 (최대 10개):")
    for e in errors:
        print(f"  '{e['text'][:60]}' → 예측:{e['got']} / 정답:{e['expected']} "
              f"deltas={e['deltas']}")

    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    eval_feedback(limit=args.limit)
