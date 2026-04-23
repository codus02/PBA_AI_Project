"""피드백 분류 평가 — LLM analyze_feedback 기반.

메트릭:
  - intent 정확도 (전체 + 클래스별)
  - ADJUST 케이스에서 vector_deltas 비어있지 않은 비율 (LLM이 실제로 축을 짚어내는지)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.agents.preference_agent import analyze_feedback
from scripts._eval_save import save_eval_result, short_model_name


def eval_feedback(limit: int | None = None, tag: str | None = None) -> dict:
    df = pd.read_csv("data/eval/feedback_eval_v2_500.csv")
    if limit:
        df = df.head(limit)

    correct = 0
    errors = []
    per_item: list[dict] = []
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
        ok = (pred == gold)
        if ok:
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

        per_item.append({
            "case_id": i,
            "text": row.feedback_text,
            "gold_intent": gold,
            "pred_intent": pred,
            "correct": ok,
            "deltas": json.dumps(result["deltas"], ensure_ascii=False),
        })

        if i % 25 == 0:
            dt = time.perf_counter() - t0
            print(f"  progress {i}/{n}  elapsed={dt:.1f}s")

    total = len(df)
    acc = correct / total * 100 if total else 0.0
    elapsed = time.perf_counter() - t0

    lines: list[str] = []

    def _log(msg: str = ""):
        print(msg)
        lines.append(msg)

    _log("\n" + "=" * 60)
    _log(f"FEEDBACK EVAL — {total} cases ({elapsed:.1f}s, {elapsed/max(total,1):.2f}s/case)")
    _log("=" * 60)

    per_class_pct: dict[str, float] = {}
    for intent, (c, t) in per_class.items():
        if t:
            pct = c / t * 100
            per_class_pct[intent] = pct
            _log(f"  {intent}: {c}/{t} = {pct:.1f}%")

    _log(f"\n[전체 intent 정확도] {correct}/{total} = {acc:.1f}%")
    adjust_delta_rate = (adjust_with_deltas / adjust_total * 100) if adjust_total else 0.0
    if adjust_total:
        _log(f"[ADJUST 델타 검출률] {adjust_with_deltas}/{adjust_total} = "
             f"{adjust_delta_rate:.1f}%  (LLM이 축을 짚어낸 비율)")

    _log("\n오답 샘플 (최대 10개):")
    for e in errors:
        _log(f"  '{e['text'][:60]}' → 예측:{e['got']} / 정답:{e['expected']} "
             f"deltas={e['deltas']}")

    if tag:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_name = os.getenv("LLM_MODEL", "")
        out_dir = Path("eval_results/cases/feedback")
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{short_model_name(model_name)}_feedback_{tag}_{stamp}.csv"
        pd.DataFrame(per_item).to_csv(csv_path, index=False)
        _log(f"\n[per-item] {len(per_item)} cases → {csv_path}  (model={model_name})")

    return {
        "accuracy_pct": acc,
        "correct": correct,
        "total": total,
        "elapsed_sec": elapsed,
        "per_class": {k: {"correct": v[0], "total": v[1]} for k, v in per_class.items()},
        "per_class_pct": per_class_pct,
        "adjust_total": adjust_total,
        "adjust_with_deltas": adjust_with_deltas,
        "adjust_delta_rate_pct": adjust_delta_rate,
        "error_samples": errors,
        "_summary_lines": lines,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", type=str, default=None,
                    help="저장 라벨. 지정 시 eval_results/summary/feedback/{model}_feedback_{tag}_{stamp}.{json,txt} 저장.")
    ap.add_argument("--model", type=str, default=None,
                    help="결과 메타에 기록할 모델명 (미지정 시 env LLM_MODEL)")
    args = ap.parse_args()

    result = eval_feedback(limit=args.limit, tag=args.tag)
    if args.tag:
        summary_lines = result.pop("_summary_lines", [])
        json_path, txt_path = save_eval_result(
            kind="feedback",
            tag=args.tag,
            limit=args.limit,
            payload=result,
            summary_lines=summary_lines,
            model=args.model,
        )
        print(f"\n[saved] {json_path}")
        print(f"[saved] {txt_path}")
