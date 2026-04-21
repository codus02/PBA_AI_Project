"""LLM 통합 정량 평가 — 슬롯/피드백/추천.

모델은 env LLM_MODEL 로 결정 (load_llm 가 이 값을 읽음).
EXAONE 등 다른 모델로 돌릴 때:
  python scripts/eval_all_llm.py --tag exaone --model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
결과는 eval_results/quantitative/{tag}_{timestamp}.json 에 저장.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(tag: str, model: str | None, limit: int | None = None):
    if model:
        os.environ["LLM_MODEL"] = model

    # env 설정 이후 import (config.py 가 읽히는 시점 이슈 방지)
    from scripts.eval_feedback import eval_feedback
    from scripts.eval_slots import run_eval as eval_slots_llm
    from scripts.eval_recommendation import eval_recommendation

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    print("=" * 60)
    print(f" LLM 통합 평가 | model={os.getenv('LLM_MODEL')} | tag={tag} | {stamp}")
    if limit:
        print(f" (eval limit={limit})")
    print("=" * 60)

    print("\n[1/3] 피드백 인텐트 분류")
    feedback_acc = eval_feedback(limit=limit)

    print("\n[2/3] 슬롯 추출")
    slot_summary = eval_slots_llm(limit=limit)

    print("\n[3/3] 추천 매칭 (RAG + rerank)")
    rec_result = eval_recommendation(limit=limit)

    out = {
        "tag": tag,
        "model": os.getenv("LLM_MODEL"),
        "timestamp": stamp,
        "limit": limit,
        "feedback_acc_pct": feedback_acc,
        "slots": slot_summary,
        "recommendation": rec_result,
    }
    out_dir = Path("eval_results/quantitative")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}_{stamp}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))

    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append(f" 최종 요약 — tag={tag}  model={os.getenv('LLM_MODEL')}  {stamp}")
    summary_lines.append("=" * 60)
    summary_lines.append(f"  피드백 분류 정확도        : {feedback_acc:.1f}%")
    summary_lines.append(f"  슬롯 scalar enum 평균     : {slot_summary['scalar_avg']:.1f}%")
    for k, v in slot_summary["scalar_per_slot"].items():
        summary_lines.append(f"    └ {k:22s}: {v:.1f}%")
    summary_lines.append(f"  taste_profile KV F1       : {slot_summary['taste_kv_f1']:.1f}")
    summary_lines.append(f"  aroma_profile KV F1       : {slot_summary['aroma_kv_f1']:.1f}")
    summary_lines.append(f"  disliked_bases F1         : {slot_summary['bases_f1']:.1f}")
    summary_lines.append(f"  favorite_drinks 검출률    : {slot_summary['favs_detected_pct']:.1f}%")
    summary_lines.append(f"  추천 Hit@1                : {rec_result['hit@1']*100:.1f}%")
    summary_lines.append(f"  추천 Hit@3                : {rec_result['hit@3']*100:.1f}%")
    summary_lines.append(f"  카테고리 Hit@3            : {rec_result['cat_hit@3']*100:.1f}%")
    summary_lines.append(f"  후보 없음 비율            : {rec_result['no_candidate_rate']*100:.1f}%")
    summary = "\n".join(summary_lines)

    txt_path = out_dir / f"{tag}_{stamp}.txt"
    txt_path.write_text(summary + "\n")

    print("\n" + summary)
    print(f"\n저장: {path}")
    print(f"      {txt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exaone", help="결과 파일 접미사 (예: qwen, exaone)")
    ap.add_argument("--model", default=None, help="HF model id (생략 시 env LLM_MODEL 사용)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(tag=args.tag, model=args.model, limit=args.limit)
