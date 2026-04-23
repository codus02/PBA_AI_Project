"""LLM 통합 정량 평가 — 슬롯 / 피드백 / 추천.

모델은 `--model` 로 지정 (생략 시 env LLM_MODEL). 세 평가를 같은 모델로 순차 실행.

  # EXAONE (기본, env LLM_MODEL 로 이미 세팅돼 있을 때)
  python -m scripts.eval_all_llm --tag exaone_v2 --limit 100

  # Qwen — 같은 코드/데이터로 비교
  python -m scripts.eval_all_llm \\
      --tag qwen_v2 \\
      --model Qwen/Qwen3-8B \\
      --limit 100

저장:
  - 각 단계별: eval_results/summary/{kind}/{model}_{kind}_{tag}_{stamp}.{json,txt}
    (feedback/slots/rec 개별 — _eval_save 헬퍼가 씀)
  - 통합 요약: eval_results/summary/all/{model}_all_{tag}_{stamp}.{json,txt}
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
    # 모델 env 주입 — eval 함수 import 전에 해야 함 (LLM 로더가 이 값을 읽기 때문)
    if model:
        os.environ["LLM_MODEL"] = model

    from scripts.eval_feedback import eval_feedback
    from scripts.eval_slots import run_eval as eval_slots_llm
    from scripts.eval_recommendation import eval_recommendation
    from scripts._eval_save import save_eval_result, short_model_name

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_name = os.getenv("LLM_MODEL", "")

    banner = [
        "=" * 60,
        f" LLM 통합 평가 | model={model_name} | tag={tag} | {stamp}",
        f" limit={limit if limit is not None else 'full'}",
        "=" * 60,
    ]
    print("\n".join(banner))

    # ---------------- 1/3 피드백 ----------------
    print("\n[1/3] 피드백 인텐트 분류")
    feedback = eval_feedback(limit=limit, tag=tag)
    fb_lines = feedback.pop("_summary_lines", [])
    save_eval_result(
        kind="feedback", tag=tag, limit=limit,
        payload=feedback, summary_lines=fb_lines, model=model_name,
    )

    # ---------------- 2/3 슬롯 ----------------
    print("\n[2/3] 슬롯 추출")
    slots = eval_slots_llm(limit=limit, tag=tag)
    sl_lines = slots.pop("_summary_lines", [])
    save_eval_result(
        kind="slots", tag=tag, limit=limit,
        payload=slots, summary_lines=sl_lines, model=model_name,
    )

    # ---------------- 3/3 추천 ----------------
    print("\n[3/3] 추천 매칭 (RAG + rerank)")
    rec = eval_recommendation(limit=limit, tag=tag)
    rc_lines = rec.pop("_summary_lines", [])
    save_eval_result(
        kind="rec", tag=tag, limit=limit,
        payload=rec, summary_lines=rc_lines, model=model_name,
    )

    # ---------------- 통합 요약 ----------------
    summary_lines = [
        "=" * 60,
        f" 최종 요약 — tag={tag}  model={model_name}  {stamp}",
        f" limit={limit if limit is not None else 'full'}",
        "=" * 60,
        f"  피드백 전체 정확도        : {feedback['accuracy_pct']:.1f}%",
        f"    └ ADJUST 델타 검출률    : {feedback['adjust_delta_rate_pct']:.1f}%",
        f"  슬롯 scalar enum 평균     : {slots['scalar_avg']:.1f}%",
    ]
    for k, v in slots["scalar_per_slot"].items():
        summary_lines.append(f"    └ {k:22s}: {v:.1f}%")
    summary_lines += [
        f"  taste_profile KV F1       : {slots['taste_kv_f1']:.1f}",
        f"  aroma_profile KV F1       : {slots['aroma_kv_f1']:.1f}",
        f"  disliked_bases F1         : {slots['bases_f1']:.1f}",
        f"  favorite_drinks 검출률    : {slots['favs_detected_pct']:.1f}%",
        f"  추천 Hit@1                : {rec['hit@1']*100:.1f}%",
        f"  추천 Hit@3                : {rec['hit@3']*100:.1f}%",
        f"  카테고리 Hit@3            : {rec['cat_hit@3']*100:.1f}%",
        f"  후보 없음 비율            : {rec['no_candidate_rate']*100:.1f}%",
    ]
    summary = "\n".join(summary_lines)
    print("\n" + summary)

    out_dir = Path("eval_results/summary/all")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_short = short_model_name(model_name)

    all_payload = {
        "tag": tag,
        "model": model_name,
        "timestamp": stamp,
        "limit": limit,
        "feedback": feedback,
        "slots": slots,
        "recommendation": rec,
    }
    base = f"{model_short}_all_{tag}_{stamp}"
    all_json = out_dir / f"{base}.json"
    all_txt = out_dir / f"{base}.txt"
    all_json.write_text(json.dumps(all_payload, ensure_ascii=False, indent=2, default=str))
    all_txt.write_text(summary + "\n")
    print(f"\n저장: {all_json}")
    print(f"      {all_txt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="결과 파일 접미사 (예: exaone_v2, qwen_v2)")
    ap.add_argument("--model", default=None,
                    help="HF model id. 생략 시 env LLM_MODEL 사용")
    ap.add_argument("--limit", type=int, default=None,
                    help="각 eval 당 처음 N건만 평가. 생략 시 전체(500건)")
    args = ap.parse_args()
    main(tag=args.tag, model=args.model, limit=args.limit)
